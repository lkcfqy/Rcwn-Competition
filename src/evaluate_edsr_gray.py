from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any


def _prefer_torch_cuda_libs() -> None:
    base = "/usr/local/lib/python3.12/dist-packages"
    libs = [
        os.path.join(base, "torch", "lib"),
        os.path.join(base, "nvidia", "cublas", "lib"),
        os.path.join(base, "nvidia", "cuda_runtime", "lib"),
        os.path.join(base, "nvidia", "cudnn", "lib"),
    ]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    prefix = ":".join([p for p in libs if os.path.isdir(p)])
    if not prefix:
        return
    if os.environ.get("RCWN_CUDA_LIBS_OK") != "1":
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ":".join([prefix, existing])
        env["RCWN_CUDA_LIBS_OK"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_prefer_torch_cuda_libs()

import torch
from torch import nn
from tqdm import tqdm

from dataset import load_pair, make_split
from metrics import EdgeMetric, measure_batch, metric_proxy


class ResidualBlockNoBN(nn.Module):
    def __init__(self, channels: int, res_scale: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.relu(self.conv1(x)))
        return x + out * self.res_scale


class EDSR(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 64,
        num_blocks: int = 16,
        scale: int = 2,
        res_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError("This lightweight evaluator only supports x2 EDSR weights")
        self.conv_first = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.body = nn.Sequential(*[ResidualBlockNoBN(channels, res_scale) for _ in range(num_blocks)])
        self.conv_after_body = nn.Conv2d(channels, channels, 3, 1, 1)
        self.upsample = nn.Sequential(nn.Conv2d(channels, channels * scale * scale, 3, 1, 1), nn.PixelShuffle(scale))
        self.conv_last = nn.Conv2d(channels, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body = self.conv_after_body(self.body(feat))
        feat = feat + body
        return self.conv_last(self.upsample(feat))


def rgb_to_gray(x: torch.Tensor, mode: str) -> torch.Tensor:
    if x.shape[1] == 1:
        return x
    if mode == "avg":
        return x.mean(dim=1, keepdim=True)
    if mode == "y":
        weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (x * weights).sum(dim=1, keepdim=True)
    if mode == "r":
        return x[:, 0:1]
    if mode == "g":
        return x[:, 1:2]
    if mode == "b":
        return x[:, 2:3]
    raise ValueError(f"Unsupported gray mode: {mode}")


def infer_shape(state: dict[str, torch.Tensor]) -> tuple[int, int, int, int]:
    conv_first = state["conv_first.weight"]
    conv_last = state["conv_last.weight"]
    channels = conv_first.shape[0]
    in_channels = conv_first.shape[1]
    out_channels = conv_last.shape[0]
    block_ids = {
        int(key.split(".")[1])
        for key in state
        if key.startswith("body.") and key.endswith(".conv1.weight")
    }
    return in_channels, out_channels, channels, len(block_ids)


def load_state(path: str, prefer_ema: bool) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        return obj
    if prefer_ema and "params_ema" in obj:
        return obj["params_ema"]
    if "params" in obj:
        return obj["params"]
    if "state_dict" in obj:
        return obj["state_dict"]
    return obj


def read_names(args: Any) -> list[str]:
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    else:
        names = split.val
    return names[: args.limit] if args.limit else names


def write_metrics_csv(path: str, rows: list[dict[str, float | str]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "psnr", "ssim", "edge", "lpips", "proxy"])
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def validate(args: Any) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    names = read_names(args)
    print(f"device={device}")
    print(f"val={len(names)} scale=x{args.scale}")

    state = load_state(args.weights, args.prefer_ema)
    in_channels, out_channels, channels, num_blocks = infer_shape(state)
    model = EDSR(
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        num_blocks=num_blocks,
        scale=args.scale,
        res_scale=args.res_scale,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state mismatch missing={missing} unexpected={unexpected}")
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"params={n_params:.2f}M in_ch={in_channels} out_ch={out_channels} "
        f"channels={channels} blocks={num_blocks} res_scale={args.res_scale}"
    )
    print(f"loaded={args.weights}")

    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    sums: dict[str, float] = {}
    rows: list[dict[str, float | str]] = []
    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(args.data, name, args.scale, device)
        inp = lr.repeat(1, in_channels, 1, 1) if in_channels != 1 else lr
        pred = rgb_to_gray(model(inp), args.gray_mode)
        metrics = measure_batch(pred.float(), hr, edge_metric, lpips_fn)
        one: dict[str, float | str] = {"name": name, **metrics, "proxy": metric_proxy(metrics)}
        rows.append(one)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value

    out = {key: value / len(names) for key, value in sums.items()}
    out["proxy"] = metric_proxy(out)
    if args.metrics_csv:
        write_metrics_csv(args.metrics_csv, rows)
        print(f"metrics_csv={args.metrics_csv}")
    print("val_result " + " ".join(f"{k}={v:.5f}" for k, v in out.items()))


def parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--res-scale", type=float, default=1.0)
    parser.add_argument("--prefer-ema", action="store_true", default=True)
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names-file", default="")
    parser.add_argument("--metrics-csv", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
