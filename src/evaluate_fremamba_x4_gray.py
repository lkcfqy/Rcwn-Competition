from __future__ import annotations

import argparse
import csv
import os
import sys
import types
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
import torch.nn.functional as F
from tqdm import tqdm

from dataset import load_pair, make_split
from metrics import EdgeMetric, measure_batch, metric_proxy


def _load_names(args: Any) -> list[str]:
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    return split.val[: args.limit] if args.limit else split.val


def _make_input(lr: torch.Tensor, channels: int) -> torch.Tensor:
    if channels == 1:
        return lr
    if channels == 3:
        return lr.repeat(1, 3, 1, 1)
    raise ValueError(f"Unsupported channel count: {channels}")


def _to_gray(pred: torch.Tensor, mode: str) -> torch.Tensor:
    if pred.shape[1] == 1:
        return pred
    if mode == "avg":
        return pred.mean(dim=1, keepdim=True)
    if mode == "y":
        weights = pred.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (pred[:, :3] * weights).sum(dim=1, keepdim=True)
    if mode == "r":
        return pred[:, 0:1]
    if mode == "g":
        return pred[:, 1:2]
    if mode == "b":
        return pred[:, 2:3]
    raise ValueError(f"Unsupported gray mode: {mode}")


def _downsample(pred: torch.Tensor, target_size: tuple[int, int], mode: str) -> torch.Tensor:
    if pred.shape[-2:] == target_size:
        return pred
    if mode == "area":
        return F.interpolate(pred, size=target_size, mode="area")
    return F.interpolate(pred, size=target_size, mode="bicubic", align_corners=False)


@torch.no_grad()
def validate(args: Any) -> None:
    external_dir = Path(args.model_dir).resolve()
    sys.path.insert(0, str(external_dir))
    thop_stub = types.ModuleType("thop")
    thop_stub.profile = lambda *a, **k: (0, 0)
    sys.modules.setdefault("thop", thop_stub)

    from fremamba import FreMamba

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = FreMamba(upscale=4, in_chans=args.channels).to(device).eval()
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    print(f"device={device} weights={args.weights}")

    names = _load_names(args)
    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    rows: list[dict[str, float | str]] = []
    sums: dict[str, float] = {}
    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(args.data, name, args.scale, device)
        pred = model(_make_input(lr, args.channels)).clamp(0.0, 1.0)
        pred = _to_gray(pred, args.gray_mode)
        pred = _downsample(pred, hr.shape[-2:], args.downsample)
        metrics = measure_batch(pred.float(), hr, edge_metric, lpips_fn)
        row = {"name": name, **metrics, "proxy": metric_proxy(metrics)}
        rows.append(row)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value

    means = {key: value / len(names) for key, value in sums.items()}
    means["proxy"] = metric_proxy(means)
    print("val_result " + " ".join(f"{key}={value:.5f}" for key, value in means.items()))

    if args.metrics_csv:
        Path(args.metrics_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_csv, "w", newline="", encoding="utf-8") as handle:
            fieldnames = ["name", "psnr", "ssim", "edge", "lpips", "proxy"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        print(f"metrics_csv={args.metrics_csv}")


def parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--model-dir", default="external_models/NTIRE2026_infraredSR/models/team06_KLETech-CEVI")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--channels", type=int, choices=[1, 3], default=3)
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--downsample", choices=["area", "bicubic"], default="area")
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
