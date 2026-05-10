from __future__ import annotations

import argparse
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
import torch.nn.functional as F
from spandrel import ModelLoader
from tqdm import tqdm

from dataset import load_pair, make_split
from metrics import EdgeMetric, measure_batch, metric_proxy


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


def make_input(lr_gray: torch.Tensor, channels: int) -> torch.Tensor:
    if channels == 1:
        return lr_gray
    if channels == 3:
        return lr_gray.repeat(1, 3, 1, 1)
    raise ValueError(f"Unsupported model input channels: {channels}")


def downsample_to_target(out: torch.Tensor, target_size: tuple[int, int], mode: str) -> torch.Tensor:
    if out.shape[-2:] == target_size:
        return out
    if mode == "area":
        return F.interpolate(out, size=target_size, mode="area")
    return F.interpolate(out, size=target_size, mode="bicubic", align_corners=False)


@torch.no_grad()
def validate(args: Any) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    names = split.val[: args.limit] if args.limit else split.val
    print(f"val={len(names)} target_scale=x{args.scale}")

    descriptor = ModelLoader(device=device).load_from_file(args.weights)
    descriptor.eval()
    arch_name = getattr(descriptor.architecture, "id", descriptor.architecture.__class__.__name__)
    n_params = sum(p.numel() for p in descriptor.model.parameters()) / 1e6
    print(
        f"params={n_params:.2f}M arch={arch_name} "
        f"model_scale=x{descriptor.scale} in_ch={descriptor.input_channels} out_ch={descriptor.output_channels}"
    )
    print(f"loaded={args.weights}")

    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    sums: dict[str, float] = {}
    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(args.data, name, args.scale, device)
        inp = make_input(lr, descriptor.input_channels)
        out = descriptor(inp)
        out = downsample_to_target(out, hr.shape[-2:], args.downsample)
        pred = rgb_to_gray(out, args.gray_mode)
        metrics = measure_batch(pred.float(), hr, edge_metric, lpips_fn)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value

    out_metrics = {key: value / len(names) for key, value in sums.items()}
    out_metrics["proxy"] = metric_proxy(out_metrics)
    print("val_result " + " ".join(f"{k}={v:.5f}" for k, v in out_metrics.items()))


def parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--downsample", choices=["bicubic", "area"], default="bicubic")
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
