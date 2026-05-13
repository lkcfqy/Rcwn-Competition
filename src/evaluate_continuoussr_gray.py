from __future__ import annotations

import argparse
import csv
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any


def _prefer_torch_cuda_libs() -> None:
    py312 = "/usr/local/lib/python3.12/dist-packages"
    libs = [
        os.path.join(py312, "torch", "lib"),
        os.path.join(py312, "nvidia", "cublas", "lib"),
        os.path.join(py312, "nvidia", "cuda_runtime", "lib"),
        os.path.join(py312, "nvidia", "cudnn", "lib"),
    ]
    prefix = ":".join([p for p in libs if os.path.isdir(p)])
    if not prefix:
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if os.environ.get("RCWN_CONTINUOUSSR_CUDA_LIBS_OK") != "1":
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ":".join([prefix, existing]) if existing else prefix
        env["RCWN_CONTINUOUSSR_CUDA_LIBS_OK"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_prefer_torch_cuda_libs()

import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from dataset import load_pair, make_split
from metrics import EdgeMetric, measure_batch, metric_proxy

ROOT = Path(__file__).resolve().parents[1]
CONTINUOUSSR_ROOT = ROOT / "external_models" / "ContinuousSR"
sys.path.insert(0, str(CONTINUOUSSR_ROOT))

import models  # noqa: E402


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def rgb_to_gray(x: torch.Tensor, mode: str) -> torch.Tensor:
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


def read_names(args: argparse.Namespace) -> list[str]:
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    else:
        names = split.val
    return names[: args.limit] if args.limit else names


def write_metrics_csv(path: str, rows: list[dict[str, float | str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "psnr", "ssim", "edge", "lpips", "proxy"])
        writer.writeheader()
        writer.writerows(rows)


def load_continuoussr_model(model_path: str, device: torch.device) -> torch.nn.Module:
    if not model_path:
        model_path = hf_hub_download(repo_id="pey12/ContinuousSR", filename="ContinuousSR.pth")
    ckpt: dict[str, Any] = torch.load(model_path, map_location="cpu", weights_only=False)
    model_spec = ckpt["model"]
    model = models.make(model_spec, load_sd=True).to(device).eval()
    return model


@torch.no_grad()
def validate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type != "cuda":
        raise RuntimeError("ContinuousSR upstream code uses hard-coded .cuda() tensors; CUDA is required.")
    names = read_names(args)
    print(f"device={device} val={len(names)} scale=x{args.scale}")
    print(f"model_path={args.model_path or 'hf://pey12/ContinuousSR/ContinuousSR.pth'} gray_mode={args.gray_mode}")

    model = load_continuoussr_model(args.model_path, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params={n_params:.2f}M")

    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    scale_tensor = torch.tensor([[float(args.scale), float(args.scale)]], device=device)
    sums: dict[str, float] = {}
    metric_rows: list[dict[str, float | str]] = []
    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(args.data, name, args.scale, device)
        lr_rgb = lr.repeat(1, 3, 1, 1)
        with autocast_context(device, args.amp):
            pred_rgb = model(lr_rgb, scale_tensor)
        pred = rgb_to_gray(pred_rgb.float().clamp(0, 1), args.gray_mode)
        metrics = measure_batch(pred, hr, edge_metric, lpips_fn)
        one = dict(metrics)
        one.setdefault("lpips", 0.0)
        one["proxy"] = metric_proxy(one)
        row: dict[str, float | str] = {"name": name}
        row.update(one)
        metric_rows.append(row)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value
        torch.cuda.empty_cache()

    out = {key: value / len(names) for key, value in sums.items()}
    out["proxy"] = metric_proxy(out)
    if args.metrics_csv:
        write_metrics_csv(args.metrics_csv, metric_rows)
        print(f"metrics_csv={args.metrics_csv}")
    print("val_result " + " ".join(f"{key}={value:.5f}" for key, value in out.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names-file")
    parser.add_argument("--metrics-csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
