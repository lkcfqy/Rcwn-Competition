from __future__ import annotations

import argparse
import csv
import os
import sys
from contextlib import nullcontext
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
from runtime import _aug, _deaug


ROOT = Path(__file__).resolve().parents[1]
EAMAMBA_ROOT = ROOT / "external_models" / "EAMamba"


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def load_external_models_module():
    # EAMamba's code expects its own package to be imported as plain `models`.
    for key in list(sys.modules):
        if key == "models" or key.startswith("models."):
            del sys.modules[key]
    sys.path.insert(0, str(EAMAMBA_ROOT))
    import models as eamamba_models  # type: ignore

    return eamamba_models


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


def build_model(weights: str, device: torch.device) -> torch.nn.Module:
    eamamba_models = load_external_models_module()
    ckpt = torch.load(weights, map_location="cpu")
    model_spec = ckpt["model"]
    model, model_e = eamamba_models.make(model_spec, load_sd=True)
    del model
    model_e = model_e.to(device).eval()
    n_params = sum(p.numel() for p in model_e.parameters()) / 1e6
    current_iter = ckpt.get("current_iter", "unknown")
    print(f"params={n_params:.2f}M current_iter={current_iter} model={model_spec.get('name')}")
    return model_e


@torch.no_grad()
def forward_once(model: torch.nn.Module, lr_gray: torch.Tensor, gray_mode: str) -> torch.Tensor:
    pred = model(lr_gray.repeat(1, 3, 1, 1))
    return rgb_to_gray(pred, gray_mode)


@torch.no_grad()
def forward_gray(model: torch.nn.Module, lr_gray: torch.Tensor, gray_mode: str, tta: bool) -> torch.Tensor:
    if not tta:
        return forward_once(model, lr_gray, gray_mode)
    preds = []
    for mode in range(8):
        pred = forward_once(model, _aug(lr_gray, mode), gray_mode)
        preds.append(_deaug(pred, mode))
    return torch.stack(preds).mean(dim=0)


@torch.no_grad()
def validate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    else:
        names = split.val[: args.limit] if args.limit else split.val
    print(f"val={len(names)} scale=x{args.scale}")

    model = build_model(args.weights, device)
    print(f"loaded={args.weights}")

    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    sums: dict[str, float] = {}
    metric_rows: list[dict[str, float | str]] = []
    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(args.data, name, args.scale, device)
        with autocast_context(device, args.amp):
            pred = forward_gray(model, lr, args.gray_mode, args.tta)
        pred = pred.float().clamp(0, 1)
        if pred.shape[-2:] != hr.shape[-2:]:
            mode = "area" if args.resize == "area" else "bicubic"
            kwargs: dict[str, Any] = {} if mode == "area" else {"align_corners": False}
            pred = F.interpolate(pred, size=hr.shape[-2:], mode=mode, **kwargs).clamp(0, 1)
        metrics = measure_batch(pred, hr, edge_metric, lpips_fn)
        one = dict(metrics)
        one["proxy"] = metric_proxy(one)
        if args.metrics_csv:
            row: dict[str, float | str] = {"name": name}
            row.update(one)
            metric_rows.append(row)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value

    out = {key: value / len(names) for key, value in sums.items()}
    out["proxy"] = metric_proxy(out)
    if args.metrics_csv:
        out_path = Path(args.metrics_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["name", "psnr", "ssim", "edge", "lpips", "proxy"])
            writer.writeheader()
            writer.writerows(metric_rows)
        print(f"metrics_csv={out_path}")
    print("val_result " + " ".join(f"{key}={value:.5f}" for key, value in out.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--resize", choices=["area", "bicubic"], default="bicubic")
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
