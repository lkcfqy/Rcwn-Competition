from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dataset import load_pair, make_split
from evaluate_hat_gray import build_hat, forward_gray as forward_hat_gray, normalize_state as normalize_hat_state
from evaluate_mambairv2_gray import (
    build_mambairv2,
    forward_gray as forward_mambair_gray,
    normalize_state as normalize_mambair_state,
)
from metrics import EdgeMetric, measure_batch, metric_proxy
from runtime import _aug, _deaug


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def load_state(path: str, key: str) -> Any:
    return torch.load(path, map_location="cpu")


def forward_mambair_tta(model: torch.nn.Module, lr: torch.Tensor, gray_mode: str) -> torch.Tensor:
    preds = []
    for mode in range(8):
        pred = forward_mambair_gray(model, _aug(lr, mode), gray_mode)
        preds.append(_deaug(pred, mode))
    return torch.stack(preds).mean(dim=0)


@torch.no_grad()
def validate(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    names = split.val[: args.limit] if args.limit else split.val
    print(f"device={device} val={len(names)} scale=x{args.scale}")

    hat = build_hat(args.hat_variant, args.scale, args.use_checkpoint).to(device).eval()
    hat_ckpt = load_state(args.hat_weights, args.hat_param_key)
    hat.load_state_dict(normalize_hat_state(hat_ckpt, args.hat_param_key), strict=True)
    print(f"loaded_hat={args.hat_weights} key={args.hat_param_key}")

    mamba = build_mambairv2(args.mamba_variant, args.scale, args.use_checkpoint).to(device).eval()
    mamba_ckpt = load_state(args.mamba_weights, args.mamba_param_key)
    mamba.load_state_dict(normalize_mambair_state(mamba_ckpt, args.mamba_param_key), strict=True)
    print(f"loaded_mamba={args.mamba_weights} key={args.mamba_param_key}")

    alphas = [float(x) for x in args.alphas.split(",")]
    sums = {alpha: {} for alpha in alphas}
    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(args.data, name, args.scale, device)
        with autocast_context(device, args.amp):
            hat_pred = forward_hat_gray(hat, lr, args.scale, args.gray_mode, tta=args.hat_tta).float()
            if args.mamba_tta:
                mamba_pred = forward_mambair_tta(mamba, lr, args.gray_mode).float()
            else:
                mamba_pred = forward_mambair_gray(mamba, lr, args.gray_mode).float()
        for alpha in alphas:
            pred = hat_pred * alpha + mamba_pred * (1.0 - alpha)
            metrics = measure_batch(pred, hr, edge_metric, lpips_fn)
            for key, value in metrics.items():
                sums[alpha][key] = sums[alpha].get(key, 0.0) + value

    for alpha in alphas:
        out = {key: value / len(names) for key, value in sums[alpha].items()}
        out["proxy"] = metric_proxy(out)
        print(f"alpha_hat={alpha:.4f} " + " ".join(f"{key}={value:.5f}" for key, value in out.items()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--hat-weights", required=True)
    parser.add_argument("--hat-variant", choices=["l", "m"], default="l")
    parser.add_argument("--hat-param-key", default="state_dict")
    parser.add_argument("--mamba-weights", required=True)
    parser.add_argument("--mamba-variant", choices=["light", "base", "large"], default="base")
    parser.add_argument("--mamba-param-key", default="state_dict")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--alphas", default="0.8,0.9,0.95")
    parser.add_argument("--hat-tta", action="store_true")
    parser.add_argument("--mamba-tta", action="store_true")
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
