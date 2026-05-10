from __future__ import annotations

import argparse
from contextlib import nullcontext
from typing import Any

import torch
from tqdm import tqdm

from dataset import load_pair, make_split
from evaluate_hat_gray import build_hat, forward_gray, normalize_state
from metrics import EdgeMetric, measure_batch, metric_proxy


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def load_hat(path: str, variant: str, param_key: str, scale: int, device: torch.device, use_checkpoint: bool):
    model = build_hat(variant, scale, use_checkpoint).to(device).eval()
    ckpt: Any = torch.load(path, map_location="cpu")
    model.load_state_dict(normalize_state(ckpt, param_key), strict=True)
    return model


@torch.no_grad()
def validate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    names = split.val[: args.limit] if args.limit else split.val
    print(f"device={device} val={len(names)} scale=x{args.scale}")

    model_a = load_hat(args.weights_a, args.variant_a, args.param_key_a, args.scale, device, args.use_checkpoint)
    model_b = load_hat(args.weights_b, args.variant_b, args.param_key_b, args.scale, device, args.use_checkpoint)
    print(f"loaded_a={args.weights_a} variant={args.variant_a} key={args.param_key_a}")
    print(f"loaded_b={args.weights_b} variant={args.variant_b} key={args.param_key_b}")

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
            pred_a = forward_gray(model_a, lr, args.scale, args.gray_mode, tta=args.tta_a).float()
            pred_b = forward_gray(model_b, lr, args.scale, args.gray_mode, tta=args.tta_b).float()
        for alpha in alphas:
            pred = pred_a * alpha + pred_b * (1.0 - alpha)
            metrics = measure_batch(pred, hr, edge_metric, lpips_fn)
            for key, value in metrics.items():
                sums[alpha][key] = sums[alpha].get(key, 0.0) + value

    for alpha in alphas:
        out = {key: value / len(names) for key, value in sums[alpha].items()}
        out["proxy"] = metric_proxy(out)
        print(f"alpha_a={alpha:.4f} " + " ".join(f"{key}={value:.5f}" for key, value in out.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--weights-a", required=True)
    parser.add_argument("--variant-a", choices=["l", "m"], default="l")
    parser.add_argument("--param-key-a", default="state_dict")
    parser.add_argument("--weights-b", required=True)
    parser.add_argument("--variant-b", choices=["l", "m"], default="m")
    parser.add_argument("--param-key-b", default="params_ema")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--alphas", default="0.7,0.8,0.9,0.95")
    parser.add_argument("--tta-a", action="store_true")
    parser.add_argument("--tta-b", action="store_true")
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
