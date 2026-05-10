from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any


def _prefer_torch_cuda_libs() -> None:
    base = "/usr/local/lib/python3.12/dist-packages/nvidia"
    libs = [
        os.path.join(base, "cublas", "lib"),
        os.path.join(base, "cuda_runtime", "lib"),
        os.path.join(base, "cudnn", "lib"),
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
from tqdm import tqdm

from dataset import load_pair, make_split
from hat_gray_common import build_hat as build_hat_common, forward_hat_gray, load_hat_weights as load_hat_weights_common
from metrics import EdgeMetric, measure_batch, metric_proxy
from runtime import INTERP, _aug, _deaug, _gaussian_blur, apply_postprocess, interp_tensor


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


def build_hat(variant: str, scale: int, use_checkpoint: bool, native_io: bool = False) -> torch.nn.Module:
    return build_hat_common(variant, scale, use_checkpoint, native_io=native_io)


def load_hat_weights(
    model: torch.nn.Module,
    path: str,
    param_key: str,
    native_io: bool = False,
    gray_mode: str = "avg",
) -> None:
    load_hat_weights_common(model, path, param_key, native_io=native_io, gray_mode=gray_mode)


def _forward_gray_once(
    model: torch.nn.Module,
    lr_gray: torch.Tensor,
    scale: int,
    gray_mode: str,
    window_size: int = 16,
    native_io: bool = False,
):
    return forward_hat_gray(model, lr_gray, scale, gray_mode, window_size=window_size, native_io=native_io)


def forward_gray(
    model: torch.nn.Module,
    lr_gray: torch.Tensor,
    scale: int,
    gray_mode: str,
    window_size: int = 16,
    tta: bool = False,
    native_io: bool = False,
):
    if not tta:
        return _forward_gray_once(model, lr_gray, scale, gray_mode, window_size, native_io=native_io)
    preds = []
    for mode in range(8):
        pred = _forward_gray_once(model, _aug(lr_gray, mode), scale, gray_mode, window_size, native_io=native_io)
        preds.append(_deaug(pred, mode))
    return torch.stack(preds).mean(dim=0)


def preprocess_lr(
    lr: torch.Tensor,
    sharpen_amount: float,
    sharpen_radius: float,
    blur_sigma: float,
    contrast: float,
    bias: float,
    gamma: float,
) -> torch.Tensor:
    out = lr.float()
    if blur_sigma > 0:
        out = _gaussian_blur(out, blur_sigma)
    if sharpen_amount > 0:
        blur = _gaussian_blur(out, sharpen_radius)
        out = out + sharpen_amount * (out - blur)
    if contrast != 1.0 or bias != 0.0:
        out = (out - 0.5) * contrast + 0.5 + bias
    out = out.clamp(0, 1)
    if gamma != 1.0:
        out = out.clamp_min(1e-6).pow(gamma).clamp(0, 1)
    return out


@torch.no_grad()
def validate(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    else:
        names = split.val[: args.limit] if args.limit else split.val
    print(f"val={len(names)} scale=x{args.scale}")

    model = build_hat(args.variant, args.scale, args.use_checkpoint, native_io=args.native_io).to(device).eval()
    load_hat_weights(model, args.weights, args.param_key, native_io=args.native_io, gray_mode=args.gray_mode)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params={n_params:.2f}M preset=hat_{args.variant}")
    print(f"loaded={args.weights} param_key={args.param_key}")

    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    sums = {}
    metric_rows: list[dict[str, float | str]] = []
    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(args.data, name, args.scale, device)
        model_lr = preprocess_lr(
            lr,
            args.input_sharpen_amount,
            args.input_sharpen_radius,
            args.input_blur_sigma,
            args.input_contrast,
            args.input_bias,
            args.input_gamma,
        )
        with autocast_context(device, args.amp):
            pred = forward_gray(model, model_lr, args.scale, args.gray_mode, tta=args.tta, native_io=args.native_io)
        if (
            args.blend_interp > 0
            or args.sharpen_amount > 0
            or args.back_project_iters > 0
            or args.clip_mode != "hard"
        ):
            base = interp_tensor(lr, args.scale, args.interp)
            pred = apply_postprocess(
                pred.float(),
                base,
                lr=lr,
                blend_interp=args.blend_interp,
                sharpen_amount=args.sharpen_amount,
                sharpen_radius=args.sharpen_radius,
                back_project_iters=args.back_project_iters,
                back_project_alpha=args.back_project_alpha,
                back_project_down=args.back_project_down,
                back_project_up=args.back_project_up,
                back_project_down_sigma=args.back_project_down_sigma,
                clip_mode=args.clip_mode,
            )
        metrics = measure_batch(pred.float(), hr, edge_metric, lpips_fn)
        one = dict(metrics)
        one["proxy"] = metric_proxy(one)
        if args.print_per_image:
            print("image", name, " ".join(f"{k}={v:.6f}" for k, v in one.items()))
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
    print("val_result " + " ".join(f"{k}={v:.5f}" for k, v in out.items()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", choices=["l", "m"], default="l")
    parser.add_argument("--param-key", default="params_ema")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--native-io", action="store_true")
    parser.add_argument("--interp", choices=sorted(INTERP), default="lanczos")
    parser.add_argument("--blend-interp", type=float, default=0.0)
    parser.add_argument("--sharpen-amount", type=float, default=0.0)
    parser.add_argument("--sharpen-radius", type=float, default=1.0)
    parser.add_argument("--back-project-iters", type=int, default=0)
    parser.add_argument("--back-project-alpha", type=float, default=1.0)
    parser.add_argument("--back-project-down", choices=["nearest", "linear", "cubic", "area"], default="area")
    parser.add_argument("--back-project-up", choices=["nearest", "linear", "cubic", "area"], default="cubic")
    parser.add_argument("--back-project-down-sigma", type=float, default=0.0)
    parser.add_argument("--clip-mode", choices=["hard", "match-base", "none"], default="hard")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--input-sharpen-amount", type=float, default=0.0)
    parser.add_argument("--input-sharpen-radius", type=float, default=1.0)
    parser.add_argument("--input-blur-sigma", type=float, default=0.0)
    parser.add_argument("--input-contrast", type=float, default=1.0)
    parser.add_argument("--input-bias", type=float, default=0.0)
    parser.add_argument("--input-gamma", type=float, default=1.0)
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names-file")
    parser.add_argument("--print-per-image", action="store_true")
    parser.add_argument("--metrics-csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
