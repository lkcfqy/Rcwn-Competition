from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dataset import load_pair, make_split
from evaluate_hat_gray import build_hat, forward_gray as forward_hat_gray, load_hat_weights
from evaluate_mambairv2_gray import (
    build_mambairv2,
    forward_gray as forward_mambair_gray,
    normalize_state as normalize_mambair_state,
)
from evaluate_hat_postprocess_sweep import Route, parse_route
from metrics import EdgeMetric, measure_batch, metric_proxy
from runtime import _aug, _deaug, apply_postprocess, interp_tensor


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


def read_names(args) -> list[str]:
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    else:
        names = split.val
    return names[: args.limit] if args.limit else names


def safe_name(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)


def write_metrics_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "psnr", "ssim", "edge", "lpips", "proxy"])
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "route",
                "alpha_hat",
                "interp",
                "blend_interp",
                "sharpen_amount",
                "sharpen_radius",
                "clip_mode",
                "psnr",
                "ssim",
                "edge",
                "lpips",
                "proxy",
                "metrics_csv",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def validate(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    names = read_names(args)
    print(f"device={device} val={len(names)} scale=x{args.scale}")

    hat = build_hat(args.hat_variant, args.scale, args.use_checkpoint, native_io=args.hat_native_io).to(device).eval()
    load_hat_weights(hat, args.hat_weights, args.hat_param_key, native_io=args.hat_native_io, gray_mode=args.gray_mode)
    print(f"loaded_hat={args.hat_weights} key={args.hat_param_key}")

    mamba = build_mambairv2(args.mamba_variant, args.scale, args.use_checkpoint).to(device).eval()
    mamba_ckpt = load_state(args.mamba_weights, args.mamba_param_key)
    mamba.load_state_dict(normalize_mambair_state(mamba_ckpt, args.mamba_param_key), strict=True)
    print(f"loaded_mamba={args.mamba_weights} key={args.mamba_param_key}")

    alphas = [float(x) for x in args.alphas.split(",")]
    routes = [parse_route(raw) for raw in args.route] if args.route else [Route(name="raw")]
    jobs: list[tuple[str, float, Route]] = []
    for alpha in alphas:
        alpha_tag = f"a{alpha:.4f}".rstrip("0").rstrip(".").replace(".", "p")
        for route in routes:
            jobs.append((f"{alpha_tag}_{route.name}", alpha, route))
    sums = {job_name: {} for job_name, _, _ in jobs}
    metric_rows: dict[str, list[dict[str, float | str]]] = {job_name: [] for job_name, _, _ in jobs}
    interp_methods = sorted({route.interp for _, _, route in jobs if route.needs_base and route.interp is not None})
    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(args.data, name, args.scale, device)
        with autocast_context(device, args.amp):
            hat_pred = forward_hat_gray(
                hat,
                lr,
                args.scale,
                args.gray_mode,
                tta=args.hat_tta,
                native_io=args.hat_native_io,
            ).float()
            if args.mamba_tta:
                mamba_pred = forward_mambair_tta(mamba, lr, args.gray_mode).float()
            else:
                mamba_pred = forward_mambair_gray(mamba, lr, args.gray_mode).float()
        bases = {method: interp_tensor(lr, args.scale, method) for method in interp_methods}
        for job_name, alpha, route in jobs:
            pred = hat_pred * alpha + mamba_pred * (1.0 - alpha)
            base = bases.get(route.interp) if route.needs_base else None
            pred = apply_postprocess(
                pred,
                base,
                lr=lr,
                blend_interp=route.blend_interp,
                sharpen_amount=route.sharpen_amount,
                sharpen_radius=route.sharpen_radius,
                clip_mode=route.clip_mode,
            )
            metrics = measure_batch(pred.float(), hr, edge_metric, lpips_fn)
            one = dict(metrics)
            one.setdefault("lpips", 0.0)
            one["proxy"] = metric_proxy(one)
            row: dict[str, float | str] = {"name": name}
            row.update(one)
            metric_rows[job_name].append(row)
            for key, value in metrics.items():
                sums[job_name][key] = sums[job_name].get(key, 0.0) + value

    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else None
    summary_rows: list[dict[str, float | str]] = []
    for job_name, alpha, route in jobs:
        out = {key: value / len(names) for key, value in sums[job_name].items()}
        out["proxy"] = metric_proxy(out)
        metrics_csv = ""
        if metrics_dir is not None:
            metrics_path = metrics_dir / f"{safe_name(job_name)}.csv"
            write_metrics_csv(metrics_path, metric_rows[job_name])
            metrics_csv = str(metrics_path)
            print(f"metrics_csv route={job_name} alpha_hat={alpha:.4f} path={metrics_path}")
        summary_rows.append(
            {
                "route": job_name,
                "alpha_hat": alpha,
                "interp": route.interp or "",
                "blend_interp": route.blend_interp,
                "sharpen_amount": route.sharpen_amount,
                "sharpen_radius": route.sharpen_radius,
                "clip_mode": route.clip_mode,
                **out,
                "metrics_csv": metrics_csv,
            }
        )
    summary_rows.sort(key=lambda row: float(row["proxy"]), reverse=True)
    if args.summary_csv:
        write_summary_csv(Path(args.summary_csv), summary_rows)
        print(f"summary_csv={args.summary_csv}")
    for row in summary_rows:
        print(
            "val_result "
            + f"route={row['route']} alpha_hat={float(row['alpha_hat']):.4f} "
            + f"interp={row['interp']} blend_interp={float(row['blend_interp']):.5f} "
            + f"sharpen_amount={float(row['sharpen_amount']):.5f} clip_mode={row['clip_mode']} "
            + " ".join(f"{key}={float(row[key]):.5f}" for key in ["psnr", "ssim", "edge", "lpips", "proxy"])
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--hat-weights", required=True)
    parser.add_argument("--hat-variant", choices=["l", "m"], default="l")
    parser.add_argument("--hat-param-key", default="state_dict")
    parser.add_argument("--hat-native-io", action="store_true")
    parser.add_argument("--mamba-weights", required=True)
    parser.add_argument("--mamba-variant", choices=["light", "base", "large"], default="base")
    parser.add_argument("--mamba-param-key", default="state_dict")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--alphas", default="0.8,0.9,0.95")
    parser.add_argument("--route", action="append", help="name[:interp[:blend[:sharpen[:radius[:clip]]]]]")
    parser.add_argument("--hat-tta", action="store_true")
    parser.add_argument("--mamba-tta", action="store_true")
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names-file")
    parser.add_argument("--metrics-dir")
    parser.add_argument("--summary-csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
