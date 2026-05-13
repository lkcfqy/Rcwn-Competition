from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dataset import load_pair
from evaluate_hat_external_ensemble import build_external, read_names
from evaluate_hat_gray import autocast_context, build_hat, forward_gray, load_hat_weights
from evaluate_hat_postprocess_sweep import parse_route, safe_name
from metrics import EdgeMetric, measure_batch, metric_proxy
from runtime import _gaussian_blur, apply_postprocess, interp_tensor


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
                "mix_mode",
                "beta",
                "residual_sigma",
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


class HatExternal:
    def __init__(self, args: Any, device: torch.device):
        self.args = args
        self.model = build_hat(
            args.external_hat_variant,
            args.scale,
            args.external_use_checkpoint,
            native_io=args.external_native_io,
        ).to(device).eval()
        load_hat_weights(
            self.model,
            args.external_weights,
            args.external_param_key,
            native_io=args.external_native_io,
            gray_mode=args.external_gray_mode,
        )
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(
            f"external=hat params={n_params:.2f}M variant={args.external_hat_variant} "
            f"native_io={args.external_native_io} tta={args.external_tta}"
        )

    @torch.no_grad()
    def __call__(self, lr: torch.Tensor, hr_size: tuple[int, int], device: torch.device) -> torch.Tensor:
        with autocast_context(device, self.args.amp):
            out = forward_gray(
                self.model,
                lr,
                self.args.scale,
                self.args.external_gray_mode,
                tta=self.args.external_tta,
                native_io=self.args.external_native_io,
            )
        if out.shape[-2:] != hr_size:
            raise ValueError(f"External HAT output size {tuple(out.shape[-2:])} != target {hr_size}")
        return out.float()


def build_residual_external(args: Any, device: torch.device):
    if args.external_kind == "hat":
        return HatExternal(args, device)
    return build_external(args, device)


def mix_prediction(hat_pred: torch.Tensor, ext_pred: torch.Tensor, args: Any, beta: float) -> torch.Tensor:
    residual = ext_pred.float() - hat_pred.float()
    if args.mix_mode == "residual":
        return hat_pred.float() + beta * residual
    if args.mix_mode == "highpass":
        low = _gaussian_blur(residual, args.residual_sigma)
        return hat_pred.float() + beta * (residual - low)
    raise ValueError(args.mix_mode)


@torch.no_grad()
def validate(args: Any) -> None:
    routes = [parse_route(raw) for raw in args.route]
    betas = [float(x) for x in args.betas.split(",") if x.strip()]
    if not betas:
        raise ValueError("At least one beta is required.")
    if args.mix_mode == "highpass" and args.residual_sigma <= 0:
        raise ValueError("--residual-sigma must be positive for highpass mode.")

    jobs = []
    for beta in betas:
        beta_tag = f"b{beta:.4f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")
        for route in routes:
            jobs.append((f"{args.mix_mode}_{beta_tag}_{route.name}", beta, route))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")
    names = read_names(args)
    print(f"val={len(names)} scale=x{args.scale} jobs={len(jobs)} mix_mode={args.mix_mode}")

    model = build_hat(args.variant, args.scale, args.use_checkpoint, native_io=args.native_io).to(device).eval()
    load_hat_weights(model, args.weights, args.param_key, native_io=args.native_io, gray_mode=args.gray_mode)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"hat_params={n_params:.2f}M preset=hat_{args.variant}")
    print(f"hat_loaded={args.weights} param_key={args.param_key}")
    external = build_residual_external(args, device)

    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    interp_methods = sorted({route.interp for _, _, route in jobs if route.needs_base and route.interp is not None})
    sums: dict[str, dict[str, float]] = {name: {} for name, _, _ in jobs}
    metric_rows: dict[str, list[dict[str, float | str]]] = {name: [] for name, _, _ in jobs}

    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(args.data, name, args.scale, device)
        with autocast_context(device, args.amp):
            hat_pred = forward_gray(model, lr, args.scale, args.gray_mode, tta=args.tta, native_io=args.native_io).float()
        ext_pred = external(lr, hr.shape[-2:], device)
        bases = {method: interp_tensor(lr, args.scale, method) for method in interp_methods}
        mixed: dict[float, torch.Tensor] = {
            beta: mix_prediction(hat_pred, ext_pred, args, beta).clamp(0, 1) for beta in betas
        }
        for job_name, beta, route in jobs:
            base = bases.get(route.interp) if route.needs_base else None
            pred = apply_postprocess(
                mixed[beta],
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
    for job_name, beta, route in jobs:
        out = {key: value / len(names) for key, value in sums[job_name].items()}
        out.setdefault("lpips", 0.0)
        out["proxy"] = metric_proxy(out)
        metrics_csv = ""
        if metrics_dir is not None:
            metrics_path = metrics_dir / f"{safe_name(job_name)}.csv"
            write_metrics_csv(metrics_path, metric_rows[job_name])
            metrics_csv = str(metrics_path)
        summary_rows.append(
            {
                "route": job_name,
                "mix_mode": args.mix_mode,
                "beta": beta,
                "residual_sigma": args.residual_sigma,
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
    if metrics_dir is not None:
        print(f"metrics_dir={metrics_dir}")
    for row in summary_rows:
        print(
            "val_result "
            + f"route={row['route']} mix_mode={row['mix_mode']} beta={float(row['beta']):.5f} "
            + f"sigma={float(row['residual_sigma']):.5f} interp={row['interp']} "
            + f"blend_interp={float(row['blend_interp']):.5f} "
            + " ".join(f"{key}={float(row[key]):.5f}" for key in ["psnr", "ssim", "edge", "lpips", "proxy"])
        )


def parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", choices=["l", "m"], default="l")
    parser.add_argument("--param-key", default="state_dict")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--native-io", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--external-kind", choices=["onnx", "spandrel", "cat", "swinfir", "hat"], required=True)
    parser.add_argument("--external-weights", required=True)
    parser.add_argument("--external-param-key", default="params")
    parser.add_argument("--external-hat-variant", choices=["l", "m"], default="l")
    parser.add_argument("--external-native-io", action="store_true")
    parser.add_argument("--external-tta", action="store_true")
    parser.add_argument("--external-use-checkpoint", action="store_true")
    parser.add_argument("--external-cat-variant", choices=["r", "a", "r2", "a2"], default="a2")
    parser.add_argument("--external-swinfir-variant", choices=["swinfir_t", "swinfir", "hatfir"], default="hatfir")
    parser.add_argument("--external-gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--external-downsample", choices=["bicubic", "area"], default="bicubic")
    parser.add_argument("--mix-mode", choices=["residual", "highpass"], default="highpass")
    parser.add_argument("--betas", default="-0.2,-0.1,-0.05,0.05,0.1,0.2")
    parser.add_argument("--residual-sigma", type=float, default=1.0)
    parser.add_argument("--route", action="append", required=True)
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names-file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--summary-csv")
    parser.add_argument("--metrics-dir")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--ort-threads", type=int, default=4)
    parser.add_argument("--input-layout", choices=["auto", "nchw", "nhwc"], default="auto")
    parser.add_argument("--output-layout", choices=["auto", "nchw", "nhwc"], default="auto")
    parser.add_argument("--input-channels", choices=["auto", "1", "3"], default="auto")
    parser.add_argument("--input-range", choices=["0_1", "0_255"], default="0_1")
    parser.add_argument("--output-range", choices=["auto", "0_1", "0_255"], default="auto")
    parser.add_argument("--allow-resize-output", action="store_true")
    parser.add_argument("--auto-tile-fixed", action="store_true")
    parser.add_argument("--tile-overlap", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
