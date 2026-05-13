from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from dataset import load_pair, make_split
from evaluate_hat_gray import (
    _forward_gray_once,
    autocast_context,
    build_hat,
    load_hat_weights,
    preprocess_lr,
)
from evaluate_hat_postprocess_sweep import parse_route, safe_name
from metrics import EdgeMetric, measure_batch, metric_proxy
from runtime import _aug, _deaug, apply_postprocess, interp_tensor


@dataclass(frozen=True)
class Subset:
    name: str
    modes: tuple[int, ...]


def parse_subset(raw: str) -> Subset:
    if "=" in raw:
        name, modes_raw = raw.split("=", 1)
    else:
        modes_raw = raw
        name = "m" + "_".join(part.strip() for part in raw.replace(":", ",").split(",") if part.strip())
    modes = tuple(int(part.strip()) for part in modes_raw.replace(":", ",").split(",") if part.strip())
    if not modes:
        raise ValueError(f"Empty TTA subset: {raw}")
    if any(mode < 0 or mode > 7 for mode in modes):
        raise ValueError(f"TTA modes must be in [0, 7]: {raw}")
    if len(set(modes)) != len(modes):
        raise ValueError(f"TTA modes must be unique within a subset: {raw}")
    name = name.strip()
    if not name:
        raise ValueError(f"Empty TTA subset name: {raw}")
    return Subset(name=name, modes=modes)


def read_names(args) -> list[str]:
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    else:
        names = split.val
    return names[: args.limit] if args.limit else names


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
                "subset",
                "modes",
                "route",
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
def forward_tta_modes(
    model: torch.nn.Module,
    lr: torch.Tensor,
    scale: int,
    gray_mode: str,
    native_io: bool,
    modes: list[int],
    amp: str,
) -> dict[int, torch.Tensor]:
    preds: dict[int, torch.Tensor] = {}
    for mode in modes:
        with autocast_context(lr.device, amp):
            pred = _forward_gray_once(model, _aug(lr, mode), scale, gray_mode, native_io=native_io)
        preds[mode] = _deaug(pred, mode).float()
    return preds


@torch.no_grad()
def validate(args) -> None:
    subsets = [parse_subset(raw) for raw in args.subset]
    subset_names = [subset.name for subset in subsets]
    if len(subset_names) != len(set(subset_names)):
        raise ValueError(f"Subset names must be unique: {subset_names}")

    routes = [parse_route(raw) for raw in args.route]
    route_names = [route.name for route in routes]
    if len(route_names) != len(set(route_names)):
        raise ValueError(f"Route names must be unique: {route_names}")

    jobs = [(subset, route) for subset in subsets for route in routes]
    job_names = [f"{subset.name}_{route.name}" for subset, route in jobs]
    if len(job_names) != len(set(job_names)):
        raise ValueError("Combined subset/route names must be unique.")

    modes = sorted({mode for subset in subsets for mode in subset.modes})
    interp_methods = sorted({route.interp for route in routes if route.needs_base and route.interp is not None})

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")
    names = read_names(args)
    print(f"val={len(names)} scale=x{args.scale} modes={modes} jobs={len(jobs)}")

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

    sums: dict[str, dict[str, float]] = {job_name: {} for job_name in job_names}
    metric_rows: dict[str, list[dict[str, float | str]]] = {job_name: [] for job_name in job_names}

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
        mode_preds = forward_tta_modes(
            model,
            model_lr,
            args.scale,
            args.gray_mode,
            args.native_io,
            modes,
            args.amp,
        )
        bases = {method: interp_tensor(lr, args.scale, method) for method in interp_methods}

        subset_preds: dict[str, torch.Tensor] = {}
        for subset in subsets:
            subset_preds[subset.name] = torch.stack([mode_preds[mode] for mode in subset.modes]).mean(dim=0)

        for subset, route in jobs:
            job_name = f"{subset.name}_{route.name}"
            base = bases.get(route.interp) if route.needs_base else None
            pred = apply_postprocess(
                subset_preds[subset.name],
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
    for subset, route in jobs:
        job_name = f"{subset.name}_{route.name}"
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
                "subset": subset.name,
                "modes": ",".join(str(mode) for mode in subset.modes),
                "route": route.name,
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
            + f"subset={row['subset']} modes={row['modes']} route={row['route']} "
            + f"interp={row['interp']} blend_interp={float(row['blend_interp']):.5f} "
            + f"sharpen_amount={float(row['sharpen_amount']):.5f} clip_mode={row['clip_mode']} "
            + " ".join(f"{key}={float(row[key]):.5f}" for key in ["psnr", "ssim", "edge", "lpips", "proxy"])
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", choices=["l", "m"], default="l")
    parser.add_argument("--param-key", default="params_ema")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--native-io", action="store_true")
    parser.add_argument("--subset", action="append", required=True, help="name=0,1,2 style TTA subset")
    parser.add_argument("--route", action="append", required=True, help="name[:interp[:blend[:sharpen[:radius[:clip]]]]]")
    parser.add_argument("--input-sharpen-amount", type=float, default=0.0)
    parser.add_argument("--input-sharpen-radius", type=float, default=1.0)
    parser.add_argument("--input-blur-sigma", type=float, default=0.0)
    parser.add_argument("--input-contrast", type=float, default=1.0)
    parser.add_argument("--input-bias", type=float, default=0.0)
    parser.add_argument("--input-gamma", type=float, default=1.0)
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
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
