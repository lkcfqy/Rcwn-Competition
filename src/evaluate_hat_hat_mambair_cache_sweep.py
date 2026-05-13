from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dataset import load_pair, make_split
from evaluate_hat_gray import build_hat, forward_gray as forward_hat_gray, load_hat_weights
from evaluate_hat_hat_mambair_ensemble import (
    autocast_context,
    forward_mambair_tta,
    parse_float_list,
    parse_triplets,
    safe_name,
    tag_float,
    write_metrics_csv,
    write_summary_csv,
)
from evaluate_hat_postprocess_sweep import Route, parse_route
from evaluate_mambairv2_gray import (
    build_mambairv2,
    forward_gray as forward_mambair_gray,
    normalize_state as normalize_mambair_state,
)
from metrics import EdgeMetric, measure_batch, metric_proxy
from runtime import _gaussian_blur, apply_postprocess, interp_tensor


def read_names(args: Any) -> list[str]:
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    else:
        names = split.val
    return names[: args.limit] if args.limit else names


def cache_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{safe_name(Path(name).stem)}.pt"


def load_models(args: Any, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    hat_a = build_hat(args.hat_a_variant, args.scale, args.use_checkpoint, native_io=args.hat_a_native_io).to(device).eval()
    load_hat_weights(
        hat_a,
        args.hat_a_weights,
        args.hat_a_param_key,
        native_io=args.hat_a_native_io,
        gray_mode=args.gray_mode,
    )
    print(f"loaded_hat_a={args.hat_a_weights} key={args.hat_a_param_key} native_io={args.hat_a_native_io}")

    hat_b = build_hat(args.hat_b_variant, args.scale, args.use_checkpoint, native_io=args.hat_b_native_io).to(device).eval()
    load_hat_weights(
        hat_b,
        args.hat_b_weights,
        args.hat_b_param_key,
        native_io=args.hat_b_native_io,
        gray_mode=args.gray_mode,
    )
    print(f"loaded_hat_b={args.hat_b_weights} key={args.hat_b_param_key} native_io={args.hat_b_native_io}")

    mamba = build_mambairv2(args.mamba_variant, args.scale, args.use_checkpoint).to(device).eval()
    mamba_ckpt = torch.load(args.mamba_weights, map_location="cpu")
    mamba.load_state_dict(normalize_mambair_state(mamba_ckpt, args.mamba_param_key), strict=True)
    print(f"loaded_mamba={args.mamba_weights} key={args.mamba_param_key}")
    return hat_a, hat_b, mamba


@torch.no_grad()
def load_or_build_prediction(
    args: Any,
    name: str,
    device: torch.device,
    cache_dir: Path | None,
    models: tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    path = cache_path(cache_dir, name) if cache_dir is not None else None
    if path is not None and path.exists() and not args.refresh_cache:
        payload = torch.load(path, map_location="cpu")
        lr, hr = load_pair(args.data, name, args.scale, device)
        return (
            lr,
            hr,
            payload["pred_a"].to(device=device, dtype=torch.float32),
            payload["pred_b"].to(device=device, dtype=torch.float32),
            payload["pred_m"].to(device=device, dtype=torch.float32),
        )

    if models is None:
        raise FileNotFoundError(f"Missing cache for {name}: {path}")
    hat_a, hat_b, mamba = models
    lr, hr = load_pair(args.data, name, args.scale, device)
    with autocast_context(device, args.amp):
        pred_a = forward_hat_gray(
            hat_a,
            lr,
            args.scale,
            args.gray_mode,
            tta=args.hat_a_tta,
            native_io=args.hat_a_native_io,
        ).float()
        pred_b = forward_hat_gray(
            hat_b,
            lr,
            args.scale,
            args.gray_mode,
            tta=args.hat_b_tta,
            native_io=args.hat_b_native_io,
        ).float()
        if args.mamba_tta:
            pred_m = forward_mambair_tta(mamba, lr, args.gray_mode).float()
        else:
            pred_m = forward_mambair_gray(mamba, lr, args.gray_mode).float()

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "name": name,
                "scale": args.scale,
                "gray_mode": args.gray_mode,
                "pred_a": pred_a.detach().cpu(),
                "pred_b": pred_b.detach().cpu(),
                "pred_m": pred_m.detach().cpu(),
            },
            path,
        )
    return lr, hr, pred_a, pred_b, pred_m


@torch.no_grad()
def validate(args: Any) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    names = read_names(args)
    print(f"device={device} val={len(names)} scale=x{args.scale}")

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    cache_complete = (
        cache_dir is not None
        and not args.refresh_cache
        and all(cache_path(cache_dir, name).exists() for name in names)
    )
    models = None if cache_complete else load_models(args, device)
    if cache_complete:
        print(f"using_prediction_cache={cache_dir}")

    triplets = parse_triplets(args.weights)
    routes = [parse_route(raw) for raw in args.route] if args.route else [Route(name="raw")]
    residual_sigmas = [0.0] if args.mamba_mix_mode == "blend" else parse_float_list(args.mamba_residual_sigmas)
    if args.mamba_mix_mode == "highpass" and any(value <= 0 for value in residual_sigmas):
        raise ValueError("--mamba-residual-sigmas must be positive for highpass mode.")

    jobs: list[tuple[str, tuple[float, float, float], Route, float]] = []
    for weights in triplets:
        wa, wb, wm = weights
        weight_tag = f"wa{tag_float(wa)}_wb{tag_float(wb)}_wm{tag_float(wm)}"
        for route in routes:
            for sigma in residual_sigmas:
                sigma_tag = f"_hp{tag_float(sigma)}" if args.mamba_mix_mode == "highpass" else ""
                jobs.append((f"{weight_tag}_{route.name}{sigma_tag}", weights, route, sigma))
    job_names = [job_name for job_name, *_ in jobs]
    if len(set(job_names)) != len(job_names):
        duplicates = sorted({name for name in job_names if job_names.count(name) > 1})
        raise ValueError(f"Duplicate route names would corrupt accumulated metrics: {duplicates}")

    sums = {job_name: {} for job_name, _, _, _ in jobs}
    metric_rows: dict[str, list[dict[str, float | str]]] = {job_name: [] for job_name, _, _, _ in jobs}
    interp_methods = sorted({route.interp for _, _, route, _ in jobs if route.needs_base and route.interp is not None})
    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    for name in tqdm(names, desc="val", leave=False):
        lr, hr, pred_a, pred_b, pred_m = load_or_build_prediction(args, name, device, cache_dir, models)
        bases = {method: interp_tensor(lr, args.scale, method) for method in interp_methods}
        for job_name, weights, route, sigma in jobs:
            wa, wb, wm = weights
            if args.mamba_mix_mode == "highpass":
                hat_total = wa + wb
                if hat_total <= 0:
                    raise ValueError("Highpass Mamba mix requires positive HAT weights.")
                hat_base = (pred_a * wa + pred_b * wb) / hat_total
                residual = pred_m - hat_base
                pred = hat_base + wm * (residual - _gaussian_blur(residual, sigma))
            else:
                pred = pred_a * wa + pred_b * wb + pred_m * wm
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
    for job_name, weights, route, sigma in jobs:
        out = {key: value / len(names) for key, value in sums[job_name].items()}
        out.setdefault("lpips", 0.0)
        out["proxy"] = metric_proxy(out)
        metrics_csv = ""
        if metrics_dir is not None:
            metrics_path = metrics_dir / f"{safe_name(job_name)}.csv"
            write_metrics_csv(metrics_path, metric_rows[job_name])
            metrics_csv = str(metrics_path)
            print(f"metrics_csv route={job_name} weights={weights} path={metrics_path}")
        wa, wb, wm = weights
        summary_rows.append(
            {
                "route": job_name,
                "weight_hat_a": wa,
                "weight_hat_b": wb,
                "weight_mamba": wm,
                "interp": route.interp or "",
                "blend_interp": route.blend_interp,
                "sharpen_amount": route.sharpen_amount,
                "sharpen_radius": route.sharpen_radius,
                "clip_mode": route.clip_mode,
                "mamba_mix_mode": args.mamba_mix_mode,
                "mamba_residual_sigma": sigma,
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
            + f"route={row['route']} "
            + f"weights={float(row['weight_hat_a']):.4f},{float(row['weight_hat_b']):.4f},{float(row['weight_mamba']):.4f} "
            + f"interp={row['interp']} blend_interp={float(row['blend_interp']):.5f} "
            + f"sharpen_amount={float(row['sharpen_amount']):.5f} clip_mode={row['clip_mode']} "
            + f"mamba_mix_mode={row['mamba_mix_mode']} mamba_residual_sigma={float(row['mamba_residual_sigma']):.5f} "
            + " ".join(f"{key}={float(row[key]):.5f}" for key in ["psnr", "ssim", "edge", "lpips", "proxy"])
        )


def parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--hat-a-weights", required=True)
    parser.add_argument("--hat-a-variant", choices=["l", "m"], default="l")
    parser.add_argument("--hat-a-param-key", default="state_dict")
    parser.add_argument("--hat-a-native-io", action="store_true")
    parser.add_argument("--hat-a-tta", action="store_true")
    parser.add_argument("--hat-b-weights", required=True)
    parser.add_argument("--hat-b-variant", choices=["l", "m"], default="l")
    parser.add_argument("--hat-b-param-key", default="state_dict")
    parser.add_argument("--hat-b-native-io", action="store_true")
    parser.add_argument("--hat-b-tta", action="store_true")
    parser.add_argument("--mamba-weights", required=True)
    parser.add_argument("--mamba-variant", choices=["light", "base", "large"], default="large")
    parser.add_argument("--mamba-param-key", default="state_dict")
    parser.add_argument("--mamba-tta", action="store_true")
    parser.add_argument("--mamba-mix-mode", choices=["blend", "highpass"], default="blend")
    parser.add_argument("--mamba-residual-sigmas", default="1.0")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--weights", default="0.6,0.3,0.1;0.65,0.25,0.1;0.7,0.2,0.1")
    parser.add_argument("--route", action="append", help="name[:interp[:blend[:sharpen[:radius[:clip]]]]]")
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names-file")
    parser.add_argument("--cache-dir")
    parser.add_argument("--refresh-cache", action="store_true")
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
