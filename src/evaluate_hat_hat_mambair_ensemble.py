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
from evaluate_hat_postprocess_sweep import Route, parse_route
from evaluate_mambairv2_gray import (
    build_mambairv2,
    forward_gray as forward_mambair_gray,
    normalize_state as normalize_mambair_state,
)
from metrics import EdgeMetric, measure_batch, metric_proxy
from runtime import _aug, _deaug, _gaussian_blur, apply_postprocess, interp_tensor


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def read_names(args: Any) -> list[str]:
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    else:
        names = split.val
    return names[: args.limit] if args.limit else names


def safe_name(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)


def parse_weight_sets(raw: str, include_atd: bool) -> list[tuple[float, float, float, float]]:
    weight_sets: list[tuple[float, float, float, float]] = []
    expected = 4 if include_atd else 3
    label = "quadruplet 'hat_a,hat_b,mamba,atd'" if include_atd else "triplet 'hat_a,hat_b,mamba'"
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [float(x.strip()) for x in chunk.split(",")]
        if len(parts) != expected:
            raise ValueError(f"Expected {label}, got: {chunk}")
        total = sum(parts)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total:.8f}: {chunk}")
        if include_atd:
            weight_sets.append((parts[0], parts[1], parts[2], parts[3]))
        else:
            weight_sets.append((parts[0], parts[1], parts[2], 0.0))
    if not weight_sets:
        raise ValueError("No weight sets parsed")
    return weight_sets


def parse_triplets(raw: str) -> list[tuple[float, float, float]]:
    return [(wa, wb, wm) for wa, wb, wm, _ in parse_weight_sets(raw, include_atd=False)]


def tag_float(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def parse_float_list(raw: str) -> list[float]:
    values = [float(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


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
                "weight_hat_a",
                "weight_hat_b",
                "weight_mamba",
                "weight_atd",
                "interp",
                "blend_interp",
                "sharpen_amount",
                "sharpen_radius",
                "clip_mode",
                "mamba_mix_mode",
                "mamba_residual_sigma",
                "mamba_gray_mode",
                "tone_gamma",
                "tone_gain",
                "tone_bias",
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


def forward_mambair_tta(model: torch.nn.Module, lr: torch.Tensor, gray_mode: str) -> torch.Tensor:
    preds = []
    for mode in range(8):
        pred = forward_mambair_gray(model, _aug(lr, mode), gray_mode)
        preds.append(_deaug(pred, mode))
    return torch.stack(preds).mean(dim=0)


def apply_tone(pred: torch.Tensor, gamma: float, gain: float, bias: float) -> torch.Tensor:
    if abs(gamma - 1.0) <= 1e-12 and abs(gain - 1.0) <= 1e-12 and abs(bias) <= 1e-12:
        return pred
    pred = pred.float().clamp(0, 1)
    if abs(gamma - 1.0) > 1e-12:
        pred = pred.pow(gamma)
    if abs(gain - 1.0) > 1e-12 or abs(bias) > 1e-12:
        pred = pred * gain + bias
    return pred.clamp(0, 1)


@torch.no_grad()
def validate(args: Any) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    names = read_names(args)
    print(f"device={device} val={len(names)} scale=x{args.scale}")

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
    mamba_gray_mode = args.mamba_gray_mode or args.gray_mode
    print(f"loaded_mamba={args.mamba_weights} key={args.mamba_param_key} gray_mode={mamba_gray_mode}")

    atd = None
    forward_atd_gray = None
    if args.atd_weights:
        from evaluate_atd_gray import build_atd, forward_gray as _forward_atd_gray, normalize_state as normalize_atd_state

        atd = build_atd(args.atd_variant, args.scale, args.atd_use_checkpoint).to(device).eval()
        atd_ckpt = torch.load(args.atd_weights, map_location="cpu")
        atd.load_state_dict(normalize_atd_state(atd_ckpt, args.atd_param_key), strict=True)
        forward_atd_gray = _forward_atd_gray
        print(f"loaded_atd={args.atd_weights} key={args.atd_param_key} variant={args.atd_variant}")

    weight_sets = parse_weight_sets(args.weights, include_atd=atd is not None)
    routes = [parse_route(raw) for raw in args.route] if args.route else [Route(name="raw")]
    residual_sigmas = [0.0] if args.mamba_mix_mode == "blend" else parse_float_list(args.mamba_residual_sigmas)
    if args.mamba_mix_mode == "highpass" and any(value <= 0 for value in residual_sigmas):
        raise ValueError("--mamba-residual-sigmas must be positive for highpass mode.")
    tone_gammas = parse_float_list(args.tone_gammas)
    tone_gains = parse_float_list(args.tone_gains)
    tone_biases = parse_float_list(args.tone_biases)
    jobs: list[tuple[str, tuple[float, float, float, float], Route, float, float, float, float]] = []
    for weights in weight_sets:
        wa, wb, wm, watd = weights
        weight_tag = f"wa{tag_float(wa)}_wb{tag_float(wb)}_wm{tag_float(wm)}"
        if atd is not None:
            weight_tag += f"_watd{tag_float(watd)}"
        if mamba_gray_mode != args.gray_mode:
            weight_tag += f"_mgray{mamba_gray_mode}"
        for route in routes:
            for sigma in residual_sigmas:
                sigma_tag = f"_hp{tag_float(sigma)}" if args.mamba_mix_mode == "highpass" else ""
                for gamma in tone_gammas:
                    for gain in tone_gains:
                        for bias in tone_biases:
                            tone_tag = ""
                            if abs(gamma - 1.0) > 1e-12:
                                tone_tag += f"_g{tag_float(gamma)}"
                            if abs(gain - 1.0) > 1e-12:
                                tone_tag += f"_k{tag_float(gain)}"
                            if abs(bias) > 1e-12:
                                tone_tag += f"_b{tag_float(bias)}"
                            jobs.append((f"{weight_tag}_{route.name}{sigma_tag}{tone_tag}", weights, route, sigma, gamma, gain, bias))

    sums = {job_name: {} for job_name, *_ in jobs}
    metric_rows: dict[str, list[dict[str, float | str]]] = {job_name: [] for job_name, *_ in jobs}
    interp_methods = sorted({route.interp for _, _, route, *_ in jobs if route.needs_base and route.interp is not None})
    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    for name in tqdm(names, desc="val", leave=False):
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
                pred_m = forward_mambair_tta(mamba, lr, mamba_gray_mode).float()
            else:
                pred_m = forward_mambair_gray(mamba, lr, mamba_gray_mode).float()
            pred_atd = None
            if atd is not None:
                if forward_atd_gray is None:
                    raise RuntimeError("ATD forward function was not initialized.")
                pred_atd = forward_atd_gray(atd, lr, args.atd_gray_mode).float()

        bases = {method: interp_tensor(lr, args.scale, method) for method in interp_methods}
        for job_name, weights, route, sigma, gamma, gain, bias in jobs:
            wa, wb, wm, watd = weights
            if args.mamba_mix_mode == "highpass":
                if watd:
                    raise ValueError("ATD four-way mode only supports --mamba-mix-mode blend.")
                hat_total = wa + wb
                if hat_total <= 0:
                    raise ValueError("Highpass Mamba mix requires positive HAT weights.")
                hat_base = (pred_a * wa + pred_b * wb) / hat_total
                residual = pred_m - hat_base
                pred = hat_base + wm * (residual - _gaussian_blur(residual, sigma))
            else:
                pred = pred_a * wa + pred_b * wb + pred_m * wm
                if watd:
                    if pred_atd is None:
                        raise RuntimeError("ATD prediction was not computed.")
                    pred = pred + pred_atd * watd
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
            pred = apply_tone(pred, gamma, gain, bias)
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
    for job_name, weights, route, sigma, gamma, gain, bias in jobs:
        out = {key: value / len(names) for key, value in sums[job_name].items()}
        out["proxy"] = metric_proxy(out)
        metrics_csv = ""
        if metrics_dir is not None:
            metrics_path = metrics_dir / f"{safe_name(job_name)}.csv"
            write_metrics_csv(metrics_path, metric_rows[job_name])
            metrics_csv = str(metrics_path)
            print(f"metrics_csv route={job_name} weights={weights} path={metrics_path}")
        wa, wb, wm, watd = weights
        summary_rows.append(
            {
                "route": job_name,
                "weight_hat_a": wa,
                "weight_hat_b": wb,
                "weight_mamba": wm,
                "weight_atd": watd,
                "interp": route.interp or "",
                "blend_interp": route.blend_interp,
                "sharpen_amount": route.sharpen_amount,
                "sharpen_radius": route.sharpen_radius,
                "clip_mode": route.clip_mode,
                "mamba_mix_mode": args.mamba_mix_mode,
                "mamba_residual_sigma": sigma,
                "mamba_gray_mode": mamba_gray_mode,
                "tone_gamma": gamma,
                "tone_gain": gain,
                "tone_bias": bias,
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
            + f"weights={float(row['weight_hat_a']):.4f},{float(row['weight_hat_b']):.4f},"
            + f"{float(row['weight_mamba']):.4f},{float(row['weight_atd']):.4f} "
            + f"interp={row['interp']} blend_interp={float(row['blend_interp']):.5f} "
            + f"sharpen_amount={float(row['sharpen_amount']):.5f} clip_mode={row['clip_mode']} "
            + f"mamba_mix_mode={row['mamba_mix_mode']} mamba_residual_sigma={float(row['mamba_residual_sigma']):.5f} "
            + f"mamba_gray_mode={row['mamba_gray_mode']} "
            + f"tone_gamma={float(row['tone_gamma']):.5f} tone_gain={float(row['tone_gain']):.5f} "
            + f"tone_bias={float(row['tone_bias']):.5f} "
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
    parser.add_argument("--mamba-gray-mode", choices=["avg", "y", "r", "g", "b"], default="")
    parser.add_argument("--atd-weights", default="")
    parser.add_argument("--atd-variant", choices=["base", "light"], default="base")
    parser.add_argument("--atd-param-key", default="params_ema")
    parser.add_argument("--atd-gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--atd-use-checkpoint", action="store_true")
    parser.add_argument("--tone-gammas", default="1.0")
    parser.add_argument("--tone-gains", default="1.0")
    parser.add_argument("--tone-biases", default="0.0")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--weights", default="0.6,0.3,0.1;0.65,0.25,0.1;0.7,0.2,0.1")
    parser.add_argument("--route", action="append", help="name[:interp[:blend[:sharpen[:radius[:clip]]]]]")
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
