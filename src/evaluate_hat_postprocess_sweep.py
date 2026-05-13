from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import load_pair, make_split
from evaluate_hat_gray import (
    autocast_context,
    build_hat,
    forward_gray,
    load_hat_weights,
    preprocess_lr,
)
from metrics import EdgeMetric, measure_batch, metric_proxy
from runtime import INTERP, apply_postprocess, interp_tensor


@dataclass(frozen=True)
class Route:
    name: str
    interp: str | None = None
    blend_interp: float = 0.0
    sharpen_amount: float = 0.0
    sharpen_radius: float = 1.0
    clip_mode: str = "hard"

    @property
    def needs_base(self) -> bool:
        return self.blend_interp > 0.0 or self.clip_mode == "match-base"


def parse_route(raw: str) -> Route:
    """Parse route specs like raw or cubic_b002:cubic:0.02:0.0:hard."""
    if "=" in raw:
        name, rest = raw.split("=", 1)
        parts = [name, *rest.replace(",", ":").split(":")]
    else:
        parts = raw.replace(",", ":").split(":")
    parts = [part.strip() for part in parts if part.strip() != ""]
    if not parts:
        raise ValueError("Empty route spec.")

    name = parts[0]
    interp = parts[1] if len(parts) >= 2 else None
    blend_interp = float(parts[2]) if len(parts) >= 3 else 0.0
    sharpen_amount = float(parts[3]) if len(parts) >= 4 else 0.0
    sharpen_radius = float(parts[4]) if len(parts) >= 5 else 1.0
    clip_mode = parts[5] if len(parts) >= 6 else "hard"
    if len(parts) > 6:
        raise ValueError(f"Too many fields in route spec: {raw}")
    if interp is not None and interp not in INTERP:
        raise ValueError(f"Unsupported interp for route {name}: {interp}")
    if clip_mode not in {"hard", "match-base", "none"}:
        raise ValueError(f"Unsupported clip mode for route {name}: {clip_mode}")
    if blend_interp < 0.0:
        raise ValueError(f"Negative blend for route {name}: {blend_interp}")
    if sharpen_amount < 0.0:
        raise ValueError(f"Negative sharpen for route {name}: {sharpen_amount}")
    if sharpen_radius <= 0.0:
        raise ValueError(f"Non-positive sharpen radius for route {name}: {sharpen_radius}")
    return Route(
        name=name,
        interp=interp,
        blend_interp=blend_interp,
        sharpen_amount=sharpen_amount,
        sharpen_radius=sharpen_radius,
        clip_mode=clip_mode,
    )


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)


def parse_phase_pad(raw: str) -> tuple[int, int]:
    if "," in raw:
        parts = raw.split(",", 1)
    elif ":" in raw:
        parts = raw.split(":", 1)
    else:
        parts = [raw, raw]
    top = int(parts[0])
    left = int(parts[1])
    if top < 0 or left < 0:
        raise ValueError(f"Phase pads must be non-negative: {raw}")
    return top, left


def read_names(args) -> list[str]:
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    else:
        names = split.val
    return names[: args.limit] if args.limit else names


@torch.no_grad()
def forward_with_phase_pads(
    model: torch.nn.Module,
    lr: torch.Tensor,
    scale: int,
    gray_mode: str,
    tta: bool,
    native_io: bool,
    phase_pads: list[tuple[int, int]],
) -> torch.Tensor:
    if not phase_pads:
        phase_pads = [(0, 0)]
    _, _, h, w = lr.shape
    preds = []
    for pad_top, pad_left in phase_pads:
        if pad_top == 0 and pad_left == 0:
            phase_lr = lr
            crop_top = crop_left = 0
        else:
            phase_lr = F.pad(lr, (pad_left, 0, pad_top, 0), mode="reflect")
            crop_top = pad_top * scale
            crop_left = pad_left * scale
        pred = forward_gray(model, phase_lr, scale, gray_mode, tta=tta, native_io=native_io).float()
        preds.append(pred[..., crop_top : crop_top + h * scale, crop_left : crop_left + w * scale])
    return torch.stack(preds).mean(dim=0)


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
                "alpha_a",
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
    routes = [parse_route(raw) for raw in args.route]
    phase_pads = [parse_phase_pad(raw) for raw in args.phase_pad]
    phase_pads_b = [parse_phase_pad(raw) for raw in (args.phase_pad_b or args.phase_pad)]
    route_names = [route.name for route in routes]
    if len(route_names) != len(set(route_names)):
        raise ValueError(f"Route names must be unique: {route_names}")
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    if not alphas:
        raise ValueError("At least one alpha is required.")
    if not args.weights_b:
        if any(abs(alpha - 1.0) > 1e-9 for alpha in alphas):
            raise ValueError("--alphas other than 1.0 require --weights-b.")
        alphas = [1.0]
    if any(alpha < 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError(f"Alphas must be in [0, 1]: {alphas}")

    route_jobs: list[tuple[str, float, Route]] = []
    for alpha in alphas:
        alpha_tag = f"a{alpha:.3f}".rstrip("0").rstrip(".").replace(".", "p")
        for route in routes:
            job_name = route.name if not args.weights_b else f"{alpha_tag}_{route.name}"
            route_jobs.append((job_name, alpha, route))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")
    names = read_names(args)
    print(f"val={len(names)} scale=x{args.scale} routes={len(route_jobs)}")

    model = build_hat(args.variant, args.scale, args.use_checkpoint, native_io=args.native_io).to(device).eval()
    load_hat_weights(model, args.weights, args.param_key, native_io=args.native_io, gray_mode=args.gray_mode)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params={n_params:.2f}M preset=hat_{args.variant}")
    print(f"loaded={args.weights} param_key={args.param_key}")
    model_b = None
    if args.weights_b:
        model_b = build_hat(args.variant_b, args.scale, args.use_checkpoint_b, native_io=args.native_io_b).to(device).eval()
        load_hat_weights(model_b, args.weights_b, args.param_key_b, native_io=args.native_io_b, gray_mode=args.gray_mode)
        n_params_b = sum(p.numel() for p in model_b.parameters()) / 1e6
        print(f"params_b={n_params_b:.2f}M preset=hat_{args.variant_b}")
        print(f"loaded_b={args.weights_b} param_key={args.param_key_b}")

    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    sums: dict[str, dict[str, float]] = {name: {} for name, _, _ in route_jobs}
    metric_rows: dict[str, list[dict[str, float | str]]] = {name: [] for name, _, _ in route_jobs}
    interp_methods = sorted({route.interp for route in routes if route.needs_base and route.interp is not None})

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
            pred = forward_with_phase_pads(
                model,
                model_lr,
                args.scale,
                args.gray_mode,
                args.tta,
                args.native_io,
                phase_pads,
            )
        pred = pred.float()
        pred_b = None
        if model_b is not None:
            with autocast_context(device, args.amp):
                pred_b = forward_with_phase_pads(
                    model_b,
                    model_lr,
                    args.scale,
                    args.gray_mode,
                    args.tta_b,
                    args.native_io_b,
                    phase_pads_b,
                ).float()
        bases = {method: interp_tensor(lr, args.scale, method) for method in interp_methods}

        for job_name, alpha, route in route_jobs:
            job_pred = pred if pred_b is None else pred * alpha + pred_b * (1.0 - alpha)
            base = bases.get(route.interp) if route.needs_base else None
            route_pred = apply_postprocess(
                job_pred,
                base,
                lr=lr,
                blend_interp=route.blend_interp,
                sharpen_amount=route.sharpen_amount,
                sharpen_radius=route.sharpen_radius,
                clip_mode=route.clip_mode,
            )
            metrics = measure_batch(route_pred.float(), hr, edge_metric, lpips_fn)
            one = dict(metrics)
            one.setdefault("lpips", 0.0)
            one["proxy"] = metric_proxy(one)
            row: dict[str, float | str] = {"name": name}
            row.update(one)
            metric_rows[job_name].append(row)
            for key, value in metrics.items():
                route_sum = sums[job_name]
                route_sum[key] = route_sum.get(key, 0.0) + value

    summary_rows: list[dict[str, float | str]] = []
    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else None
    for job_name, alpha, route in route_jobs:
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
                "alpha_a": alpha,
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
            + f"route={row['route']} alpha_a={float(row['alpha_a']):.5f} "
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
    parser.add_argument("--weights-b")
    parser.add_argument("--variant-b", choices=["l", "m"], default="l")
    parser.add_argument("--param-key-b", default="state_dict")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--native-io", action="store_true")
    parser.add_argument("--native-io-b", action="store_true")
    parser.add_argument("--alphas", default="1.0")
    parser.add_argument("--route", action="append", required=True, help="name[:interp[:blend[:sharpen[:radius[:clip]]]]]")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--tta-b", action="store_true")
    parser.add_argument(
        "--phase-pad",
        action="append",
        default=[],
        help="Reflect-pad top,left pixels before inference and crop back after x2, e.g. 4,4. Can be repeated.",
    )
    parser.add_argument(
        "--phase-pad-b",
        action="append",
        default=[],
        help="Optional phase pads for model B; defaults to --phase-pad.",
    )
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
    parser.add_argument("--use-checkpoint-b", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
