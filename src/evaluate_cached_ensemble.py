from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dataset import load_pair, make_split
from evaluate_hat_postprocess_sweep import Route, parse_route, safe_name
from metrics import EdgeMetric, measure_batch, metric_proxy
from runtime import apply_postprocess, interp_tensor


@dataclass(frozen=True)
class Source:
    label: str
    cache_dir: Path
    key: str


def tag_float(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def parse_source(raw: str) -> Source:
    if "=" not in raw:
        raise ValueError(f"Source must be label=cache_dir:key, got {raw}")
    label, rest = raw.split("=", 1)
    if ":" not in rest:
        raise ValueError(f"Source must be label=cache_dir:key, got {raw}")
    cache_dir, key = rest.rsplit(":", 1)
    return Source(label=label, cache_dir=Path(cache_dir), key=key)


def parse_weight_sets(raw: str, n_sources: int) -> list[tuple[float, ...]]:
    weight_sets: list[tuple[float, ...]] = []
    for group in raw.split(";"):
        group = group.strip()
        if not group:
            continue
        weights = tuple(float(part.strip()) for part in group.split(",") if part.strip())
        if len(weights) != n_sources:
            raise ValueError(f"Expected {n_sources} weights, got {len(weights)} in {group}")
        total = sum(weights)
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Weights must sum to 1.0, got {total:.6f} in {group}")
        if any(value < 0.0 for value in weights):
            raise ValueError(f"Weights must be non-negative: {group}")
        weight_sets.append(weights)
    if not weight_sets:
        raise ValueError("No weight sets provided.")
    return weight_sets


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


def load_source_prediction(source: Source, name: str, device: torch.device) -> torch.Tensor:
    path = cache_path(source.cache_dir, name)
    if not path.exists():
        raise FileNotFoundError(f"Missing cache for {source.label}/{name}: {path}")
    payload = torch.load(path, map_location="cpu")
    if source.key not in payload:
        raise KeyError(f"Missing key {source.key!r} in {path}; keys={list(payload.keys())}")
    return payload[source.key].to(device=device, dtype=torch.float32)


def write_metrics_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "psnr", "ssim", "edge", "lpips", "proxy"])
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: list[dict[str, float | str]], labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "route",
        *[f"weight_{label}" for label in labels],
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
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def validate(args: Any) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    names = read_names(args)
    sources = [parse_source(raw) for raw in args.source]
    labels = [source.label for source in sources]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate source labels: {labels}")
    weight_sets = parse_weight_sets(args.weights, len(sources))
    routes = [parse_route(raw) for raw in args.route]
    jobs: list[tuple[str, tuple[float, ...], Route]] = []
    for weights in weight_sets:
        weight_tag = "_".join(f"w{label}{tag_float(value)}" for label, value in zip(labels, weights))
        for route in routes:
            jobs.append((f"{weight_tag}_{route.name}", weights, route))
    job_names = [job_name for job_name, _, _ in jobs]
    if len(set(job_names)) != len(job_names):
        duplicates = sorted({name for name in job_names if job_names.count(name) > 1})
        raise ValueError(f"Duplicate job names: {duplicates}")

    print(f"device={device} val={len(names)} sources={','.join(labels)} routes={len(jobs)}")

    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    interp_methods = sorted({route.interp for _, _, route in jobs if route.needs_base and route.interp is not None})
    sums: dict[str, dict[str, float]] = {job_name: {} for job_name, _, _ in jobs}
    metric_rows: dict[str, list[dict[str, float | str]]] = {job_name: [] for job_name, _, _ in jobs}

    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(args.data, name, args.scale, device)
        preds = [load_source_prediction(source, name, device) for source in sources]
        bases = {method: interp_tensor(lr, args.scale, method) for method in interp_methods}
        for job_name, weights, route in jobs:
            pred = sum(pred_i * weight for pred_i, weight in zip(preds, weights))
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
                route_sum = sums[job_name]
                route_sum[key] = route_sum.get(key, 0.0) + value

    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else None
    summary_rows: list[dict[str, float | str]] = []
    for job_name, weights, route in jobs:
        out = {key: value / len(names) for key, value in sums[job_name].items()}
        out.setdefault("lpips", 0.0)
        out["proxy"] = metric_proxy(out)
        metrics_csv = ""
        if metrics_dir is not None:
            metrics_path = metrics_dir / f"{safe_name(job_name)}.csv"
            write_metrics_csv(metrics_path, metric_rows[job_name])
            metrics_csv = str(metrics_path)
        row: dict[str, float | str] = {
            "route": job_name,
            "interp": route.interp or "",
            "blend_interp": route.blend_interp,
            "sharpen_amount": route.sharpen_amount,
            "sharpen_radius": route.sharpen_radius,
            "clip_mode": route.clip_mode,
            **out,
            "metrics_csv": metrics_csv,
        }
        for label, weight in zip(labels, weights):
            row[f"weight_{label}"] = weight
        summary_rows.append(row)

    summary_rows.sort(key=lambda row: float(row["proxy"]), reverse=True)
    if args.summary_csv:
        write_summary_csv(Path(args.summary_csv), summary_rows, labels)
        print(f"summary_csv={args.summary_csv}")
    if metrics_dir:
        print(f"metrics_dir={metrics_dir}")
    for row in summary_rows:
        weight_text = ",".join(f"{label}={float(row[f'weight_{label}']):.4f}" for label in labels)
        print(
            "val_result "
            + f"route={row['route']} weights={weight_text} "
            + f"interp={row['interp']} blend_interp={float(row['blend_interp']):.5f} "
            + " ".join(f"{key}={float(row[key]):.5f}" for key in ["psnr", "ssim", "edge", "lpips", "proxy"])
        )


def parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--source", action="append", required=True, help="label=cache_dir:key")
    parser.add_argument("--weights", required=True, help="Semicolon-separated weight sets matching --source order.")
    parser.add_argument("--route", action="append", required=True)
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names-file", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--summary-csv", default="")
    parser.add_argument("--metrics-dir", default="")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
