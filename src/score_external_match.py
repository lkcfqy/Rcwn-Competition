from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gray_utils import GRAY_MODE_CHOICES
from lr_features import FEATURE_COLUMNS, extract_lr_features, list_png_names, load_gray
from prepare_external_sr import (
    crop_to_aspect,
    degrade_x2,
    gradient_mean,
    list_images,
    load_stats,
    match_sample_stats,
    to_gray_u8,
)


HR_SIZE = (640, 512)
LR_SIZE = (320, 256)
ASPECT = HR_SIZE[0] / HR_SIZE[1]


@dataclass(frozen=True)
class SourceSpec:
    label: str
    path: str


def parse_source(raw: str) -> SourceSpec:
    if "::" not in raw:
        raise ValueError(f"Invalid --source spec: {raw}")
    label, path = raw.split("::", 1)
    return SourceSpec(label=label, path=path)


def parse_float_list(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x != ""]


def parse_str_list(raw: str) -> list[str]:
    return [x for x in raw.split(",") if x != ""]


def feature_summary(rows: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for key in FEATURE_COLUMNS:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        out[key] = (float(values.mean()), float(values.std()) if len(values) > 1 else 0.0)
    return out


def load_target_feature_rows(input_dir: str, limit: int = 0) -> list[dict[str, float]]:
    names = list_png_names(input_dir)
    if limit:
        names = names[:limit]
    rows = []
    for name in names:
        img = load_gray(Path(input_dir) / name)
        rows.append(extract_lr_features(img))
    return rows


def build_source_images(
    source: SourceSpec,
    top_grad_count: int,
    sample_count: int,
    seed: int,
    gray_mode: str,
) -> list[Path]:
    images = list_images([source.path], [], [])
    if top_grad_count > 0:
        scored = [(gradient_mean(path, gray_mode=gray_mode), path) for path in images]
        scored.sort(key=lambda item: item[0], reverse=True)
        images = [path for _, path in scored[:top_grad_count]]
    rng = random.Random(seed)
    rng.shuffle(images)
    if sample_count > 0:
        images = images[:sample_count]
    return images


def make_candidate_feature_rows(
    images: list[Path],
    stats: list[tuple[float, float]],
    match_strength: float,
    sigma: float,
    interp: str,
    sigma_jitter: float,
    noise_std: float,
    gamma_jitter: float,
    seed: int,
    gray_mode: str,
) -> list[dict[str, float]]:
    rng = random.Random(seed)
    rows: list[dict[str, float]] = []
    for path in images:
        img = to_gray_u8(path, gray_mode=gray_mode)
        if img is None:
            continue
        crop = crop_to_aspect(img, ASPECT, rng, random_crop=False)
        if min(crop.shape[:2]) < 32:
            continue
        matched = match_sample_stats(crop, rng, stats, match_strength)
        hr = cv2.resize(matched, HR_SIZE, interpolation=cv2.INTER_AREA)
        lr = degrade_x2(
            hr,
            rng,
            LR_SIZE,
            "official",
            sigma,
            interp,
            sigma_jitter,
            noise_std,
            gamma_jitter,
        )
        rows.append(extract_lr_features(lr))
    return rows


def summary_distance(
    candidate: dict[str, tuple[float, float]],
    target: dict[str, tuple[float, float]],
    mean_weight: float = 1.0,
    std_weight: float = 0.5,
) -> float:
    total = 0.0
    for key in FEATURE_COLUMNS:
        cand_mean, cand_std = candidate[key]
        tgt_mean, tgt_std = target[key]
        scale = max(tgt_std, 1e-6)
        total += mean_weight * abs(cand_mean - tgt_mean) / scale
        total += std_weight * abs(cand_std - tgt_std) / scale
    return total / len(FEATURE_COLUMNS)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, help="label::path")
    parser.add_argument("--official-target", required=True)
    parser.add_argument("--test-target")
    parser.add_argument("--match-stats-from", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--top-grad-count", type=int, default=256)
    parser.add_argument("--official-limit", type=int, default=256)
    parser.add_argument("--test-limit", type=int, default=100)
    parser.add_argument("--match-strength", type=float, default=1.0)
    parser.add_argument("--sigmas", default="0.5,0.65,0.8")
    parser.add_argument("--interps", default="area,linear,cubic")
    parser.add_argument("--sigma-jitters", default="0,0.1")
    parser.add_argument("--noise-stds", default="0,0.25")
    parser.add_argument("--gamma-jitters", default="0,0.03")
    parser.add_argument("--official-weight", type=float, default=0.7)
    parser.add_argument("--test-weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gray-mode", choices=GRAY_MODE_CHOICES, default="avg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    official_rows = load_target_feature_rows(args.official_target, limit=args.official_limit)
    if not official_rows:
        raise ValueError("No official target rows loaded.")
    official_summary = feature_summary(official_rows)

    test_summary = None
    if args.test_target:
        test_rows = load_target_feature_rows(args.test_target, limit=args.test_limit)
        if not test_rows:
            raise ValueError("No test target rows loaded.")
        test_summary = feature_summary(test_rows)

    match_stats = load_stats(args.match_stats_from, gray_mode=args.gray_mode)
    sigmas = parse_float_list(args.sigmas)
    interps = parse_str_list(args.interps)
    sigma_jitters = parse_float_list(args.sigma_jitters)
    noise_stds = parse_float_list(args.noise_stds)
    gamma_jitters = parse_float_list(args.gamma_jitters)

    results: list[dict[str, float | int | str]] = []
    for source_index, raw_source in enumerate(args.source):
        source = parse_source(raw_source)
        images = build_source_images(
            source,
            args.top_grad_count,
            args.sample_count,
            args.seed + source_index,
            gray_mode=args.gray_mode,
        )
        print(f"source={source.label} images={len(images)} path={source.path}")
        for sigma in sigmas:
            for interp in interps:
                for sigma_jitter in sigma_jitters:
                    for noise_std in noise_stds:
                        for gamma_jitter in gamma_jitters:
                            rows = make_candidate_feature_rows(
                                images=images,
                                stats=match_stats,
                                match_strength=args.match_strength,
                                sigma=sigma,
                                interp=interp,
                                sigma_jitter=sigma_jitter,
                                noise_std=noise_std,
                                gamma_jitter=gamma_jitter,
                                seed=args.seed + source_index,
                                gray_mode=args.gray_mode,
                            )
                            if not rows:
                                continue
                            candidate_summary = feature_summary(rows)
                            official_score = summary_distance(candidate_summary, official_summary)
                            test_score = summary_distance(candidate_summary, test_summary) if test_summary else 0.0
                            total_score = args.official_weight * official_score + args.test_weight * test_score
                            results.append(
                                {
                                    "source": source.label,
                                    "path": source.path,
                                    "images": len(rows),
                                    "match_strength": args.match_strength,
                                    "sigma": sigma,
                                    "interp": interp,
                                    "sigma_jitter": sigma_jitter,
                                    "noise_std": noise_std,
                                    "gamma_jitter": gamma_jitter,
                                    "official_score": official_score,
                                    "test_score": test_score,
                                    "total_score": total_score,
                                    "edge_mean": candidate_summary["edge_mean"][0],
                                    "fft_high_ratio": candidate_summary["fft_high_ratio"][0],
                                    "std": candidate_summary["std"][0],
                                }
                            )

    results.sort(key=lambda row: float(row["total_score"]))
    fieldnames = [
        "source",
        "path",
        "images",
        "match_strength",
        "sigma",
        "interp",
        "sigma_jitter",
        "noise_std",
        "gamma_jitter",
        "official_score",
        "test_score",
        "total_score",
        "edge_mean",
        "fft_high_ratio",
        "std",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"wrote={out_path}")
    for row in results[:10]:
        print(
            f"top source={row['source']} score={float(row['total_score']):.4f} "
            f"sigma={row['sigma']} interp={row['interp']} "
            f"sigjit={row['sigma_jitter']} noise={row['noise_std']} gamma={row['gamma_jitter']}"
        )


if __name__ == "__main__":
    main()
