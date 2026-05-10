from __future__ import annotations

import argparse
import csv
import random
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gray_utils import GRAY_MODE_CHOICES
from lr_features import FEATURE_COLUMNS, extract_lr_features, list_png_names, load_gray
from prepare_external_sr import crop_to_aspect, degrade_x2, list_images, to_gray_u8


HR_SIZE = (640, 512)
LR_SIZE = (320, 256)
ASPECT = HR_SIZE[0] / HR_SIZE[1]


@dataclass(frozen=True)
class FeatureSummary:
    means: dict[str, float]
    stds: dict[str, float]


def feature_summary(rows: list[dict[str, float]]) -> FeatureSummary:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for key in FEATURE_COLUMNS:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        means[key] = float(values.mean())
        stds[key] = max(float(values.std()) if len(values) > 1 else 0.0, 1e-6)
    return FeatureSummary(means=means, stds=stds)


def load_target_feature_rows(input_dir: str, limit: int = 0) -> list[dict[str, float]]:
    names = list_png_names(input_dir)
    if limit:
        names = names[:limit]
    rows = []
    for name in names:
        img = load_gray(Path(input_dir) / name)
        rows.append(extract_lr_features(img))
    return rows


def image_score(
    features: dict[str, float],
    official: FeatureSummary,
    test: FeatureSummary | None,
    official_weight: float,
    test_weight: float,
) -> tuple[float, float, float]:
    official_score = 0.0
    test_score = 0.0
    for key in FEATURE_COLUMNS:
        official_score += abs(features[key] - official.means[key]) / official.stds[key]
        if test is not None:
            test_score += abs(features[key] - test.means[key]) / test.stds[key]
    official_score /= len(FEATURE_COLUMNS)
    if test is not None:
        test_score /= len(FEATURE_COLUMNS)
    total = official_weight * official_score + test_weight * test_score
    return total, official_score, test_score


def build_lr_features(
    path: Path,
    sigma: float,
    interp: str,
    sigma_jitter: float,
    noise_std: float,
    gamma_jitter: float,
    gray_mode: str,
) -> dict[str, float] | None:
    rng = random.Random(zlib.crc32(str(path).encode("utf-8")) & 0xFFFFFFFF)
    img = to_gray_u8(path, gray_mode=gray_mode)
    if img is None:
        return None
    crop = crop_to_aspect(img, ASPECT, rng=rng, random_crop=False)
    if min(crop.shape[:2]) < 32:
        return None
    hr = cv2.resize(crop, HR_SIZE, interpolation=cv2.INTER_AREA)
    # Fixed seed for deterministic ranking.
    lr = degrade_x2(
        hr,
        rng=rng,
        out_size=LR_SIZE,
        mode="official",
        sigma=sigma,
        interp=interp,
        sigma_jitter=sigma_jitter,
        noise_std=noise_std,
        gamma_jitter=gamma_jitter,
    )
    return extract_lr_features(lr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--official-target", required=True)
    parser.add_argument("--test-target")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-paths", required=True)
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument("--official-limit", type=int, default=256)
    parser.add_argument("--test-limit", type=int, default=100)
    parser.add_argument("--path-contains", action="append", default=[])
    parser.add_argument("--path-excludes", action="append", default=[])
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--interp", choices=["area", "linear", "cubic", "lanczos"], default="area")
    parser.add_argument("--sigma-jitter", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--gamma-jitter", type=float, default=0.0)
    parser.add_argument("--official-weight", type=float, default=0.7)
    parser.add_argument("--test-weight", type=float, default=0.3)
    parser.add_argument("--gray-mode", choices=GRAY_MODE_CHOICES, default="avg")
    args = parser.parse_args()

    official_rows = load_target_feature_rows(args.official_target, limit=args.official_limit)
    test_rows = (
        load_target_feature_rows(args.test_target, limit=args.test_limit) if args.test_target else []
    )
    official_summary = feature_summary(official_rows)
    test_summary = feature_summary(test_rows) if test_rows else None

    images = list_images([args.source_root], args.path_contains, args.path_excludes)
    rows: list[dict[str, str | float]] = []
    for idx, path in enumerate(images, start=1):
        features = build_lr_features(
            path=path,
            sigma=args.sigma,
            interp=args.interp,
            sigma_jitter=args.sigma_jitter,
            noise_std=args.noise_std,
            gamma_jitter=args.gamma_jitter,
            gray_mode=args.gray_mode,
        )
        if features is None:
            continue
        total_score, official_score, test_score = image_score(
            features=features,
            official=official_summary,
            test=test_summary,
            official_weight=args.official_weight,
            test_weight=args.test_weight,
        )
        rows.append(
            {
                "rank_score": total_score,
                "official_score": official_score,
                "test_score": test_score,
                "gradient_mean": float(np.hypot(features["gx_abs_mean"], features["gy_abs_mean"])),
                "path": str(path.resolve()),
                "name": path.name,
            }
        )
        if idx % 500 == 0:
            print(f"scored={idx}/{len(images)}")

    rows.sort(key=lambda row: float(row["rank_score"]))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "rank_score", "official_score", "test_score", "gradient_mean", "name", "path"],
        )
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    **row,
                }
            )

    top_rows = rows[: args.top_k]
    out_paths = Path(args.out_paths)
    out_paths.parent.mkdir(parents=True, exist_ok=True)
    out_paths.write_text(
        "\n".join(str(row["path"]) for row in top_rows) + ("\n" if top_rows else ""),
        encoding="utf-8",
    )
    print(
        f"wrote_csv={out_csv} wrote_paths={out_paths} total={len(rows)} top_k={len(top_rows)} "
        f"best={top_rows[0]['rank_score'] if top_rows else 'na'}"
    )


if __name__ == "__main__":
    main()
