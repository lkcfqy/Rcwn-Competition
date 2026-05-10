from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


FEATURE_COLUMNS = [
    "mean",
    "std",
    "p01",
    "p05",
    "p25",
    "p50",
    "p75",
    "p95",
    "p99",
    "contrast_95_05",
    "entropy",
    "edge_mean",
    "edge_std",
    "edge_p95",
    "lap_var",
    "fft_high_ratio",
    "gx_abs_mean",
    "gy_abs_mean",
    "grad_anisotropy",
]


def list_png_names(input_dir: str | Path, names: list[str] | None = None) -> list[str]:
    root = Path(input_dir)
    if names is None:
        return sorted(path.name for path in root.iterdir() if path.suffix.lower() == ".png")
    return [name for name in names if (root / name).is_file()]


def load_gray(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _entropy_u8(img: np.ndarray) -> float:
    hist = np.bincount(img.reshape(-1), minlength=256).astype(np.float64)
    probs = hist / max(float(hist.sum()), 1.0)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def _fft_high_ratio(img01: np.ndarray) -> float:
    centered = img01 - float(img01.mean())
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    h, w = power.shape
    yy, xx = np.ogrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = min(h, w) * 0.18
    mask_low = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    total = float(power.sum())
    if total <= 0:
        return 0.0
    high = float(power[~mask_low].sum())
    return high / total


def extract_lr_features(img_u8: np.ndarray) -> dict[str, float]:
    img01 = img_u8.astype(np.float32) / 255.0
    flat = img01.reshape(-1)
    percentiles = np.percentile(flat, [1, 5, 25, 50, 75, 95, 99])

    gx = cv2.Sobel(img01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img01, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    lap = cv2.Laplacian(img01, cv2.CV_32F, ksize=3)

    gx_abs_mean = float(np.abs(gx).mean())
    gy_abs_mean = float(np.abs(gy).mean())
    grad_sum = gx_abs_mean + gy_abs_mean
    anisotropy = 0.0 if grad_sum <= 1e-12 else abs(gx_abs_mean - gy_abs_mean) / grad_sum

    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p50": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
        "contrast_95_05": float(percentiles[5] - percentiles[1]),
        "entropy": _entropy_u8(img_u8),
        "edge_mean": float(edge.mean()),
        "edge_std": float(edge.std()),
        "edge_p95": float(np.percentile(edge.reshape(-1), 95)),
        "lap_var": float(lap.var()),
        "fft_high_ratio": _fft_high_ratio(img01),
        "gx_abs_mean": gx_abs_mean,
        "gy_abs_mean": gy_abs_mean,
        "grad_anisotropy": anisotropy,
    }


def summarize_feature_ranges(rows: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for key in FEATURE_COLUMNS:
        values = [float(row[key]) for row in rows]
        if not values:
            out[key] = (math.nan, math.nan)
            continue
        out[key] = (float(min(values)), float(max(values)))
    return out
