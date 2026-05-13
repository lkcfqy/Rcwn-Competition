from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


METRIC_KEYS = ("psnr", "ssim", "edge", "lpips")
ALL_KEYS = (*METRIC_KEYS, "proxy")


@dataclass(frozen=True)
class Candidate:
    source: str
    route: str
    psnr: float
    ssim: float
    edge: float
    lpips: float
    proxy: float

    @property
    def metrics(self) -> np.ndarray:
        return np.array([self.psnr, self.ssim, self.edge, self.lpips], dtype=np.float64)


def _float_or_none(raw: str | None) -> float | None:
    try:
        return float(raw) if raw not in {None, ""} else None
    except ValueError:
        return None


def read_submission_fit_rows(path: Path) -> list[tuple[str, np.ndarray, float, float]]:
    rows: list[tuple[str, np.ndarray, float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            score = _float_or_none(row.get("platform_score"))
            proxy = _float_or_none(row.get("local_proxy"))
            metrics = [_float_or_none(row.get(f"local_{key}")) for key in METRIC_KEYS]
            if score is None or proxy is None or any(value is None for value in metrics):
                continue
            rows.append((row.get("zip", ""), np.array(metrics, dtype=np.float64), score, proxy))
    if len(rows) < 3:
        raise ValueError(f"Need at least 3 submitted rows with scores, got {len(rows)} from {path}")
    return rows


def fit_proxy(rows: list[tuple[str, np.ndarray, float, float]]) -> np.ndarray:
    y = np.array([row[2] for row in rows], dtype=np.float64)
    x = np.array([[1.0, row[3]] for row in rows], dtype=np.float64)
    return np.linalg.lstsq(x, y, rcond=None)[0]


def fit_ridge(rows: list[tuple[str, np.ndarray, float, float]], ridge_lambda: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.array([row[2] for row in rows], dtype=np.float64)
    features = np.stack([row[1] for row in rows])
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0.0] = 1.0
    z = (features - mean) / std
    x = np.column_stack([np.ones(len(z), dtype=np.float64), z])
    penalty = np.eye(x.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(x.T @ x + ridge_lambda * penalty, x.T @ y)
    return coef, mean, std


def rmse(values: np.ndarray, predictions: np.ndarray) -> float:
    return math.sqrt(float(np.mean((values - predictions) ** 2)))


def read_candidate_csv(path: Path) -> list[Candidate]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except UnicodeDecodeError:
        return []
    if not rows or not all(key in rows[0] for key in ALL_KEYS):
        return []

    candidates: list[Candidate] = []
    has_name = "name" in rows[0]
    if has_name and len(rows) > 1:
        values_by_key: dict[str, list[float]] = {key: [] for key in ALL_KEYS}
        for row in rows:
            for key in ALL_KEYS:
                value = _float_or_none(row.get(key))
                if value is None:
                    return []
                values_by_key[key].append(value)
        means = {key: sum(values) / len(values) for key, values in values_by_key.items()}
        candidates.append(
            Candidate(
                source=str(path),
                route=f"aggregate_{len(rows)}",
                psnr=means["psnr"],
                ssim=means["ssim"],
                edge=means["edge"],
                lpips=means["lpips"],
                proxy=means["proxy"],
            )
        )
        return candidates

    for index, row in enumerate(rows):
        values = {key: _float_or_none(row.get(key)) for key in ALL_KEYS}
        if any(value is None for value in values.values()):
            continue
        route = row.get("route") or row.get("recipe") or row.get("name") or f"row{index}"
        candidates.append(
            Candidate(
                source=str(path),
                route=route,
                psnr=float(values["psnr"]),
                ssim=float(values["ssim"]),
                edge=float(values["edge"]),
                lpips=float(values["lpips"]),
                proxy=float(values["proxy"]),
            )
        )
    return candidates


def collect_candidates(paths: list[Path], require_text: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    for path in paths:
        if path.is_dir():
            csv_paths = sorted(path.rglob("*.csv"))
        else:
            csv_paths = [path]
        for csv_path in csv_paths:
            if require_text and require_text not in str(csv_path):
                continue
            if csv_path.name in {"submission_log.csv", "probe_summary_20260510.csv"}:
                continue
            candidates.extend(read_candidate_csv(csv_path))
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-log", default="experiments/submission_log.csv")
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--require-text", default="full120")
    parser.add_argument("--ridge-lambda", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    fit_rows = read_submission_fit_rows(Path(args.submission_log))
    proxy_coef = fit_proxy(fit_rows)
    ridge_coef, ridge_mean, ridge_std = fit_ridge(fit_rows, args.ridge_lambda)

    y = np.array([row[2] for row in fit_rows], dtype=np.float64)
    proxy_x = np.array([[1.0, row[3]] for row in fit_rows], dtype=np.float64)
    ridge_features = np.stack([row[1] for row in fit_rows])
    ridge_x = np.column_stack([np.ones(len(fit_rows)), (ridge_features - ridge_mean) / ridge_std])

    print(
        "fit_proxy "
        f"score={proxy_coef[0]:.6f}+{proxy_coef[1]:.6f}*proxy "
        f"rmse={rmse(y, proxy_x @ proxy_coef):.5f}"
    )
    print(
        "fit_ridge "
        f"lambda={args.ridge_lambda:g} rmse={rmse(y, ridge_x @ ridge_coef):.5f} "
        f"coef_std={','.join(f'{value:.6f}' for value in ridge_coef)}"
    )

    paths = [Path(raw) for raw in (args.candidate or ["experiments"])]
    candidates = collect_candidates(paths, args.require_text)
    if not candidates:
        print("no_candidates")
        return

    def predict_proxy(candidate: Candidate) -> float:
        return float(np.array([1.0, candidate.proxy]) @ proxy_coef)

    def predict_ridge(candidate: Candidate) -> float:
        z = (candidate.metrics - ridge_mean) / ridge_std
        return float(np.r_[1.0, z] @ ridge_coef)

    ranked = sorted(candidates, key=predict_ridge, reverse=True)
    print("ranked_by_ridge")
    for candidate in ranked[: args.top_k]:
        print(
            f"ridge={predict_ridge(candidate):.4f} "
            f"proxy_fit={predict_proxy(candidate):.4f} "
            f"proxy={candidate.proxy:.5f} "
            f"psnr={candidate.psnr:.5f} ssim={candidate.ssim:.5f} "
            f"edge={candidate.edge:.5f} lpips={candidate.lpips:.5f} "
            f"source={candidate.source} route={candidate.route}"
        )


if __name__ == "__main__":
    main()
