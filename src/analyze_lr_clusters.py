from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dataset import make_split
from lr_features import FEATURE_COLUMNS, extract_lr_features, list_png_names, load_gray


def parse_dataset_source(spec: str) -> tuple[str, str, int, str]:
    parts = spec.split("::")
    if len(parts) != 4:
        raise ValueError(f"Invalid --dataset-source spec: {spec}")
    label, root_dir, scale_raw, split_name = parts
    return label, root_dir, int(scale_raw), split_name


def parse_dir_source(spec: str) -> tuple[str, str]:
    parts = spec.split("::")
    if len(parts) != 2:
        raise ValueError(f"Invalid --dir-source spec: {spec}")
    return parts[0], parts[1]


def source_rows(args) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []

    for spec in args.dataset_source:
        label, root_dir, scale, split_name = parse_dataset_source(spec)
        lr_dir = Path(root_dir) / f"input_{640 // scale}"
        if split_name == "all":
            names = list_png_names(lr_dir)
        else:
            split = make_split(root_dir, scale, val_count=args.val_count, seed=args.seed)
            if split_name == "train":
                names = split.train
            elif split_name == "val":
                names = split.val
            else:
                raise ValueError(f"Unsupported split: {split_name}")
        for name in names:
            img = load_gray(lr_dir / name)
            row: dict[str, float | str | int] = {
                "dataset": label,
                "name": name,
                "path": str(lr_dir / name),
                "height": int(img.shape[0]),
                "width": int(img.shape[1]),
            }
            row.update(extract_lr_features(img))
            rows.append(row)

    for spec in args.dir_source:
        label, input_dir = parse_dir_source(spec)
        for name in list_png_names(input_dir):
            img = load_gray(Path(input_dir) / name)
            row = {
                "dataset": label,
                "name": name,
                "path": str(Path(input_dir) / name),
                "height": int(img.shape[0]),
                "width": int(img.shape[1]),
            }
            row.update(extract_lr_features(img))
            rows.append(row)

    return rows


def standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (matrix - mean) / std, mean, std


def init_kmeans_pp(matrix: np.ndarray, k: int, rng: random.Random) -> np.ndarray:
    count = matrix.shape[0]
    centers = [matrix[rng.randrange(count)]]
    while len(centers) < k:
        dist2 = np.min(
            np.stack([np.sum((matrix - center) ** 2, axis=1) for center in centers], axis=0),
            axis=0,
        )
        total = float(dist2.sum())
        if total <= 0:
            centers.append(matrix[rng.randrange(count)])
            continue
        target = rng.random() * total
        acc = 0.0
        index = count - 1
        for idx, value in enumerate(dist2):
            acc += float(value)
            if acc >= target:
                index = idx
                break
        centers.append(matrix[index])
    return np.stack(centers, axis=0)


def run_kmeans(matrix: np.ndarray, k: int, seed: int, max_iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    if matrix.shape[0] < k:
        raise ValueError(f"Need at least {k} rows for clustering, got {matrix.shape[0]}.")
    rng = random.Random(seed)
    centers = init_kmeans_pp(matrix, k, rng)

    for _ in range(max_iters):
        dist2 = np.sum((matrix[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = dist2.argmin(axis=1)
        new_centers = centers.copy()
        changed = False
        for idx in range(k):
            mask = labels == idx
            if not np.any(mask):
                farthest = int(np.argmax(np.min(dist2, axis=1)))
                new_centers[idx] = matrix[farthest]
                changed = True
                continue
            cluster_mean = matrix[mask].mean(axis=0)
            if not np.allclose(cluster_mean, centers[idx]):
                changed = True
            new_centers[idx] = cluster_mean
        centers = new_centers
        if not changed:
            break

    dist2 = np.sum((matrix[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = dist2.argmin(axis=1)
    return labels, centers


def write_rows(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "name", "path", "height", "width", *FEATURE_COLUMNS, "cluster"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, float | str | int]], k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cluster_sets: dict[int, list[dict[str, float | str | int]]] = defaultdict(list)
    for row in rows:
        cluster_sets[int(row["cluster"])].append(row)

    datasets = sorted({str(row["dataset"]) for row in rows})
    fieldnames = ["cluster", "count", *[f"count_{label}" for label in datasets], *FEATURE_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cluster in range(k):
            members = cluster_sets.get(cluster, [])
            counts = Counter(str(row["dataset"]) for row in members)
            out: dict[str, float | str | int] = {
                "cluster": cluster,
                "count": len(members),
            }
            for label in datasets:
                out[f"count_{label}"] = counts.get(label, 0)
            for key in FEATURE_COLUMNS:
                values = [float(row[key]) for row in members]
                out[key] = float(np.mean(values)) if values else float("nan")
            writer.writerow(out)


def write_cluster_name_lists(out_dir: Path, rows: list[dict[str, float | str | int]], k: int) -> None:
    grouped: dict[tuple[str, int], list[str]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), int(row["cluster"]))].append(str(row["name"]))
    for (label, cluster), names in grouped.items():
        path = out_dir / label / f"cluster_{cluster}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{name}\n" for name in sorted(names)), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-source",
        action="append",
        default=[],
        help="label::root_dir::scale::split, where split is train/val/all",
    )
    parser.add_argument(
        "--dir-source",
        action="append",
        default=[],
        help="label::input_dir",
    )
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--cluster-name-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = source_rows(args)
    if not rows:
        raise ValueError("No source rows loaded.")

    matrix = np.array([[float(row[key]) for key in FEATURE_COLUMNS] for row in rows], dtype=np.float64)
    matrix, _, _ = standardize(matrix)
    labels, _ = run_kmeans(matrix, args.k, args.seed)

    for row, label in zip(rows, labels.tolist()):
        row["cluster"] = int(label)

    out_csv = Path(args.out_csv)
    summary_csv = Path(args.summary_csv)
    write_rows(out_csv, rows)
    write_summary(summary_csv, rows, args.k)
    if args.cluster_name_dir:
        write_cluster_name_lists(Path(args.cluster_name_dir), rows, args.k)
    counts = Counter(int(row["cluster"]) for row in rows)
    print(f"wrote_features={out_csv}")
    print(f"wrote_summary={summary_csv}")
    print("cluster_counts " + " ".join(f"{idx}={counts.get(idx, 0)}" for idx in range(args.k)))


if __name__ == "__main__":
    main()
