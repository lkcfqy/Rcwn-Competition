from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from dataset import list_image_names, make_split
from nearest_hybrid import build_nearest_index, feature_from_image, psnr_u8


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def topk_neighbors(
    query: np.ndarray,
    pool_feats: np.ndarray,
    pool_names: list[str],
    pool_lrs: dict[str, np.ndarray],
    feature_size: tuple[int, int],
    topk: int,
) -> list[tuple[int, str, float, float]]:
    q = feature_from_image(query, feature_size)
    sims = pool_feats @ q
    k = min(topk, len(pool_names))
    candidate_idx = np.argpartition(-sims, np.arange(k))[:k]
    candidate_idx = sorted(candidate_idx.tolist(), key=lambda idx: float(sims[idx]), reverse=True)
    rows = []
    for rank, idx in enumerate(candidate_idx, start=1):
        name = pool_names[idx]
        rows.append((rank, name, float(sims[idx]), psnr_u8(query, pool_lrs[name])))
    return rows


def parse_size(raw: str) -> tuple[int, int]:
    w, h = raw.lower().split("x", 1)
    return int(w), int(h)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "test"], required=True)
    parser.add_argument("--train-root", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--test-input", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/初赛测试集/input_320")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-size", type=parse_size, default=(80, 64))
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--select-topk", type=int, default=1)
    parser.add_argument("--lr-psnr-threshold", type=float, default=40.0)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--names-out", required=True)
    args = parser.parse_args()

    train_root = Path(args.train_root)
    lr_dir = train_root / f"input_{640 // args.scale}"
    if args.mode == "val":
        split = make_split(args.train_root, args.scale, val_count=args.val_count, seed=args.seed)
        query_items = [(name, lr_dir / name) for name in split.val]
        pool_names = split.train
    else:
        query_items = [(path.name, path) for path in sorted(Path(args.test_input).glob("*.png"))]
        pool_names = list_image_names(args.train_root, args.scale)

    pool_feats, pool_lrs = build_nearest_index(train_root, pool_names, args.scale, args.feature_size)
    selected: list[str] = []
    rows: list[dict[str, str | int | float]] = []

    for query_name, query_path in query_items:
        query = read_gray(query_path)
        neighbors = topk_neighbors(query, pool_feats, pool_names, pool_lrs, args.feature_size, args.topk)
        top_lr_psnr = neighbors[0][3] if neighbors else 0.0
        high_confidence = top_lr_psnr >= args.lr_psnr_threshold
        for rank, train_name, sim, lr_psnr in neighbors:
            use_for_support = high_confidence and rank <= args.select_topk
            if use_for_support:
                selected.append(train_name)
            rows.append(
                {
                    "mode": args.mode,
                    "query": query_name,
                    "rank": rank,
                    "train_name": train_name,
                    "feature_sim": sim,
                    "lr_psnr": lr_psnr,
                    "top_lr_psnr": top_lr_psnr,
                    "selected": int(use_for_support),
                }
            )

    Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv_out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["mode", "query", "rank", "train_name", "feature_sim", "lr_psnr", "top_lr_psnr", "selected"],
        )
        writer.writeheader()
        writer.writerows(rows)

    unique_selected = sorted(dict.fromkeys(selected))
    Path(args.names_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.names_out).write_text("\n".join(unique_selected) + ("\n" if unique_selected else ""), encoding="utf-8")
    print(
        f"mode={args.mode} queries={len(query_items)} pool={len(pool_names)} "
        f"selected={len(unique_selected)} threshold={args.lr_psnr_threshold} select_topk={args.select_topk}"
    )
    for row in rows:
        if row["rank"] == 1 and float(row["top_lr_psnr"]) >= args.lr_psnr_threshold:
            print(
                f"hit query={row['query']} train={row['train_name']} "
                f"sim={float(row['feature_sim']):.6f} lr_psnr={float(row['lr_psnr']):.2f}"
            )


if __name__ == "__main__":
    main()
