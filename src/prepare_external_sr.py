from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np

from gray_utils import GRAY_MODE_CHOICES, load_gray_u8


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
INTERPS = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def path_matches(path: Path, contains: list[str], excludes: list[str]) -> bool:
    text = str(path).replace(os.sep, "/").lower()
    if contains and not all(token.lower() in text for token in contains):
        return False
    if excludes and any(token.lower() in text for token in excludes):
        return False
    return True


def list_images(paths: list[str], contains: list[str], excludes: list[str]) -> list[Path]:
    images: list[Path] = []
    for raw in paths:
        root = Path(raw)
        if root.is_file() and root.suffix.lower() in IMAGE_EXTS:
            if path_matches(root, contains, excludes):
                images.append(root)
            continue
        if root.is_dir():
            images.extend(
                p
                for p in root.rglob("*")
                if p.suffix.lower() in IMAGE_EXTS and path_matches(p, contains, excludes)
            )
    return sorted(set(images))


def load_paths_file(path: str | None) -> set[Path]:
    if not path:
        return set()
    root = Path(path)
    selected: set[Path] = set()
    for raw in root.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        selected.add(Path(raw).resolve())
    return selected


def to_gray_u8(path: Path, gray_mode: str = "avg") -> np.ndarray | None:
    return load_gray_u8(path, gray_mode=gray_mode)


def load_stats(path: str | None, gray_mode: str = "avg") -> list[tuple[float, float]]:
    if not path:
        return []
    stats: list[tuple[float, float]] = []
    for image_path in list_images([path], [], []):
        img = to_gray_u8(image_path, gray_mode=gray_mode)
        if img is None:
            continue
        x = img.astype(np.float32)
        stats.append((float(x.mean()), max(float(x.std()), 1.0)))
    return stats


def match_sample_stats(
    img: np.ndarray,
    rng: random.Random,
    stats: list[tuple[float, float]],
    strength: float,
) -> np.ndarray:
    if not stats or strength <= 0:
        return img
    target_mean, target_std = rng.choice(stats)
    x = img.astype(np.float32)
    src_mean = float(x.mean())
    src_std = max(float(x.std()), 1.0)
    matched = (x - src_mean) * (target_std / src_std) + target_mean
    if strength < 1.0:
        matched = x * (1.0 - strength) + matched * strength
    return np.round(np.clip(matched, 0, 255)).astype(np.uint8)


def gradient_mean(path: Path, gray_mode: str = "avg") -> float:
    img = to_gray_u8(path, gray_mode=gray_mode)
    if img is None:
        return -1.0
    x = img.astype(np.float32)
    grad_x = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(grad_x * grad_x + grad_y * grad_y)))


def crop_to_aspect(img: np.ndarray, aspect: float, rng: random.Random, random_crop: bool) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 1 or w <= 1:
        return img
    if w / h > aspect:
        crop_h = h
        crop_w = max(1, int(round(h * aspect)))
    else:
        crop_w = w
        crop_h = max(1, int(round(w / aspect)))
    if random_crop:
        scale = rng.uniform(0.72, 1.0)
        crop_w = max(1, int(round(crop_w * scale)))
        crop_h = max(1, int(round(crop_h * scale)))
    left = rng.randint(0, max(0, w - crop_w)) if random_crop else max(0, (w - crop_w) // 2)
    top = rng.randint(0, max(0, h - crop_h)) if random_crop else max(0, (h - crop_h) // 2)
    return img[top : top + crop_h, left : left + crop_w]


def degrade_x2(
    hr: np.ndarray,
    rng: random.Random,
    out_size: tuple[int, int],
    mode: str,
    sigma: float,
    interp: str,
    sigma_jitter: float,
    noise_std: float,
    gamma_jitter: float,
) -> np.ndarray:
    x = hr.astype(np.float32)
    if mode == "random":
        if rng.random() < 0.55:
            sigma = rng.uniform(0.15, 0.75)
            x = cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, sigmaY=sigma)
        interp_code = rng.choice(list(INTERPS.values()))
        lr = cv2.resize(x, out_size, interpolation=interp_code)
        if rng.random() < 0.35:
            noise = rng.normalvariate(0.0, rng.uniform(0.15, 1.2))
            lr = lr + np.random.default_rng(rng.randrange(2**32)).normal(0.0, abs(noise), lr.shape)
        if rng.random() < 0.20:
            gamma = rng.uniform(0.92, 1.08)
            lr = 255.0 * np.power(np.clip(lr / 255.0, 0, 1), gamma)
        return np.round(np.clip(lr, 0, 255)).astype(np.uint8)

    if mode == "official":
        if sigma_jitter > 0:
            sigma = max(0.0, rng.uniform(sigma - sigma_jitter, sigma + sigma_jitter))
        if sigma > 0:
            x = cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, sigmaY=sigma)
        lr = cv2.resize(x, out_size, interpolation=INTERPS[interp])
        if noise_std > 0:
            lr = lr + np.random.default_rng(rng.randrange(2**32)).normal(0.0, noise_std, lr.shape)
        if gamma_jitter > 0:
            gamma = rng.uniform(max(0.01, 1.0 - gamma_jitter), 1.0 + gamma_jitter)
            lr = 255.0 * np.power(np.clip(lr / 255.0, 0, 1), gamma)
        return np.round(np.clip(lr, 0, 255)).astype(np.uint8)

    raise ValueError(f"Unsupported degrade mode: {mode}")


def write_pair(
    img: np.ndarray,
    name: str,
    out_dir: Path,
    rng: random.Random,
    hr_size: tuple[int, int],
    lr_size: tuple[int, int],
    args: argparse.Namespace,
    stats: list[tuple[float, float]],
) -> None:
    img = match_sample_stats(img, rng, stats, args.match_stats_strength)
    hr = cv2.resize(img, hr_size, interpolation=cv2.INTER_AREA)
    lr = degrade_x2(
        hr,
        rng,
        lr_size,
        args.degrade_mode,
        args.degrade_sigma,
        args.degrade_interp,
        args.degrade_sigma_jitter,
        args.noise_std,
        args.gamma_jitter,
    )
    cv2.imwrite(str(out_dir / "target_640" / name), hr)
    cv2.imwrite(str(out_dir / "input_320" / name), lr)


def main(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    stats = load_stats(args.match_stats_from, gray_mode=args.gray_mode)
    out_dir = Path(args.out_dir)
    (out_dir / "input_320").mkdir(parents=True, exist_ok=True)
    (out_dir / "target_640").mkdir(parents=True, exist_ok=True)

    images = list_images(args.inputs, args.path_contains, args.path_excludes)
    selected_paths = load_paths_file(args.paths_file)
    if selected_paths:
        images = [path for path in images if path.resolve() in selected_paths]
        print(f"paths_file kept={len(images)} from={args.paths_file}")
    if args.min_grad > 0 or args.top_grad_count:
        scored = [(gradient_mean(path, gray_mode=args.gray_mode), path) for path in images]
        scored = [(score, path) for score, path in scored if score >= args.min_grad]
        scored.sort(key=lambda item: item[0], reverse=True)
        if args.top_grad_count:
            scored = scored[: args.top_grad_count]
        images = [path for _, path in scored]
        if scored:
            kept_scores = [score for score, _ in scored]
            print(
                f"grad_filter kept={len(images)} min={min(kept_scores):.3f} "
                f"median={float(np.median(kept_scores)):.3f} max={max(kept_scores):.3f}"
            )
        else:
            print("grad_filter kept=0")
    if args.shuffle:
        rng.shuffle(images)
    if args.limit:
        images = images[: args.limit]
    print(
        f"found={len(images)} out={out_dir} degrade={args.degrade_mode} "
        f"sigma={args.degrade_sigma} interp={args.degrade_interp} match_stats={len(stats)}"
    )

    hr_size = (640, 512)
    lr_size = (320, 256)
    aspect = hr_size[0] / hr_size[1]
    written = 0
    for path in images:
        img = to_gray_u8(path, gray_mode=args.gray_mode)
        if img is None:
            continue
        for patch_idx in range(args.patches_per_image):
            crop = crop_to_aspect(img, aspect, rng, random_crop=args.patches_per_image > 1)
            if min(crop.shape[:2]) < 32:
                continue
            stem = path.stem.replace(" ", "_")
            name = f"ext_{written:07d}_{stem[:40]}_{patch_idx}.png"
            write_pair(crop, name, out_dir, rng, hr_size, lr_size, args, stats)
            written += 1
    print(f"written={written}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--patches-per-image", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--path-contains", action="append", default=[])
    parser.add_argument("--path-excludes", action="append", default=[])
    parser.add_argument("--paths-file")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--min-grad", type=float, default=0.0)
    parser.add_argument("--top-grad-count", type=int, default=0)
    parser.add_argument("--match-stats-from")
    parser.add_argument("--match-stats-strength", type=float, default=1.0)
    parser.add_argument("--degrade-mode", choices=["random", "official"], default="random")
    parser.add_argument("--degrade-sigma", type=float, default=0.5)
    parser.add_argument("--degrade-interp", choices=sorted(INTERPS), default="area")
    parser.add_argument("--degrade-sigma-jitter", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--gamma-jitter", type=float, default=0.0)
    parser.add_argument("--gray-mode", choices=GRAY_MODE_CHOICES, default="avg")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
