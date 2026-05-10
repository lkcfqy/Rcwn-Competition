from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path


def _prefer_torch_cuda_libs() -> None:
    base = "/usr/local/lib/python3.12/dist-packages/nvidia"
    libs = [
        os.path.join(base, "cublas", "lib"),
        os.path.join(base, "cuda_runtime", "lib"),
        os.path.join(base, "cudnn", "lib"),
    ]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    prefix = ":".join([p for p in libs if os.path.isdir(p)])
    if not prefix:
        return
    if os.environ.get("RCWN_CUDA_LIBS_OK") != "1":
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ":".join([prefix, existing])
        env["RCWN_CUDA_LIBS_OK"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_prefer_torch_cuda_libs()

import cv2
import numpy as np
import torch
from tqdm import tqdm

from checkpoint import load_model_from_checkpoint
from dataset import load_pair, make_split
from metrics import EdgeMetric, measure_batch, metric_proxy
from models import MODEL_PRESETS
from runtime import apply_postprocess, forward_ensemble, image_to_tensor, interp_tensor, parse_coeffs, tensor_to_image


def feature_from_image(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    small = cv2.resize(img, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    small = (small - small.mean()) / (small.std() + 1e-6)
    vec = small.reshape(-1)
    return vec / (np.linalg.norm(vec) + 1e-6)


def psnr_u8(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse <= 0:
        return 99.0
    return float(10.0 * np.log10((255.0 * 255.0) / mse))


def build_nearest_index(train_root: Path, pool_names: list[str], scale: int, feature_size: tuple[int, int]):
    lr_dir = train_root / f"input_{640 // scale}"
    feats = []
    lrs = {}
    for name in pool_names:
        img = cv2.imread(str(lr_dir / name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(lr_dir / name)
        lrs[name] = img
        feats.append(feature_from_image(img, feature_size))
    return np.stack(feats), lrs


def nearest_match(query: np.ndarray, pool_feats: np.ndarray, pool_names: list[str], pool_lrs: dict[str, np.ndarray], feature_size):
    q = feature_from_image(query, feature_size)
    sims = pool_feats @ q
    idx = int(np.argmax(sims))
    name = pool_names[idx]
    return name, float(sims[idx]), psnr_u8(query, pool_lrs[name])


def nearest_tensor(train_root: Path, name: str, device: torch.device) -> torch.Tensor:
    img = cv2.imread(str(train_root / "target_640" / name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(train_root / "target_640" / name)
    return image_to_tensor(img, device)


def should_use_nearest(lr_psnr: float, sim: float, args) -> bool:
    return lr_psnr >= args.lr_psnr_threshold and sim >= args.sim_threshold


def blend_pred(model_pred: torch.Tensor, nn_pred: torch.Tensor, lr_psnr: float, args) -> torch.Tensor:
    alpha = args.nearest_alpha
    if args.alpha_ramp > 0:
        alpha = min(1.0, max(0.0, (lr_psnr - args.lr_psnr_threshold) / args.alpha_ramp)) * alpha
    return model_pred * (1.0 - alpha) + nn_pred * alpha


@torch.no_grad()
def model_predict(models, lr: torch.Tensor, args):
    base = interp_tensor(lr, args.scale, args.interp)
    if models:
        pred = forward_ensemble(models, lr, args.amp, args.tta, args.coeffs)
    else:
        pred = base
    return apply_postprocess(
        pred,
        base,
        lr=lr,
        blend_interp=args.blend_interp,
        sharpen_amount=args.sharpen_amount,
        sharpen_radius=args.sharpen_radius,
        clip_mode=args.clip_mode,
    )


@torch.no_grad()
def run_val(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    split = make_split(args.train_root, args.scale, val_count=args.val_count, seed=args.seed)
    query_names = split.val[: args.limit] if args.limit else split.val
    pool_names = split.train
    pool_feats, pool_lrs = build_nearest_index(Path(args.train_root), pool_names, args.scale, args.feature_size)
    print(f"device={device} val={len(query_names)} pool={len(pool_names)}")

    models = []
    for weights in args.weights or []:
        model, _ = load_model_from_checkpoint(weights, device, scale=args.scale, preset=args.preset)
        model.eval()
        models.append(model)
    args.coeffs = parse_coeffs(args.ensemble_coeffs, len(models)) if models else None

    lpips_fn = None
    if args.lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device).eval()
    edge_metric = EdgeMetric().to(device)
    sums = {}
    used = []

    for name in tqdm(query_names):
        lr, hr = load_pair(args.train_root, name, args.scale, device)
        lr_img = cv2.imread(str(Path(args.train_root) / f"input_{640 // args.scale}" / name), cv2.IMREAD_GRAYSCALE)
        nn_name, sim, lr_psnr = nearest_match(lr_img, pool_feats, pool_names, pool_lrs, args.feature_size)
        pred = model_predict(models, lr, args)
        if should_use_nearest(lr_psnr, sim, args):
            pred = blend_pred(pred, nearest_tensor(Path(args.train_root), nn_name, device), lr_psnr, args)
            used.append((name, nn_name, sim, lr_psnr))
        metrics = measure_batch(pred, hr, edge_metric, lpips_fn)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value

    out = {key: value / len(query_names) for key, value in sums.items()}
    out["proxy"] = metric_proxy(out)
    print("metrics " + " ".join(f"{key}={value:.6f}" for key, value in out.items()))
    print(f"nearest_used={len(used)} threshold_lr_psnr={args.lr_psnr_threshold} threshold_sim={args.sim_threshold} alpha={args.nearest_alpha}")
    for item in used[:80]:
        print("used", item[0], "->", item[1], f"sim={item[2]:.6f}", f"lr_psnr={item[3]:.2f}")


@torch.no_grad()
def run_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_root = Path(args.train_root)
    test_input = Path(args.test_input)
    out_dir = Path(args.out_dir)
    prelim = out_dir / "preliminary"
    prelim.mkdir(parents=True, exist_ok=True)
    pool_names = sorted([p.name for p in (train_root / f"input_{640 // args.scale}").glob("*.png")])
    pool_feats, pool_lrs = build_nearest_index(train_root, pool_names, args.scale, args.feature_size)
    query_paths = sorted(test_input.glob("*.png"))
    print(f"device={device} test={len(query_paths)} pool={len(pool_names)}")

    models = []
    for weights in args.weights or []:
        model, _ = load_model_from_checkpoint(weights, device, scale=args.scale, preset=args.preset)
        model.eval()
        models.append(model)
    args.coeffs = parse_coeffs(args.ensemble_coeffs, len(models)) if models else None
    used = []

    for path in tqdm(query_paths):
        lr_img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if lr_img is None:
            raise FileNotFoundError(path)
        lr = image_to_tensor(lr_img, device)
        nn_name, sim, lr_psnr = nearest_match(lr_img, pool_feats, pool_names, pool_lrs, args.feature_size)
        pred = model_predict(models, lr, args)
        if should_use_nearest(lr_psnr, sim, args):
            pred = blend_pred(pred, nearest_tensor(train_root, nn_name, device), lr_psnr, args)
            used.append((path.name, nn_name, sim, lr_psnr))
        cv2.imwrite(str(prelim / path.name), tensor_to_image(pred))

    readme = out_dir / "README.md"
    readme.write_text(
        "# RCWN first-stage x2 submission\n\n"
        "Generated by nearest-hybrid inference: model ensemble fallback plus high-confidence example-based HR retrieval from the provided training set.\n",
        encoding="utf-8",
    )
    if args.zip_path:
        with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(readme, "README.md")
            for p in sorted(prelim.glob("*.png")):
                zf.write(p, f"preliminary/{p.name}")
        print(f"zip={args.zip_path}")
    print(f"nearest_used={len(used)}")
    for item in used:
        print("used", item[0], "->", item[1], f"sim={item[2]:.6f}", f"lr_psnr={item[3]:.2f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    parser.add_argument("--train-root", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--test-input", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/初赛测试集/input_320")
    parser.add_argument("--out-dir", default="submission/fqy_nearest_hybrid")
    parser.add_argument("--zip-path")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--weights", nargs="+")
    parser.add_argument("--ensemble-coeffs")
    parser.add_argument("--preset", choices=["auto", *sorted(MODEL_PRESETS)], default="auto")
    parser.add_argument("--interp", default="lanczos")
    parser.add_argument("--blend-interp", type=float, default=0.0)
    parser.add_argument("--sharpen-amount", type=float, default=0.0)
    parser.add_argument("--sharpen-radius", type=float, default=1.0)
    parser.add_argument("--clip-mode", choices=["hard", "match-base", "none"], default="hard")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-size", type=lambda s: tuple(int(x) for x in s.split("x")), default=(80, 64))
    parser.add_argument("--lr-psnr-threshold", type=float, default=40.0)
    parser.add_argument("--sim-threshold", type=float, default=0.0)
    parser.add_argument("--nearest-alpha", type=float, default=1.0)
    parser.add_argument("--alpha-ramp", type=float, default=0.0)
    parser.add_argument("--lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "val":
        run_val(args)
    else:
        run_test(args)
