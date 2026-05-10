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
import torch
from tqdm import tqdm

from dataset import load_pair, make_split
from evaluate_hat_gray import build_hat, forward_gray, normalize_state
from metrics import EdgeMetric, measure_batch, metric_proxy
from nearest_hybrid import (
    build_nearest_index,
    blend_pred,
    nearest_match,
    nearest_tensor,
    should_use_nearest,
)
from runtime import image_to_tensor, tensor_to_image


def load_hat(args, device):
    model = build_hat(args.variant, args.scale, args.use_checkpoint).to(device).eval()
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(normalize_state(ckpt, args.param_key), strict=True)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params={n_params:.2f}M preset=hat_{args.variant}")
    print(f"loaded={args.weights} param_key={args.param_key}")
    return model


@torch.no_grad()
def run_val(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    split = make_split(args.train_root, args.scale, val_count=args.val_count, seed=args.seed)
    query_names = split.val[: args.limit] if args.limit else split.val
    pool_names = split.train
    pool_feats, pool_lrs = build_nearest_index(Path(args.train_root), pool_names, args.scale, args.feature_size)
    print(f"device={device} val={len(query_names)} pool={len(pool_names)}")
    model = load_hat(args, device)

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
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda" and args.amp == "bf16")):
            pred = forward_gray(model, lr, args.scale, args.gray_mode)
        pred = pred.float().clamp(0, 1)
        if should_use_nearest(lr_psnr, sim, args):
            pred = blend_pred(pred, nearest_tensor(Path(args.train_root), nn_name, device), lr_psnr, args)
            used.append((name, nn_name, sim, lr_psnr))
        metrics = measure_batch(pred, hr, edge_metric, lpips_fn)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value

    out = {key: value / len(query_names) for key, value in sums.items()}
    out["proxy"] = metric_proxy(out)
    print("metrics " + " ".join(f"{key}={value:.6f}" for key, value in out.items()))
    print(f"nearest_used={len(used)} threshold_lr_psnr={args.lr_psnr_threshold} alpha={args.nearest_alpha}")
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
    model = load_hat(args, device)
    used = []

    for path in tqdm(query_paths):
        lr_img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if lr_img is None:
            raise FileNotFoundError(path)
        lr = image_to_tensor(lr_img, device)
        nn_name, sim, lr_psnr = nearest_match(lr_img, pool_feats, pool_names, pool_lrs, args.feature_size)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda" and args.amp == "bf16")):
            pred = forward_gray(model, lr, args.scale, args.gray_mode)
        pred = pred.float().clamp(0, 1)
        if should_use_nearest(lr_psnr, sim, args):
            pred = blend_pred(pred, nearest_tensor(train_root, nn_name, device), lr_psnr, args)
            used.append((path.name, nn_name, sim, lr_psnr))
        cv2.imwrite(str(prelim / path.name), tensor_to_image(pred))

    readme = out_dir / "README.md"
    if used:
        method_text = (
            "Generated by HAT-L ImageNet-pretrained inference plus high-confidence "
            "example-based HR retrieval from the provided training set.\n"
        )
    else:
        method_text = (
            "Generated by pure HAT-L ImageNet-pretrained model inference after legal "
            "low-learning-rate fine-tuning on the official training split. No training HR "
            "image is directly copied or substituted into the test outputs.\n"
        )
    readme.write_text("# RCWN first-stage x2 submission\n\n" + method_text, encoding="utf-8")
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
    parser.add_argument("--out-dir", default="submission/fqy_hat_l_nearest_thr40")
    parser.add_argument("--zip-path")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", choices=["l", "m"], default="l")
    parser.add_argument("--param-key", default="params_ema")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-size", type=lambda s: tuple(int(x) for x in s.split("x")), default=(80, 64))
    parser.add_argument("--lr-psnr-threshold", type=float, default=40.0)
    parser.add_argument("--sim-threshold", type=float, default=0.0)
    parser.add_argument("--nearest-alpha", type=float, default=1.0)
    parser.add_argument("--alpha-ramp", type=float, default=0.0)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--amp", choices=["off", "bf16"], default="bf16")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "val":
        run_val(args)
    else:
        run_test(args)
