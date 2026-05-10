from __future__ import annotations

import argparse
import os
import sys
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
from nearest_hybrid import build_nearest_index, nearest_match


def load_hat(path: str, variant: str, scale: int, param_key: str, device: torch.device, use_checkpoint: bool):
    model = build_hat(variant, scale, use_checkpoint).to(device).eval()
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(normalize_state(ckpt, param_key), strict=True)
    print(f"loaded={path} param_key={param_key}")
    return model


@torch.no_grad()
def predict(model, lr, scale: int, gray_mode: str, amp: str, device: torch.device):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda" and amp == "bf16")):
        return forward_gray(model, lr, scale, gray_mode).float().clamp(0, 1)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--base-weights", required=True)
    parser.add_argument("--target-weights", required=True)
    parser.add_argument("--variant", choices=["l", "m"], default="l")
    parser.add_argument("--base-param-key", default="state_dict")
    parser.add_argument("--target-param-key", default="state_dict")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-size", type=lambda s: tuple(int(x) for x in s.split("x")), default=(80, 64))
    parser.add_argument("--lr-psnr-threshold", type=float, default=40.0)
    parser.add_argument("--sim-threshold", type=float, default=0.0)
    parser.add_argument("--lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--amp", choices=["off", "bf16"], default="bf16")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    split = make_split(args.train_root, args.scale, val_count=args.val_count, seed=args.seed)
    pool_feats, pool_lrs = build_nearest_index(Path(args.train_root), split.train, args.scale, args.feature_size)
    print(f"device={device} val={len(split.val)} pool={len(split.train)} threshold={args.lr_psnr_threshold}")

    base = load_hat(args.base_weights, args.variant, args.scale, args.base_param_key, device, args.use_checkpoint)
    target = load_hat(args.target_weights, args.variant, args.scale, args.target_param_key, device, args.use_checkpoint)
    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device).eval()

    sums = {}
    used = []
    lr_dir = Path(args.train_root) / f"input_{640 // args.scale}"
    for name in tqdm(split.val, desc="val", leave=False):
        lr, hr = load_pair(args.train_root, name, args.scale, device)
        lr_img = cv2.imread(str(lr_dir / name), cv2.IMREAD_GRAYSCALE)
        if lr_img is None:
            raise FileNotFoundError(lr_dir / name)
        nn_name, sim, lr_psnr = nearest_match(lr_img, pool_feats, split.train, pool_lrs, args.feature_size)
        use_target = lr_psnr >= args.lr_psnr_threshold and sim >= args.sim_threshold
        pred = predict(target if use_target else base, lr, args.scale, args.gray_mode, args.amp, device)
        if use_target:
            used.append((name, nn_name, sim, lr_psnr))
        metrics = measure_batch(pred, hr, edge_metric, lpips_fn)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value

    out = {key: value / len(split.val) for key, value in sums.items()}
    out["proxy"] = metric_proxy(out)
    print("metrics " + " ".join(f"{key}={value:.6f}" for key, value in out.items()))
    print(f"target_used={len(used)}")
    for item in used:
        print("target", item[0], "nearest", item[1], f"sim={item[2]:.6f}", f"lr_psnr={item[3]:.2f}")


if __name__ == "__main__":
    main()
