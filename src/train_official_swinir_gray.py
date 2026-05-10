from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any


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

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SWINIR_ROOT = ROOT / "external_models" / "SwinIR"
spec = importlib.util.spec_from_file_location(
    "official_network_swinir",
    SWINIR_ROOT / "models" / "network_swinir.py",
)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load official SwinIR from {SWINIR_ROOT}")
official_network_swinir = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official_network_swinir)
OfficialSwinIR = official_network_swinir.SwinIR

from dataset import SRDataset, load_pair, make_split  # noqa: E402
from losses import CompositeLoss  # noqa: E402
from metrics import EdgeMetric, measure_batch, metric_proxy  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def build_official_swinir(scale: int, training_patch_size: int) -> OfficialSwinIR:
    return OfficialSwinIR(
        upscale=scale,
        in_chans=3,
        img_size=training_patch_size,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )


def normalize_swinir_state(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("params", "params_ema", "state_dict"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint must be a state dict or contain params/state_dict.")
    state = {}
    for key, value in obj.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        state[key] = value
    return state


def load_swinir_weights(model: torch.nn.Module, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu")
    state = normalize_swinir_state(ckpt)
    model.load_state_dict(state, strict=True)


def save_swinir_checkpoint(
    path: str,
    model: torch.nn.Module,
    config: dict[str, Any],
    metrics: dict[str, float] | None = None,
    half: bool = False,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {}
    for key, tensor in model.state_dict().items():
        tensor = tensor.detach().cpu()
        state[key] = tensor.half() if half and tensor.is_floating_point() else tensor
    torch.save(
        {
            "kind": "official_swinir_gray",
            "state_dict": state,
            "config": config,
            "metrics": metrics or {},
        },
        path,
    )


def rgb_to_gray(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "avg":
        return x.mean(dim=1, keepdim=True)
    if mode == "y":
        weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (x * weights).sum(dim=1, keepdim=True)
    if mode == "r":
        return x[:, 0:1]
    if mode == "g":
        return x[:, 1:2]
    if mode == "b":
        return x[:, 2:3]
    raise ValueError(f"Unsupported gray mode: {mode}")


def forward_gray(
    model: torch.nn.Module,
    lr_gray: torch.Tensor,
    scale: int,
    gray_mode: str,
    window_size: int = 8,
) -> torch.Tensor:
    lr_rgb = lr_gray.repeat(1, 3, 1, 1)
    _, _, h_old, w_old = lr_rgb.shape
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    if h_pad:
        lr_rgb = torch.cat([lr_rgb, torch.flip(lr_rgb, [2])], dim=2)[:, :, : h_old + h_pad, :]
    if w_pad:
        lr_rgb = torch.cat([lr_rgb, torch.flip(lr_rgb, [3])], dim=3)[:, :, :, : w_old + w_pad]
    pred_rgb = model(lr_rgb)
    pred_rgb = pred_rgb[..., : h_old * scale, : w_old * scale]
    return rgb_to_gray(pred_rgb, gray_mode)


@torch.no_grad()
def validate(model, root_dir, names, scale, device, amp, gray_mode, lpips_net="alex"):
    model.eval()
    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=lpips_net).to(device).eval()

    sums = {}
    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(root_dir, name, scale, device)
        with autocast_context(device, amp):
            pred = forward_gray(model, lr, scale, gray_mode)
        metrics = measure_batch(pred.float(), hr, edge_metric, lpips_fn)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value

    out = {key: value / len(names) for key, value in sums.items()}
    out["proxy"] = metric_proxy(out)
    model.train()
    return out


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"device={device}")

    split = make_split(
        args.data,
        args.scale,
        val_count=0 if args.val_data else args.val_count,
        seed=args.seed,
        train_all=args.train_all,
    )
    val_root = args.val_data or args.data
    val_names = split.val
    if args.val_data:
        val_names = make_split(
            args.val_data,
            args.scale,
            val_count=args.val_count,
            seed=args.seed,
        ).val
    if args.train_limit:
        split = type(split)(train=split.train[: args.train_limit], val=split.val)
    print(f"train={len(split.train)} val={len(val_names)} scale=x{args.scale}")
    if args.val_data:
        print(f"train_data={args.data}")
        print(f"val_data={args.val_data}")

    model = build_official_swinir(args.scale, args.training_patch_size).to(device)
    print("params=11.90M preset=official_swinir_m")
    if args.weights:
        load_swinir_weights(model, args.weights)
        print(f"loaded={args.weights}")

    cfg = vars(args).copy()
    best = -1e9
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.no_initial_val:
        metrics = validate(
            model,
            val_root,
            val_names,
            args.scale,
            device,
            args.amp,
            args.gray_mode,
            lpips_net=args.val_lpips_net,
        )
        print("initial_val " + " ".join(f"{k}={v:.5f}" for k, v in metrics.items()))
        best = metrics["proxy"]
        save_swinir_checkpoint(os.path.join(args.out_dir, f"best_x{args.scale}.pth"), model, cfg, metrics, half=False)
        save_swinir_checkpoint(os.path.join(args.out_dir, f"best_x{args.scale}_fp16.pth"), model, cfg, metrics, half=True)

    if args.eval_only:
        return

    dataset = SRDataset(
        args.data,
        scale=args.scale,
        patch_size=args.patch_size,
        names=split.train,
        repeat=args.repeat,
        augment=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        drop_last=True,
    )

    criterion = CompositeLoss(
        pixel_weight=args.pixel_weight,
        ssim_weight=args.ssim_weight,
        edge_weight=args.edge_weight,
        lpips_weight=args.lpips_weight,
        lpips_net=args.lpips_net,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs * len(loader)),
        eta_min=args.min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp == "fp16"))
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {}
        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for lr, hr in pbar:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.amp):
                pred = forward_gray(model, lr, args.scale, args.gray_mode)
                loss, logs = criterion(pred.float(), hr)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running["loss"] = running.get("loss", 0.0) + float(loss.detach().cpu())
            for key, value in logs.items():
                running[key] = running.get(key, 0.0) + value
            step = max(1, pbar.n + 1)
            pbar.set_postfix(loss=running["loss"] / step, lr=scheduler.get_last_lr()[0])

        elapsed = (time.time() - start_time) / 60
        print(f"epoch={epoch} train_loss={running['loss'] / len(loader):.5f} elapsed={elapsed:.1f}m")

        if epoch % args.val_every == 0 or epoch == args.epochs:
            metrics = validate(
                model,
                val_root,
                val_names,
                args.scale,
                device,
                args.amp,
                args.gray_mode,
                lpips_net=args.val_lpips_net,
            )
            print("val " + " ".join(f"{k}={v:.5f}" for k, v in metrics.items()))
            save_swinir_checkpoint(os.path.join(args.out_dir, f"latest_x{args.scale}.pth"), model, cfg, metrics, half=False)
            if args.save_every_val:
                epoch_path = os.path.join(args.out_dir, f"epoch{epoch:03d}_x{args.scale}.pth")
                epoch_half_path = os.path.join(args.out_dir, f"epoch{epoch:03d}_x{args.scale}_fp16.pth")
                save_swinir_checkpoint(epoch_path, model, cfg, metrics, half=False)
                save_swinir_checkpoint(epoch_half_path, model, cfg, metrics, half=True)
            if metrics["proxy"] > best:
                best = metrics["proxy"]
                save_swinir_checkpoint(os.path.join(args.out_dir, f"best_x{args.scale}.pth"), model, cfg, metrics, half=False)
                save_swinir_checkpoint(os.path.join(args.out_dir, f"best_x{args.scale}_fp16.pth"), model, cfg, metrics, half=True)
                print(f"saved best proxy={best:.5f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--val-data")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--training-patch-size", type=int, default=64)
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=2)
    parser.add_argument("--no-initial-val", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--save-every-val", action="store_true")
    parser.add_argument("--train-all", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min-lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--pixel-weight", type=float, default=1.0)
    parser.add_argument("--ssim-weight", type=float, default=0.0)
    parser.add_argument("--edge-weight", type=float, default=0.0)
    parser.add_argument("--lpips-weight", type=float, default=0.0)
    parser.add_argument("--lpips-net", choices=["alex", "vgg"], default="alex")
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--out-dir", default="checkpoints_official_swinir_gray")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
