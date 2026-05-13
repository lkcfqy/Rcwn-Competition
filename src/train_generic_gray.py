from __future__ import annotations

import argparse
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any


def _prefer_torch_cuda_libs() -> None:
    base = "/usr/local/lib/python3.12/dist-packages"
    libs = [
        os.path.join(base, "torch", "lib"),
        os.path.join(base, "nvidia", "cublas", "lib"),
        os.path.join(base, "nvidia", "cuda_runtime", "lib"),
        os.path.join(base, "nvidia", "cudnn", "lib"),
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SRDataset, SyntheticDegradeDataset, list_image_names, load_pair, make_split
from losses import CompositeLoss
from metrics import EdgeMetric, measure_batch, metric_proxy
from model_registry_gray import build_model, default_param_key, forward_gray, load_model_weights
from probe_summary import append_probe_summary, probe_status


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, amp: str, family: str | None = None):
    if family == "pft":
        return nullcontext()
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def save_checkpoint(
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
    torch.save({"kind": "generic_gray", "state_dict": state, "config": config, "metrics": metrics or {}}, path)


@torch.no_grad()
def validate(model, args, val_root: str, val_names: list[str], device: torch.device) -> dict[str, float]:
    model.eval()
    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    sums: dict[str, float] = {}
    for name in tqdm(val_names, desc="val", leave=False):
        lr, hr = load_pair(val_root, name, args.scale, device)
        with autocast_context(device, args.amp, args.family):
            pred = forward_gray(model, args.family, lr, args.scale, args.gray_mode)
        metrics = measure_batch(pred.float(), hr, edge_metric, lpips_fn)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value

    out = {key: value / len(val_names) for key, value in sums.items()}
    out["proxy"] = metric_proxy(out)
    model.train()
    return out


def train(args) -> None:
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
        val_names = make_split(args.val_data, args.scale, val_count=args.val_count, seed=args.seed).val
    if args.val_names_file:
        available_val = set(list_image_names(val_root, args.scale))
        with open(args.val_names_file, "r", encoding="utf-8") as handle:
            custom_val_names = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
        missing_val = sorted(set(custom_val_names) - available_val)
        if missing_val:
            raise FileNotFoundError(
                f"{len(missing_val)} names from {args.val_names_file} are not in {val_root}: {missing_val[:5]}"
            )
        val_names = sorted(dict.fromkeys(custom_val_names))
        print(f"val_names_file={args.val_names_file}")
    if args.train_names_file:
        available = set(list_image_names(args.data, args.scale))
        with open(args.train_names_file, "r", encoding="utf-8") as handle:
            custom_names = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
        missing = sorted(set(custom_names) - available)
        if missing:
            raise FileNotFoundError(f"{len(missing)} names from {args.train_names_file} are not in {args.data}: {missing[:5]}")
        split = type(split)(train=sorted(dict.fromkeys(custom_names)), val=split.val)
        print(f"train_names_file={args.train_names_file}")
    if args.train_limit:
        split = type(split)(train=split.train[: args.train_limit], val=split.val)
    print(f"train={len(split.train)} val={len(val_names)} scale=x{args.scale}")

    model = build_model(args.family, args.variant, args.scale, use_checkpoint=args.use_checkpoint).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params={n_params:.2f}M preset={args.family}_{args.variant}")
    if args.family == "pft" and args.amp != "off":
        print(f"amp_override=off family={args.family} reason=custom_cuda_ops_require_fp32")
    if args.weights:
        used_key = load_model_weights(model, args.family, args.weights, args.param_key or default_param_key(args.family))
        print(f"loaded={args.weights} param_key={used_key}")

    cfg = vars(args).copy()
    best = -1e9
    os.makedirs(args.out_dir, exist_ok=True)
    initial_proxy: float | None = None
    final_proxy: float | None = None
    last_train_loss: float | None = None

    if not args.no_initial_val:
        metrics = validate(model, args, val_root, val_names, device)
        initial_proxy = metrics["proxy"]
        print("initial_val " + " ".join(f"{k}={v:.5f}" for k, v in metrics.items()))
        best = metrics["proxy"]
        save_checkpoint(os.path.join(args.out_dir, f"best_x{args.scale}.pth"), model, cfg, metrics, half=False)
        save_checkpoint(os.path.join(args.out_dir, f"best_x{args.scale}_fp16.pth"), model, cfg, metrics, half=True)
        final_proxy = metrics["proxy"]

    if args.eval_only:
        append_probe_summary(
            args.summary_csv,
            {
                "source": args.summary_source,
                "recipe": args.summary_recipe,
                "static_score": args.summary_static_score,
                "model_family": args.family,
                "model_variant": args.variant,
                "weights": args.weights,
                "out_dir": args.out_dir,
                "initial_val": initial_proxy,
                "final_val": final_proxy,
                "best_val": best if best > -1e8 else "",
                "delta": (final_proxy - initial_proxy) if final_proxy is not None and initial_proxy is not None else "",
                "train_loss": "",
                "promoted_or_closed": probe_status(final_proxy, args.summary_baseline_proxy, args.summary_promote_threshold),
            },
        )
        return

    if args.synthetic_degrade:
        dataset = SyntheticDegradeDataset(
            args.data,
            scale=args.scale,
            patch_size=args.patch_size,
            names=split.train,
            repeat=args.repeat,
            augment=True,
            sigma=args.degrade_sigma,
            interp=args.degrade_interp,
        )
        print(f"dataset=synthetic_degrade sigma={args.degrade_sigma} interp={args.degrade_interp}")
    else:
        dataset = SRDataset(args.data, scale=args.scale, patch_size=args.patch_size, names=split.train, repeat=args.repeat)
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
        running: dict[str, float] = {}
        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for lr, hr in pbar:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.amp, args.family):
                pred = forward_gray(model, args.family, lr, args.scale, args.gray_mode)
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

        last_train_loss = running["loss"] / len(loader)
        elapsed = (time.time() - start_time) / 60
        print(f"epoch={epoch} train_loss={last_train_loss:.5f} elapsed={elapsed:.1f}m")

        if epoch % args.val_every == 0 or epoch == args.epochs:
            metrics = validate(model, args, val_root, val_names, device)
            final_proxy = metrics["proxy"]
            print("val " + " ".join(f"{k}={v:.5f}" for k, v in metrics.items()))
            save_checkpoint(os.path.join(args.out_dir, f"latest_x{args.scale}.pth"), model, cfg, metrics, half=False)
            if args.save_every_val:
                save_checkpoint(
                    os.path.join(args.out_dir, f"epoch{epoch:03d}_x{args.scale}.pth"),
                    model,
                    cfg,
                    metrics,
                    half=False,
                )
                save_checkpoint(
                    os.path.join(args.out_dir, f"epoch{epoch:03d}_x{args.scale}_fp16.pth"),
                    model,
                    cfg,
                    metrics,
                    half=True,
                )
            if metrics["proxy"] > best:
                best = metrics["proxy"]
                save_checkpoint(os.path.join(args.out_dir, f"best_x{args.scale}.pth"), model, cfg, metrics, half=False)
                save_checkpoint(os.path.join(args.out_dir, f"best_x{args.scale}_fp16.pth"), model, cfg, metrics, half=True)
                print(f"saved best proxy={best:.5f}")

    append_probe_summary(
        args.summary_csv,
        {
            "source": args.summary_source,
            "recipe": args.summary_recipe,
            "static_score": args.summary_static_score,
            "model_family": args.family,
            "model_variant": args.variant,
            "weights": args.weights,
            "out_dir": args.out_dir,
            "initial_val": initial_proxy,
            "final_val": final_proxy,
            "best_val": best if best > -1e8 else "",
            "delta": (final_proxy - initial_proxy) if final_proxy is not None and initial_proxy is not None else "",
            "train_loss": last_train_loss,
            "promoted_or_closed": probe_status(final_proxy, args.summary_baseline_proxy, args.summary_promote_threshold),
        },
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--val-data")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--family", choices=["grl", "pft", "rgt", "swin2sr", "catanet", "sst"], required=True)
    parser.add_argument("--variant", default="base")
    parser.add_argument("--param-key", default="")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--patch-size", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--train-names-file")
    parser.add_argument("--val-names-file")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--synthetic-degrade", action="store_true")
    parser.add_argument("--degrade-sigma", type=float, default=0.5)
    parser.add_argument("--degrade-interp", choices=["area", "linear", "cubic", "lanczos"], default="area")
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--no-initial-val", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--save-every-val", action="store_true")
    parser.add_argument("--train-all", action="store_true")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--pixel-weight", type=float, default=1.0)
    parser.add_argument("--ssim-weight", type=float, default=0.0)
    parser.add_argument("--edge-weight", type=float, default=0.0)
    parser.add_argument("--lpips-weight", type=float, default=0.0)
    parser.add_argument("--lpips-net", choices=["alex", "vgg"], default="alex")
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--out-dir", default="checkpoints_generic_gray")
    parser.add_argument("--summary-csv")
    parser.add_argument("--summary-source", default="")
    parser.add_argument("--summary-recipe", default="")
    parser.add_argument("--summary-static-score", type=float)
    parser.add_argument("--summary-baseline-proxy", type=float)
    parser.add_argument("--summary-promote-threshold", type=float)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
