from __future__ import annotations

import argparse
import os
import random
import sys
import time
from contextlib import nullcontext


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

from checkpoint import normalize_state_dict, save_checkpoint
from dataset import SRDataset, list_image_names, load_pair, make_split
from hat_gray_common import build_hat, forward_hat_gray, load_hat_weights
from losses import CompositeLoss
from metrics import EdgeMetric, measure_batch, metric_proxy
from models import MODEL_PRESETS, build_model, count_parameters


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


def build_teacher(args, device: torch.device) -> torch.nn.Module | None:
    if not args.teacher_weights:
        return None

    if args.teacher_kind == "hat":
        teacher = build_hat(args.teacher_variant, args.scale, args.teacher_use_checkpoint).to(device)
        load_hat_weights(teacher, args.teacher_weights, args.teacher_param_key)
        label = f"hat_{args.teacher_variant}"
    else:
        teacher = build_model(
            scale=args.scale,
            preset=args.teacher_preset,
            num_features=args.teacher_num_features,
            num_groups=args.teacher_num_groups,
            num_blocks=args.teacher_num_blocks,
        ).to(device)
        ckpt = torch.load(args.teacher_weights, map_location="cpu")
        teacher.load_state_dict(normalize_state_dict(ckpt), strict=True)
        label = args.teacher_preset

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    print(f"teacher_loaded={args.teacher_weights} teacher_kind={args.teacher_kind} teacher_preset={label}")
    return teacher


def forward_teacher(args, teacher: torch.nn.Module, lr: torch.Tensor) -> torch.Tensor:
    if args.teacher_kind == "hat":
        return forward_hat_gray(teacher, lr, args.scale, args.teacher_gray_mode)
    return teacher(lr)


def distill_loss(pred: torch.Tensor, teacher_pred: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "l1":
        return F.l1_loss(pred, teacher_pred)
    if mode == "smooth_l1":
        return F.smooth_l1_loss(pred, teacher_pred, beta=0.01)
    if mode == "charb":
        return torch.sqrt((pred - teacher_pred).square() + 1e-6).mean()
    raise ValueError(f"Unsupported teacher loss: {mode}")


@torch.no_grad()
def validate(model, root_dir, names, scale, device, amp, lpips_net="alex"):
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
            pred = model(lr)
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
    if args.val_data:
        print(f"train_data={args.data}")
        print(f"val_data={args.val_data}")

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

    model = build_model(
        scale=args.scale,
        preset=args.preset,
        num_features=args.num_features,
        num_groups=args.num_groups,
        num_blocks=args.num_blocks,
    ).to(device)
    print(f"params={count_parameters(model) / 1e6:.2f}M preset={args.preset}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(normalize_state_dict(ckpt), strict=True)
        print(f"resumed={args.resume}")

    teacher = build_teacher(args, device)

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

    cfg = vars(args).copy()
    best = -1e9
    os.makedirs(args.out_dir, exist_ok=True)
    start_time = time.time()

    if not args.no_initial_val:
        metrics = validate(
            model,
            val_root,
            val_names,
            args.scale,
            device,
            args.amp,
            lpips_net=args.val_lpips_net,
        )
        print(
            "initial_val "
            + " ".join(f"{k}={v:.5f}" for k, v in metrics.items())
        )
        best = metrics["proxy"]
        save_checkpoint(os.path.join(args.out_dir, f"best_x{args.scale}.pth"), model, cfg, metrics, half=False)
        save_checkpoint(os.path.join(args.out_dir, f"best_x{args.scale}_fp16.pth"), model, cfg, metrics, half=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {}
        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for lr, hr in pbar:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.amp):
                pred = model(lr)
                loss, logs = criterion(pred.float(), hr)
            if teacher is not None and args.teacher_weight > 0:
                with torch.no_grad():
                    with autocast_context(device, args.teacher_amp):
                        teacher_pred = forward_teacher(args, teacher, lr)
                kd = distill_loss(pred.float(), teacher_pred.float(), args.teacher_loss)
                loss = loss + args.teacher_weight * kd
                logs["teacher"] = float(kd.detach().cpu())
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
                lpips_net=args.val_lpips_net,
            )
            print(
                "val "
                + " ".join(f"{k}={v:.5f}" for k, v in metrics.items())
            )
            latest = os.path.join(args.out_dir, f"latest_x{args.scale}.pth")
            save_checkpoint(latest, model, cfg, metrics, half=False)
            if args.save_every_val:
                epoch_path = os.path.join(args.out_dir, f"epoch{epoch:03d}_x{args.scale}.pth")
                epoch_half_path = os.path.join(args.out_dir, f"epoch{epoch:03d}_x{args.scale}_fp16.pth")
                save_checkpoint(epoch_path, model, cfg, metrics, half=False)
                save_checkpoint(epoch_half_path, model, cfg, metrics, half=True)
            if metrics["proxy"] > best:
                best = metrics["proxy"]
                save_checkpoint(os.path.join(args.out_dir, f"best_x{args.scale}.pth"), model, cfg, metrics, half=False)
                save_checkpoint(os.path.join(args.out_dir, f"best_x{args.scale}_fp16.pth"), model, cfg, metrics, half=True)
                print(f"saved best proxy={best:.5f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--val-data", help="Optional validation root; useful for external-data training with official validation.")
    parser.add_argument("--scale", type=int, choices=[2, 4], default=2)
    parser.add_argument("--preset", choices=sorted(MODEL_PRESETS), default="base")
    parser.add_argument("--num-features", type=int)
    parser.add_argument("--num-groups", type=int)
    parser.add_argument("--num-blocks", type=int)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--train-names-file")
    parser.add_argument("--val-names-file")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--no-initial-val", action="store_true")
    parser.add_argument("--save-every-val", action="store_true")
    parser.add_argument("--train-all", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--pixel-weight", type=float, default=1.0)
    parser.add_argument("--ssim-weight", type=float, default=0.05)
    parser.add_argument("--edge-weight", type=float, default=0.03)
    parser.add_argument("--lpips-weight", type=float, default=0.0)
    parser.add_argument("--lpips-net", choices=["alex", "vgg"], default="alex")
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--teacher-weights")
    parser.add_argument("--teacher-kind", choices=["native", "hat"], default="hat")
    parser.add_argument("--teacher-preset", choices=sorted(MODEL_PRESETS), default="base")
    parser.add_argument("--teacher-variant", choices=["m", "l"], default="l")
    parser.add_argument("--teacher-param-key", default="state_dict")
    parser.add_argument("--teacher-gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--teacher-num-features", type=int)
    parser.add_argument("--teacher-num-groups", type=int)
    parser.add_argument("--teacher-num-blocks", type=int)
    parser.add_argument("--teacher-use-checkpoint", action="store_true")
    parser.add_argument("--teacher-weight", type=float, default=0.0)
    parser.add_argument("--teacher-loss", choices=["l1", "smooth_l1", "charb"], default="l1")
    parser.add_argument("--teacher-amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--resume")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
