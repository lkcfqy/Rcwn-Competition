from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SRDataset, make_split, load_pair
from evaluate_hat_gray import autocast_context, build_hat, forward_gray, load_hat_weights
from hat_refiner import HatRefiner
from metrics import EdgeMetric, measure_batch, metric_proxy
from runtime import interp_tensor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_names(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


def write_metrics_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "psnr", "ssim", "edge", "lpips", "proxy"])
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def validate(hat: torch.nn.Module, refiner: HatRefiner, names: list[str], args, device: torch.device) -> dict[str, float]:
    refiner.eval()
    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    sums: dict[str, float] = {}
    rows: list[dict[str, float | str]] = []
    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(args.data, name, args.scale, device)
        with autocast_context(device, args.amp):
            pred = forward_gray(hat, lr, args.scale, args.gray_mode, tta=args.tta_eval, native_io=args.native_io).float()
        base = interp_tensor(lr, args.scale, args.interp)
        refined = refiner(pred, base)
        if args.blend_interp > 0:
            refined = (refined * (1.0 - args.blend_interp) + base * args.blend_interp).clamp(0, 1)
        metrics = measure_batch(refined.float(), hr, edge_metric, lpips_fn)
        one = dict(metrics)
        one["proxy"] = metric_proxy(one)
        row: dict[str, float | str] = {"name": name}
        row.update(one)
        rows.append(row)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value
    out = {key: value / len(names) for key, value in sums.items()}
    out["proxy"] = metric_proxy(out)
    if args.metrics_csv:
        write_metrics_csv(Path(args.metrics_csv), rows)
    return out


def train(args) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    val_names = read_names(args.val_names_file) if args.val_names_file else split.val
    val_set = set(val_names)
    train_names = [name for name in split.train if name not in val_set]
    if args.train_limit:
        train_names = train_names[: args.train_limit]
    print(f"device={device} train={len(train_names)} val={len(val_names)}")

    hat = build_hat(args.variant, args.scale, args.use_checkpoint, native_io=args.native_io).to(device).eval()
    load_hat_weights(hat, args.weights, args.param_key, native_io=args.native_io, gray_mode=args.gray_mode)
    for param in hat.parameters():
        param.requires_grad_(False)
    print(f"loaded_hat={args.weights}")

    refiner = HatRefiner(args.channels, args.depth, args.residual_scale).to(device)
    opt = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = SRDataset(
        args.data,
        scale=args.scale,
        patch_size=args.patch_size,
        names=train_names,
        repeat=args.repeat,
        augment=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    best_proxy = -1e9
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        refiner.train()
        loss_sum = 0.0
        for lr, hr in tqdm(loader, desc=f"train e{epoch}", leave=False):
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            with torch.no_grad(), autocast_context(device, args.amp):
                pred = forward_gray(hat, lr, args.scale, args.gray_mode, tta=False, native_io=args.native_io).float()
                base = F.interpolate(lr, scale_factor=args.scale, mode="bicubic", align_corners=False).clamp(0, 1)
            refined = refiner(pred, base)
            loss = F.l1_loss(refined, hr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(refiner.parameters(), args.grad_clip)
            opt.step()
            loss_sum += float(loss.detach().cpu())

        metrics = validate(hat, refiner, val_names, args, device)
        avg_loss = loss_sum / max(1, len(loader))
        print(
            f"epoch={epoch} train_l1={avg_loss:.6f} "
            + " ".join(f"{key}={value:.5f}" for key, value in metrics.items()),
            flush=True,
        )
        ckpt = {
            "kind": "hat_refiner",
            "state_dict": refiner.state_dict(),
            "config": {
                "channels": args.channels,
                "depth": args.depth,
                "residual_scale": args.residual_scale,
            },
            "metrics": metrics,
        }
        torch.save(ckpt, out_dir / "latest_refiner.pth")
        if metrics["proxy"] > best_proxy:
            best_proxy = metrics["proxy"]
            torch.save(ckpt, out_dir / "best_refiner.pth")
            print(f"saved_best={out_dir / 'best_refiner.pth'} proxy={best_proxy:.5f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", choices=["l", "m"], default="l")
    parser.add_argument("--param-key", default="state_dict")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--native-io", action="store_true")
    parser.add_argument("--interp", choices=["cubic", "lanczos"], default="cubic")
    parser.add_argument("--blend-interp", type=float, default=0.0)
    parser.add_argument("--tta-eval", action="store_true")
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--val-names-file", default="")
    parser.add_argument("--metrics-csv", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="none")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
