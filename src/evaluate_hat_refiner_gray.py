from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm

from dataset import load_pair, make_split
from evaluate_hat_gray import autocast_context, build_hat, forward_gray, load_hat_weights
from hat_refiner import HatRefiner
from metrics import EdgeMetric, measure_batch, metric_proxy
from runtime import interp_tensor


def read_names(args) -> list[str]:
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    return split.val[: args.limit] if args.limit else split.val


def load_refiner(path: str, device: torch.device) -> HatRefiner:
    ckpt = torch.load(path, map_location="cpu", mmap=True)
    config = ckpt.get("config", {})
    model = HatRefiner(
        channels=int(config.get("channels", 32)),
        depth=int(config.get("depth", 4)),
        residual_scale=float(config.get("residual_scale", 0.05)),
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device).eval()


@torch.no_grad()
def validate(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    names = read_names(args)
    print(f"device={device} val={len(names)}")
    hat = build_hat(args.variant, args.scale, args.use_checkpoint, native_io=args.native_io).to(device).eval()
    load_hat_weights(hat, args.weights, args.param_key, native_io=args.native_io, gray_mode=args.gray_mode)
    refiner = load_refiner(args.refiner, device)
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
            pred = forward_gray(hat, lr, args.scale, args.gray_mode, tta=args.tta, native_io=args.native_io).float()
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
        Path(args.metrics_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["name", "psnr", "ssim", "edge", "lpips", "proxy"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"metrics_csv={args.metrics_csv}")
    print("val_result " + " ".join(f"{key}={value:.5f}" for key, value in out.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--refiner", required=True)
    parser.add_argument("--variant", choices=["l", "m"], default="l")
    parser.add_argument("--param-key", default="state_dict")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--native-io", action="store_true")
    parser.add_argument("--interp", choices=["cubic", "lanczos"], default="cubic")
    parser.add_argument("--blend-interp", type=float, default=0.0)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names-file", default="")
    parser.add_argument("--metrics-csv", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
