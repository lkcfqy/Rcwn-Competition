from __future__ import annotations

import argparse
import csv
import os
import sys
from contextlib import nullcontext
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

import torch
from tqdm import tqdm

import super_image
from dataset import load_pair, make_split
from gray_utils import rgb_tensor_to_gray
from metrics import EdgeMetric, measure_batch, metric_proxy
from runtime import _aug, _deaug, apply_postprocess, interp_tensor


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def read_names(args) -> list[str]:
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    else:
        names = split.val
    return names[: args.limit] if args.limit else names


@torch.no_grad()
def forward_gray(model: torch.nn.Module, lr_gray: torch.Tensor, scale: int, gray_mode: str, tta: bool) -> torch.Tensor:
    def once(x: torch.Tensor) -> torch.Tensor:
        out = model(x.repeat(1, 3, 1, 1))
        out = out[..., : x.shape[-2] * scale, : x.shape[-1] * scale]
        return rgb_tensor_to_gray(out, gray_mode)

    if not tta:
        return once(lr_gray)
    preds = []
    for mode in range(8):
        pred = once(_aug(lr_gray, mode))
        preds.append(_deaug(pred, mode))
    return torch.stack(preds).mean(dim=0)


@torch.no_grad()
def validate(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")
    names = read_names(args)
    print(f"val={len(names)} scale=x{args.scale}")

    model_cls = getattr(super_image, args.model_class)
    model = model_cls.from_pretrained(args.repo_id, scale=args.scale).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params={n_params:.2f}M class={args.model_class} repo={args.repo_id}")

    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    sums = {}
    metric_rows: list[dict[str, float | str]] = []
    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(args.data, name, args.scale, device)
        with autocast_context(device, args.amp):
            pred = forward_gray(model, lr, args.scale, args.gray_mode, args.tta)
        if args.blend_interp > 0 or args.sharpen_amount > 0 or args.clip_mode != "hard":
            base = interp_tensor(lr, args.scale, args.interp)
            pred = apply_postprocess(
                pred.float(),
                base,
                lr=lr,
                blend_interp=args.blend_interp,
                sharpen_amount=args.sharpen_amount,
                sharpen_radius=args.sharpen_radius,
                clip_mode=args.clip_mode,
            )
        metrics = measure_batch(pred.float(), hr, edge_metric, lpips_fn)
        one = dict(metrics)
        one["proxy"] = metric_proxy(one)
        if args.metrics_csv:
            row: dict[str, float | str] = {"name": name}
            row.update(one)
            metric_rows.append(row)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value

    out = {key: value / len(names) for key, value in sums.items()}
    out["proxy"] = metric_proxy(out)
    if args.metrics_csv:
        out_path = Path(args.metrics_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["name", "psnr", "ssim", "edge", "lpips", "proxy"])
            writer.writeheader()
            writer.writerows(metric_rows)
        print(f"metrics_csv={out_path}")
    print("val_result " + " ".join(f"{key}={value:.5f}" for key, value in out.items()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--model-class", required=True)
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--interp", choices=["nearest", "linear", "cubic", "lanczos", "area"], default="cubic")
    parser.add_argument("--blend-interp", type=float, default=0.0)
    parser.add_argument("--sharpen-amount", type=float, default=0.0)
    parser.add_argument("--sharpen-radius", type=float, default=1.0)
    parser.add_argument("--clip-mode", choices=["hard", "match-base", "none"], default="hard")
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names-file")
    parser.add_argument("--metrics-csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
