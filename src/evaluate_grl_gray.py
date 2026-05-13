from __future__ import annotations

import argparse
import csv
import os
import sys
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

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import load_pair, make_split
from metrics import EdgeMetric, measure_batch, metric_proxy

ROOT = Path(__file__).resolve().parents[1]
GRL_ROOT = ROOT / "external_models" / "GRL-Image-Restoration"
sys.path.insert(0, str(GRL_ROOT))

from models.networks.grl import GRL  # noqa: E402


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def build_grl(variant: str, scale: int) -> GRL:
    common = dict(
        upscale=scale,
        in_channels=3,
        img_range=1.0,
        img_size=256,
        window_size=8,
        stripe_size=[8, None],
        stripe_groups=[None, 4],
        stripe_shift=True,
        mlp_ratio=2,
        qkv_proj_type="linear",
        anchor_proj_type="avgpool",
        anchor_one_stage=True,
        out_proj_type="linear",
        conv_type="1conv",
        init_method="n",
        fairscale_checkpoint=False,
        offload_to_cpu=False,
        double_window=False,
        stripe_square=False,
        separable_conv_act=True,
        use_buffer=True,
        use_efficient_buffer=True,
        euclidean_dist=False,
    )
    if variant == "tiny":
        return GRL(
            **common,
            embed_dim=64,
            upsampler="pixelshuffledirect",
            depths=[4, 4, 4, 4],
            num_heads_window=[2, 2, 2, 2],
            num_heads_stripe=[2, 2, 2, 2],
            anchor_window_down_factor=4,
            local_connection=False,
        )
    if variant == "small":
        return GRL(
            **common,
            embed_dim=128,
            upsampler="pixelshuffle",
            depths=[4, 4, 4, 4],
            num_heads_window=[2, 2, 2, 2],
            num_heads_stripe=[2, 2, 2, 2],
            anchor_window_down_factor=4,
            local_connection=False,
        )
    if variant == "base":
        return GRL(
            **common,
            embed_dim=180,
            upsampler="pixelshuffle",
            depths=[4, 4, 8, 8, 8, 4, 4],
            num_heads_window=[3, 3, 3, 3, 3, 3, 3],
            num_heads_stripe=[3, 3, 3, 3, 3, 3, 3],
            anchor_window_down_factor=2,
            local_connection=True,
        )
    raise ValueError(f"Unsupported GRL variant: {variant}")


def normalize_state(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError("GRL checkpoint must be a state dict or contain state_dict.")
    out = {}
    for name, value in obj.items():
        if name.startswith("model."):
            name = name[len("model.") :]
        if name.startswith("module."):
            name = name[len("module.") :]
        out[name] = value
    return out


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


@torch.no_grad()
def forward_gray(model: torch.nn.Module, lr_gray: torch.Tensor, gray_mode: str) -> torch.Tensor:
    _, _, h, w = lr_gray.shape
    pad_unit = int(getattr(model, "pad_size", 16))
    side = max(h, w)
    side = ((side + pad_unit - 1) // pad_unit) * pad_unit
    pad_h = side - h
    pad_w = side - w
    lr_rgb = lr_gray.repeat(1, 3, 1, 1)
    if pad_h or pad_w:
        mode = "reflect" if h > 1 and w > 1 else "replicate"
        lr_rgb = F.pad(lr_rgb, (0, pad_w, 0, pad_h), mode=mode)
    out = model(lr_rgb)
    out = out[..., : h * model.upscale, : w * model.upscale]
    return rgb_to_gray(out, gray_mode)


@torch.no_grad()
def validate(args: Any) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    else:
        names = split.val
    names = names[: args.limit] if args.limit else names
    print(f"val={len(names)} scale=x{args.scale}")

    model = build_grl(args.variant, args.scale).to(device).eval()
    ckpt = torch.load(args.weights, map_location="cpu")
    state = normalize_state(ckpt)
    state = model.convert_checkpoint(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params={n_params:.2f}M preset=grl_{args.variant}")
    print(f"loaded={args.weights} missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("missing_keys_head=" + ",".join(missing[:8]))
    if unexpected:
        print("unexpected_keys_head=" + ",".join(unexpected[:8]))

    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    sums: dict[str, float] = {}
    metric_rows: list[dict[str, float | str]] = []
    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(args.data, name, args.scale, device)
        with autocast_context(device, args.amp):
            pred = forward_gray(model, lr, args.gray_mode)
        metrics = measure_batch(pred.float(), hr, edge_metric, lpips_fn)
        if args.metrics_csv:
            row: dict[str, float | str] = {"name": name}
            row.update(metrics)
            row["proxy"] = metric_proxy(metrics)
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
    print("val_result " + " ".join(f"{k}={v:.5f}" for k, v in out.items()))


def parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--variant", choices=["tiny", "small", "base"], required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
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
