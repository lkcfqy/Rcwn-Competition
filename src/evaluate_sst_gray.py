from __future__ import annotations

import argparse
import csv
import os
import sys
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

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import load_pair, make_split
from metrics import EdgeMetric, measure_batch, metric_proxy
from runtime import _aug, _deaug

ROOT = Path(__file__).resolve().parents[1]
SST_ROOT = ROOT / "external_models" / "SST"
sys.path.insert(0, str(SST_ROOT))
from sst.archs.sst_arch import SST  # noqa: E402


PRESETS: dict[str, dict[str, Any]] = {
    "light": {
        "dim": 48,
        "window_sizes": [8, 16, 32, 16, 32, 64],
        "num_heads": [3, 3, 3, 3, 3, 3],
        "ranks": [16, 16, 16, 24, 24, 24],
        "n_blocks": 5,
        "exp_ratio": 1.5,
        "attn_type": "RIB",
        "rib_hidden_dim": 32,
        "rib_n_freqs": 10,
        "upscaling_factor": 2,
        "upsampler_type": "pixelshuffle_direct",
        "gate_type": "DWC",
    },
    "light_plus": {
        "dim": 48,
        "window_sizes": [16, 32, 48, 32, 48, 96],
        "num_heads": [3, 3, 3, 3, 3, 3],
        "ranks": [16, 16, 16, 24, 24, 24],
        "n_blocks": 5,
        "exp_ratio": 1.5,
        "attn_type": "RIB",
        "rib_hidden_dim": 32,
        "rib_n_freqs": 10,
        "upscaling_factor": 2,
        "upsampler_type": "pixelshuffle_direct",
        "gate_type": "DWC",
    },
    "base": {
        "dim": 180,
        "window_sizes": [16, 32, 64, 16, 32, 64],
        "num_heads": [6, 6, 6, 6, 6, 6],
        "ranks": [18, 18, 18, 34, 34, 34],
        "n_blocks": 6,
        "exp_ratio": 1.25,
        "attn_type": "RIB",
        "rib_hidden_dim": 32,
        "rib_n_freqs": 10,
        "upscaling_factor": 2,
        "upsampler_type": "pixelshuffle",
        "gate_type": "DWC",
    },
    "base_plus": {
        "dim": 180,
        "window_sizes": [16, 32, 48, 32, 48, 96],
        "num_heads": [6, 6, 6, 6, 6, 6],
        "ranks": [18, 18, 18, 34, 34, 34],
        "n_blocks": 6,
        "exp_ratio": 1.25,
        "attn_type": "RIB",
        "rib_hidden_dim": 32,
        "rib_n_freqs": 10,
        "upscaling_factor": 2,
        "upsampler_type": "pixelshuffle",
        "gate_type": "DWC",
    },
    "large": {
        "dim": 192,
        "window_sizes": [16, 32, 64, 16, 32, 64],
        "num_heads": [6, 6, 6, 6, 6, 6],
        "ranks": [16, 16, 16, 32, 32, 32],
        "n_blocks": 8,
        "exp_ratio": 2,
        "attn_type": "RIB",
        "rib_hidden_dim": 32,
        "rib_n_freqs": 10,
        "upscaling_factor": 2,
        "upsampler_type": "pixelshuffle",
        "gate_type": "DWC",
        "intermediate_dim": 96,
    },
    "large_plus": {
        "dim": 192,
        "window_sizes": [16, 32, 48, 32, 48, 96],
        "num_heads": [6, 6, 6, 6, 6, 6],
        "ranks": [16, 16, 16, 32, 32, 32],
        "n_blocks": 8,
        "exp_ratio": 2,
        "attn_type": "RIB",
        "rib_hidden_dim": 32,
        "rib_n_freqs": 10,
        "upscaling_factor": 2,
        "upsampler_type": "pixelshuffle",
        "gate_type": "DWC",
        "intermediate_dim": 96,
    },
    "xl_plus": {
        "dim": 224,
        "window_sizes": [16, 32, 48, 32, 48, 96],
        "num_heads": [7, 7, 7, 7, 7, 7],
        "ranks": [16, 16, 16, 32, 32, 32],
        "n_blocks": 10,
        "exp_ratio": 2,
        "attn_type": "RIB",
        "rib_hidden_dim": 32,
        "rib_n_freqs": 10,
        "upscaling_factor": 2,
        "upsampler_type": "pixelshuffle",
        "gate_type": "DWC",
        "intermediate_dim": 96,
    },
}


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def normalize_state(obj: Any, key: str) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict) and key in obj:
        obj = obj[key]
    elif isinstance(obj, dict):
        for fallback in ("params_ema", "params", "state_dict"):
            if fallback in obj and isinstance(obj[fallback], dict):
                obj = obj[fallback]
                break
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint must be a state dict or contain params/params_ema/state_dict.")
    return {name.removeprefix("module."): value for name, value in obj.items()}


def build_sst(variant: str, model_scale: int = 2) -> SST:
    kwargs = dict(PRESETS[variant])
    kwargs["upscaling_factor"] = model_scale
    return SST(**kwargs)


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


def forward_sst_gray(model: torch.nn.Module, lr_gray: torch.Tensor, gray_mode: str, tta: bool = False) -> torch.Tensor:
    if not tta:
        return rgb_to_gray(model(lr_gray.repeat(1, 3, 1, 1)).float(), gray_mode)
    preds = []
    for mode in range(8):
        aug = _aug(lr_gray, mode)
        pred = rgb_to_gray(model(aug.repeat(1, 3, 1, 1)).float(), gray_mode)
        preds.append(_deaug(pred, mode))
    return torch.stack(preds).mean(dim=0)


def read_names(args: Any) -> list[str]:
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    else:
        names = split.val
    return names[: args.limit] if args.limit else names


def write_metrics_csv(path: str, rows: list[dict[str, float | str]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "psnr", "ssim", "edge", "lpips", "proxy"])
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def validate(args: Any) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    names = read_names(args)
    print(f"device={device}")
    print(f"val={len(names)} scale=x{args.scale}")

    model = build_sst(args.variant, args.model_scale).to(device).eval()
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(normalize_state(ckpt, args.param_key), strict=True)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params={n_params:.2f}M preset=sst_{args.variant}")
    print(f"loaded={args.weights} param_key={args.param_key}")

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
            pred = forward_sst_gray(model, lr, args.gray_mode, tta=args.tta)
        if pred.shape[-2:] != hr.shape[-2:]:
            mode = "area" if args.resize_output == "area" else "bicubic"
            kwargs = {} if mode == "area" else {"align_corners": False}
            pred = F.interpolate(pred, size=hr.shape[-2:], mode=mode, **kwargs)
        metrics = measure_batch(pred, hr, edge_metric, lpips_fn)
        one: dict[str, float | str] = {"name": name, **metrics, "proxy": metric_proxy(metrics)}
        rows.append(one)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value

    out = {key: value / len(names) for key, value in sums.items()}
    out["proxy"] = metric_proxy(out)
    if args.metrics_csv:
        write_metrics_csv(args.metrics_csv, rows)
        print(f"metrics_csv={args.metrics_csv}")
    print("val_result " + " ".join(f"{key}={value:.5f}" for key, value in out.items()))


def parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--model-scale", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", choices=sorted(PRESETS), default="base")
    parser.add_argument("--param-key", default="params_ema")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--resize-output", choices=["area", "bicubic"], default="area")
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names-file", default="")
    parser.add_argument("--metrics-csv", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
