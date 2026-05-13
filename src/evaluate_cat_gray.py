from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import types
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
from tqdm import tqdm

from dataset import load_pair, make_split
from metrics import EdgeMetric, measure_batch, metric_proxy

ROOT = Path(__file__).resolve().parents[1]
CAT_ROOT = ROOT / "external_models" / "CAT"


class _Registry:
    def __init__(self, name: str):
        self._name = name
        self._obj_map = {}

    def register(self, obj=None, suffix=None):
        def deco(cls):
            name = cls.__name__ if suffix is None else f"{cls.__name__}_{suffix}"
            self._obj_map[name] = cls
            return cls

        return deco if obj is None else deco(obj)


def _load_cat_class():
    basicsr_mod = types.ModuleType("basicsr")
    utils_mod = types.ModuleType("basicsr.utils")
    registry_mod = types.ModuleType("basicsr.utils.registry")
    registry_mod.ARCH_REGISTRY = _Registry("arch")
    sys.modules.setdefault("basicsr", basicsr_mod)
    sys.modules.setdefault("basicsr.utils", utils_mod)
    sys.modules["basicsr.utils.registry"] = registry_mod
    spec = importlib.util.spec_from_file_location("cat_arch_local", CAT_ROOT / "basicsr" / "archs" / "cat_arch.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load CAT from {CAT_ROOT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CAT


CAT = _load_cat_class()


def _with_pad_divisor(model: torch.nn.Module, divisor: int) -> torch.nn.Module:
    model.rcwn_pad_divisor = divisor
    return model


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
    out = {}
    for name, value in obj.items():
        if name.startswith("module."):
            name = name[len("module.") :]
        out[name] = value
    return out


def build_cat(variant: str, scale: int) -> torch.nn.Module:
    if variant == "r":
        return _with_pad_divisor(CAT(
            upscale=scale,
            in_chans=3,
            img_size=64,
            split_size_0=[4, 4, 4, 4, 4, 4],
            split_size_1=[16, 16, 16, 16, 16, 16],
            img_range=1.0,
            depth=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=4,
            resi_connection="1conv",
            block_name="CATB_regular",
            upsampler="pixelshuffle",
        ), 16)
    if variant == "a":
        return _with_pad_divisor(CAT(
            upscale=scale,
            in_chans=3,
            img_size=64,
            split_size_0=[2, 2, 2, 4, 4, 4],
            split_size_1=[0, 0, 0, 0, 0, 0],
            img_range=1.0,
            depth=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=4,
            resi_connection="1conv",
            block_name="CATB_axial",
            upsampler="pixelshuffle",
        ), 4)
    if variant == "r2":
        return _with_pad_divisor(CAT(
            upscale=scale,
            in_chans=3,
            img_size=64,
            split_size_0=[4, 4, 4, 4, 4, 4],
            split_size_1=[16, 16, 16, 16, 16, 16],
            img_range=1.0,
            depth=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            resi_connection="1conv",
            block_name="CATB_regular",
            upsampler="pixelshuffle",
        ), 16)
    if variant == "a2":
        return _with_pad_divisor(CAT(
            upscale=scale,
            in_chans=3,
            img_size=64,
            split_size_0=[4, 4, 4, 4, 4, 4],
            split_size_1=[0, 0, 0, 0, 0, 0],
            img_range=1.0,
            depth=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=4,
            resi_connection="1conv",
            block_name="CATB_axial",
            upsampler="pixelshuffle",
        ), 4)
    raise ValueError(f"Unsupported CAT variant: {variant}")


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
def forward_gray(model: torch.nn.Module, lr_gray: torch.Tensor, scale: int, gray_mode: str) -> torch.Tensor:
    lr_rgb = lr_gray.repeat(1, 3, 1, 1)
    _, _, h_old, w_old = lr_rgb.shape
    divisor = int(getattr(model, "rcwn_pad_divisor", 4))
    h_pad = (divisor - h_old % divisor) % divisor
    w_pad = (divisor - w_old % divisor) % divisor
    if h_pad:
        lr_rgb = torch.cat([lr_rgb, torch.flip(lr_rgb, [2])], dim=2)[:, :, : h_old + h_pad, :]
    if w_pad:
        lr_rgb = torch.cat([lr_rgb, torch.flip(lr_rgb, [3])], dim=3)[:, :, :, : w_old + w_pad]
    out = model(lr_rgb)
    out = out[..., : h_old * scale, : w_old * scale]
    return rgb_to_gray(out, gray_mode)


@torch.no_grad()
def validate(args) -> None:
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

    model = build_cat(args.variant, args.scale).to(device).eval()
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(normalize_state(ckpt, args.param_key), strict=True)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params={n_params:.2f}M preset=cat_{args.variant}")
    print(f"loaded={args.weights} param_key={args.param_key}")

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
            pred = forward_gray(model, lr, args.scale, args.gray_mode)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", choices=["r", "a", "r2", "a2"], default="a2")
    parser.add_argument("--param-key", default="params")
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
