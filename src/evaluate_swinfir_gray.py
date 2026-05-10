from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types
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
from tqdm import tqdm

from dataset import load_pair, make_split
from metrics import EdgeMetric, measure_batch, metric_proxy

ROOT = Path(__file__).resolve().parents[1]
SWINFIR_ROOT = ROOT / "external_models" / "SwinFIR"


class _ArchRegistry:
    def register(self):
        def decorator(cls):
            return cls

        return decorator


def _install_minimal_basicsr() -> None:
    basicsr = types.ModuleType("basicsr")
    basicsr.__path__ = []
    sys.modules["basicsr"] = basicsr

    archs = types.ModuleType("basicsr.archs")
    archs.__path__ = []
    sys.modules["basicsr.archs"] = archs

    arch_util = types.ModuleType("basicsr.archs.arch_util")

    def to_2tuple(x):
        if isinstance(x, tuple):
            return x
        return (x, x)

    arch_util.to_2tuple = to_2tuple
    arch_util.trunc_normal_ = torch.nn.init.trunc_normal_
    sys.modules["basicsr.archs.arch_util"] = arch_util

    utils = types.ModuleType("basicsr.utils")
    sys.modules["basicsr.utils"] = utils

    registry = types.ModuleType("basicsr.utils.registry")
    registry.ARCH_REGISTRY = _ArchRegistry()
    sys.modules["basicsr.utils.registry"] = registry


def _load_swinfir_archs():
    _install_minimal_basicsr()

    swinfir = types.ModuleType("swinfir")
    swinfir.__path__ = [str(SWINFIR_ROOT / "swinfir")]
    sys.modules["swinfir"] = swinfir

    archs = types.ModuleType("swinfir.archs")
    archs.__path__ = [str(SWINFIR_ROOT / "swinfir" / "archs")]
    sys.modules["swinfir.archs"] = archs

    modules = {}
    for name in ("swinfir_arch", "hatfir_arch"):
        qualified = f"swinfir.archs.{name}"
        spec = importlib.util.spec_from_file_location(
            qualified,
            SWINFIR_ROOT / "swinfir" / "archs" / f"{name}.py",
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load {qualified}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["swinfir_arch"].SwinFIR, modules["hatfir_arch"].HATFIR


SwinFIR, HATFIR = _load_swinfir_archs()


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
    state = {}
    for name, value in obj.items():
        if name.startswith("module."):
            name = name[len("module.") :]
        state[name] = value
    return state


def build_model(variant: str, scale: int) -> tuple[torch.nn.Module, int]:
    if variant == "swinfir":
        return (
            SwinFIR(
                upscale=scale,
                in_chans=3,
                img_size=60,
                window_size=12,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler="pixelshuffle",
                resi_connection="SFB",
            ),
            12,
        )
    if variant == "swinfir_t":
        return (
            SwinFIR(
                upscale=scale,
                in_chans=3,
                img_size=60,
                window_size=12,
                img_range=1.0,
                depths=[6, 5, 5, 6],
                embed_dim=60,
                num_heads=[6, 6, 6, 6],
                mlp_ratio=2,
                upsampler="pixelshuffledirect",
                resi_connection="HSFB",
            ),
            12,
        )
    if variant == "hatfir":
        return (
            HATFIR(
                upscale=scale,
                in_chans=3,
                img_size=64,
                window_size=16,
                compress_ratio=3,
                squeeze_factor=30,
                conv_scale=0.01,
                overlap_ratio=0.5,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler="pixelshuffle",
                resi_connection="SFB",
            ),
            16,
        )
    if variant == "hatfir_l":
        return (
            HATFIR(
                upscale=scale,
                in_chans=3,
                img_size=64,
                window_size=16,
                compress_ratio=3,
                squeeze_factor=30,
                conv_scale=0.01,
                overlap_ratio=0.5,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler="pixelshuffle",
                resi_connection="SFB",
            ),
            16,
        )
    raise ValueError(f"Unsupported SwinFIR variant: {variant}")


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


def pad_to_window(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, int, int]:
    _, _, h_old, w_old = x.shape
    h_pad = (window_size - h_old % window_size) % window_size
    w_pad = (window_size - w_old % window_size) % window_size
    if h_pad:
        x = torch.cat([x, torch.flip(x, [2])], dim=2)[:, :, : h_old + h_pad, :]
    if w_pad:
        x = torch.cat([x, torch.flip(x, [3])], dim=3)[:, :, :, : w_old + w_pad]
    return x, h_old, w_old


def forward_gray(
    model: torch.nn.Module,
    lr_gray: torch.Tensor,
    scale: int,
    window_size: int,
    gray_mode: str,
) -> torch.Tensor:
    lr_rgb, h_old, w_old = pad_to_window(lr_gray.repeat(1, 3, 1, 1), window_size)
    out = model(lr_rgb)
    out = out[..., : h_old * scale, : w_old * scale]
    return rgb_to_gray(out, gray_mode)


@torch.no_grad()
def validate(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    names = split.val[: args.limit] if args.limit else split.val
    print(f"val={len(names)} scale=x{args.scale}")

    model, window_size = build_model(args.variant, args.scale)
    model = model.to(device).eval()
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(normalize_state(ckpt, args.param_key), strict=True)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params={n_params:.2f}M preset={args.variant}")
    print(f"loaded={args.weights} param_key={args.param_key}")

    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    sums = {}
    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(args.data, name, args.scale, device)
        with autocast_context(device, args.amp):
            pred = forward_gray(model, lr, args.scale, window_size, args.gray_mode)
        metrics = measure_batch(pred.float(), hr, edge_metric, lpips_fn)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value

    out = {key: value / len(names) for key, value in sums.items()}
    out["proxy"] = metric_proxy(out)
    print("val_result " + " ".join(f"{k}={v:.5f}" for k, v in out.items()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", choices=["swinfir", "swinfir_t", "hatfir", "hatfir_l"], default="swinfir")
    parser.add_argument("--param-key", default="params_ema")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
