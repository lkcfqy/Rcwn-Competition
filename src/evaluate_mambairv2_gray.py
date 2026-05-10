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
MAMBAIR_ROOT = ROOT / "external_models" / "MambaIR"


class _ArchRegistry:
    def register(self):
        def decorator(cls):
            return cls

        return decorator


def _prepare_basicsr_shim() -> None:
    basicsr = types.ModuleType("basicsr")
    basicsr.__path__ = [str(MAMBAIR_ROOT / "basicsr")]
    sys.modules["basicsr"] = basicsr

    archs = types.ModuleType("basicsr.archs")
    archs.__path__ = [str(MAMBAIR_ROOT / "basicsr" / "archs")]
    sys.modules["basicsr.archs"] = archs

    utils = types.ModuleType("basicsr.utils")

    def get_root_logger(*args, **kwargs):
        import logging

        return logging.getLogger("mambairv2_eval")

    utils.get_root_logger = get_root_logger
    sys.modules["basicsr.utils"] = utils

    registry = types.ModuleType("basicsr.utils.registry")
    registry.ARCH_REGISTRY = _ArchRegistry()
    sys.modules["basicsr.utils.registry"] = registry


def _load_arch(module_name: str, class_name: str):
    _prepare_basicsr_shim()
    spec = importlib.util.spec_from_file_location(
        f"basicsr.archs.{module_name}",
        MAMBAIR_ROOT / "basicsr" / "archs" / f"{module_name}.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {MAMBAIR_ROOT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"basicsr.archs.{module_name}"] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


MambaIRv2 = _load_arch("mambairv2_arch", "MambaIRv2")
MambaIRv2Light = _load_arch("mambairv2light_arch", "MambaIRv2Light")


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


def build_mambairv2(variant: str, scale: int, use_checkpoint: bool) -> torch.nn.Module:
    if variant == "light":
        return MambaIRv2Light(
            upscale=scale,
            in_chans=3,
            img_size=64,
            img_range=1.0,
            embed_dim=48,
            d_state=8,
            depths=[5, 5, 5, 5],
            num_heads=[4, 4, 4, 4],
            window_size=16,
            inner_rank=32,
            num_tokens=64,
            convffn_kernel_size=5,
            mlp_ratio=1.0,
            upsampler="pixelshuffledirect",
            resi_connection="1conv",
            use_checkpoint=use_checkpoint,
        )
    if variant in {"base", "large"}:
        depths = [6, 6, 6, 6, 6, 6]
        heads = [6, 6, 6, 6, 6, 6]
        if variant == "large":
            depths = [6, 6, 6, 6, 6, 6, 6, 6, 6]
            heads = [6, 6, 6, 6, 6, 6, 6, 6, 6]
        return MambaIRv2(
            upscale=scale,
            in_chans=3,
            img_size=64,
            img_range=1.0,
            embed_dim=174,
            d_state=16,
            depths=depths,
            num_heads=heads,
            window_size=16,
            inner_rank=64,
            num_tokens=128,
            convffn_kernel_size=5,
            mlp_ratio=2.0,
            upsampler="pixelshuffle",
            resi_connection="1conv",
            use_checkpoint=use_checkpoint,
        )
    raise ValueError(f"Unsupported MambaIRv2 variant: {variant}")


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


def forward_gray(model: torch.nn.Module, lr_gray: torch.Tensor, gray_mode: str) -> torch.Tensor:
    out = model(lr_gray.repeat(1, 3, 1, 1))
    return rgb_to_gray(out, gray_mode)


@torch.no_grad()
def validate(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    names = split.val[: args.limit] if args.limit else split.val
    print(f"val={len(names)} scale=x{args.scale}")

    model = build_mambairv2(args.variant, args.scale, args.use_checkpoint).to(device).eval()
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
            pred = forward_gray(model, lr, args.gray_mode)
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
    parser.add_argument("--variant", choices=["light", "base", "large"], default="base")
    parser.add_argument("--param-key", default="params")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
