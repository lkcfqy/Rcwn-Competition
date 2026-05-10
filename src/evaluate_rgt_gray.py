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
RGT_ROOT = ROOT / "external_models" / "RGT"


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


def _load_rgt_class():
    basicsr_mod = types.ModuleType("basicsr")
    utils_mod = types.ModuleType("basicsr.utils")
    registry_mod = types.ModuleType("basicsr.utils.registry")
    registry_mod.ARCH_REGISTRY = _Registry("arch")
    sys.modules.setdefault("basicsr", basicsr_mod)
    sys.modules.setdefault("basicsr.utils", utils_mod)
    sys.modules["basicsr.utils.registry"] = registry_mod
    spec = importlib.util.spec_from_file_location("rgt_arch_local", RGT_ROOT / "basicsr" / "archs" / "rgt_arch.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load RGT from {RGT_ROOT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RGT


RGT = _load_rgt_class()


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def normalize_state(obj: Any, key: str) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict) and key in obj:
        obj = obj[key]
    elif isinstance(obj, dict) and "params_ema" in obj:
        obj = obj["params_ema"]
    elif isinstance(obj, dict) and "params" in obj:
        obj = obj["params"]
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint must be a state dict or contain the requested param key.")
    state = {}
    for name, value in obj.items():
        if name.startswith("module."):
            name = name[len("module.") :]
        state[name] = value
    return state


def build_rgt(variant: str, scale: int) -> torch.nn.Module:
    if variant == "rgt":
        depth = [6] * 8
    elif variant == "s":
        depth = [6] * 6
    else:
        raise ValueError(f"Unsupported RGT variant: {variant}")
    return RGT(
        upscale=scale,
        in_chans=3,
        img_size=64,
        img_range=1.0,
        depth=depth,
        embed_dim=180,
        num_heads=[6] * len(depth),
        mlp_ratio=2,
        resi_connection="1conv",
        split_size=[8, 32],
        c_ratio=0.5,
    )


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
    out = model(lr_rgb)
    out = out[..., : lr_gray.shape[-2] * scale, : lr_gray.shape[-1] * scale]
    return rgb_to_gray(out, gray_mode)


@torch.no_grad()
def validate(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    names = split.val[: args.limit] if args.limit else split.val
    print(f"val={len(names)} scale=x{args.scale}")

    model = build_rgt(args.variant, args.scale).to(device).eval()
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(normalize_state(ckpt, args.param_key), strict=True)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params={n_params:.2f}M preset=rgt_{args.variant}")
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
            pred = forward_gray(model, lr, args.scale, args.gray_mode)
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
    parser.add_argument("--variant", choices=["rgt", "s"], default="rgt")
    parser.add_argument("--param-key", default="params")
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
