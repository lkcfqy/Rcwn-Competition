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
import torch.nn.functional as F
from tqdm import tqdm

from dataset import load_pair, make_split
from metrics import EdgeMetric, measure_batch, metric_proxy

ROOT = Path(__file__).resolve().parents[1]
DRCT_ROOT = ROOT / "external_models" / "DRCT"


class _Registry:
    def register(self, *args, **kwargs):
        def deco(cls):
            return cls

        return deco


def _install_drct_stubs() -> None:
    basicsr = types.ModuleType("basicsr")
    basicsr_utils = types.ModuleType("basicsr.utils")
    basicsr_registry = types.ModuleType("basicsr.utils.registry")
    basicsr_archs = types.ModuleType("basicsr.archs")
    basicsr_arch_util = types.ModuleType("basicsr.archs.arch_util")

    basicsr_registry.ARCH_REGISTRY = _Registry()

    def to_2tuple(x):
        return x if isinstance(x, tuple) else (x, x)

    try:
        from timm.layers import trunc_normal_
    except Exception:
        from timm.models.layers import trunc_normal_

    basicsr_arch_util.to_2tuple = to_2tuple
    basicsr_arch_util.trunc_normal_ = trunc_normal_

    sys.modules.setdefault("basicsr", basicsr)
    sys.modules.setdefault("basicsr.utils", basicsr_utils)
    sys.modules.setdefault("basicsr.utils.registry", basicsr_registry)
    sys.modules.setdefault("basicsr.archs", basicsr_archs)
    sys.modules.setdefault("basicsr.archs.arch_util", basicsr_arch_util)


def load_drct_arch():
    _install_drct_stubs()
    spec = importlib.util.spec_from_file_location("drct_arch_local", DRCT_ROOT / "drct" / "archs" / "DRCT_arch.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load DRCT from {DRCT_ROOT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DRCT


DRCT = load_drct_arch()


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
    elif isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint must be a state dict or contain the requested param key.")
    state = {}
    for name, value in obj.items():
        if name.startswith("module."):
            name = name[len("module.") :]
        state[name] = value
    return state


def build_drct(variant: str, scale: int, use_checkpoint: bool) -> torch.nn.Module:
    if variant == "l":
        depths = [6] * 12
    elif variant == "base":
        depths = [6] * 6
    else:
        raise ValueError(f"Unsupported DRCT variant: {variant}")
    return DRCT(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depths=depths,
        embed_dim=180,
        num_heads=[6] * len(depths),
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        use_checkpoint=use_checkpoint,
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


def forward_gray(
    model: torch.nn.Module,
    lr_gray: torch.Tensor,
    model_scale: int,
    target_scale: int,
    gray_mode: str,
    downsample: str,
    window_size: int = 16,
):
    lr_rgb = lr_gray.repeat(1, 3, 1, 1)
    _, _, h_old, w_old = lr_rgb.shape
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    if h_pad:
        lr_rgb = torch.cat([lr_rgb, torch.flip(lr_rgb, [2])], dim=2)[:, :, : h_old + h_pad, :]
    if w_pad:
        lr_rgb = torch.cat([lr_rgb, torch.flip(lr_rgb, [3])], dim=3)[:, :, :, : w_old + w_pad]
    out = model(lr_rgb)
    out = out[..., : h_old * model_scale, : w_old * model_scale]
    if model_scale != target_scale:
        mode = "area" if downsample == "area" else "bicubic"
        kwargs = {} if mode == "area" else {"align_corners": False}
        out = F.interpolate(out, size=(h_old * target_scale, w_old * target_scale), mode=mode, **kwargs)
    return rgb_to_gray(out, gray_mode)


@torch.no_grad()
def validate(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")
    split = make_split(args.data, args.target_scale, val_count=args.val_count, seed=args.seed)
    names = split.val[: args.limit] if args.limit else split.val
    print(f"val={len(names)} target_scale=x{args.target_scale} model_scale=x{args.model_scale}")

    model = build_drct(args.variant, args.model_scale, args.use_checkpoint).to(device).eval()
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(normalize_state(ckpt, args.param_key), strict=True)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params={n_params:.2f}M preset=drct_{args.variant}")
    print(f"loaded={args.weights} param_key={args.param_key}")

    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    sums = {}
    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(args.data, name, args.target_scale, device)
        with autocast_context(device, args.amp):
            pred = forward_gray(
                model,
                lr,
                args.model_scale,
                args.target_scale,
                args.gray_mode,
                args.downsample,
            )
        metrics = measure_batch(pred.float(), hr, edge_metric, lpips_fn)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value

    out = {key: value / len(names) for key, value in sums.items()}
    out["proxy"] = metric_proxy(out)
    print("val_result " + " ".join(f"{k}={v:.5f}" for k, v in out.items()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--target-scale", type=int, choices=[2], default=2)
    parser.add_argument("--model-scale", type=int, choices=[2, 4], default=4)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", choices=["base", "l"], default="base")
    parser.add_argument("--param-key", default="params_ema")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--downsample", choices=["area", "bicubic"], default="area")
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
