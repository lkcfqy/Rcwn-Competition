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
from runtime import _aug, _deaug

ROOT = Path(__file__).resolve().parents[1]
IRSR_ROOT = ROOT / "external_models" / "IRSRMamba"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_irsrmamba_class():
    for key in list(sys.modules):
        if key == "basicsr" or key.startswith("basicsr."):
            del sys.modules[key]

    basicsr_pkg = types.ModuleType("basicsr")
    basicsr_pkg.__path__ = [str(IRSR_ROOT / "basicsr")]
    sys.modules["basicsr"] = basicsr_pkg

    utils_pkg = types.ModuleType("basicsr.utils")
    utils_pkg.__path__ = [str(IRSR_ROOT / "basicsr" / "utils")]
    sys.modules["basicsr.utils"] = utils_pkg

    sys.modules.setdefault("pywt", types.ModuleType("pywt"))
    _load_module("basicsr.utils.registry", IRSR_ROOT / "basicsr" / "utils" / "registry.py")
    arch = _load_module("basicsr.archs.irsrmamba_arch", IRSR_ROOT / "basicsr" / "archs" / "irsrmamba_arch.py")
    return arch.IRSRMamba


IRSRMamba = _load_irsrmamba_class()


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def normalize_state(obj: Any, key: str) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
        obj = obj[key]
    elif isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint must be a state dict or contain params/state_dict.")
    state = {}
    for name, value in obj.items():
        if name.startswith("module."):
            name = name[len("module.") :]
        if name.startswith("model."):
            name = name[len("model.") :]
        state[name] = value
    return state


def build_irsrmamba(scale: int) -> torch.nn.Module:
    return IRSRMamba(
        upscale=scale,
        in_chans=3,
        img_size=64,
        img_range=1.0,
        d_state=16,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
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


def forward_gray_once(model: torch.nn.Module, lr_gray: torch.Tensor, gray_mode: str) -> torch.Tensor:
    out = model(lr_gray.repeat(1, 3, 1, 1))
    return rgb_to_gray(out, gray_mode)


def _starts(size: int, tile: int, stride: int) -> list[int]:
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def forward_gray_tiled(
    model: torch.nn.Module,
    lr_gray: torch.Tensor,
    gray_mode: str,
    scale: int,
    tile: int,
    overlap: int,
) -> torch.Tensor:
    if tile <= 0:
        return forward_gray_once(model, lr_gray, gray_mode)
    if lr_gray.shape[0] != 1:
        raise ValueError("Tiled IRSRMamba inference currently expects batch size 1.")

    _, _, h, w = lr_gray.shape
    tile_h = min(tile, h)
    tile_w = min(tile, w)
    stride = max(1, tile - overlap)
    out = lr_gray.new_zeros((1, 1, h * scale, w * scale))
    weight = lr_gray.new_zeros((1, 1, h * scale, w * scale))
    for y in _starts(h, tile_h, stride):
        for x in _starts(w, tile_w, stride):
            patch = lr_gray[..., y : y + tile_h, x : x + tile_w]
            pred = forward_gray_once(model, patch, gray_mode)
            yy = y * scale
            xx = x * scale
            ph = tile_h * scale
            pw = tile_w * scale
            out[..., yy : yy + ph, xx : xx + pw] += pred[..., :ph, :pw]
            weight[..., yy : yy + ph, xx : xx + pw] += 1
    return out / weight.clamp_min(1)


def forward_gray(
    model: torch.nn.Module,
    lr_gray: torch.Tensor,
    gray_mode: str,
    scale: int,
    tta: bool,
    tile: int,
    tile_overlap: int,
) -> torch.Tensor:
    if not tta:
        return forward_gray_tiled(model, lr_gray, gray_mode, scale, tile, tile_overlap)
    preds = []
    for mode in range(8):
        pred = forward_gray_tiled(model, _aug(lr_gray, mode), gray_mode, scale, tile, tile_overlap)
        preds.append(_deaug(pred, mode))
    return torch.stack(preds).mean(dim=0)


@torch.no_grad()
def validate(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    names = split.val[: args.limit] if args.limit else split.val
    print(f"val={len(names)} scale=x{args.scale}")

    model = build_irsrmamba(args.scale).to(device).eval()
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(normalize_state(ckpt, args.param_key), strict=True)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params={n_params:.2f}M preset=irsrmamba_x{args.scale}")
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
            pred = forward_gray(model, lr, args.gray_mode, args.scale, args.tta, args.tile, args.tile_overlap)
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
    parser.add_argument("--param-key", default="params")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--tile", type=int, default=64)
    parser.add_argument("--tile-overlap", type=int, default=16)
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="off")
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
