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
from timm.models.layers import to_2tuple, trunc_normal_
from tqdm import tqdm

from dataset import load_pair, make_split
from metrics import EdgeMetric, measure_batch, metric_proxy

ROOT = Path(__file__).resolve().parents[1]
SWINFIR_ROOT = ROOT / "external_models" / "SwinFIR"


class _Registry:
    def __init__(self, name: str):
        self.name = name
        self._obj_map = {}

    def register(self, obj=None, suffix=None):
        def deco(cls):
            key = cls.__name__ if suffix is None else f"{cls.__name__}_{suffix}"
            self._obj_map[key] = cls
            return cls

        return deco if obj is None else deco(obj)


def _install_basicsr_stubs() -> None:
    basicsr_mod = types.ModuleType("basicsr")
    utils_mod = types.ModuleType("basicsr.utils")
    registry_mod = types.ModuleType("basicsr.utils.registry")
    archs_mod = types.ModuleType("basicsr.archs")
    arch_util_mod = types.ModuleType("basicsr.archs.arch_util")
    registry_mod.ARCH_REGISTRY = _Registry("arch")
    arch_util_mod.to_2tuple = to_2tuple
    arch_util_mod.trunc_normal_ = trunc_normal_

    def scandir(dir_path: str, suffix: str | tuple[str, ...] | None = None, recursive: bool = False, full_path: bool = False):
        for root, _, files in os.walk(dir_path):
            for file_name in files:
                if suffix is not None and not file_name.endswith(suffix):
                    continue
                path = os.path.join(root, file_name)
                yield path if full_path else os.path.relpath(path, dir_path)
            if not recursive:
                break

    utils_mod.scandir = scandir
    sys.modules.setdefault("basicsr", basicsr_mod)
    sys.modules.setdefault("basicsr.utils", utils_mod)
    sys.modules["basicsr.utils.registry"] = registry_mod
    sys.modules.setdefault("basicsr.archs", archs_mod)
    sys.modules["basicsr.archs.arch_util"] = arch_util_mod


def _load_module(module_name: str, module_path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_swinfir_arch_stubs() -> None:
    swinfir_mod = types.ModuleType("swinfir")
    swinfir_mod.__path__ = [str(SWINFIR_ROOT / "swinfir")]
    archs_mod = types.ModuleType("swinfir.archs")
    archs_mod.__path__ = [str(SWINFIR_ROOT / "swinfir" / "archs")]
    sys.modules.setdefault("swinfir", swinfir_mod)
    sys.modules.setdefault("swinfir.archs", archs_mod)
    _load_module("swinfir.archs.swinfir_utils", SWINFIR_ROOT / "swinfir" / "archs" / "swinfir_utils.py")
    _load_module("swinfir.archs.local_arch", SWINFIR_ROOT / "swinfir" / "archs" / "local_arch.py")


def _load_class(module_path: Path, class_name: str):
    _install_basicsr_stubs()
    _install_swinfir_arch_stubs()
    module = _load_module(f"swinfir_local_{class_name.lower()}", module_path)
    return getattr(module, class_name)


SwinFIR = _load_class(SWINFIR_ROOT / "swinfir" / "archs" / "swinfir_arch.py", "SwinFIR")
HATFIR = _load_class(SWINFIR_ROOT / "swinfir" / "archs" / "hatfir_arch.py", "HATFIR")


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def build_model(variant: str, scale: int) -> torch.nn.Module:
    if variant == "swinfir_t":
        model = SwinFIR(
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
        )
        model.rcwn_window_size = 12
        return model
    if variant == "swinfir":
        model = SwinFIR(
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
        )
        model.rcwn_window_size = 12
        return model
    if variant == "hatfir":
        model = HATFIR(
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
        )
        model.rcwn_window_size = 16
        return model
    raise ValueError(f"Unsupported SwinFIR variant: {variant}")


def normalize_state(obj: Any, key: str) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict) and key in obj:
        obj = obj[key]
    elif isinstance(obj, dict):
        for fallback in ("params_ema", "params", "state_dict"):
            if fallback in obj and isinstance(obj[fallback], dict):
                obj = obj[fallback]
                break
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint must be a state dict or contain params_ema/params/state_dict.")
    out = {}
    for name, value in obj.items():
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
    win = int(getattr(model, "rcwn_window_size", 16))
    pad_h = (win - h % win) % win
    pad_w = (win - w % win) % win
    lr_rgb = lr_gray.repeat(1, 3, 1, 1)
    if pad_h or pad_w:
        lr_rgb = F.pad(lr_rgb, (0, pad_w, 0, pad_h), mode="reflect")
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

    model = build_model(args.variant, args.scale).to(device).eval()
    ckpt = torch.load(args.weights, map_location="cpu")
    missing, unexpected = model.load_state_dict(normalize_state(ckpt, args.param_key), strict=False)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params={n_params:.2f}M preset={args.variant}")
    print(f"loaded={args.weights} param_key={args.param_key} missing={len(missing)} unexpected={len(unexpected)}")
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
        for key_name, value in metrics.items():
            sums[key_name] = sums.get(key_name, 0.0) + value

    out = {key_name: value / len(names) for key_name, value in sums.items()}
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
    parser.add_argument("--variant", choices=["swinfir_t", "swinfir", "hatfir"], required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--param-key", default="params_ema")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names-file")
    parser.add_argument("--metrics-csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="off")
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
