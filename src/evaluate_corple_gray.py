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
CORPLE_ROOT = ROOT / "external_models" / "CoRPLE"


def _install_corple_import_stubs() -> None:
    basicsr = types.ModuleType("basicsr")
    basicsr.__path__ = [str(CORPLE_ROOT / "basicsr")]
    basicsr_utils = types.ModuleType("basicsr.utils")
    basicsr_registry = types.ModuleType("basicsr.utils.registry")
    basicsr_archs = types.ModuleType("basicsr.archs")
    basicsr_archs.__path__ = [str(CORPLE_ROOT / "basicsr" / "archs")]
    contourlet = types.ModuleType("basicsr.archs.contourlet_transform")
    contourlet.__path__ = [str(CORPLE_ROOT / "basicsr" / "archs" / "contourlet_transform")]

    class _Registry:
        def register(self, *args, **kwargs):
            def deco(cls):
                return cls

            return deco

    basicsr_registry.ARCH_REGISTRY = _Registry()
    sys.modules["basicsr"] = basicsr
    sys.modules["basicsr.utils"] = basicsr_utils
    sys.modules["basicsr.utils.registry"] = basicsr_registry
    sys.modules["basicsr.archs"] = basicsr_archs
    sys.modules["basicsr.archs.contourlet_transform"] = contourlet


_install_corple_import_stubs()


def _load_corple_dat():
    spec = importlib.util.spec_from_file_location(
        "basicsr.archs.dat_arch",
        CORPLE_ROOT / "basicsr" / "archs" / "dat_arch.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load CoRPLE DAT architecture.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["basicsr.archs.dat_arch"] = module
    spec.loader.exec_module(module)
    return module.DAT


DAT = _load_corple_dat()


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


def build_corple(scale: int, use_checkpoint: bool) -> torch.nn.Module:
    return DAT(
        upscale=scale,
        in_chans=3,
        img_size=64,
        img_range=1.0,
        depth=[18],
        embed_dim=60,
        num_heads=[6],
        expansion_factor=2,
        resi_connection="3conv",
        split_size=[8, 32],
        upsampler="pixelshuffledirect",
        use_chk=use_checkpoint,
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


def forward_corple_no_unused_contourlet(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.mean = model.mean.type_as(x)
    x = (x - model.mean) * model.img_range
    if model.upsampler != "pixelshuffledirect":
        raise ValueError("This evaluator is intended for CoRPLE light pixelshuffledirect checkpoints.")
    x = model.conv_first(x)
    x = model.conv_after_body(model.forward_features(x)) + x
    x = model.upsample(x)
    return x / model.img_range + model.mean


def forward_gray(model: torch.nn.Module, lr_gray: torch.Tensor, scale: int, gray_mode: str, pad_multiple: int) -> torch.Tensor:
    lr_rgb = lr_gray.repeat(1, 3, 1, 1)
    _, _, h_old, w_old = lr_rgb.shape
    h_pad = (h_old + pad_multiple - 1) // pad_multiple * pad_multiple - h_old
    w_pad = (w_old + pad_multiple - 1) // pad_multiple * pad_multiple - w_old
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
        names = split.val[: args.limit] if args.limit else split.val
    print(f"val={len(names)} scale=x{args.scale}")

    model = build_corple(args.scale, args.use_checkpoint).to(device).eval()
    ckpt = torch.load(args.weights, map_location="cpu")
    missing, unexpected = model.load_state_dict(normalize_state(ckpt, args.param_key), strict=False)
    if missing:
        print(f"missing_keys={len(missing)}")
    if unexpected:
        print(f"unexpected_keys={len(unexpected)}")
    if args.strict and (missing or unexpected):
        raise RuntimeError("Checkpoint did not strictly match CoRPLE light architecture.")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params={n_params:.2f}M preset=corple_light")
    print(f"loaded={args.weights} param_key={args.param_key}")

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
            pred = forward_gray(model, lr, args.scale, args.gray_mode, args.pad_multiple)
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
    print("val_result " + " ".join(f"{k}={v:.5f}" for k, v in out.items()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--param-key", default="params")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--pad-multiple", type=int, default=32)
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names-file")
    parser.add_argument("--metrics-csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
