from __future__ import annotations

import argparse
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
from tqdm import tqdm

from dataset import load_pair, make_split
from metrics import EdgeMetric, measure_batch, metric_proxy

ROOT = Path(__file__).resolve().parents[1]
CHASNET_ROOT = ROOT / "external_models" / "ChasNet" / "x2"
sys.path.insert(0, str(CHASNET_ROOT))
from models.modules.architecture import OurGen, OurGen2  # noqa: E402


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def normalize_state(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("state_dict", "params_ema", "params"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint must be a state dict or contain state_dict/params.")
    state = {}
    for name, value in obj.items():
        if name.startswith("module."):
            name = name[len("module.") :]
        state[name] = value
    return state


def build_chasnet(branch: str) -> torch.nn.Module:
    if branch == "g1":
        return OurGen(in_nc=3, nf=64)
    if branch == "g2":
        return OurGen2(in_nc=3, nf=64)
    raise ValueError(f"Unsupported branch: {branch}")


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


def _load_branch(weights: str, branch: str, device: torch.device) -> torch.nn.Module:
    model = build_chasnet(branch).to(device).eval()
    state = normalize_state(torch.load(weights, map_location="cpu"))
    model.load_state_dict(state, strict=True)
    return model


def _ensure_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} produced non-finite values")


def forward_gray(
    model_g1: torch.nn.Module,
    model_g2: torch.nn.Module | None,
    lr_gray: torch.Tensor,
    gray_mode: str,
) -> torch.Tensor:
    lr_rgb = lr_gray.repeat(1, 3, 1, 1)
    pred_g1 = model_g1(lr_rgb)
    _ensure_finite("ChasNet G1", pred_g1)
    if model_g2 is None:
        return rgb_to_gray(pred_g1, gray_mode)
    pred_g2 = model_g2(lr_rgb)
    _ensure_finite("ChasNet G2", pred_g2)
    return rgb_to_gray((pred_g1 + pred_g2) / 2.0, gray_mode)


@torch.no_grad()
def validate(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    names = split.val[: args.limit] if args.limit else split.val
    print(f"val={len(names)} scale=x{args.scale}")

    model_g1 = _load_branch(args.weights_g1, "g1", device)
    model_g2 = _load_branch(args.weights_g2, "g2", device) if args.weights_g2 else None
    n_params = sum(p.numel() for p in model_g1.parameters()) / 1e6
    print(f"params={n_params:.2f}M preset=chasnet_x2")
    print(f"loaded_g1={args.weights_g1}")
    if args.weights_g2:
        print(f"loaded_g2={args.weights_g2}")
    else:
        print("loaded_g2=none")

    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    sums = {}
    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(args.data, name, args.scale, device)
        with autocast_context(device, args.amp):
            pred = forward_gray(model_g1, model_g2, lr, args.gray_mode)
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
    parser.add_argument("--weights-g1", required=True)
    parser.add_argument("--weights-g2", default="")
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
