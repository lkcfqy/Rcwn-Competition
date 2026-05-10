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
RAMIT_ROOT = ROOT / "external_models" / "RAMiT"
sys.path.insert(0, str(RAMIT_ROOT))

from my_model.ramit import RAMiT as RAMiTBase  # noqa: E402
from my_model.ramit_slimsr import RAMiT as RAMiTSlimSR  # noqa: E402


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def normalize_state(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("params_ema", "params", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint must be a state dict or contain params/params_ema/state_dict/model.")
    state = {}
    for name, value in obj.items():
        if name.startswith("module."):
            name = name[len("module.") :]
        state[name] = value
    return state


def build_model(variant: str) -> torch.nn.Module:
    common = dict(
        target_mode="light_x2",
        img_norm=True,
        in_chans=3,
        dim=64,
        depths=[6, 4, 4, 6],
        num_heads=[4, 4, 4, 4],
        head_dim=None,
        chsa_head_ratio=0.25,
        window_size=8,
        hidden_ratio=2.0,
        qkv_bias=True,
        act_layer=torch.nn.GELU,
        norm_layer="ReshapeLayerNorm",
        tail_mv=2,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        helper=True,
        mv_act=torch.nn.LeakyReLU,
    )
    if variant == "ramit":
        return RAMiTBase(mv_ver=2, exp_factor=1.2, expand_groups=4, **common)
    if variant == "ramit_1":
        return RAMiTBase(mv_ver=1, exp_factor=None, expand_groups=None, **common)
    if variant == "ramit_slimsr":
        common.update(dim=48, depths=[8, 2, 2, 8])
        return RAMiTSlimSR(mv_ver=2, exp_factor=1.2, expand_groups=4, **common)
    raise ValueError(f"Unsupported RAMiT variant: {variant}")


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
    split = make_split(args.data, 2, val_count=args.val_count, seed=args.seed)
    names = split.val[: args.limit] if args.limit else split.val
    print(f"val={len(names)} scale=x2")

    model = build_model(args.variant).to(device).eval()
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(normalize_state(ckpt), strict=True)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params={n_params:.2f}M preset={args.variant}")
    print(f"loaded={args.weights}")

    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    sums = {}
    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(args.data, name, 2, device)
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
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", choices=["ramit", "ramit_1", "ramit_slimsr"], default="ramit")
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
