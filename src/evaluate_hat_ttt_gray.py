from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import random
import sys
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
from runtime import _gaussian_blur, resize_tensor

ROOT = Path(__file__).resolve().parents[1]
HAT_ROOT = ROOT / "external_models" / "HAT"
spec = importlib.util.spec_from_file_location("hat_arch_local", HAT_ROOT / "hat" / "archs" / "hat_arch.py")
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load HAT from {HAT_ROOT}")
hat_arch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hat_arch)
HAT = hat_arch.HAT


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def normalize_state(obj: Any, key: str) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict) and key in obj:
        obj = obj[key]
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


def build_hat(variant: str, scale: int, use_checkpoint: bool) -> torch.nn.Module:
    if variant == "l":
        depths = [6] * 12
    elif variant == "m":
        depths = [6] * 6
    else:
        raise ValueError(f"Unsupported HAT variant: {variant}")
    return HAT(
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
    scale: int,
    gray_mode: str,
    window_size: int = 16,
) -> torch.Tensor:
    lr_rgb = lr_gray.repeat(1, 3, 1, 1)
    _, _, h_old, w_old = lr_rgb.shape
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    if h_pad:
        lr_rgb = torch.cat([lr_rgb, torch.flip(lr_rgb, [2])], dim=2)[:, :, : h_old + h_pad, :]
    if w_pad:
        lr_rgb = torch.cat([lr_rgb, torch.flip(lr_rgb, [3])], dim=3)[:, :, :, : w_old + w_pad]
    out = model(lr_rgb)
    out = out[..., : h_old * scale, : w_old * scale]
    return rgb_to_gray(out, gray_mode)


def set_train_scope(model: torch.nn.Module, scope: str) -> list[torch.nn.Parameter]:
    if scope == "all":
        for param in model.parameters():
            param.requires_grad_(True)
    else:
        for param in model.parameters():
            param.requires_grad_(False)
        prefixes = {
            "tail": ("conv_before_upsample", "upsample", "conv_last"),
            "last": ("conv_last",),
        }[scope]
        for name, param in model.named_parameters():
            if name.startswith(prefixes):
                param.requires_grad_(True)
    return [param for param in model.parameters() if param.requires_grad]


def random_lr_target_patch(lr: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, _, h, w = lr.shape
    ps = min(patch_size, h, w)
    top = random.randint(0, h - ps)
    left = random.randint(0, w - ps)
    target = lr[..., top : top + ps, left : left + ps]
    if random.random() < 0.5:
        target = torch.flip(target, [3])
    if random.random() < 0.5:
        target = torch.flip(target, [2])
    if random.random() < 0.5:
        target = torch.rot90(target, random.randint(1, 3), [2, 3])
    return target.contiguous()


def make_self_lr(target: torch.Tensor, sigma: float, interp: str, scale: int) -> torch.Tensor:
    degraded = _gaussian_blur(target, sigma) if sigma > 0 else target
    h, w = target.shape[-2:]
    return resize_tensor(degraded, (h // scale, w // scale), interp).clamp(0, 1)


def adapt_on_lr(model: torch.nn.Module, lr: torch.Tensor, args) -> float:
    if args.ttt_steps <= 0:
        return 0.0
    params = set_train_scope(model, args.train_scope)
    model.train()
    opt = torch.optim.AdamW(params, lr=args.ttt_lr, weight_decay=args.ttt_weight_decay)
    loss_sum = 0.0
    for _ in range(args.ttt_steps):
        target = random_lr_target_patch(lr, args.self_patch_size)
        source = make_self_lr(target, args.self_down_sigma, args.self_down_interp, args.scale)
        opt.zero_grad(set_to_none=True)
        with autocast_context(lr.device, args.amp):
            pred = forward_gray(model, source, args.scale, args.gray_mode)
            loss = F.l1_loss(pred.float(), target.float())
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        loss_sum += float(loss.detach().cpu())
    return loss_sum / args.ttt_steps


@torch.no_grad()
def eval_one(model, lr, hr, edge_metric, lpips_fn, args) -> dict[str, float]:
    model.eval()
    with autocast_context(lr.device, args.amp):
        pred = forward_gray(model, lr, args.scale, args.gray_mode)
    return measure_batch(pred.float(), hr, edge_metric, lpips_fn)


def validate(args) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    names = split.val[: args.limit] if args.limit else split.val
    print(f"val={len(names)} scale=x{args.scale}")

    base_model = build_hat(args.variant, args.scale, args.use_checkpoint).to(device)
    ckpt = torch.load(args.weights, map_location="cpu")
    base_state = normalize_state(ckpt, args.param_key)
    base_model.load_state_dict(base_state, strict=True)
    n_params = sum(p.numel() for p in base_model.parameters()) / 1e6
    print(f"params={n_params:.2f}M preset=hat_{args.variant}")
    print(f"loaded={args.weights} param_key={args.param_key}")
    print(
        f"ttt steps={args.ttt_steps} lr={args.ttt_lr:g} scope={args.train_scope} "
        f"patch={args.self_patch_size} down=sigma{args.self_down_sigma:g}+{args.self_down_interp}"
    )

    model = copy.deepcopy(base_model).to(device)
    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    sums = {}
    self_loss_sum = 0.0
    for idx, name in enumerate(tqdm(names, desc="val", leave=False)):
        lr, hr = load_pair(args.data, name, args.scale, device)
        if idx > 0 or args.reload_first:
            model.load_state_dict(base_state, strict=True)
        self_loss = adapt_on_lr(model, lr, args)
        self_loss_sum += self_loss
        metrics = eval_one(model, lr, hr, edge_metric, lpips_fn, args)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value
        print(
            f"image {name} self_loss={self_loss:.6f} "
            + " ".join(f"{k}={v:.5f}" for k, v in metrics.items()),
            flush=True,
        )

    out = {key: value / len(names) for key, value in sums.items()}
    out["proxy"] = metric_proxy(out)
    out["self_loss"] = self_loss_sum / len(names)
    print("val_result " + " ".join(f"{k}={v:.5f}" for k, v in out.items()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", choices=["l", "m"], default="l")
    parser.add_argument("--param-key", default="state_dict")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="none")
    parser.add_argument("--ttt-steps", type=int, default=8)
    parser.add_argument("--ttt-lr", type=float, default=1e-5)
    parser.add_argument("--ttt-weight-decay", type=float, default=0.0)
    parser.add_argument("--train-scope", choices=["all", "tail", "last"], default="tail")
    parser.add_argument("--self-patch-size", type=int, default=96)
    parser.add_argument("--self-down-sigma", type=float, default=0.5)
    parser.add_argument("--self-down-interp", choices=["area", "linear", "cubic", "nearest"], default="area")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--reload-first", action="store_true")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
