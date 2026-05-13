from __future__ import annotations

import argparse
import csv
import copy
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path


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
from hat_gray_common import build_hat as build_hat_common
from hat_gray_common import forward_hat_gray, load_hat_weights as load_hat_weights_common
from metrics import EdgeMetric, measure_batch, metric_proxy
from runtime import INTERP, _gaussian_blur, apply_postprocess, interp_tensor, resize_tensor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def build_hat(variant: str, scale: int, use_checkpoint: bool, native_io: bool = False) -> torch.nn.Module:
    return build_hat_common(variant, scale, use_checkpoint, native_io=native_io)


def forward_gray(
    model: torch.nn.Module,
    lr_gray: torch.Tensor,
    scale: int,
    gray_mode: str,
    window_size: int = 16,
    native_io: bool = False,
) -> torch.Tensor:
    return forward_hat_gray(model, lr_gray, scale, gray_mode, window_size=window_size, native_io=native_io)


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
            pred = forward_gray(model, source, args.scale, args.gray_mode, native_io=args.native_io)
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
        pred = forward_gray(model, lr, args.scale, args.gray_mode, native_io=args.native_io)
    if args.blend_interp > 0:
        base = interp_tensor(lr, args.scale, args.interp)
        pred = apply_postprocess(pred.float(), base, lr=lr, blend_interp=args.blend_interp)
    return measure_batch(pred.float(), hr, edge_metric, lpips_fn)


def validate(args) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    else:
        names = split.val[: args.limit] if args.limit else split.val
    print(f"val={len(names)} scale=x{args.scale}")

    base_model = build_hat(args.variant, args.scale, args.use_checkpoint, native_io=args.native_io).to(device)
    load_hat_weights_common(base_model, args.weights, args.param_key, native_io=args.native_io, gray_mode=args.gray_mode)
    base_state = {key: value.detach().cpu().clone() for key, value in base_model.state_dict().items()}
    n_params = sum(p.numel() for p in base_model.parameters()) / 1e6
    print(f"params={n_params:.2f}M preset=hat_{args.variant}")
    print(f"loaded={args.weights} param_key={args.param_key} native_io={args.native_io}")
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
    metric_rows: list[dict[str, float | str]] = []
    self_loss_sum = 0.0
    for idx, name in enumerate(tqdm(names, desc="val", leave=False)):
        lr, hr = load_pair(args.data, name, args.scale, device)
        if idx > 0 or args.reload_first:
            model.load_state_dict(base_state, strict=True)
        self_loss = adapt_on_lr(model, lr, args)
        self_loss_sum += self_loss
        metrics = eval_one(model, lr, hr, edge_metric, lpips_fn, args)
        one = dict(metrics)
        one["proxy"] = metric_proxy(one)
        if args.metrics_csv:
            row: dict[str, float | str] = {"name": name}
            row.update(one)
            metric_rows.append(row)
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
    parser.add_argument("--variant", choices=["l", "m"], default="l")
    parser.add_argument("--param-key", default="state_dict")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--native-io", action="store_true")
    parser.add_argument("--interp", choices=sorted(INTERP), default="lanczos")
    parser.add_argument("--blend-interp", type=float, default=0.0)
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names-file")
    parser.add_argument("--metrics-csv")
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
