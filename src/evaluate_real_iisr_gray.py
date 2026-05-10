from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import load_pair, make_split
from metrics import EdgeMetric, measure_batch, metric_proxy
from runtime import apply_postprocess, autocast_context, interp_tensor


ROOT = Path(__file__).resolve().parents[1]
REAL_IISR_ROOT = ROOT / "external_models" / "Real-IISR"


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


def _load_real_iisr_module():
    sys.path.insert(0, str(REAL_IISR_ROOT))
    from models import build_var  # type: ignore

    return build_var


def normalize_real_state(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict) and "trainer" in obj:
        obj = obj["trainer"]
    if isinstance(obj, dict) and "var_wo_ddp" in obj:
        obj = obj["var_wo_ddp"]
    if not isinstance(obj, dict):
        raise TypeError("Real-IISR checkpoint must contain trainer.var_wo_ddp or be a state dict.")
    return {k.removeprefix("module."): v for k, v in obj.items()}


def build_real_iisr(args, device: torch.device) -> torch.nn.Module:
    build_var = _load_real_iisr_module()
    patch_nums = tuple(int(x) for x in args.pn.replace("-", "_").split("_"))
    _, var = build_var(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        controlnet_depth=args.depth,
        device="cpu",
        patch_nums=patch_nums,
        control_patch_nums=patch_nums,
        num_classes=2,
        depth=args.depth,
        shared_aln=False,
        attn_l2_norm=True,
        flash_if_available=False,
        fused_if_available=False,
        init_adaln=0.5,
        init_adaln_gamma=1e-5,
        init_head=0.02,
        init_std=-1,
    )
    if args.half and device.type == "cuda":
        var = var.half()
    ckpt = torch.load(args.weights, map_location="cpu", mmap=True)
    state = normalize_real_state(ckpt)
    model_keys = set(var.state_dict().keys())
    state_keys = set(state.keys())
    extra = state_keys - model_keys
    for key in extra:
        state.pop(key)
    missing = model_keys - set(state.keys())
    if missing:
        sample = ", ".join(sorted(missing)[:10])
        raise KeyError(f"Missing {len(missing)} Real-IISR keys, first keys: {sample}")
    var.load_state_dict(state, strict=True)
    del ckpt, state
    gc.collect()
    return var.to(device).eval()


def gray_to_real_condition(base: torch.Tensor) -> torch.Tensor:
    return base.repeat(1, 3, 1, 1).mul(2.0).sub(1.0)


def gray_from_real_output(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "avg":
        return x.mean(dim=1, keepdim=True)
    if mode == "y":
        w = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (x * w).sum(dim=1, keepdim=True)
    if mode == "r":
        return x[:, 0:1]
    if mode == "g":
        return x[:, 1:2]
    if mode == "b":
        return x[:, 2:3]
    raise ValueError(mode)


def match_mean_std(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    dims = (2, 3)
    pred_mean = pred.mean(dim=dims, keepdim=True)
    pred_std = pred.std(dim=dims, keepdim=True).clamp_min(1e-6)
    ref_mean = ref.mean(dim=dims, keepdim=True)
    ref_std = ref.std(dim=dims, keepdim=True).clamp_min(1e-6)
    return (pred - pred_mean) / pred_std * ref_std + ref_mean


def tile_starts(size: int, tile: int) -> list[int]:
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, tile))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


@torch.no_grad()
def forward_tiled(model: torch.nn.Module, base: torch.Tensor, args, device: torch.device) -> torch.Tensor:
    _, _, h, w = base.shape
    tile = args.tile_size
    if h > tile:
        raise ValueError(f"Only horizontal tiling is implemented; got height={h}, tile={tile}.")
    ys = tile_starts(h, tile)
    xs = tile_starts(w, tile)
    pred_sum = base.new_zeros(1, 1, h, w)
    weight_sum = base.new_zeros(1, 1, h, w)
    label = torch.zeros(1, dtype=torch.long, device=device)
    for y in ys:
        for x in xs:
            patch = base[:, :, y : y + tile, x : x + tile]
            pad_h = tile - patch.shape[-2]
            pad_w = tile - patch.shape[-1]
            if pad_h or pad_w:
                patch = F.pad(patch, (0, pad_w, 0, pad_h), mode="reflect")
            cond = gray_to_real_condition(patch)
            with autocast_context(device, args.amp):
                out = model.autoregressive_infer_cfg(
                    B=1,
                    cfg=args.cfg,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    text_hidden=None,
                    lr_inp=cond,
                    negative_text=None,
                    label_B=label,
                    lr_inp_scale=None,
                    more_smooth=False,
                )
            gray = gray_from_real_output(out.float(), args.gray_mode)
            gray = gray[:, :, : patch.shape[-2], : patch.shape[-1]]
            gray = gray[:, :, : min(tile, h - y), : min(tile, w - x)]
            if args.match_stats:
                ref = base[:, :, y : y + gray.shape[-2], x : x + gray.shape[-1]]
                gray = match_mean_std(gray, ref)
            pred_sum[:, :, y : y + gray.shape[-2], x : x + gray.shape[-1]] += gray
            weight_sum[:, :, y : y + gray.shape[-2], x : x + gray.shape[-1]] += 1
    return pred_sum / weight_sum.clamp_min(1e-6)


@torch.no_grad()
def validate(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    names = split.val[: args.limit] if args.limit else split.val
    print(f"device={device} val={len(names)} scale=x{args.scale}")
    print(f"loaded={args.weights}")

    model = build_real_iisr(args, device)
    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    sums: dict[str, float] = {}
    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(args.data, name, args.scale, device)
        base = interp_tensor(lr, args.scale, args.interp)
        pred = forward_tiled(model, base, args, device)
        pred = apply_postprocess(
            pred.float(),
            base,
            lr=lr,
            blend_interp=args.blend_interp,
            sharpen_amount=args.sharpen_amount,
            sharpen_radius=args.sharpen_radius,
            back_project_iters=args.back_project_iters,
            back_project_alpha=args.back_project_alpha,
            back_project_down=args.back_project_down,
            back_project_up=args.back_project_up,
            back_project_down_sigma=args.back_project_down_sigma,
            clip_mode=args.clip_mode,
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
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--pn", default="1_2_3_4_6_9_13_18_24_32")
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--interp", choices=["nearest", "linear", "cubic", "area", "lanczos"], default="cubic")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--match-stats", action="store_true")
    parser.add_argument("--blend-interp", type=float, default=0.0)
    parser.add_argument("--sharpen-amount", type=float, default=0.0)
    parser.add_argument("--sharpen-radius", type=float, default=1.0)
    parser.add_argument("--back-project-iters", type=int, default=0)
    parser.add_argument("--back-project-alpha", type=float, default=1.0)
    parser.add_argument("--back-project-down", choices=["nearest", "linear", "cubic", "area"], default="area")
    parser.add_argument("--back-project-up", choices=["nearest", "linear", "cubic", "area"], default="cubic")
    parser.add_argument("--back-project-down-sigma", type=float, default=0.0)
    parser.add_argument("--clip-mode", choices=["hard", "match-base", "none"], default="hard")
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.75)
    parser.add_argument("--val-count", type=int, default=40)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="none")
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="fp16")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
