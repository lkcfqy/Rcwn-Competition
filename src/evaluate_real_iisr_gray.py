from __future__ import annotations

import argparse
import csv
import os
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
from metrics import EdgeMetric, measure_batch, metric_proxy

ROOT = Path(__file__).resolve().parents[1]
REAL_IISR_ROOT = ROOT / "external_models" / "Real-IISR"
sys.path.insert(0, str(REAL_IISR_ROOT))

from models import VQVAE, build_var  # noqa: E402


PATCH_NUMS = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, cache_enabled=True)


def build_model(device: torch.device, weights: str, fuse: bool):
    vae, var = build_var(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        controlnet_depth=24,
        device=device,
        patch_nums=PATCH_NUMS,
        control_patch_nums=PATCH_NUMS,
        num_classes=2,
        depth=24,
        shared_aln=False,
        attn_l2_norm=True,
        flash_if_available=fuse,
        fused_if_available=fuse,
        init_adaln=0.5,
        init_adaln_gamma=1e-5,
        init_head=0.02,
        init_std=-1,
    )
    _ = vae
    ckpt = torch.load(weights, map_location="cpu", mmap=True)
    state = ckpt["trainer"]["var_wo_ddp"]
    model_keys = set(var.state_dict().keys())
    extra_keys = sorted(set(state.keys()) - model_keys)
    for key in extra_keys:
        state.pop(key)
    missing = sorted(model_keys - set(state.keys()))
    if missing:
        raise KeyError(f"Missing Real-IISR parameters: {missing[:20]} ({len(missing)} total)")
    var.load_state_dict(state, strict=True)
    return var.eval(), len(extra_keys)


@torch.no_grad()
def forward_gray(model, lr_gray: torch.Tensor, out_hw: tuple[int, int], input_size: int, amp: str) -> torch.Tensor:
    device = lr_gray.device
    lr_rgb = lr_gray.repeat(1, 3, 1, 1)
    lr_rgb = F.interpolate(lr_rgb, size=(input_size, input_size), mode="bicubic", align_corners=False)
    lr_rgb = lr_rgb.clamp(0, 1) * 2.0 - 1.0
    label = torch.zeros((lr_rgb.shape[0],), dtype=torch.long, device=device)
    with autocast_context(device, amp):
        pred = model.autoregressive_infer_cfg(
            B=lr_rgb.shape[0],
            cfg=1,
            top_k=1,
            top_p=0.75,
            text_hidden=None,
            lr_inp=lr_rgb,
            negative_text=None,
            label_B=label,
            lr_inp_scale=None,
            more_smooth=False,
        )
    pred = pred.float().clamp(0, 1)
    pred = pred.mean(dim=1, keepdim=True)
    return F.interpolate(pred, size=out_hw, mode="bicubic", align_corners=False).clamp(0, 1)


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
    print(f"val={len(names)} scale=x{args.scale} input_size={args.input_size}")

    model, extra_count = build_model(device, args.weights, args.fuse)
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"params={n_params:.3f}B preset=real_iisr extra_skipped={extra_count}")
    print(f"loaded={args.weights}")

    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    sums = {}
    metric_rows: list[dict[str, float | str]] = []
    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(args.data, name, args.scale, device)
        pred = forward_gray(model, lr, hr.shape[-2:], args.input_size, args.amp)
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
    print("val_result " + " ".join(f"{key}={value:.5f}" for key, value in out.items()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names-file")
    parser.add_argument("--metrics-csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="fp16")
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--fuse", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
