from __future__ import annotations

import argparse
import csv
import os
import sys
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
DIFIISR_ROOT = ROOT / "external_models" / "DifIISR"
sys.path.insert(0, str(DIFIISR_ROOT))

from omegaconf import OmegaConf  # noqa: E402
from sampler import DifIISRSampler  # noqa: E402


def autocast_enabled(amp: str) -> bool:
    return amp != "off"


def build_sampler(args: argparse.Namespace) -> DifIISRSampler:
    config = OmegaConf.load(args.config)
    config.model.ckpt_path = args.weights
    config.autoencoder.ckpt_path = args.autoencoder
    config.diffusion.params.sf = args.model_scale
    if args.steps:
        config.diffusion.params.steps = args.steps
        config.diffusion.params.timestep_respacing = args.steps
    if args.chop_size == 512:
        chop_stride = 448
    elif args.chop_size == 256:
        chop_stride = 224
    else:
        raise ValueError("chop_size must be 256 or 512")
    return DifIISRSampler(
        config,
        sf=args.model_scale,
        chop_size=args.chop_size,
        chop_stride=chop_stride,
        chop_bs=1,
        use_fp16=autocast_enabled(args.amp),
        seed=args.seed,
        ddim=True,
    )


@torch.no_grad()
def predict_gray(sampler: DifIISRSampler, lr: torch.Tensor, out_hw: tuple[int, int], args: argparse.Namespace) -> torch.Tensor:
    lr_rgb = lr.repeat(1, 3, 1, 1)
    pred = sampler.sample_func(
        (lr_rgb - 0.5) / 0.5,
        noise_repeat=args.noise_repeat,
        one_step=args.one_step,
        apply_decoder=True,
    )
    pred = (pred.float() * 0.5 + 0.5).clamp(0, 1)
    pred = pred.mean(dim=1, keepdim=True)
    if pred.shape[-2:] != out_hw:
        pred = F.interpolate(pred, size=out_hw, mode=args.resize, align_corners=False if args.resize in {"bilinear", "bicubic"} else None)
    return pred.clamp(0, 1)


@torch.no_grad()
def validate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() and not args.cpu:
        raise RuntimeError("DifIISR evaluation requires CUDA unless --cpu is explicitly set.")
    device = torch.device("cpu" if args.cpu else "cuda")
    if device.type != "cuda":
        raise RuntimeError("CPU DifIISR evaluation is not practical for this probe.")
    print(f"device={device} scale=x{args.scale} model_scale=x{args.model_scale} steps={args.steps or 'config'}")

    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    else:
        names = split.val[: args.limit] if args.limit else split.val
    print(f"val={len(names)} one_step={args.one_step} resize={args.resize}")

    sampler = build_sampler(args)
    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    sums: dict[str, float] = {}
    metric_rows: list[dict[str, float | str]] = []
    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(args.data, name, args.scale, device)
        pred = predict_gray(sampler, lr, hr.shape[-2:], args)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--model-scale", type=int, choices=[2, 4], default=4)
    parser.add_argument("--weights", default=str(DIFIISR_ROOT / "weights" / "DifIISR.pth"))
    parser.add_argument("--autoencoder", default=str(DIFIISR_ROOT / "weights" / "autoencoder_vq_f4.pth"))
    parser.add_argument("--config", default=str(DIFIISR_ROOT / "configs" / "DifIISR_test.yaml"))
    parser.add_argument("--steps", type=int, default=0, help="Override diffusion steps for a quick smoke probe.")
    parser.add_argument("--chop-size", type=int, choices=[256, 512], default=512)
    parser.add_argument("--one-step", action="store_true", default=True)
    parser.add_argument("--noise-repeat", action="store_true")
    parser.add_argument("--resize", choices=["area", "bilinear", "bicubic"], default="area")
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names-file")
    parser.add_argument("--metrics-csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16"], default="fp16")
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
