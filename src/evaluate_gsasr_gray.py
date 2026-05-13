from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path


def _prefer_torch_cuda_libs() -> None:
    py312 = "/usr/local/lib/python3.12/dist-packages"
    libs = [
        os.path.join(py312, "torch", "lib"),
        os.path.join(py312, "nvidia", "cublas", "lib"),
        os.path.join(py312, "nvidia", "cuda_runtime", "lib"),
        os.path.join(py312, "nvidia", "cudnn", "lib"),
    ]
    prefix = ":".join([p for p in libs if os.path.isdir(p)])
    if not prefix:
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if os.environ.get("RCWN_GSASR_CUDA_LIBS_OK") != "1":
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ":".join([prefix, existing]) if existing else prefix
        env["RCWN_GSASR_CUDA_LIBS_OK"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_prefer_torch_cuda_libs()

import torch
from tqdm import tqdm

from dataset import load_pair, make_split
from metrics import EdgeMetric, measure_batch, metric_proxy

ROOT = Path(__file__).resolve().parents[1]
GSASR_ROOT = ROOT / "external_models" / "GSASR"
sys.path.insert(0, str(GSASR_ROOT))

from inference_enhenced import (  # noqa: E402
    generate_2D_gaussian_splatting_step,
    load_model,
    postprocess,
    preprocess,
    split_and_joint_image,
)


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


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


def read_names(args: argparse.Namespace) -> list[str]:
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    else:
        names = split.val
    return names[: args.limit] if args.limit else names


def write_metrics_csv(path: str, rows: list[dict[str, float | str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "psnr", "ssim", "edge", "lpips", "proxy"])
        writer.writeheader()
        writer.writerows(rows)


def denominator_for(model_name: str) -> int:
    if model_name in {"EDSR_DIV2K", "EDSR_DF2K", "RDN_DIV2K", "RDN_DF2K"}:
        return 12
    return 16


def forward_gsasr(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    lr_gray: torch.Tensor,
    scale: int,
    denominator: int,
    tile_process: bool,
    tile_size: int,
    tile_overlap: int,
    crop_size: int,
    dmax: float,
) -> torch.Tensor:
    lq = lr_gray.repeat(1, 3, 1, 1)
    scale_float = float(scale)
    gt_h = math.floor(scale_float * lq.shape[2])
    gt_w = math.floor(scale_float * lq.shape[3])
    scale_modify = torch.tensor([scale_float, scale_float], device=lq.device)

    if tile_process:
        output = split_and_joint_image(
            lq=lq,
            scale_factor=scale_float,
            split_size=tile_size,
            overlap_size=tile_overlap,
            model_g=encoder,
            model_fea2gs=decoder,
            crop_size=crop_size,
            scale_modify=scale_modify,
            default_step_size=1.2,
            cuda_rendering=True,
            mode="scale_modify",
            if_dmax=True,
            dmax_mode="fix",
            dmax=dmax,
        )
        return postprocess(output, gt_h, gt_w)

    lq_pad = preprocess(lq, denominator)
    gt_size_pad = torch.tensor(
        [math.floor(scale_float * lq_pad.shape[2]), math.floor(scale_float * lq_pad.shape[3])]
    )
    encoder_output = encoder(lq_pad)
    scale_vector = torch.tensor(scale_float, dtype=torch.float32, device=lq.device).unsqueeze(0)
    batch_gs_parameters = decoder(encoder_output, scale_vector)
    gs_parameters = batch_gs_parameters[0, :]
    b_output = generate_2D_gaussian_splatting_step(
        gs_parameters=gs_parameters,
        sr_size=gt_size_pad,
        scale=scale_float,
        sample_coords=None,
        scale_modify=scale_modify,
        default_step_size=1.2,
        cuda_rendering=True,
        mode="scale_modify",
        if_dmax=True,
        dmax_mode="fix",
        dmax=dmax,
    )
    return postprocess(b_output.unsqueeze(0), gt_h, gt_w)


@torch.no_grad()
def validate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    names = read_names(args)
    print(f"device={device} val={len(names)} scale=x{args.scale}")
    print(
        f"model={args.model_name} source={args.model_path} gray_mode={args.gray_mode} "
        f"tile={args.tile_process} dmax={args.dmax}"
    )

    encoder, decoder = load_model(
        pretrained_model_name_or_path=args.model_path,
        model_name=args.model_name,
        device=device,
    )
    denominator = denominator_for(args.model_name)
    n_params = (sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())) / 1e6
    print(f"params={n_params:.2f}M denominator={denominator}")

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
            pred_rgb = forward_gsasr(
                encoder=encoder,
                decoder=decoder,
                lr_gray=lr,
                scale=args.scale,
                denominator=denominator,
                tile_process=args.tile_process,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                crop_size=args.crop_size,
                dmax=args.dmax,
            )
        pred = rgb_to_gray(pred_rgb.float().clamp(0, 1), args.gray_mode)
        metrics = measure_batch(pred, hr, edge_metric, lpips_fn)
        one = dict(metrics)
        one.setdefault("lpips", 0.0)
        one["proxy"] = metric_proxy(one)
        row: dict[str, float | str] = {"name": name}
        row.update(one)
        metric_rows.append(row)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = {key: value / len(names) for key, value in sums.items()}
    out["proxy"] = metric_proxy(out)
    if args.metrics_csv:
        write_metrics_csv(args.metrics_csv, metric_rows)
        print(f"metrics_csv={args.metrics_csv}")
    print("val_result " + " ".join(f"{key}={value:.5f}" for key, value in out.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument(
        "--model-name",
        choices=["EDSR_DIV2K", "EDSR_DF2K", "RDN_DIV2K", "RDN_DF2K", "SWIN_DIV2K", "SWIN_DF2K", "HATL_SA1B"],
        default="EDSR_DF2K",
    )
    parser.add_argument("--model-path", default="mutou0308/GSASR")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names-file")
    parser.add_argument("--metrics-csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--tile-process", action="store_true")
    parser.add_argument("--tile-size", type=int, default=240)
    parser.add_argument("--tile-overlap", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=4)
    parser.add_argument("--dmax", type=float, default=0.1)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
