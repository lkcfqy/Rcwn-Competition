from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import load_pair, make_split
from metrics import EdgeMetric, measure_batch, metric_proxy


def read_names(args: Any) -> list[str]:
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    return split.val[: args.limit] if args.limit else split.val


def dim_value(dim: Any) -> int | None:
    return dim if isinstance(dim, int) and dim > 0 else None


def infer_layout(shape: list[Any], requested: str) -> str:
    if requested != "auto":
        return requested
    if len(shape) != 4:
        return "nchw"
    c_first = dim_value(shape[1])
    c_last = dim_value(shape[3])
    if c_first in (1, 3):
        return "nchw"
    if c_last in (1, 3):
        return "nhwc"
    return "nchw"


def infer_channels(shape: list[Any], layout: str, requested: str) -> int:
    if requested != "auto":
        return int(requested)
    if len(shape) == 4:
        channel_dim = shape[1] if layout == "nchw" else shape[3]
        channel = dim_value(channel_dim)
        if channel in (1, 3):
            return channel
    return 3


def input_dtype(type_name: str) -> np.dtype:
    if "float16" in type_name:
        return np.float16
    if "uint8" in type_name:
        return np.uint8
    return np.float32


def make_onnx_input(
    lr_gray: torch.Tensor,
    layout: str,
    channels: int,
    value_range: str,
    dtype: np.dtype,
) -> np.ndarray:
    if channels == 1:
        arr = lr_gray.detach().cpu().numpy()
    elif channels == 3:
        arr = lr_gray.repeat(1, 3, 1, 1).detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported input channels: {channels}")
    if value_range == "0_255":
        arr = arr * 255.0
    if layout == "nhwc":
        arr = np.transpose(arr, (0, 2, 3, 1))
    if dtype == np.uint8:
        arr = np.clip(np.rint(arr), 0, 255).astype(dtype)
    else:
        arr = arr.astype(dtype)
    return arr


def fixed_hw(shape: list[Any], layout: str) -> tuple[int | None, int | None]:
    if len(shape) != 4:
        return None, None
    if layout == "nhwc":
        return dim_value(shape[1]), dim_value(shape[2])
    return dim_value(shape[2]), dim_value(shape[3])


def output_to_nchw(output: np.ndarray, layout: str) -> torch.Tensor:
    out = np.asarray(output)
    if out.ndim == 2:
        out = out[None, None, :, :]
    elif out.ndim == 3:
        if out.shape[0] in (1, 3):
            out = out[None, :, :, :]
        elif out.shape[-1] in (1, 3):
            out = np.transpose(out[None, :, :, :], (0, 3, 1, 2))
        else:
            raise ValueError(f"Cannot infer 3D output layout for shape={out.shape}")
    elif out.ndim == 4:
        if layout == "auto":
            if out.shape[1] in (1, 3):
                layout = "nchw"
            elif out.shape[-1] in (1, 3):
                layout = "nhwc"
            else:
                layout = "nchw"
        if layout == "nhwc":
            out = np.transpose(out, (0, 3, 1, 2))
    else:
        raise ValueError(f"Unsupported ONNX output rank: shape={out.shape}")
    return torch.from_numpy(np.ascontiguousarray(out)).float()


def rgb_to_gray(x: torch.Tensor, mode: str) -> torch.Tensor:
    if x.shape[1] == 1:
        return x
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


def normalize_output(out: torch.Tensor, value_range: str) -> torch.Tensor:
    if value_range == "0_255":
        return out / 255.0
    if value_range == "auto" and float(out.detach().amax().cpu()) > 2.0:
        return out / 255.0
    return out


def build_session(args: Any) -> ort.InferenceSession:
    available = ort.get_available_providers()
    if args.provider == "cpu":
        providers = ["CPUExecutionProvider"]
    elif args.provider == "cuda" and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif args.provider == "cuda":
        raise RuntimeError(f"CUDAExecutionProvider unavailable; available={available}")
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.ort_threads
    print(f"providers={providers} available={available}")
    return ort.InferenceSession(args.weights, sess_options=opts, providers=providers)


def tile_starts(size: int, tile: int, overlap: int) -> list[int]:
    if size <= tile:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, size - tile + 1, stride))
    last = size - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def run_onnx_whole(
    session: ort.InferenceSession,
    input_info: Any,
    output_info: Any,
    lr: torch.Tensor,
    input_layout: str,
    input_channels: int,
    dtype: np.dtype,
    args: Any,
) -> torch.Tensor:
    ort_input = make_onnx_input(lr, input_layout, input_channels, args.input_range, dtype)
    ort_outputs = session.run([output_info.name], {input_info.name: ort_input})
    return output_to_nchw(ort_outputs[0], args.output_layout)


def run_onnx_tiled(
    session: ort.InferenceSession,
    input_info: Any,
    output_info: Any,
    lr: torch.Tensor,
    input_layout: str,
    input_channels: int,
    dtype: np.dtype,
    input_hw: tuple[int | None, int | None],
    args: Any,
) -> torch.Tensor:
    tile_h, tile_w = input_hw
    if tile_h is None or tile_w is None:
        return run_onnx_whole(session, input_info, output_info, lr, input_layout, input_channels, dtype, args)
    _, _, h, w = lr.shape
    if h == tile_h and w == tile_w:
        return run_onnx_whole(session, input_info, output_info, lr, input_layout, input_channels, dtype, args)
    if not args.auto_tile_fixed:
        return run_onnx_whole(session, input_info, output_info, lr, input_layout, input_channels, dtype, args)
    out_h = h * args.scale
    out_w = w * args.scale
    acc = torch.zeros((1, input_channels, out_h, out_w), dtype=torch.float32)
    weight = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32)
    for top in tile_starts(h, tile_h, args.tile_overlap):
        for left in tile_starts(w, tile_w, args.tile_overlap):
            patch = lr[..., top : top + tile_h, left : left + tile_w]
            valid_h, valid_w = patch.shape[-2:]
            if valid_h != tile_h or valid_w != tile_w:
                patch = F.pad(patch, (0, tile_w - valid_w, 0, tile_h - valid_h), mode="replicate")
            pred = run_onnx_whole(session, input_info, output_info, patch, input_layout, input_channels, dtype, args)
            pred = pred[..., : valid_h * args.scale, : valid_w * args.scale]
            y0 = top * args.scale
            x0 = left * args.scale
            ph, pw = pred.shape[-2:]
            acc[..., y0 : y0 + ph, x0 : x0 + pw] += pred
            weight[..., y0 : y0 + ph, x0 : x0 + pw] += 1.0
    return acc / weight.clamp_min(1.0)


@torch.no_grad()
def validate(args: Any) -> None:
    names = read_names(args)
    print(f"val={len(names)} scale=x{args.scale}")
    session = build_session(args)
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    input_layout = infer_layout(list(input_info.shape), args.input_layout)
    input_channels = infer_channels(list(input_info.shape), input_layout, args.input_channels)
    input_hw = fixed_hw(list(input_info.shape), input_layout)
    dtype = input_dtype(input_info.type)
    print(
        f"input={input_info.name} shape={input_info.shape} type={input_info.type} "
        f"layout={input_layout} channels={input_channels}"
    )
    print(f"output={output_info.name} shape={output_info.shape} type={output_info.type}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu_metrics else "cpu")
    print(f"metrics_device={device}")
    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    sums: dict[str, float] = {}
    rows: list[dict[str, float | str]] = []
    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(args.data, name, args.scale, device)
        pred = run_onnx_tiled(
            session,
            input_info,
            output_info,
            lr,
            input_layout,
            input_channels,
            dtype,
            input_hw,
            args,
        ).to(device)
        pred = normalize_output(pred, args.output_range)
        pred = rgb_to_gray(pred, args.gray_mode)
        if pred.shape[-2:] != hr.shape[-2:]:
            if not args.allow_resize_output:
                raise ValueError(f"{name}: output size {tuple(pred.shape[-2:])} != target {tuple(hr.shape[-2:])}")
            pred = F.interpolate(pred, size=hr.shape[-2:], mode="bicubic", align_corners=False)
        metrics = measure_batch(pred.float(), hr, edge_metric, lpips_fn)
        row = {"name": name, **metrics, "proxy": metric_proxy(metrics)}
        rows.append(row)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value

    out_metrics = {key: value / len(names) for key, value in sums.items()}
    out_metrics["proxy"] = metric_proxy(out_metrics)
    print("val_result " + " ".join(f"{key}={value:.5f}" for key, value in out_metrics.items()))
    if args.metrics_csv:
        Path(args.metrics_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_csv, "w", newline="", encoding="utf-8") as handle:
            fieldnames = ["name", "psnr", "ssim", "edge", "lpips", "proxy"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        print(f"metrics_csv={args.metrics_csv}")


def parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names-file", default="")
    parser.add_argument("--metrics-csv", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--ort-threads", type=int, default=4)
    parser.add_argument("--input-layout", choices=["auto", "nchw", "nhwc"], default="auto")
    parser.add_argument("--output-layout", choices=["auto", "nchw", "nhwc"], default="auto")
    parser.add_argument("--input-channels", choices=["auto", "1", "3"], default="auto")
    parser.add_argument("--input-range", choices=["0_1", "0_255"], default="0_1")
    parser.add_argument("--output-range", choices=["auto", "0_1", "0_255"], default="auto")
    parser.add_argument("--allow-resize-output", action="store_true")
    parser.add_argument("--auto-tile-fixed", action="store_true")
    parser.add_argument("--tile-overlap", type=int, default=8)
    parser.add_argument("--cpu-metrics", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
