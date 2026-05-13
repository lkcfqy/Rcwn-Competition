from __future__ import annotations

import argparse
import csv
import os
import sys
import types
from contextlib import nullcontext
from pathlib import Path


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
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset import load_pair, make_split
from metrics import EdgeMetric, measure_batch, metric_proxy
from runtime import _aug, _deaug


ROOT = Path(__file__).resolve().parents[1]
LDFF_ROOT = ROOT / "external_models" / "LDFF-Net"


class _BaseModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _ConvModule(nn.Module):
    """Tiny mmcv ConvModule compatibility shim for LDFF-Net inference."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool | str = "auto",
        norm_cfg=None,
        act_cfg: object | None = {"type": "ReLU"},
        **kwargs,
    ):
        super().__init__()
        if bias == "auto":
            bias = norm_cfg is None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bool(bias),
        )
        if act_cfg is None:
            self.activate = None
        else:
            self.activate = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.activate is not None:
            x = self.activate(x)
        return x


def install_ldff_dependency_stubs() -> None:
    mmengine = types.ModuleType("mmengine")
    mmengine_model = types.ModuleType("mmengine.model")
    mmengine_model.BaseModule = _BaseModule
    mmengine.model = mmengine_model
    sys.modules.setdefault("mmengine", mmengine)
    sys.modules.setdefault("mmengine.model", mmengine_model)

    mmcv = types.ModuleType("mmcv")
    mmcv_cnn = types.ModuleType("mmcv.cnn")
    mmcv_cnn.ConvModule = _ConvModule
    mmcv.cnn = mmcv_cnn
    sys.modules.setdefault("mmcv", mmcv)
    sys.modules.setdefault("mmcv.cnn", mmcv_cnn)


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


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


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    install_ldff_dependency_stubs()
    for key in list(sys.modules):
        if key == "model" or key.startswith("model."):
            del sys.modules[key]
    sys.path.insert(0, str(LDFF_ROOT))
    from model.LDFF_Net import MYMODEL  # type: ignore

    model = MYMODEL(up_scale=args.model_scale).to(device).eval()
    state = torch.load(args.weights, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    important_missing = [key for key in missing if "spatial_scale" not in key]
    if important_missing or unexpected:
        print(f"load_state important_missing={important_missing[:8]} unexpected={unexpected[:8]}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params={n_params:.2f}M loaded={args.weights}")
    return model


@torch.no_grad()
def forward_once(model: torch.nn.Module, lr_gray: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    lr_rgb = lr_gray.repeat(1, 3, 1, 1)
    lr_norm = lr_rgb * 2.0 - 1.0
    pred = model(lr_norm)
    if args.match_lr_stats:
        pred = (pred - pred.mean(dim=[2, 3], keepdim=True)) / (pred.std(dim=[2, 3], keepdim=True) + 1e-6)
        pred = pred * lr_norm.std(dim=[2, 3], keepdim=True) + lr_norm.mean(dim=[2, 3], keepdim=True)
    pred = (pred / 2.0 + 0.5).clamp(0, 1)
    return rgb_to_gray(pred, args.gray_mode)


@torch.no_grad()
def forward_gray(model: torch.nn.Module, lr_gray: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if not args.tta:
        return forward_once(model, lr_gray, args)
    preds = []
    for mode in range(8):
        pred = forward_once(model, _aug(lr_gray, mode), args)
        preds.append(_deaug(pred, mode))
    return torch.stack(preds).mean(dim=0)


@torch.no_grad()
def validate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    else:
        names = split.val[: args.limit] if args.limit else split.val
    print(
        f"val={len(names)} scale=x{args.scale} model_scale=x{args.model_scale} "
        f"tta={args.tta} match_lr_stats={args.match_lr_stats}"
    )

    model = build_model(args, device)
    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    sums: dict[str, float] = {}
    rows: list[dict[str, float | str]] = []
    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(args.data, name, args.scale, device)
        with autocast_context(device, args.amp):
            pred = forward_gray(model, lr, args)
        pred = pred.float().clamp(0, 1)
        if pred.shape[-2:] != hr.shape[-2:]:
            kwargs = {} if args.resize == "area" else {"align_corners": False}
            pred = F.interpolate(pred, size=hr.shape[-2:], mode=args.resize, **kwargs).clamp(0, 1)
        metrics = measure_batch(pred, hr, edge_metric, lpips_fn)
        one = dict(metrics)
        one["proxy"] = metric_proxy(one)
        if args.metrics_csv:
            row: dict[str, float | str] = {"name": name}
            row.update(one)
            rows.append(row)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value

    out = {key: value / len(names) for key, value in sums.items()}
    out["proxy"] = metric_proxy(out)
    if args.metrics_csv:
        path = Path(args.metrics_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["name", "psnr", "ssim", "edge", "lpips", "proxy"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"metrics_csv={path}")
    print("val_result " + " ".join(f"{key}={value:.5f}" for key, value in out.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--model-scale", type=int, choices=[2, 4], default=4)
    parser.add_argument("--weights", default=str(LDFF_ROOT / "weight" / "LDFF_Net" / "net_params_1000.pkl"))
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--match-lr-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--resize", choices=["area", "bicubic"], default="area")
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names-file")
    parser.add_argument("--metrics-csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
