from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import load_pair, make_split
from evaluate_hat_gray import autocast_context, build_hat, forward_gray, load_hat_weights
from evaluate_hat_postprocess_sweep import Route, parse_route, safe_name
from evaluate_onnx_sr_gray import (
    build_session,
    fixed_hw,
    infer_channels,
    infer_layout,
    input_dtype,
    normalize_output,
    run_onnx_tiled,
)
from metrics import EdgeMetric, measure_batch, metric_proxy
from runtime import apply_postprocess, interp_tensor


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


def read_names(args: Any) -> list[str]:
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    else:
        names = split.val
    return names[: args.limit] if args.limit else names


def write_metrics_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "psnr", "ssim", "edge", "lpips", "proxy"])
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "route",
                "alpha_hat",
                "interp",
                "blend_interp",
                "sharpen_amount",
                "sharpen_radius",
                "clip_mode",
                "psnr",
                "ssim",
                "edge",
                "lpips",
                "proxy",
                "metrics_csv",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def cache_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{safe_name(Path(name).stem)}.pt"


class OnnxExternal:
    def __init__(self, args: Any):
        self.args = args
        session_args = SimpleNamespace(**vars(args))
        session_args.weights = args.external_weights
        self.session = build_session(session_args)
        self.input_info = self.session.get_inputs()[0]
        self.output_info = self.session.get_outputs()[0]
        self.input_layout = infer_layout(list(self.input_info.shape), args.input_layout)
        self.input_channels = infer_channels(list(self.input_info.shape), self.input_layout, args.input_channels)
        self.input_hw = fixed_hw(list(self.input_info.shape), self.input_layout)
        self.dtype = input_dtype(self.input_info.type)
        print(
            f"external=onnx input={self.input_info.shape} layout={self.input_layout} "
            f"channels={self.input_channels} output={self.output_info.shape}"
        )

    @torch.no_grad()
    def __call__(self, lr: torch.Tensor, hr_size: tuple[int, int], device: torch.device) -> torch.Tensor:
        out = run_onnx_tiled(
            self.session,
            self.input_info,
            self.output_info,
            lr,
            self.input_layout,
            self.input_channels,
            self.dtype,
            self.input_hw,
            self.args,
        ).to(device)
        out = normalize_output(out, self.args.output_range)
        out = rgb_to_gray(out, self.args.external_gray_mode)
        if out.shape[-2:] != hr_size:
            if not self.args.allow_resize_output:
                raise ValueError(f"External output size {tuple(out.shape[-2:])} != target {hr_size}")
            out = F.interpolate(out, size=hr_size, mode="bicubic", align_corners=False)
        return out.float()


class SpandrelExternal:
    def __init__(self, args: Any, device: torch.device):
        from spandrel import ModelLoader

        self.args = args
        self.descriptor = ModelLoader(device=device).load_from_file(args.external_weights)
        self.descriptor.eval()
        arch_name = getattr(self.descriptor.architecture, "id", self.descriptor.architecture.__class__.__name__)
        n_params = sum(p.numel() for p in self.descriptor.model.parameters()) / 1e6
        print(
            f"external=spandrel params={n_params:.2f}M arch={arch_name} "
            f"scale=x{self.descriptor.scale} in_ch={self.descriptor.input_channels}"
        )

    @torch.no_grad()
    def __call__(self, lr: torch.Tensor, hr_size: tuple[int, int], device: torch.device) -> torch.Tensor:
        if self.descriptor.input_channels == 1:
            inp = lr
        elif self.descriptor.input_channels == 3:
            inp = lr.repeat(1, 3, 1, 1)
        else:
            raise ValueError(f"Unsupported Spandrel input channels: {self.descriptor.input_channels}")
        out = self.descriptor(inp)
        if out.shape[-2:] != hr_size:
            mode = "area" if self.args.external_downsample == "area" else "bicubic"
            kwargs = {} if mode == "area" else {"align_corners": False}
            out = F.interpolate(out, size=hr_size, mode=mode, **kwargs)
        return rgb_to_gray(out, self.args.external_gray_mode).float()


class CatExternal:
    def __init__(self, args: Any, device: torch.device):
        from evaluate_cat_gray import build_cat, normalize_state

        self.args = args
        self.device = device
        self.model = build_cat(args.external_cat_variant, args.scale).to(device).eval()
        ckpt = torch.load(args.external_weights, map_location="cpu")
        self.model.load_state_dict(normalize_state(ckpt, args.external_param_key), strict=True)
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"external=cat params={n_params:.2f}M variant={args.external_cat_variant}")

    @torch.no_grad()
    def __call__(self, lr: torch.Tensor, hr_size: tuple[int, int], device: torch.device) -> torch.Tensor:
        from evaluate_cat_gray import forward_gray as forward_cat_gray

        with autocast_context(device, self.args.amp):
            out = forward_cat_gray(self.model, lr, self.args.scale, self.args.external_gray_mode)
        if out.shape[-2:] != hr_size:
            if not self.args.allow_resize_output:
                raise ValueError(f"External output size {tuple(out.shape[-2:])} != target {hr_size}")
            out = F.interpolate(out, size=hr_size, mode="bicubic", align_corners=False)
        return out.float()


class SwinFIRExternal:
    def __init__(self, args: Any, device: torch.device):
        from evaluate_swinfir_gray import build_model, normalize_state

        self.args = args
        self.model = build_model(args.external_swinfir_variant, args.scale).to(device).eval()
        ckpt = torch.load(args.external_weights, map_location="cpu")
        self.model.load_state_dict(normalize_state(ckpt, args.external_param_key), strict=True)
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"external=swinfir params={n_params:.2f}M variant={args.external_swinfir_variant}")

    @torch.no_grad()
    def __call__(self, lr: torch.Tensor, hr_size: tuple[int, int], device: torch.device) -> torch.Tensor:
        from evaluate_swinfir_gray import forward_gray as forward_swinfir_gray

        out = forward_swinfir_gray(self.model, lr, self.args.external_gray_mode)
        if out.shape[-2:] != hr_size:
            if not self.args.allow_resize_output:
                raise ValueError(f"External output size {tuple(out.shape[-2:])} != target {hr_size}")
            out = F.interpolate(out, size=hr_size, mode="bicubic", align_corners=False)
        return out.float()


class ATDExternal:
    def __init__(self, args: Any, device: torch.device):
        from evaluate_atd_gray import build_atd, normalize_state

        self.args = args
        self.model = build_atd(args.external_atd_variant, args.scale, args.external_use_checkpoint).to(device).eval()
        ckpt = torch.load(args.external_weights, map_location="cpu")
        self.model.load_state_dict(normalize_state(ckpt, args.external_param_key), strict=True)
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"external=atd params={n_params:.2f}M variant={args.external_atd_variant}")

    @torch.no_grad()
    def __call__(self, lr: torch.Tensor, hr_size: tuple[int, int], device: torch.device) -> torch.Tensor:
        from evaluate_atd_gray import forward_gray as forward_atd_gray

        with autocast_context(device, self.args.amp):
            out = forward_atd_gray(self.model, lr, self.args.external_gray_mode)
        if out.shape[-2:] != hr_size:
            if not self.args.allow_resize_output:
                raise ValueError(f"External output size {tuple(out.shape[-2:])} != target {hr_size}")
            out = F.interpolate(out, size=hr_size, mode="bicubic", align_corners=False)
        return out.float()


class GPSMambaExternal:
    def __init__(self, args: Any, device: torch.device):
        from evaluate_gpsmamba_gray import build_gpsmamba, normalize_state

        self.args = args
        self.model = build_gpsmamba(args.scale).to(device).eval()
        ckpt = torch.load(args.external_weights, map_location="cpu")
        self.model.load_state_dict(normalize_state(ckpt, args.external_param_key), strict=True)
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"external=gpsmamba params={n_params:.2f}M")

    @torch.no_grad()
    def __call__(self, lr: torch.Tensor, hr_size: tuple[int, int], device: torch.device) -> torch.Tensor:
        from evaluate_gpsmamba_gray import forward_gray as forward_gpsmamba_gray

        with autocast_context(device, self.args.amp):
            out = forward_gpsmamba_gray(
                self.model,
                lr,
                self.args.external_gray_mode,
                self.args.scale,
                self.args.external_tta,
                self.args.external_tile,
                self.args.external_tile_overlap,
            )
        if out.shape[-2:] != hr_size:
            if not self.args.allow_resize_output:
                raise ValueError(f"External output size {tuple(out.shape[-2:])} != target {hr_size}")
            out = F.interpolate(out, size=hr_size, mode="bicubic", align_corners=False)
        return out.float()


class IRSRMambaExternal:
    def __init__(self, args: Any, device: torch.device):
        from evaluate_irsrmamba_gray import build_irsrmamba, normalize_state

        self.args = args
        self.model = build_irsrmamba(args.scale).to(device).eval()
        ckpt = torch.load(args.external_weights, map_location="cpu")
        self.model.load_state_dict(normalize_state(ckpt, args.external_param_key), strict=True)
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"external=irsrmamba params={n_params:.2f}M")

    @torch.no_grad()
    def __call__(self, lr: torch.Tensor, hr_size: tuple[int, int], device: torch.device) -> torch.Tensor:
        from evaluate_irsrmamba_gray import forward_gray as forward_irsrmamba_gray

        with autocast_context(device, self.args.amp):
            out = forward_irsrmamba_gray(
                self.model,
                lr,
                self.args.external_gray_mode,
                self.args.scale,
                self.args.external_tta,
                self.args.external_tile,
                self.args.external_tile_overlap,
            )
        if out.shape[-2:] != hr_size:
            if not self.args.allow_resize_output:
                raise ValueError(f"External output size {tuple(out.shape[-2:])} != target {hr_size}")
            out = F.interpolate(out, size=hr_size, mode="bicubic", align_corners=False)
        return out.float()


class SSTExternal:
    def __init__(self, args: Any, device: torch.device):
        from evaluate_sst_gray import build_sst, forward_sst_gray, normalize_state

        self.args = args
        self.forward_sst_gray = forward_sst_gray
        self.model = build_sst(args.external_sst_variant).to(device).eval()
        ckpt = torch.load(args.external_weights, map_location="cpu")
        self.model.load_state_dict(normalize_state(ckpt, args.external_param_key), strict=True)
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"external=sst params={n_params:.2f}M variant={args.external_sst_variant}")

    @torch.no_grad()
    def __call__(self, lr: torch.Tensor, hr_size: tuple[int, int], device: torch.device) -> torch.Tensor:
        with autocast_context(device, self.args.amp):
            out = self.forward_sst_gray(
                self.model,
                lr,
                self.args.external_gray_mode,
                tta=self.args.external_tta,
            )
        if out.shape[-2:] != hr_size:
            if not self.args.allow_resize_output:
                raise ValueError(f"External output size {tuple(out.shape[-2:])} != target {hr_size}")
            out = F.interpolate(out, size=hr_size, mode="bicubic", align_corners=False)
        return out.float()


class PFTExternal:
    def __init__(self, args: Any, device: torch.device):
        from evaluate_pft_gray import build_pft, forward_gray as forward_pft_gray, normalize_state

        self.args = args
        self.forward_pft_gray = forward_pft_gray
        self.model = build_pft(args.external_pft_variant, args.scale, args.external_use_checkpoint).to(device).eval()
        ckpt = torch.load(args.external_weights, map_location="cpu")
        self.model.load_state_dict(normalize_state(ckpt, args.external_param_key), strict=True)
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"external=pft params={n_params:.2f}M variant={args.external_pft_variant}")

    @torch.no_grad()
    def __call__(self, lr: torch.Tensor, hr_size: tuple[int, int], device: torch.device) -> torch.Tensor:
        with autocast_context(device, self.args.amp):
            out = self.forward_pft_gray(
                self.model,
                lr,
                self.args.external_gray_mode,
                tta=self.args.external_tta,
            ).float()
        if out.shape[-2:] != hr_size:
            if not self.args.allow_resize_output:
                raise ValueError(f"External output size {tuple(out.shape[-2:])} != target {hr_size}")
            out = F.interpolate(out, size=hr_size, mode="bicubic", align_corners=False)
        return out.float()


def build_external(args: Any, device: torch.device):
    if args.external_kind == "onnx":
        return OnnxExternal(args)
    if args.external_kind == "spandrel":
        return SpandrelExternal(args, device)
    if args.external_kind == "cat":
        return CatExternal(args, device)
    if args.external_kind == "swinfir":
        return SwinFIRExternal(args, device)
    if args.external_kind == "atd":
        return ATDExternal(args, device)
    if args.external_kind == "gpsmamba":
        return GPSMambaExternal(args, device)
    if args.external_kind == "irsrmamba":
        return IRSRMambaExternal(args, device)
    if args.external_kind == "sst":
        return SSTExternal(args, device)
    if args.external_kind == "pft":
        return PFTExternal(args, device)
    raise ValueError(args.external_kind)


@torch.no_grad()
def validate(args: Any) -> None:
    routes = [parse_route(raw) for raw in args.route]
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    if any(alpha < 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError(f"Alphas must be in [0, 1]: {alphas}")
    jobs: list[tuple[str, float, Route]] = []
    for alpha in alphas:
        alpha_tag = f"a{alpha:.3f}".rstrip("0").rstrip(".").replace(".", "p")
        for route in routes:
            jobs.append((f"{alpha_tag}_{route.name}", alpha, route))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")
    names = read_names(args)
    print(f"val={len(names)} scale=x{args.scale} routes={len(jobs)}")

    model = build_hat(args.variant, args.scale, args.use_checkpoint, native_io=args.native_io).to(device).eval()
    load_hat_weights(model, args.weights, args.param_key, native_io=args.native_io, gray_mode=args.gray_mode)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"hat_params={n_params:.2f}M preset=hat_{args.variant}")
    print(f"hat_loaded={args.weights} param_key={args.param_key}")
    external = build_external(args, device)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    interp_methods = sorted({route.interp for _, _, route in jobs if route.needs_base and route.interp is not None})
    sums: dict[str, dict[str, float]] = {name: {} for name, _, _ in jobs}
    metric_rows: dict[str, list[dict[str, float | str]]] = {name: [] for name, _, _ in jobs}

    for name in tqdm(names, desc="val", leave=False):
        lr, hr = load_pair(args.data, name, args.scale, device)
        path = cache_path(cache_dir, name) if cache_dir is not None else None
        if path is not None and path.exists() and not args.refresh_cache:
            payload = torch.load(path, map_location="cpu")
            hat_pred = payload["hat_pred"].to(device=device, dtype=torch.float32)
            ext_pred = payload["ext_pred"].to(device=device, dtype=torch.float32)
        else:
            with autocast_context(device, args.amp):
                hat_pred = forward_gray(model, lr, args.scale, args.gray_mode, tta=args.tta, native_io=args.native_io).float()
            ext_pred = external(lr, hr.shape[-2:], device)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "name": name,
                        "scale": args.scale,
                        "hat_pred": hat_pred.detach().cpu(),
                        "ext_pred": ext_pred.detach().cpu(),
                    },
                    path,
                )
        bases = {method: interp_tensor(lr, args.scale, method) for method in interp_methods}
        for job_name, alpha, route in jobs:
            pred = hat_pred * alpha + ext_pred * (1.0 - alpha)
            base = bases.get(route.interp) if route.needs_base else None
            pred = apply_postprocess(
                pred,
                base,
                lr=lr,
                blend_interp=route.blend_interp,
                sharpen_amount=route.sharpen_amount,
                sharpen_radius=route.sharpen_radius,
                clip_mode=route.clip_mode,
            )
            metrics = measure_batch(pred.float(), hr, edge_metric, lpips_fn)
            one = dict(metrics)
            one.setdefault("lpips", 0.0)
            one["proxy"] = metric_proxy(one)
            row: dict[str, float | str] = {"name": name}
            row.update(one)
            metric_rows[job_name].append(row)
            for key, value in metrics.items():
                route_sum = sums[job_name]
                route_sum[key] = route_sum.get(key, 0.0) + value

    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else None
    summary_rows: list[dict[str, float | str]] = []
    for job_name, alpha, route in jobs:
        out = {key: value / len(names) for key, value in sums[job_name].items()}
        out.setdefault("lpips", 0.0)
        out["proxy"] = metric_proxy(out)
        metrics_csv = ""
        if metrics_dir is not None:
            metrics_path = metrics_dir / f"{safe_name(job_name)}.csv"
            write_metrics_csv(metrics_path, metric_rows[job_name])
            metrics_csv = str(metrics_path)
        summary_rows.append(
            {
                "route": job_name,
                "alpha_hat": alpha,
                "interp": route.interp or "",
                "blend_interp": route.blend_interp,
                "sharpen_amount": route.sharpen_amount,
                "sharpen_radius": route.sharpen_radius,
                "clip_mode": route.clip_mode,
                **out,
                "metrics_csv": metrics_csv,
            }
        )
    summary_rows.sort(key=lambda row: float(row["proxy"]), reverse=True)
    if args.summary_csv:
        write_summary_csv(Path(args.summary_csv), summary_rows)
        print(f"summary_csv={args.summary_csv}")
    if metrics_dir:
        print(f"metrics_dir={metrics_dir}")
    for row in summary_rows:
        print(
            "val_result "
            + f"route={row['route']} alpha_hat={float(row['alpha_hat']):.5f} "
            + f"interp={row['interp']} blend_interp={float(row['blend_interp']):.5f} "
            + " ".join(f"{key}={float(row[key]):.5f}" for key in ["psnr", "ssim", "edge", "lpips", "proxy"])
        )


def parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", choices=["l", "m"], default="l")
    parser.add_argument("--param-key", default="state_dict")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--native-io", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument(
        "--external-kind",
        choices=["onnx", "spandrel", "cat", "swinfir", "atd", "gpsmamba", "irsrmamba", "sst", "pft"],
        required=True,
    )
    parser.add_argument("--external-weights", required=True)
    parser.add_argument("--external-param-key", default="params")
    parser.add_argument("--external-cat-variant", choices=["r", "a", "r2", "a2"], default="a2")
    parser.add_argument("--external-swinfir-variant", choices=["swinfir_t", "swinfir", "hatfir"], default="hatfir")
    parser.add_argument("--external-atd-variant", choices=["base", "light"], default="base")
    parser.add_argument(
        "--external-sst-variant",
        choices=["light", "light_plus", "base", "base_plus", "large", "large_plus", "xl_plus"],
        default="xl_plus",
    )
    parser.add_argument("--external-pft-variant", choices=["pft", "light"], default="pft")
    parser.add_argument("--external-use-checkpoint", action="store_true")
    parser.add_argument("--external-gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--external-downsample", choices=["bicubic", "area"], default="bicubic")
    parser.add_argument("--external-tta", action="store_true")
    parser.add_argument("--external-tile", type=int, default=64)
    parser.add_argument("--external-tile-overlap", type=int, default=16)
    parser.add_argument("--alphas", default="0.9,0.95,0.975,1.0")
    parser.add_argument("--route", action="append", required=True)
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names-file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--summary-csv")
    parser.add_argument("--metrics-dir")
    parser.add_argument("--cache-dir")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--cpu", action="store_true")
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
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
