from __future__ import annotations

from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torch.nn.functional as F


INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
    "area": cv2.INTER_AREA,
}


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def image_to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(img.astype(np.float32) / 255.0).view(1, 1, img.shape[0], img.shape[1]).to(device)


def tensor_to_image(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().float().clamp(0, 1)[0, 0].cpu().numpy()
    return np.round(arr * 255.0).clip(0, 255).astype(np.uint8)


def interpolate_np(img: np.ndarray, out_size: tuple[int, int], method: str) -> np.ndarray:
    return cv2.resize(img, out_size, interpolation=INTERP[method])


def interp_tensor(lr: torch.Tensor, scale: int, method: str) -> torch.Tensor:
    if method == "lanczos":
        arr = (lr[0, 0].detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        pred = cv2.resize(arr, (640, 512), interpolation=cv2.INTER_LANCZOS4)
        return torch.from_numpy(pred.astype(np.float32) / 255.0).view(1, 1, 512, 640).to(lr.device)
    mode = "bicubic" if method == "cubic" else "bilinear"
    if method in ("nearest", "area"):
        mode = method
    kwargs = {} if mode in ("nearest", "area") else {"align_corners": False}
    return F.interpolate(lr, scale_factor=scale, mode=mode, **kwargs).clamp(0, 1)


def resize_tensor(x: torch.Tensor, size: tuple[int, int], method: str) -> torch.Tensor:
    if method == "cubic":
        return F.interpolate(x, size=size, mode="bicubic", align_corners=False)
    if method == "linear":
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    if method == "area":
        return F.interpolate(x, size=size, mode="area")
    if method == "nearest":
        return F.interpolate(x, size=size, mode="nearest")
    raise ValueError(f"Unsupported tensor resize method: {method}")


def back_project(
    pred: torch.Tensor,
    lr: torch.Tensor,
    iters: int,
    alpha: float,
    down_method: str,
    up_method: str,
    down_sigma: float = 0.0,
) -> torch.Tensor:
    if iters <= 0 or alpha <= 0:
        return pred
    target_size = lr.shape[-2:]
    pred_size = pred.shape[-2:]
    lr = lr.float()
    for _ in range(iters):
        down_source = _gaussian_blur(pred, down_sigma) if down_sigma > 0 else pred
        down = resize_tensor(down_source, target_size, down_method)
        residual = lr - down
        pred = pred + alpha * resize_tensor(residual, pred_size, up_method)
    return pred


def _aug(x: torch.Tensor, mode: int) -> torch.Tensor:
    if mode == 0:
        return x
    if mode == 1:
        return torch.flip(x, [3])
    if mode == 2:
        return torch.flip(x, [2])
    if mode == 3:
        return torch.flip(x, [2, 3])
    if mode == 4:
        return torch.rot90(x, 1, [2, 3])
    if mode == 5:
        return torch.rot90(x, 3, [2, 3])
    if mode == 6:
        return torch.flip(torch.rot90(x, 1, [2, 3]), [3])
    if mode == 7:
        return torch.flip(torch.rot90(x, 1, [2, 3]), [2])
    raise ValueError(mode)


def _deaug(x: torch.Tensor, mode: int) -> torch.Tensor:
    if mode == 0:
        return x
    if mode == 1:
        return torch.flip(x, [3])
    if mode == 2:
        return torch.flip(x, [2])
    if mode == 3:
        return torch.flip(x, [2, 3])
    if mode == 4:
        return torch.rot90(x, -1, [2, 3])
    if mode == 5:
        return torch.rot90(x, -3, [2, 3])
    if mode == 6:
        return torch.rot90(torch.flip(x, [3]), -1, [2, 3])
    if mode == 7:
        return torch.rot90(torch.flip(x, [2]), -1, [2, 3])
    raise ValueError(mode)


@torch.no_grad()
def forward_model(model, x: torch.Tensor, amp: str, tta: bool) -> torch.Tensor:
    if not tta:
        with autocast_context(x.device, amp):
            return model(x).float()
    outs = []
    for mode in range(8):
        aug = _aug(x, mode)
        with autocast_context(x.device, amp):
            pred = model(aug).float()
        outs.append(_deaug(pred, mode))
    return torch.stack(outs).mean(dim=0)


@torch.no_grad()
def forward_ensemble(
    models: list[torch.nn.Module],
    x: torch.Tensor,
    amp: str,
    tta: bool,
    coeffs: list[float] | None = None,
) -> torch.Tensor:
    if not models:
        raise ValueError("At least one model is required.")
    if coeffs is None:
        coeffs = [1.0 / len(models)] * len(models)
    total = sum(coeffs)
    coeffs = [c / total for c in coeffs]
    preds = [forward_model(model, x, amp, tta) * coeff for model, coeff in zip(models, coeffs)]
    return torch.stack(preds).sum(dim=0)


def parse_coeffs(raw: str | None, count: int) -> list[float] | None:
    if raw is None or raw == "":
        return None
    coeffs = [float(x) for x in raw.split(",")]
    if len(coeffs) != count:
        raise ValueError(f"Expected {count} ensemble coefficients, got {len(coeffs)}.")
    if any(c < 0 for c in coeffs) or sum(coeffs) <= 0:
        raise ValueError("Ensemble coefficients must be non-negative and sum to a positive value.")
    return coeffs


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    radius = max(1, int(round(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(coords**2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    kh = kernel.view(1, 1, -1, 1)
    kw = kernel.view(1, 1, 1, -1)
    x_pad = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x_blur = F.conv2d(x_pad, kh)
    x_pad = F.pad(x_blur, (radius, radius, 0, 0), mode="reflect")
    return F.conv2d(x_pad, kw)


def apply_postprocess(
    pred: torch.Tensor,
    base: torch.Tensor | None,
    lr: torch.Tensor | None = None,
    blend_interp: float = 0.0,
    sharpen_amount: float = 0.0,
    sharpen_radius: float = 1.0,
    back_project_iters: int = 0,
    back_project_alpha: float = 1.0,
    back_project_down: str = "area",
    back_project_up: str = "cubic",
    back_project_down_sigma: float = 0.0,
    clip_mode: str = "hard",
) -> torch.Tensor:
    pred = pred.float()
    if base is not None and blend_interp > 0:
        pred = pred * (1.0 - blend_interp) + base.float() * blend_interp
    if sharpen_amount > 0:
        blur = _gaussian_blur(pred, sharpen_radius)
        pred = pred + sharpen_amount * (pred - blur)
    if lr is not None and back_project_iters > 0:
        pred = back_project(
            pred,
            lr,
            back_project_iters,
            back_project_alpha,
            back_project_down,
            back_project_up,
            back_project_down_sigma,
        )
    if clip_mode == "hard":
        pred = pred.clamp(0, 1)
    elif clip_mode == "match-base":
        if base is None:
            pred = pred.clamp(0, 1)
        else:
            lo = base.amin(dim=(2, 3), keepdim=True)
            hi = base.amax(dim=(2, 3), keepdim=True)
            pred = torch.maximum(torch.minimum(pred, hi), lo)
    elif clip_mode == "none":
        pred = pred.clamp(0, 1)
    else:
        raise ValueError(f"Unsupported clip mode: {clip_mode}")
    return pred
