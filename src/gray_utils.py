from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


GRAY_MODE_CHOICES = ["avg", "y", "r", "g", "b"]


def normalize_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros(x.shape, dtype=np.uint8)
    x = (x - lo) / (hi - lo)
    return np.round(np.clip(x, 0, 1) * 255.0).astype(np.uint8)


def _bgr_to_gray(img: np.ndarray, gray_mode: str) -> np.ndarray:
    x = img.astype(np.float32)
    if gray_mode == "avg":
        gray = x.mean(axis=2)
    elif gray_mode == "y":
        weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
        gray = (x * weights.reshape(1, 1, 3)).sum(axis=2)
    elif gray_mode == "r":
        gray = x[..., 2]
    elif gray_mode == "g":
        gray = x[..., 1]
    elif gray_mode == "b":
        gray = x[..., 0]
    else:
        raise ValueError(f"Unsupported gray mode: {gray_mode}")
    return normalize_gray_u8(gray)


def load_gray_u8(path: str | Path, gray_mode: str = "avg") -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        return normalize_gray_u8(img)
    if img.ndim == 3 and img.shape[2] == 1:
        return normalize_gray_u8(img[:, :, 0])
    if img.ndim == 3 and img.shape[2] >= 3:
        return _bgr_to_gray(img[:, :, :3], gray_mode)
    raise ValueError(f"Unsupported image shape for {path}: {img.shape}")


def rgb_tensor_to_gray(x: torch.Tensor, gray_mode: str) -> torch.Tensor:
    if gray_mode == "avg":
        return x.mean(dim=1, keepdim=True)
    if gray_mode == "y":
        weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (x * weights).sum(dim=1, keepdim=True)
    if gray_mode == "r":
        return x[:, 0:1]
    if gray_mode == "g":
        return x[:, 1:2]
    if gray_mode == "b":
        return x[:, 2:3]
    raise ValueError(f"Unsupported gray mode: {gray_mode}")
