from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import torch

from gray_utils import rgb_tensor_to_gray


ROOT = Path(__file__).resolve().parents[1]
HAT_ROOT = ROOT / "external_models" / "HAT"
spec = importlib.util.spec_from_file_location("hat_arch_local", HAT_ROOT / "hat" / "archs" / "hat_arch.py")
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load HAT from {HAT_ROOT}")
hat_arch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hat_arch)
HAT = hat_arch.HAT
RGB_MEAN = (0.4488, 0.4371, 0.4040)


def normalize_hat_state(obj: Any, key: str) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict) and key in obj:
        obj = obj[key]
    elif isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint must be a state dict or contain the requested param key.")
    state = {}
    for name, value in obj.items():
        if name.startswith("module."):
            name = name[len("module.") :]
        state[name] = value
    return state


def _gray_output_weights(gray_mode: str, ref: torch.Tensor) -> torch.Tensor:
    if gray_mode == "avg":
        return ref.new_tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    if gray_mode == "y":
        return ref.new_tensor([0.299, 0.587, 0.114])
    if gray_mode == "r":
        return ref.new_tensor([1.0, 0.0, 0.0])
    if gray_mode == "g":
        return ref.new_tensor([0.0, 1.0, 0.0])
    if gray_mode == "b":
        return ref.new_tensor([0.0, 0.0, 1.0])
    raise ValueError(f"Unsupported gray mode: {gray_mode}")


def _gray_mean(gray_mode: str, ref: torch.Tensor) -> torch.Tensor:
    weights = _gray_output_weights(gray_mode, ref)
    rgb_mean = ref.new_tensor(RGB_MEAN)
    return (weights * rgb_mean).sum()


def adapt_hat_state_to_native_io(
    state: dict[str, torch.Tensor],
    gray_mode: str,
    img_range: float = 1.0,
) -> dict[str, torch.Tensor]:
    conv_first = state.get("conv_first.weight")
    conv_first_bias = state.get("conv_first.bias")
    conv_last = state.get("conv_last.weight")
    conv_last_bias = state.get("conv_last.bias")
    if conv_first is None or conv_first_bias is None or conv_last is None or conv_last_bias is None:
        raise KeyError("HAT checkpoint is missing conv_first/conv_last weights needed for native-io adaptation.")
    if conv_first.ndim != 4 or conv_first_bias.ndim != 1 or conv_last.ndim != 4 or conv_last_bias.ndim != 1:
        raise ValueError("Unexpected HAT checkpoint tensor shapes for native-io adaptation.")
    if conv_first.shape[1] == 1 and conv_last.shape[0] == 1 and conv_last_bias.shape[0] == 1:
        return state
    if conv_first.shape[1] != 3 or conv_last.shape[0] != 3 or conv_last_bias.shape[0] != 3:
        raise ValueError(
            "Only standard 3-channel HAT checkpoints can be adapted to native gray io. "
            f"Got conv_first={tuple(conv_first.shape)} conv_last={tuple(conv_last.shape)} "
            f"conv_last_bias={tuple(conv_last_bias.shape)}"
        )
    out_weights = _gray_output_weights(gray_mode, conv_last)
    gray_mean = _gray_mean(gray_mode, conv_first)
    rgb_mean = conv_first.new_tensor(RGB_MEAN)
    kernel_sums = conv_first.sum(dim=(2, 3))
    bias_shift = ((gray_mean - rgb_mean).view(1, -1) * kernel_sums).sum(dim=1) * float(img_range)
    adapted: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key == "conv_first.weight":
            adapted[key] = value.sum(dim=1, keepdim=True)
        elif key == "conv_first.bias":
            adapted[key] = value + bias_shift.to(dtype=value.dtype)
        elif key == "conv_last.weight":
            adapted[key] = (value * out_weights.view(-1, 1, 1, 1)).sum(dim=0, keepdim=True)
        elif key == "conv_last.bias":
            adapted[key] = (value * out_weights).sum().view(1)
        else:
            adapted[key] = value
    return adapted


def build_hat(variant: str, scale: int, use_checkpoint: bool, native_io: bool = False) -> torch.nn.Module:
    if variant == "l":
        depths = [6] * 12
    elif variant == "m":
        depths = [6] * 6
    else:
        raise ValueError(f"Unsupported HAT variant: {variant}")
    return HAT(
        upscale=scale,
        in_chans=1 if native_io else 3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depths=depths,
        embed_dim=180,
        num_heads=[6] * len(depths),
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        use_checkpoint=use_checkpoint,
    )


def load_hat_weights(
    model: torch.nn.Module,
    path: str,
    param_key: str,
    native_io: bool = False,
    gray_mode: str = "avg",
) -> None:
    ckpt = torch.load(path, map_location="cpu")
    state = normalize_hat_state(ckpt, param_key)
    if native_io:
        state = adapt_hat_state_to_native_io(state, gray_mode, img_range=float(getattr(model, "img_range", 1.0)))
    model.load_state_dict(state, strict=True)
    if native_io:
        gray_mean = _gray_mean(gray_mode, model.mean)
        model.mean = model.mean.new_tensor([gray_mean]).view(1, 1, 1, 1)


def forward_hat_gray(
    model: torch.nn.Module,
    lr_gray: torch.Tensor,
    scale: int,
    gray_mode: str,
    window_size: int = 16,
    native_io: bool = False,
) -> torch.Tensor:
    model_in = lr_gray if native_io else lr_gray.repeat(1, 3, 1, 1)
    _, _, h_old, w_old = model_in.shape
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    if h_pad:
        model_in = torch.cat([model_in, torch.flip(model_in, [2])], dim=2)[:, :, : h_old + h_pad, :]
    if w_pad:
        model_in = torch.cat([model_in, torch.flip(model_in, [3])], dim=3)[:, :, :, : w_old + w_pad]
    out = model(model_in)
    out = out[..., : h_old * scale, : w_old * scale]
    if out.shape[1] == 1:
        return out
    return rgb_tensor_to_gray(out, gray_mode)
