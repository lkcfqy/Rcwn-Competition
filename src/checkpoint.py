from __future__ import annotations

import os
from typing import Any

import torch

from models import build_model


def torch_cuda_library_path() -> str:
    base = "/usr/local/lib/python3.12/dist-packages/nvidia"
    parts = [
        os.path.join(base, "cublas", "lib"),
        os.path.join(base, "cuda_runtime", "lib"),
        os.path.join(base, "cudnn", "lib"),
    ]
    return ":".join(parts)


def normalize_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint must be a state_dict or contain a 'state_dict' key.")
    state = {}
    for key, value in obj.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        state[key] = value
    return state


def load_model_from_checkpoint(
    weights: str,
    device: torch.device,
    scale: int | None = None,
    preset: str | None = None,
    strict: bool = True,
):
    ckpt = torch.load(weights, map_location="cpu")
    cfg = {}
    if isinstance(ckpt, dict):
        cfg = dict(ckpt.get("config", {}))
    if preset == "auto":
        preset = None
    model_scale = scale or int(cfg.get("scale", 2))
    model_preset = preset or str(cfg.get("preset", "base"))
    model = build_model(
        scale=model_scale,
        preset=model_preset,
        num_features=cfg.get("num_features"),
        num_groups=cfg.get("num_groups"),
        num_blocks=cfg.get("num_blocks"),
    )
    state = normalize_state_dict(ckpt)
    model.load_state_dict(state, strict=strict)
    return model.to(device), cfg


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    config: dict[str, Any],
    metrics: dict[str, float] | None = None,
    half: bool = False,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {}
    for key, tensor in model.state_dict().items():
        tensor = tensor.detach().cpu()
        state[key] = tensor.half() if half and tensor.is_floating_point() else tensor
    torch.save({"state_dict": state, "config": config, "metrics": metrics or {}}, path)
