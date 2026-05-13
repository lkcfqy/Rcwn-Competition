from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch

from gray_utils import rgb_tensor_to_gray


ROOT = Path(__file__).resolve().parents[1]


def _strip_prefixes(state: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    out = {}
    for name, value in state.items():
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix) :]
        out[name] = value
    return out


def normalize_state(obj: Any, key: str | None = None) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict) and key and key in obj and isinstance(obj[key], dict):
        obj = obj[key]
    elif isinstance(obj, dict):
        for fallback in ("state_dict", "params_ema", "params"):
            if fallback in obj and isinstance(obj[fallback], dict):
                obj = obj[fallback]
                break
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint must be a state dict or contain state_dict/params_ema/params.")
    return _strip_prefixes(obj, ("module.",))


def default_param_key(family: str) -> str:
    if family == "grl":
        return ""
    if family == "rgt":
        return "params"
    if family == "pft":
        return "params_ema"
    if family == "swin2sr":
        return "params"
    if family == "catanet":
        return "params"
    if family == "sst":
        return "params_ema"
    raise ValueError(f"Unsupported family: {family}")


def _load_grl_class():
    grl_root = ROOT / "external_models" / "GRL-Image-Restoration"
    sys.path.insert(0, str(grl_root))
    from models.networks.grl import GRL  # noqa: E402

    return GRL


def build_grl(variant: str, scale: int) -> torch.nn.Module:
    GRL = _load_grl_class()
    common = dict(
        upscale=scale,
        in_channels=3,
        img_range=1.0,
        img_size=256,
        upsampler="pixelshuffle",
        window_size=32,
        stripe_size=[64, 64],
        stripe_groups=[None, None],
        stripe_shift=True,
        mlp_ratio=2,
        qkv_proj_type="linear",
        anchor_proj_type="avgpool",
        anchor_one_stage=True,
        anchor_window_down_factor=4,
        out_proj_type="linear",
        conv_type="1conv",
        init_method="n",
        fairscale_checkpoint=False,
        offload_to_cpu=False,
        use_buffer=True,
        use_efficient_buffer=True,
        euclidean_dist=False,
    )
    if variant == "base":
        return GRL(
            embed_dim=180,
            depths=[4, 4, 8, 8, 8, 4, 4],
            num_heads_window=[3, 3, 3, 3, 3, 3, 3],
            num_heads_stripe=[3, 3, 3, 3, 3, 3, 3],
            local_connection=True,
            **common,
        )
    if variant == "small":
        return GRL(
            embed_dim=128,
            depths=[4, 4, 4, 4],
            num_heads_window=[2, 2, 2, 2],
            num_heads_stripe=[2, 2, 2, 2],
            local_connection=False,
            **common,
        )
    raise ValueError(f"Unsupported GRL variant: {variant}")


def _registry_stub() -> types.ModuleType:
    class _Registry:
        def __init__(self, name: str):
            self._name = name
            self._obj_map = {}

        def register(self, obj=None, suffix=None):
            def deco(cls):
                name = cls.__name__ if suffix is None else f"{cls.__name__}_{suffix}"
                self._obj_map[name] = cls
                return cls

            return deco if obj is None else deco(obj)

    registry_mod = types.ModuleType("basicsr.utils.registry")
    registry_mod.ARCH_REGISTRY = _Registry("arch")
    return registry_mod


def _load_rgt_class():
    rgt_root = ROOT / "external_models" / "RGT"
    basicsr_mod = types.ModuleType("basicsr")
    utils_mod = types.ModuleType("basicsr.utils")
    registry_mod = _registry_stub()
    sys.modules.setdefault("basicsr", basicsr_mod)
    sys.modules.setdefault("basicsr.utils", utils_mod)
    sys.modules["basicsr.utils.registry"] = registry_mod
    spec = importlib.util.spec_from_file_location("rgt_arch_local", rgt_root / "basicsr" / "archs" / "rgt_arch.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load RGT from {rgt_root}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RGT


def build_rgt(variant: str, scale: int) -> torch.nn.Module:
    RGT = _load_rgt_class()
    if variant == "rgt":
        depth = [6] * 8
    elif variant == "s":
        depth = [6] * 6
    else:
        raise ValueError(f"Unsupported RGT variant: {variant}")
    return RGT(
        upscale=scale,
        in_chans=3,
        img_size=64,
        img_range=1.0,
        depth=depth,
        embed_dim=180,
        num_heads=[6] * len(depth),
        mlp_ratio=2,
        resi_connection="1conv",
        split_size=[8, 32],
        c_ratio=0.5,
    )


def _load_pft_class():
    pft_root = ROOT / "external_models" / "PFT-SR"
    sys.path.insert(0, str(pft_root))
    from basicsr.archs.pft_arch import PFT  # noqa: E402

    return PFT


def build_pft(variant: str, scale: int, use_checkpoint: bool) -> torch.nn.Module:
    PFT = _load_pft_class()
    if variant == "pft":
        return PFT(
            upscale=scale,
            in_chans=3,
            img_size=64,
            embed_dim=240,
            depths=[4, 4, 4, 6, 6, 6],
            num_heads=6,
            num_topk=[
                1024,
                1024,
                1024,
                1024,
                256,
                256,
                256,
                256,
                128,
                128,
                128,
                128,
                64,
                64,
                64,
                64,
                64,
                64,
                32,
                32,
                32,
                32,
                32,
                32,
                16,
                16,
                16,
                16,
                16,
                16,
            ],
            window_size=32,
            convffn_kernel_size=7,
            img_range=1.0,
            mlp_ratio=2.0,
            upsampler="pixelshuffle",
            resi_connection="1conv",
            use_checkpoint=use_checkpoint,
        )
    if variant == "light":
        return PFT(
            upscale=scale,
            in_chans=3,
            img_size=64,
            embed_dim=52,
            depths=[2, 4, 6, 6, 6],
            num_heads=4,
            num_topk=[
                1024,
                1024,
                256,
                256,
                256,
                256,
                128,
                128,
                128,
                128,
                128,
                128,
                64,
                64,
                64,
                64,
                64,
                64,
                32,
                32,
                32,
                32,
                32,
                32,
            ],
            window_size=32,
            convffn_kernel_size=7,
            img_range=1.0,
            mlp_ratio=1.0,
            upsampler="pixelshuffledirect",
            resi_connection="1conv",
            use_checkpoint=use_checkpoint,
        )
    raise ValueError(f"Unsupported PFT variant: {variant}")


def _load_swin2sr_class():
    swin2sr_root = ROOT / "external_models" / "swin2sr"
    spec = importlib.util.spec_from_file_location(
        "swin2sr_network_local",
        swin2sr_root / "models" / "network_swin2sr.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Swin2SR from {swin2sr_root}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Swin2SR


def build_swin2sr(scale: int, use_checkpoint: bool) -> torch.nn.Module:
    Swin2SR = _load_swin2sr_class()
    return Swin2SR(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        use_checkpoint=use_checkpoint,
    )


def _load_catanet_class():
    catanet_root = ROOT / "external_models" / "CATANet"
    basicsr_mod = types.ModuleType("basicsr")
    utils_mod = types.ModuleType("basicsr.utils")
    archs_mod = types.ModuleType("basicsr.archs")
    registry_mod = _registry_stub()
    arch_util_mod = types.ModuleType("basicsr.archs.arch_util")
    arch_util_mod.trunc_normal_ = torch.nn.init.trunc_normal_
    sys.modules.setdefault("basicsr", basicsr_mod)
    sys.modules.setdefault("basicsr.utils", utils_mod)
    sys.modules.setdefault("basicsr.archs", archs_mod)
    sys.modules["basicsr.utils.registry"] = registry_mod
    sys.modules["basicsr.archs.arch_util"] = arch_util_mod
    spec = importlib.util.spec_from_file_location(
        "catanet_arch_local", catanet_root / "basicsr" / "archs" / "catanet_arch.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load CATANet from {catanet_root}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CATANet


def build_catanet(scale: int) -> torch.nn.Module:
    CATANet = _load_catanet_class()
    return CATANet(upscale=scale)


def build_sst(variant: str, scale: int) -> torch.nn.Module:
    from evaluate_sst_gray import build_sst as _build_sst

    return _build_sst(variant, scale)


def build_model(family: str, variant: str, scale: int, use_checkpoint: bool = False) -> torch.nn.Module:
    if family == "grl":
        return build_grl(variant, scale)
    if family == "pft":
        return build_pft(variant, scale, use_checkpoint)
    if family == "rgt":
        return build_rgt(variant, scale)
    if family == "swin2sr":
        return build_swin2sr(scale, use_checkpoint)
    if family == "catanet":
        return build_catanet(scale)
    if family == "sst":
        return build_sst(variant, scale)
    raise ValueError(f"Unsupported family: {family}")


def load_model_weights(
    model: torch.nn.Module,
    family: str,
    path: str,
    param_key: str | None = None,
) -> str:
    ckpt = torch.load(path, map_location="cpu")
    key = param_key if param_key is not None else default_param_key(family)
    state = normalize_state(ckpt, key if key else None)
    if family == "grl":
        state = _strip_prefixes(state, ("model.",))
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [name for name in missing if not name.startswith(("table_", "index_", "mask_"))]
        if missing or unexpected:
            raise RuntimeError(f"Unexpected GRL checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    else:
        model.load_state_dict(state, strict=True)
    return key


def forward_gray(
    model: torch.nn.Module,
    family: str,
    lr_gray: torch.Tensor,
    scale: int,
    gray_mode: str,
) -> torch.Tensor:
    lr_rgb = lr_gray.repeat(1, 3, 1, 1)
    out = model(lr_rgb)
    if isinstance(out, tuple):
        out = out[0]
    if family in {"rgt", "catanet", "sst"}:
        out = out[..., : lr_gray.shape[-2] * scale, : lr_gray.shape[-1] * scale]
    return rgb_tensor_to_gray(out, gray_mode)
