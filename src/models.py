from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from thermal_models import THERMAL_PRESETS, ThermalEdgeSR


def _make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(self.pool(x))


class RCAB(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        res_scale: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            _make_activation(activation),
            nn.Conv2d(channels, channels, 3, padding=1),
            ChannelAttention(channels, reduction),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class ResidualGroup(nn.Module):
    def __init__(
        self,
        channels: int,
        num_blocks: int,
        reduction: int = 16,
        res_scale: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        blocks = [
            RCAB(
                channels=channels,
                reduction=reduction,
                res_scale=res_scale,
                activation=activation,
            )
            for _ in range(num_blocks)
        ]
        blocks.append(nn.Conv2d(channels, channels, 3, padding=1))
        self.body = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class Upsampler(nn.Sequential):
    def __init__(self, scale: int, channels: int):
        layers = []
        if scale in (2, 4):
            for _ in range(scale.bit_length() - 1):
                layers.extend(
                    [
                        nn.Conv2d(channels, channels * 4, 3, padding=1),
                        nn.PixelShuffle(2),
                    ]
                )
        else:
            raise ValueError("Only x2 and x4 upscaling are supported.")
        super().__init__(*layers)


class RCAN(nn.Module):
    """Residual Channel Attention Network for single-channel infrared SR."""

    def __init__(
        self,
        scale: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 96,
        num_groups: int = 8,
        num_blocks: int = 10,
        reduction: int = 16,
        res_scale: float = 0.1,
        activation: str = "relu",
        skip: str = "bicubic",
    ):
        super().__init__()
        self.scale = scale
        self.skip = skip
        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.body = nn.Sequential(
            *[
                ResidualGroup(
                    channels=num_features,
                    num_blocks=num_blocks,
                    reduction=reduction,
                    res_scale=res_scale,
                    activation=activation,
                )
                for _ in range(num_groups)
            ],
            nn.Conv2d(num_features, num_features, 3, padding=1),
        )
        self.tail = nn.Sequential(
            Upsampler(scale, num_features),
            nn.Conv2d(num_features, out_channels, 3, padding=1),
        )
        nn.init.zeros_(self.tail[-1].weight)
        nn.init.zeros_(self.tail[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        feat = feat + self.body(feat)
        out = self.tail(feat)
        if self.skip == "none":
            return out
        base = F.interpolate(x, scale_factor=self.scale, mode=self.skip, align_corners=False)
        return out + base


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        table_size = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))

        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_windows, n, c = x.shape
        qkv = self.qkv(x).reshape(b_windows, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(n, n, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(b_windows // num_windows, num_windows, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b_windows, n, c)
        return self.proj(x)


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def _attention_mask(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        img_mask = x.new_zeros((1, h, w, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = count
                count += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        shortcut = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm1(shortcut)

        shift_size = self.shift_size if min(h, w) > self.window_size else 0
        if shift_size > 0:
            shifted = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
            attn_mask = self._attention_mask(x, h, w)
        else:
            shifted = x
            attn_mask = None

        x_windows = window_partition(shifted, self.window_size).view(-1, self.window_size * self.window_size, c)
        attn_windows = self.attn(x_windows, attn_mask)
        shifted = window_reverse(
            attn_windows.view(-1, self.window_size, self.window_size, c),
            self.window_size,
            h,
            w,
        )

        if shift_size > 0:
            x = torch.roll(shifted, shifts=(shift_size, shift_size), dims=(1, 2))
        else:
            x = shifted

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x.permute(0, 3, 1, 2).contiguous()


class ResidualSwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if index % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                )
                for index in range(depth)
            ]
        )
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(self.blocks(x))


class SwinIR(nn.Module):
    """SwinIR-style window transformer with a bicubic residual branch."""

    def __init__(
        self,
        scale: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        embed_dim: int = 96,
        depths: tuple[int, ...] = (6, 6, 6, 6),
        num_heads: tuple[int, ...] = (6, 6, 6, 6),
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        skip: str = "bicubic",
    ):
        super().__init__()
        if len(depths) != len(num_heads):
            raise ValueError("depths and num_heads must have the same length.")
        self.scale = scale
        self.window_size = window_size
        self.skip = skip
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.body = nn.Sequential(
            *[
                ResidualSwinTransformerBlock(
                    dim=embed_dim,
                    depth=depth,
                    num_heads=heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                )
                for depth, heads in zip(depths, num_heads)
            ]
        )
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(embed_dim, out_channels, 3, padding=1),
        )
        nn.init.zeros_(self.upsample[-1].weight)
        nn.init.zeros_(self.upsample[-1].bias)

    def _pad_to_window(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        _, _, h, w = x.shape
        pad_h = (math.ceil(h / self.window_size) * self.window_size) - h
        pad_w = (math.ceil(w / self.window_size) * self.window_size) - w
        if pad_h == 0 and pad_w == 0:
            return x, h, w
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), h, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.skip == "none":
            base = None
        else:
            base = F.interpolate(x, scale_factor=self.scale, mode=self.skip, align_corners=False)
        x, h, w = self._pad_to_window(x)
        feat = self.conv_first(x)
        feat = feat + self.conv_after_body(self.body(feat))
        out = self.upsample(feat)
        out = out[:, :, : h * self.scale, : w * self.scale]
        if base is not None:
            out = out + base
        return out


RCAN_PRESETS = {
    # Good first model for L4. FP16 checkpoint is around 30 MB.
    "base": dict(num_features=96, num_groups=8, num_blocks=10, reduction=16),
    # Faster and friendlier for phase-2 model-size scoring.
    "small": dict(num_features=64, num_groups=6, num_blocks=8, reduction=16),
    # Use for the final phase-1 image-only submission if training time allows.
    "large": dict(num_features=128, num_groups=10, num_blocks=12, reduction=16),
}


SWINIR_PRESETS = {
    "swinir_tiny": dict(type="swinir", embed_dim=60, depths=(4, 4, 4, 4), num_heads=(6, 6, 6, 6)),
    "swinir_light": dict(type="swinir", embed_dim=96, depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6)),
    "swinir_medium": dict(type="swinir", embed_dim=120, depths=(6, 6, 6, 6, 6, 6), num_heads=(6, 6, 6, 6, 6, 6)),
    "swinir_base": dict(type="swinir", embed_dim=180, depths=(6, 6, 6, 6, 6, 6), num_heads=(6, 6, 6, 6, 6, 6)),
}


MODEL_PRESETS = {**RCAN_PRESETS, **SWINIR_PRESETS, **THERMAL_PRESETS}


def build_model(
    scale: int,
    preset: str = "base",
    num_features: int | None = None,
    num_groups: int | None = None,
    num_blocks: int | None = None,
    **kwargs,
) -> nn.Module:
    cfg = dict(MODEL_PRESETS[preset])
    model_type = cfg.pop("type", "rcan")
    if model_type == "swinir":
        if num_features is not None:
            cfg["embed_dim"] = num_features
        cfg.update(kwargs)
        return SwinIR(scale=scale, **cfg)
    if model_type == "thermal_edge":
        if num_features is not None:
            cfg["channels"] = num_features
        if num_blocks is not None:
            cfg["num_blocks"] = num_blocks
        cfg.update(kwargs)
        return ThermalEdgeSR(scale=scale, **cfg)

    if num_features is not None:
        cfg["num_features"] = num_features
    if num_groups is not None:
        cfg["num_groups"] = num_groups
    if num_blocks is not None:
        cfg["num_blocks"] = num_blocks
    cfg.update(kwargs)
    return RCAN(scale=scale, **cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
