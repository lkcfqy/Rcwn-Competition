from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).square().mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class ThermalNAFBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand
        padding = ((kernel_size - 1) // 2) * dilation

        self.norm1 = LayerNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, dw_channels * 2, 1)
        self.dw = nn.Conv2d(
            dw_channels * 2,
            dw_channels * 2,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=dw_channels * 2,
        )
        self.gate = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw_channels, dw_channels, 1), nn.Sigmoid())
        self.pw2 = nn.Conv2d(dw_channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.norm2 = LayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, ffn_channels * 2, 1)
        self.ffn2 = nn.Conv2d(ffn_channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.pw1(y)
        y = self.dw(y)
        y = self.gate(y)
        y = y * self.sca(y)
        x = x + self.pw2(y) * self.beta

        z = self.norm2(x)
        z = self.ffn1(z)
        z = self.gate(z)
        return x + self.ffn2(z) * self.gamma


class EdgePrior(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        laplace = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.register_buffer("laplace", laplace.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(gx.square() + gy.square() + 1e-6)
        lap = F.conv2d(x, self.laplace, padding=1).abs()
        return torch.cat([grad_mag, lap], dim=1)


class ThermalRoutingBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.base_block = ThermalNAFBlock(channels, kernel_size=5)
        self.detail_block = ThermalNAFBlock(channels, kernel_size=3)
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.base_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.detail_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(
        self,
        base: torch.Tensor,
        detail: torch.Tensor,
        edge: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.base_block(base)
        detail = self.detail_block(detail + edge)
        mix_in = torch.cat([base, detail, edge], dim=1)
        mix = self.mix(mix_in)
        gate = self.gate(mix_in)
        base = base + mix * self.base_scale
        detail = detail + mix * gate * self.detail_scale
        return base, detail


class ThermalEdgeSR(nn.Module):
    def __init__(
        self,
        scale: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: int = 64,
        num_blocks: int = 8,
        skip: str = "bicubic",
    ):
        super().__init__()
        self.scale = scale
        self.skip = skip
        self.edge_prior = EdgePrior()
        self.stem = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.edge_embed = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.detail_seed = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.blocks = nn.ModuleList([ThermalRoutingBlock(channels) for _ in range(num_blocks)])
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )
        nn.init.zeros_(self.upsample[-1].weight)
        nn.init.zeros_(self.upsample[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.stem(x)
        edge = self.edge_embed(self.edge_prior(x))
        detail = self.detail_seed(torch.cat([base, edge], dim=1))
        shortcut = base

        for block in self.blocks:
            base, detail = block(base, detail, edge)

        fused = self.fuse(torch.cat([base, detail, edge], dim=1)) + shortcut
        out = self.upsample(fused)
        if self.skip == "none":
            return out
        residual = F.interpolate(x, scale_factor=self.scale, mode=self.skip, align_corners=False)
        return out + residual


THERMAL_PRESETS = {
    "thermal_edge_tiny": dict(type="thermal_edge", channels=48, num_blocks=6),
    "thermal_edge_small": dict(type="thermal_edge", channels=64, num_blocks=8),
    "thermal_edge_base": dict(type="thermal_edge", channels=80, num_blocks=12),
}
