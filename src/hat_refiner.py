from __future__ import annotations

import torch
from torch import nn


class HatRefiner(nn.Module):
    def __init__(self, channels: int = 32, depth: int = 4, residual_scale: float = 0.05):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(max(0, depth - 1)):
            layers.extend([nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True)])
        layers.append(nn.Conv2d(channels, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)
        self.residual_scale = residual_scale
        last = self.net[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, pred: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([pred, base, pred - base], dim=1)
        residual = torch.tanh(self.net(inp)) * self.residual_scale
        return (pred + residual).clamp(0, 1)
