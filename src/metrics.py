from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from piq import ssim


class EdgeMetric(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("kernel_x", kernel_x.view(1, 1, 3, 3))
        self.register_buffer("kernel_y", kernel_y.view(1, 1, 3, 3))

    def edges(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.kernel_x, padding=1)
        gy = F.conv2d(x, self.kernel_y, padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_edge = self.edges(pred)
        target_edge = self.edges(target)
        pred_centered = pred_edge - pred_edge.mean(dim=(1, 2, 3), keepdim=True)
        target_centered = target_edge - target_edge.mean(dim=(1, 2, 3), keepdim=True)
        denom = pred_edge.std(dim=(1, 2, 3)) * target_edge.std(dim=(1, 2, 3)) + 1e-6
        return ((pred_centered * target_centered).mean(dim=(1, 2, 3)) / denom).mean()


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target).square()).item()
    if mse <= 0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def metric_proxy(metrics: dict[str, float]) -> float:
    # Stable local selector for this contest: PSNR first, then the three auxiliary metrics.
    return (
        metrics["psnr"]
        + 6.0 * metrics["ssim"]
        + 2.0 * metrics["edge"]
        - 2.0 * metrics.get("lpips", 0.0)
    )


@torch.no_grad()
def measure_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    edge_metric: EdgeMetric,
    lpips_fn=None,
) -> dict[str, float]:
    pred = pred.clamp(0, 1)
    out = {
        "psnr": psnr(pred, target),
        "ssim": float(ssim(pred, target, data_range=1.0).detach().cpu()),
        "edge": float(edge_metric(pred, target).detach().cpu()),
    }
    if lpips_fn is not None:
        out["lpips"] = float(
            lpips_fn(
                pred.repeat(1, 3, 1, 1) * 2 - 1,
                target.repeat(1, 3, 1, 1) * 2 - 1,
            )
            .mean()
            .detach()
            .cpu()
        )
    return out
