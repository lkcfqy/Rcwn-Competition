from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from piq import ssim


class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("kernel_x", kernel_x.view(1, 1, 3, 3))
        self.register_buffer("kernel_y", kernel_y.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.kernel_x, padding=1)
        gy = F.conv2d(x, self.kernel_y, padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-6)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target).square() + self.eps).mean()


class EdgeNCCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel = Sobel()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_edge = self.sobel(pred)
        target_edge = self.sobel(target)
        pred_centered = pred_edge - pred_edge.mean(dim=(1, 2, 3), keepdim=True)
        target_centered = target_edge - target_edge.mean(dim=(1, 2, 3), keepdim=True)
        denom = pred_edge.std(dim=(1, 2, 3)) * target_edge.std(dim=(1, 2, 3)) + 1e-6
        ncc = (pred_centered * target_centered).mean(dim=(1, 2, 3)) / denom
        return 1.0 - ncc.mean()


class CompositeLoss(nn.Module):
    def __init__(
        self,
        pixel_weight: float = 1.0,
        ssim_weight: float = 0.05,
        edge_weight: float = 0.03,
        lpips_weight: float = 0.0,
        lpips_net: str = "alex",
    ):
        super().__init__()
        self.pixel = CharbonnierLoss()
        self.edge = EdgeNCCLoss()
        self.pixel_weight = pixel_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.lpips_weight = lpips_weight
        self.lpips_fn = None
        if lpips_weight > 0:
            import lpips

            self.lpips_fn = lpips.LPIPS(net=lpips_net)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        pix = self.pixel(pred, target)
        pred_bounded = pred.clamp(0, 1)
        ssim_loss = 1.0 - ssim(pred_bounded, target, data_range=1.0)
        edge_loss = self.edge(pred, target)
        total = self.pixel_weight * pix + self.ssim_weight * ssim_loss + self.edge_weight * edge_loss
        logs = {
            "pix": float(pix.detach().cpu()),
            "ssim": float(ssim_loss.detach().cpu()),
            "edge": float(edge_loss.detach().cpu()),
        }
        if self.lpips_fn is not None:
            lpips_loss = self.lpips_fn(
                pred_bounded.repeat(1, 3, 1, 1) * 2 - 1,
                target.repeat(1, 3, 1, 1) * 2 - 1,
            ).mean()
            total = total + self.lpips_weight * lpips_loss
            logs["lpips"] = float(lpips_loss.detach().cpu())
        return total, logs
