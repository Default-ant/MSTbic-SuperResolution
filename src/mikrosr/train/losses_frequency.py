"""
Frequency-Aware Loss Functions (losses_frequency.py)
=====================================================
Supplementary loss functions that target high-frequency texture restoration,
designed to complement the standard L1 pixel-wise loss used in MSTbic training.

Losses provided:
  1. SobelEdgeLoss   — penalises gradient-map differences (edge preservation).
  2. LaplacianLoss   — penalises Laplacian-of-Gaussian map differences.
  3. LaplacianPyramidLoss — multi-scale frequency decomposition loss.
  4. CombinedFreqLoss — weighted combination of all above + L1 pixel loss.

Usage in training loop:
    from losses_frequency import CombinedFreqLoss
    criterion = CombinedFreqLoss(w_pixel=1.0, w_sobel=0.1, w_lap=0.05)
    loss = criterion(output, target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sobel Edge Loss
# ---------------------------------------------------------------------------

class SobelEdgeLoss(nn.Module):
    """
    Computes L1 loss between the Sobel-edge maps of the prediction and target.

    The Sobel operator approximates image gradient magnitude:
        G = sqrt(Gx^2 + Gy^2)
    where Gx and Gy are the horizontal and vertical gradient responses.

    A loss on G directly penalises the model if it blurs away edges,
    which is the primary failure mode of LR-only pseudo-target training.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self.loss_fn = nn.L1Loss(reduction=reduction)

        # Sobel kernels (3x3) applied per channel
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        # shape: (1, 1, 3, 3) — applied depthwise
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def _sobel(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)  → returns (B, C, H, W) gradient magnitude
        Applied independently per channel.
        """
        B, C, H, W = x.shape
        x_flat = x.reshape(B * C, 1, H, W)
        gx = F.conv2d(x_flat, self.kx, padding=1)
        gy = F.conv2d(x_flat, self.ky, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        return mag.reshape(B, C, H, W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(self._sobel(pred), self._sobel(target))


# ---------------------------------------------------------------------------
# Laplacian Loss
# ---------------------------------------------------------------------------

class LaplacianLoss(nn.Module):
    """
    Computes L1 loss between the Laplacian (second-derivative) maps.
    Laplacian is more sensitive to fine texture details than Sobel.

    Kernel: [[0,-1,0],[-1,4,-1],[0,-1,0]]
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)
        lap = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
        self.register_buffer("lap", lap.view(1, 1, 3, 3))

    def _laplacian(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.reshape(B * C, 1, H, W)
        out = F.conv2d(x_flat, self.lap, padding=1)
        return out.reshape(B, C, H, W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(self._laplacian(pred), self._laplacian(target))


# ---------------------------------------------------------------------------
# Laplacian Pyramid Loss
# ---------------------------------------------------------------------------

class LaplacianPyramidLoss(nn.Module):
    """
    Multi-scale frequency loss via Laplacian pyramid decomposition.

    The Laplacian pyramid decomposes an image into a set of bandpass filtered
    images at multiple scales. Loss is computed at each level, giving the model
    explicit supervision at low, mid, and high spatial frequencies.

    Args:
        n_levels (int): Number of pyramid levels (default: 3).
        reduction (str): 'mean' or 'sum'.
    """

    def __init__(self, n_levels: int = 3, reduction: str = "mean"):
        super().__init__()
        self.n_levels = n_levels
        self.loss_fn = nn.L1Loss(reduction=reduction)

        # Gaussian blur kernel for pyramid downsampling
        gauss = torch.tensor(
            [[1, 4, 6, 4, 1],
             [4, 16, 24, 16, 4],
             [6, 24, 36, 24, 6],
             [4, 16, 24, 16, 4],
             [1, 4, 6, 4, 1]], dtype=torch.float32
        ) / 256.0
        self.register_buffer("gauss", gauss.view(1, 1, 5, 5))

    def _blur_downsample(self, x: torch.Tensor) -> torch.Tensor:
        """Blur + 2x downsample (Gaussian pyramid step)."""
        B, C, H, W = x.shape
        x_flat = x.reshape(B * C, 1, H, W)
        blurred = F.conv2d(x_flat, self.gauss, padding=2)
        down = blurred[:, :, ::2, ::2]
        return down.reshape(B, C, H // 2, W // 2)

    def _upsample(self, x: torch.Tensor, target_size) -> torch.Tensor:
        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

    def _build_pyramid(self, x: torch.Tensor) -> list:
        """Build Laplacian pyramid: list of residual (bandpass) images."""
        pyramid = []
        current = x
        for _ in range(self.n_levels):
            down = self._blur_downsample(current)
            up = self._upsample(down, current.shape[-2:])
            lap = current - up          # bandpass residual at this scale
            pyramid.append(lap)
            current = down
        pyramid.append(current)         # low-frequency base
        return pyramid

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_pyr = self._build_pyramid(pred)
        tgt_pyr = self._build_pyramid(target)

        total_loss = 0.0
        # Higher pyramid levels (finer detail) get higher weight
        for level, (p_lap, t_lap) in enumerate(zip(pred_pyr, tgt_pyr)):
            weight = 2.0 ** (self.n_levels - level)   # e.g. 8, 4, 2, 1
            total_loss += weight * self.loss_fn(p_lap, t_lap)

        # Normalise by total weight sum
        total_weight = sum(2.0 ** (self.n_levels - l) for l in range(self.n_levels + 1))
        return total_loss / total_weight


# ---------------------------------------------------------------------------
# Combined Frequency-Aware Loss (plug-in replacement for L1)
# ---------------------------------------------------------------------------

class CombinedFreqLoss(nn.Module):
    """
    Unified loss combining pixel-wise L1, Sobel edge loss, and Laplacian
    pyramid loss with configurable weights.

        L_total = w_pixel * L1(pred, target)
                + w_sobel * SobelLoss(pred, target)
                + w_lap   * LaplacianPyramidLoss(pred, target)

    Recommended starting weights: w_pixel=1.0, w_sobel=0.1, w_lap=0.05
    These can be tuned per experiment.
    """

    def __init__(
        self,
        w_pixel: float = 1.0,
        w_sobel: float = 0.1,
        w_lap: float = 0.05,
        n_pyramid_levels: int = 3,
    ):
        super().__init__()
        self.w_pixel = w_pixel
        self.w_sobel = w_sobel
        self.w_lap = w_lap

        self.pixel_loss = nn.L1Loss()
        self.sobel_loss = SobelEdgeLoss()
        self.lap_loss = LaplacianPyramidLoss(n_levels=n_pyramid_levels)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.w_pixel * self.pixel_loss(pred, target)

        if self.w_sobel > 0:
            loss = loss + self.w_sobel * self.sobel_loss(pred, target)
        if self.w_lap > 0:
            loss = loss + self.w_lap * self.lap_loss(pred, target)

        return loss

    def breakdown(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """Returns individual loss components for logging."""
        return {
            "loss_pixel": self.pixel_loss(pred, target).item(),
            "loss_sobel": self.sobel_loss(pred, target).item(),
            "loss_lap":   self.lap_loss(pred, target).item(),
        }
