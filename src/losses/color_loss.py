import torch
import torch.nn.functional as F

def compute_color_loss(rgb_f: torch.Tensor, rgb_c: torch.Tensor, gt_rgb: torch.Tensor) -> torch.Tensor:
    """Standard photometric MSE loss with coarse auxiliary."""
    return (F.mse_loss(rgb_f, gt_rgb) + 0.1 * F.mse_loss(rgb_c, gt_rgb)).float()