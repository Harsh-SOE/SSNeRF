import torch
import torch.nn.functional as F

def compute_bg_suppression_loss(acc_f: torch.Tensor, acc_c: torch.Tensor, bg_mask: torch.Tensor) -> torch.Tensor:
    """Penalizes opacity in regions where the ground truth mask says empty space."""
    if bg_mask.any():
        return acc_f[bg_mask].mean() + 0.5 * acc_c[bg_mask].mean()
    return torch.tensor(0.0, device=acc_f.device)