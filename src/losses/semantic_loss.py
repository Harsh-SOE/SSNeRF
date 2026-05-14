import torch
import torch.nn.functional as F

def compute_semantic_loss(sem_f: torch.Tensor, sem_c: torch.Tensor, gt_sem: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """Negative Log-Likelihood for 3D point classification, ignoring background."""
    gt_sem_full = gt_sem.squeeze().long().clone()
    bg_mask = (gt_mask.squeeze() < 0.5)
    gt_sem_full[bg_mask] = 0  # Force background points to class 0

    eps = 1e-8
    L_sem_f = F.nll_loss(torch.log(sem_f.clamp(min=eps)), gt_sem_full)
    L_sem_c = F.nll_loss(torch.log(sem_c.clamp(min=eps)), gt_sem_full)
    return (L_sem_f + L_sem_c).float()