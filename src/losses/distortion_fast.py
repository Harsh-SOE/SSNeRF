"""O(N*S) exact replacement for your current O(N*S*S) distortion loss."""

from __future__ import annotations

import torch


def _distortion_sorted(w: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Exact Mip-NeRF-360 distortion loss for sorted samples.

    Your current code forms [N, S, S] pairwise matrices. For sorted z, the same
    cross term can be computed with cumulative sums:
        sum_i sum_j w_i w_j |z_i-z_j|
      = 2 * sum_i w_i * (z_i * sum_{j<i} w_j - sum_{j<i} w_j*z_j)
    """
    w = w.squeeze(-1).float() if w.dim() > 2 else w.float()
    z = z.squeeze(-1).float() if z.dim() > 2 else z.float()

    cw = torch.cumsum(w, dim=-1)
    cwz = torch.cumsum(w * z, dim=-1)
    cw_prev = cw - w
    cwz_prev = cwz - w * z
    cross_loss = 2.0 * (w * (z * cw_prev - cwz_prev)).sum(dim=-1)

    dz = torch.zeros_like(z)
    dz[..., :-1] = z[..., 1:] - z[..., :-1]
    dz[..., -1] = dz[..., -2]
    self_loss = ((w * w) * dz).sum(dim=-1) / 3.0

    return (cross_loss + self_loss).mean()


def compute_distortion_loss_fast(
    wf: torch.Tensor,
    z_f: torch.Tensor,
    wc: torch.Tensor,
    z_c: torch.Tensor,
) -> torch.Tensor:
    eps = 1e-6
    wf_f32 = wf.float().clamp(eps, 1.0)
    wc_f32 = wc.float().clamp(eps, 1.0)
    return _distortion_sorted(wf_f32, z_f.float()) + 0.5 * _distortion_sorted(wc_f32, z_c.float())
