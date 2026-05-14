import torch
import torch.nn.functional as F

def compute_distortion_loss(wf: torch.Tensor, z_f: torch.Tensor, wc: torch.Tensor, z_c: torch.Tensor) -> torch.Tensor:
    """
    Directly computes the Mip-NeRF 360 distortion loss to enforce compact,
    sharp surface geometry along the rays.
    """
    def distortion(w: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Flatten trailing dimensions: w -> [N, S], z -> [N, S]
        w = w.squeeze(-1) if w.dim() > 2 else w
        z = z.squeeze(-1) if z.dim() > 2 else z

        # --- 1. Cross-Term ---
        # Penalizes pairs of weights that are large but physically far apart.
        # Math: sum_i sum_j (w_i * w_j * |z_i - z_j|)
        z_diff = torch.abs(z.unsqueeze(-1) - z.unsqueeze(-2))  # Shape: [N, S, S]
        w_prod = w.unsqueeze(-1) * w.unsqueeze(-2)             # Shape: [N, S, S]
        cross_loss = (w_prod * z_diff).sum(dim=(-2, -1))       # Shape: [N]

        # --- 2. Self-Term ---
        # Penalizes individual intervals that are too wide.
        # Math: 1/3 * sum_i (w_i^2 * delta_z_i)
        # Approximate delta_z (interval width) using adjacent samples
        dz = torch.zeros_like(z)
        dz[..., :-1] = z[..., 1:] - z[..., :-1]
        dz[..., -1] = dz[..., -2]  # Duplicate the last interval for consistency

        self_loss = ((w ** 2) * dz).sum(dim=-1) / 3.0          # Shape: [N]

        # Return the mean distortion across all rays in the batch
        return torch.mean(cross_loss + self_loss)

    # Convert to float32 to prevent half-precision overflow/underflow during squared math
    eps_e = 1e-6
    wf_f32, z_f_f32 = wf.float().clamp(eps_e, 1.0), z_f.float()
    wc_f32, z_c_f32 = wc.float().clamp(eps_e, 1.0), z_c.float()

    # Standard weighting: Fine network gets full loss, coarse gets half
    return distortion(wf_f32, z_f_f32) + 0.5 * distortion(wc_f32, z_c_f32)