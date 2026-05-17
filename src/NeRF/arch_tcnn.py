from __future__ import annotations
from typing import Tuple, Dict
from src.NeRF.hash_encoding_tcnn import ProgressiveTCNNHashEncoding, SmallDirEnc


import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticNeRFTCNN(nn.Module):
    """
    Faster same-behavior version of SemanticNeRF.

    Important: the MLP sizes, heads, density bias, direction encoding, semantic
    detach, plant_bound clipping, and progressive hash schedule match your
    current architecture. The main change is replacing the slow Python hash-grid
    with tiny-cuda-nn when available.
    """

    def __init__(
        self,
        plant_bound: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.hash_enc = ProgressiveTCNNHashEncoding(
            n_levels=16,
            n_features=2,
            log2_table=18,
            base_res=8,
            max_res=512,
            start_level=4,
            warmup_start=500,
            warmup_end=6000,
            interpolation="Linear",
        )
        self.dir_enc = SmallDirEnc()
        h = self.hash_enc.out_dim
        d = self.dir_enc.out_dim
        self.plant_bound = float(plant_bound)

        self.trunk = nn.Sequential(
            nn.Linear(h, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )
        self.density_head = nn.Linear(128, 1)
        self.color_head = nn.Sequential(
            nn.Linear(128 + d, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )
        self.semantic_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, num_classes),
        )
        nn.init.constant_(self.density_head.bias, -4.0)
        self.register_buffer("current_step", torch.tensor(0, dtype=torch.long))

    def update_step(self, step: int) -> None:
        self.current_step.fill_(step)
        self.hash_enc.update_step(step)

    def forward_density_features(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_normalized = pos / self.plant_bound
        inside = (pos_normalized.abs() <= 1.0).all(dim=-1, keepdim=True)
        x = self.trunk(self.hash_enc(pos_normalized))
        density = F.softplus(self.density_head(x)) * inside.float()
        return density, x

    def query_density(self, pos: torch.Tensor) -> torch.Tensor:
        """For nerfacc occupancy-grid updates. Returns [N]."""
        density, _ = self.forward_density_features(pos)
        return density.squeeze(-1)

    def forward(self, pos: torch.Tensor, dirs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dirs = F.normalize(dirs, dim=-1)
        density, x = self.forward_density_features(pos)
        color = self.color_head(torch.cat([x, self.dir_enc(dirs)], dim=-1))
        # Keep your original behavior: semantic loss should not reshape geometry.
        semantics = self.semantic_head(x.detach())
        return density, color, semantics


def get_rays(H: int, W: int, focal: float, pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    i = torch.arange(W, dtype=torch.float32, device=pose.device)
    j = torch.arange(H, dtype=torch.float32, device=pose.device)
    gj, gi = torch.meshgrid(j, i, indexing="ij")
    dirs = torch.stack([(gi - W / 2.0) / focal, -(gj - H / 2.0) / focal, -torch.ones_like(gi)], -1)
    rays_d = (dirs[..., None, :] * pose[:3, :3]).sum(-1)
    rays_o = pose[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def sample_coarse(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    n: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    z = near + (far - near) * torch.linspace(0.0, 1.0, n, device=rays_o.device)
    z = z.expand(rays_o.shape[0], n)
    mids = 0.5 * (z[..., 1:] + z[..., :-1])
    lower = torch.cat([z[..., :1], mids], dim=-1)
    upper = torch.cat([mids, z[..., -1:]], dim=-1)
    z = lower + (upper - lower) * torch.rand_like(z)
    pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z.unsqueeze(-1)
    return pts, z


def sample_fine(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    z_c: torch.Tensor,
    w: torch.Tensor,
    n_fine: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    z_mids = 0.5 * (z_c[..., 1:] + z_c[..., :-1])
    w_mid = w[..., 1:-1].detach() + 1e-5
    pdf = w_mid / w_mid.sum(dim=-1, keepdim=True)
    cdf = torch.cat([torch.zeros_like(pdf[..., :1]), torch.cumsum(pdf, dim=-1)], dim=-1)
    u = torch.rand((*cdf.shape[:-1], n_fine), device=cdf.device)
    inds = torch.searchsorted(cdf.contiguous(), u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    i2 = torch.stack([below, above], dim=-1)

    cdf_g = torch.gather(cdf, -1, i2.view(*i2.shape[:-2], -1)).view(*i2.shape)
    bin_g = torch.gather(z_mids, -1, i2.view(*i2.shape[:-2], -1)).view(*i2.shape)
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    z_f = bin_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bin_g[..., 1] - bin_g[..., 0])
    z_all, _ = torch.sort(torch.cat([z_c, z_f.detach()], dim=-1), dim=-1)
    pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_all.unsqueeze(-1)
    return pts, z_all


def volume_render(
    density: torch.Tensor,
    color: torch.Tensor,
    semantics: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    deltas = torch.cat(
        [z_vals[..., 1:] - z_vals[..., :-1], z_vals[..., -1:] - z_vals[..., -2:-1]],
        dim=-1,
    )
    deltas = deltas * torch.norm(rays_d.unsqueeze(-2), dim=-1)
    sigma = density[..., 0]
    alpha = 1.0 - torch.exp(-sigma * deltas)
    T = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha[..., :-1] + 1e-10], dim=-1),
        dim=-1,
    )
    weights = alpha * T
    rgb = (weights.unsqueeze(-1) * color).sum(dim=-2)
    acc = weights.sum(dim=-1, keepdim=True)
    sem_probs = torch.softmax(semantics, dim=-1)
    sem = (weights.unsqueeze(-1) * sem_probs).sum(dim=-2)
    depth = (weights * z_vals).sum(dim=-1, keepdim=True)
    return rgb, acc, sem, depth, weights


def render_rays_dense(model: nn.Module, rays_o: torch.Tensor, rays_d: torch.Tensor, cfg) -> Dict[str, torch.Tensor]:
    """Your current coarse+fine pipeline, just wrapped to avoid duplicated code."""
    pts_c, z_c = sample_coarse(
        rays_o, rays_d, cfg.training.near, cfg.training.far, cfg.training.n_samples_coarse
    )
    N, S, _ = pts_c.shape
    dirs_c = rays_d[:, None, :].expand(N, S, 3).reshape(-1, 3)
    dc, cc, sc = model(pts_c.reshape(-1, 3), dirs_c)
    rgb_c, acc_c, sem_c, _, wc = volume_render(
        dc.reshape(N, S, 1),
        cc.reshape(N, S, 3),
        sc.reshape(N, S, cfg.semantic.num_classes),
        z_c,
        rays_d,
    )

    pts_f, z_f = sample_fine(rays_o, rays_d, z_c, wc.detach(), cfg.training.n_samples_fine)
    N, S2, _ = pts_f.shape
    dirs_f = rays_d[:, None, :].expand(N, S2, 3).reshape(-1, 3)
    df, cf, sf = model(pts_f.reshape(-1, 3), dirs_f)
    rgb_f, acc_f, sem_f, _, wf = volume_render(
        df.reshape(N, S2, 1),
        cf.reshape(N, S2, 3),
        sf.reshape(N, S2, cfg.semantic.num_classes),
        z_f,
        rays_d,
    )

    return {
        "rgb_f": rgb_f,
        "rgb_c": rgb_c,
        "acc_f": acc_f,
        "acc_c": acc_c,
        "sem_f": sem_f,
        "sem_c": sem_c,
        "wf": wf,
        "z_f": z_f,
        "wc": wc,
        "z_c": z_c,
    }
