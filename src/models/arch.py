import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.hash_encoding import ProgressiveHashEncoding, SmallDirEnc

class SemanticNeRF(nn.Module):
    def __init__(self, plant_bound: float, num_classes):
        super().__init__()

        self.hash_enc = ProgressiveHashEncoding(
            n_levels=16,
            n_features=2,
            log2_table=18,
            base_res=8,
            max_res=512,
            start_level=4,
            warmup_start=500,
            warmup_end=6000,
        )

        self.dir_enc  = SmallDirEnc()

        h = self.hash_enc.out_dim
        d = self.dir_enc.out_dim
        self.plant_bound = plant_bound

        self.trunk = nn.Sequential(
            nn.Linear(h, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
        )

        self.density_head = nn.Linear(128, 1)

        self.color_head = nn.Sequential(
            nn.Linear(128 + d, 64), nn.SiLU(),
            nn.Linear(64, 3), nn.Sigmoid()
        )

        self.semantic_head = nn.Sequential(
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, num_classes)
        )

        nn.init.constant_(self.density_head.bias, -4.0)

        self.register_buffer(
            "current_step",
            torch.tensor(0, dtype=torch.long)
        )

    def update_step(self, step: int):
        self.current_step.fill_(step)

        if hasattr(self.hash_enc, "update_step"):
            self.hash_enc.update_step(step)

    def forward(self, pos, dirs):
        pos_normalized = pos / self.plant_bound
        dirs = F.normalize(dirs, dim=-1)

        inside = (pos_normalized.abs() <= 1.0).all(dim=-1, keepdim=True)

        x = self.trunk(self.hash_enc(pos_normalized))

        density = F.softplus(self.density_head(x))
        density = density * inside.float()

        color = self.color_head(torch.cat([x, self.dir_enc(dirs)], -1))

        semantics = self.semantic_head(x.detach())

        return density, color, semantics

def volume_render(density, color, semantics, z_vals, rays_d):
    deltas = torch.cat([
        z_vals[..., 1:] - z_vals[..., :-1],
        (z_vals[..., -1:] - z_vals[..., -2:-1])
    ], dim=-1)

    deltas = deltas * torch.norm(rays_d.unsqueeze(-2), dim=-1)
    alpha = 1. - torch.exp(-density[..., 0] * deltas)
    T = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]),1. - alpha[..., :-1] + 1e-10],
                  dim=-1), dim=-1)

    weights = alpha * T   # [N, S]
    rgb = (weights.unsqueeze(-1) * color).sum(-2)
    acc = weights.sum(-1, keepdim=True)
    sem_probs = torch.softmax(semantics, dim=-1)
    sem = (weights.unsqueeze(-1) * sem_probs).sum(-2)
    depth = (weights * z_vals).sum(-1, keepdim=True)

    return rgb, acc, sem, depth, weights


def get_rays(H, W, focal, pose):
    i = torch.arange(W, dtype=torch.float32, device=pose.device)
    j = torch.arange(H, dtype=torch.float32, device=pose.device)
    gj, gi = torch.meshgrid(j, i, indexing='ij')
    dirs = torch.stack([(gi-W/2.)/focal, -(gj-H/2.)/focal, -torch.ones_like(gi)], -1)
    rays_d = (dirs[...,None,:] * pose[:3,:3]).sum(-1)
    return pose[:3,3].expand(rays_d.shape), rays_d


def sample_coarse(rays_o, rays_d, near, far, n):
    z = near + (far-near)*torch.linspace(0.,1.,n,device=rays_o.device)
    z = z.expand(rays_o.shape[0], n)
    mids=.5*(z[...,1:]+z[...,:-1])
    z = torch.cat([z[...,:1],mids],-1) + (torch.cat([mids,z[...,-1:]],-1)-torch.cat([z[...,:1],mids],-1))*torch.rand_like(z)
    return rays_o.unsqueeze(-2)+rays_d.unsqueeze(-2)*z.unsqueeze(-1), z


def sample_fine(rays_o, rays_d, z_c, w, n_fine):
    z_mids=.5*(z_c[...,1:]+z_c[...,:-1])
    w_mid =w[...,1:-1]+1e-5
    pdf   =w_mid/w_mid.sum(-1,keepdim=True)
    cdf   =torch.cat([torch.zeros_like(pdf[...,:1]),torch.cumsum(pdf,-1)],-1)
    u     =torch.rand(list(cdf.shape[:-1])+[n_fine],device=cdf.device)
    inds  =torch.searchsorted(cdf.contiguous(),u,right=True)
    below = torch.clamp(inds - 1, min=0); above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    i2    =torch.stack([below,above],-1)
    cdf_g =torch.gather(cdf,-1,i2.view(*i2.shape[:-2],-1)).view(*i2.shape)
    bin_g =torch.gather(z_mids,-1,i2.view(*i2.shape[:-2],-1)).view(*i2.shape)
    denom =cdf_g[...,1]-cdf_g[...,0]
    denom =torch.where(denom<1e-5,torch.ones_like(denom),denom)
    z_f   =bin_g[...,0]+(u-cdf_g[...,0])/denom*(bin_g[...,1]-bin_g[...,0])
    z_all,_=torch.sort(torch.cat([z_c,z_f.detach()],-1),-1)
    return rays_o.unsqueeze(-2)+rays_d.unsqueeze(-2)*z_all.unsqueeze(-1), z_all