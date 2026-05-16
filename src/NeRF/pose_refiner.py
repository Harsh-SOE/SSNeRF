import torch
import torch.nn as nn
from typing import List

from src.NeRF.poses import axis_angle_to_matrix

class PoseRefiner(nn.Module):
    def __init__(self, base_poses: dict, views: List[str]):
        super().__init__()
        self.views   = views
        self.key_to_idx  = {k: i for i, k in enumerate(self.views)}
        self.top_view_exists = 'top' in self.key_to_idx

        base = torch.stack([
            torch.tensor(base_poses[k], dtype=torch.float32)
            for k in self.views
        ], dim=0)

        self.register_buffer('base_poses', base)
        self.register_buffer('base_R', base[:, :3, :3].clone())
        self.register_buffer('base_t', base[:, :3, 3].clone())

        self.delta_rot   = nn.Parameter(torch.randn(len(self.views), 3) * 1e-6)
        self.delta_trans = nn.Parameter(torch.randn(len(self.views), 3) * 1e-6)

        # Softer regularisation weight for top view (uncertain pose → less penalty)
        pose_weights = torch.ones(len(self.views), dtype=torch.float32)
        if self.top_view_exists:
            pose_weights[self.key_to_idx['top']] = 0.25
        self.register_buffer('pose_weights', pose_weights)

    def top_view_regularization(self) -> torch.Tensor:
        """
        L2 regularisation for the top-view pose delta only.
        Used during Phase 2 (top_view_start_iter → pose_start_iter)
        before full pose refinement is active for side views.
        """
        if not self.top_view_exists:
            return torch.tensor(0.0, device=self.delta_rot.device)
        top_idx    = self.key_to_idx['top']
        rot_loss   = (self.delta_rot[top_idx]  ** 2).sum()
        trans_loss = (self.delta_trans[top_idx] ** 2).sum()
        return self.pose_weights[top_idx] * (rot_loss + trans_loss)

    def refined_poses(self):
        delta_R = axis_angle_to_matrix(self.delta_rot)
        R = torch.bmm(self.base_R, delta_R)
        t = self.base_t + torch.einsum('bij,bj->bi', self.base_R, self.delta_trans)
        poses = torch.eye(4, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(R.shape[0], 1, 1)
        poses[:, :3, :3] = R
        poses[:, :3, 3]  = t
        return poses

    def pose_for_key(self, view_key):
        return self.refined_poses()[self.key_to_idx[view_key]]

    def export_pose_dict(self):
        poses_np = self.refined_poses().detach().cpu().numpy()
        return {k: poses_np[i] for k, i in self.key_to_idx.items()}

    def transform_rays(self, rays_o_base, rays_d_base, view_ids):
        base_R  = self.base_R[view_ids]
        base_t  = self.base_t[view_ids]
        delta_R = axis_angle_to_matrix(self.delta_rot[view_ids])
        delta_t = self.delta_trans[view_ids]
        rays_o  = base_t + torch.einsum('bij,bj->bi', base_R, delta_t)
        d_cam   = torch.einsum('bij,bj->bi', base_R.transpose(1, 2), rays_d_base)
        rays_d  = torch.einsum('bij,bjk,bk->bi', base_R, delta_R, d_cam)
        return rays_o, rays_d

    def regularization(self):
        rot_loss   = (self.delta_rot   ** 2).sum(-1)
        trans_loss = (self.delta_trans ** 2).sum(-1)
        return (self.pose_weights * (rot_loss + trans_loss)).mean()