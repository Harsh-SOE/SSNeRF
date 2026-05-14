import math
import torch
import numpy as np
from typing import List

from src.config.project import Config

def look_at(eye, target=np.zeros(3), up=np.array([0., 1., 0.])):
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, 0] = right
    mat[:3, 1] = true_up
    mat[:3, 2] = -fwd
    mat[:3, 3] = eye
    return mat


def axis_angle_to_matrix(vec: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle vectors to rotation matrices.
    vec: [N, 3] or [3]
    """
    single = (vec.ndim == 1)
    if single:
        vec = vec.unsqueeze(0)

    theta = torch.sqrt((vec * vec).sum(dim=-1, keepdim=True) + 1e-8)
    k = vec / theta

    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    zero = torch.zeros_like(kx)
    K = torch.stack([
        torch.stack([zero, -kz, ky], dim=-1),
        torch.stack([kz, zero, -kx], dim=-1),
        torch.stack([-ky, kx, zero], dim=-1),
    ], dim=-2)

    eye = torch.eye(3, device=vec.device, dtype=vec.dtype).unsqueeze(0).expand(vec.shape[0], -1, -1)
    sin_t = torch.sin(theta)[:, None]
    cos_t = torch.cos(theta)[:, None]
    R = eye + sin_t * K + (1.0 - cos_t) * (K @ K)

    small = (theta.squeeze(-1) < 1e-6)
    if small.any():
        R = torch.where(small[:, None, None], eye, R)

    return R[0] if single else R




def build_initial_poses(image_names: List[str], plant_target: np.ndarray, cfg: Config):
    poses = {}
    radius = cfg.camera.camera_radius

    for image_name in image_names:
        if image_name == 'top.png' and cfg.rotation.use_top_view:
            eye = plant_target + np.array([0.0, radius * 1.25, 0.15], dtype=np.float32)
            poses[image_name] = look_at(
                eye,
                target=plant_target,
                up=np.array([0., 0., 1.], dtype=np.float32)
            )
        else:
            rotation_angle = image_name.split('.')[0]
            ar = math.radians(int(rotation_angle))
            eye = plant_target + np.array([
                radius * math.sin(ar),
                0.0,
                radius * math.cos(ar)
            ], dtype=np.float32)

            poses[rotation_angle] = look_at(eye, target=plant_target)

    return poses