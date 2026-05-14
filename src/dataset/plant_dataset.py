from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import torch

from src.models.arch import get_rays
from src.config.project import Config

class PlantDataset:
    def __init__(
            self, views: List[str], 
            poses: Dict, 
            resized_path: Path, 
            mask_dir: Path, 
            label_path: Path,
            cfg: Config
            ):
        """
        views: list of views to load. Defaults to all available views.
        """

        self.views = []
        str_poses_keys = [str(k) for k in poses.keys()] 

        for view in views:
            img_path = resized_path / view
            mask_path = mask_dir / view
            label_path = label_path / view.replace('.png', '.npy')

            if not img_path.exists() or not mask_path.exists() or not label_path.exists():
                print(f'  Skipping missing view files: {view}')
                continue
            if str(view) not in str_poses_keys:
                print(f'  Skipping view without pose: {view}')
                continue

            self.views.append(view)

        self.view_to_idx = {k: i for i, k in enumerate(self.views)}

        self.rays_o = []
        self.rays_d = []
        self.ray_view_ids = []
        self.gt_rgb = []
        self.gt_mask = []
        self.gt_sem = []

        H = cfg.camera.resized_h
        W = cfg.camera.resized_w
        print(f'Loading dataset ({len(self.views)} views)...')

        for view in tqdm(self.views):
            fname = view

            img_bgr = cv2.imread(resized_path / fname)
            mask = cv2.imread(label_path / fname, cv2.IMREAD_GRAYSCALE)
            lmap = np.load(label_path / fname.replace('.png', '.npy'))

            if img_bgr is None or mask is None or lmap is None:
                raise ValueError(f"Could not read image or mask or label map")


            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
            mask = (mask > 127).astype(np.float32)
            lmap = lmap.astype(np.int64)

            # Ensure we are using the correct key type for the global poses dict
            pose_key = view if view in poses else str(view)
            pose = torch.tensor(poses[pose_key], dtype=torch.float32)

            ro, rd = get_rays(H, W, cfg.camera.focal_length, pose)

            flat_count = ro.reshape(-1, 3).shape[0]
            view_idx = self.view_to_idx[view]

            self.rays_o.append(ro.reshape(-1, 3))
            self.rays_d.append(rd.reshape(-1, 3))
            self.ray_view_ids.append(torch.full((flat_count,), view_idx, dtype=torch.long))
            self.gt_rgb.append(torch.tensor(img).reshape(-1, 3))
            self.gt_mask.append(torch.tensor(mask).reshape(-1, 1))
            self.gt_sem.append(torch.tensor(lmap).reshape(-1))

        # Flatten into massive 1D tensors
        self.rays_o = torch.cat(self.rays_o)
        self.rays_d = torch.cat(self.rays_d)
        self.ray_view_ids = torch.cat(self.ray_view_ids)
        self.gt_rgb = torch.cat(self.gt_rgb)
        self.gt_mask = torch.cat(self.gt_mask)
        self.gt_sem = torch.cat(self.gt_sem)

        plant_mask = self.gt_mask.squeeze() > 0.5
        self.plant_idx = plant_mask.nonzero(as_tuple=True)[0]
        self.bg_idx = (~plant_mask).nonzero(as_tuple=True)[0]

        # Cache lengths for fast vectorized sampling
        self.n_total = len(self.rays_o)
        self.n_plant = len(self.plant_idx)
        self.n_bg = len(self.bg_idx)

        print(f'  Total rays : {self.n_total:>12,}')
        print(f'  Plant rays : {self.n_plant:>12,}  ({100*self.n_plant/self.n_total:.1f}%)')
        print(f'  BG rays    : {self.n_bg:>12,}  ({100*self.n_bg/self.n_total:.1f}%)')
        print(f'  Sampling   : 75% plant / 25% background per batch')

        self.plant_bias = 0.75

    def get_batch(self, batch_size: int, device):
        """
        Vectorized sampling: grabs the entire batch simultaneously in C++.
        Transfers immediately to the target device (GPU).
        """
        num_plant = int(batch_size * self.plant_bias)
        num_bg = batch_size - num_plant

        # Safety catch: just in case a bad mask leaves 0 background or 0 plant rays
        if self.n_bg == 0:
            num_plant = batch_size
            num_bg = 0
        elif self.n_plant == 0:
            num_bg = batch_size
            num_plant = 0

        indices = []
        if num_plant > 0:
            rand_plant_indices = torch.randint(0, self.n_plant, (num_plant,))
            indices.append(self.plant_idx[rand_plant_indices])
        if num_bg > 0:
            rand_bg_indices = torch.randint(0, self.n_bg, (num_bg,))
            indices.append(self.bg_idx[rand_bg_indices])

        # Combine plant and bg indices
        batch_idx = torch.cat(indices)

        # Slice all data simultaneously and push to GPU
        return (
            self.rays_o[batch_idx].to(device, non_blocking=True),
            self.rays_d[batch_idx].to(device, non_blocking=True),
            self.ray_view_ids[batch_idx].to(device, non_blocking=True),
            self.gt_rgb[batch_idx].to(device, non_blocking=True),
            self.gt_mask[batch_idx].to(device, non_blocking=True),
            self.gt_sem[batch_idx].to(device, non_blocking=True),
        )