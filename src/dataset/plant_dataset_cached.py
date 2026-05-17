from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import torch

from src.NeRF.arch import get_rays
from src.config.project import Config

class PlantCachedDataset:
    def __init__(
            self, 
            image_names: List[str], 
            poses: Dict, 
            resized_path: Path, 
            mask_path: Path, 
            label_path: Path,
            cfg: Config,
            device: Optional[torch.device] = None,
            cache_on_gpu: bool = True,
            plant_bias: float = 0.75,
            global_view_to_idx: Optional[Dict[str, int]] = None,
            ):
        """
        views: list of views to load. Defaults to all available views.
        """

        self.device = device if device is not None else cfg.device
        self.plant_bias = plant_bias
        self.store_device = self.device if cache_on_gpu and self.device.type == "cuda" else torch.device("cpu")

        self.image_names: List[str] = []
        str_poses_keys = [str(k) for k in poses.keys()] 

        for image_name in image_names:

            image_key = image_name.split('.')[0]
            image_path = resized_path / image_name
            image_mask_path = mask_path / image_name
            image_label_path = label_path / image_name.replace('.png', '.npy')

            if not image_path.exists():
                print(f'  Skipping missing view files: {image_name}')
                continue
            if not image_mask_path.exists():
                print(f'  Skipping missing files without masks: {image_name}')
                continue
            if not image_label_path.exists():
                print(f'  Skipping missing files without labels: {image_label_path}')
                continue
            if image_key not in str_poses_keys:
                print(f'  Skipping view without pose: {image_name}')
                continue

            self.image_names.append(image_name)
        
        if not self.image_names:
            raise RuntimeError("No valid training views found.")
        
        if global_view_to_idx is None:
            global_view_to_idx = {name.split(".")[0]: i for i, name in enumerate(self.image_names)}

        self.global_view_to_idx = global_view_to_idx

        rays_o = []
        rays_d = []
        ray_view_ids = []
        gt_rgb = []
        gt_mask = []
        gt_sem = []

        H = cfg.camera.resized_h
        W = cfg.camera.resized_w
        print(f'Loading dataset ({len(self.image_names)} views, cache={self.store_device})...')

        for image_name in tqdm(self.image_names):

            img_bgr = cv2.imread(str(resized_path / image_name))
            mask = cv2.imread(str(mask_path / image_name), cv2.IMREAD_GRAYSCALE)
            lmap = np.load(str(label_path / image_name.replace('.png', '.npy')))

            if img_bgr is None or mask is None or lmap is None:
                raise ValueError(f"Could not read image or mask or label map")


            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mask = (mask > 127).astype(np.float32)
            lmap = lmap.astype(np.int64)

            view_key = image_name.split('.')[0]
            pose = torch.tensor(poses[view_key], dtype=torch.float32)

            ro, rd = get_rays(H, W, cfg.camera.focal_length, pose)

            flat_count = H * W
            view_idx = self.global_view_to_idx[view_key]

            rays_o.append(ro.reshape(-1, 3))
            rays_d.append(rd.reshape(-1, 3))
            ray_view_ids.append(torch.full((flat_count,), view_idx, dtype=torch.long))
            gt_rgb.append(torch.tensor(img).reshape(-1, 3))
            gt_mask.append(torch.tensor(mask).reshape(-1, 1))
            gt_sem.append(torch.tensor(lmap).reshape(-1))

        # Flatten into massive 1D tensors
        self.rays_o = torch.cat(rays_o).to(self.store_device, non_blocking=True)
        self.rays_d = torch.cat(rays_d).to(self.store_device, non_blocking=True)
        self.ray_view_ids = torch.cat(ray_view_ids).to(self.store_device, non_blocking=True)
        self.gt_rgb = torch.cat(gt_rgb).to(self.store_device, non_blocking=True)
        self.gt_mask = torch.cat(gt_mask).to(self.store_device, non_blocking=True)
        self.gt_sem = torch.cat(gt_sem).to(self.store_device, non_blocking=True)

        plant_mask = self.gt_mask.squeeze(-1) > 0.5
        self.plant_idx = plant_mask.nonzero(as_tuple=True)[0]
        self.bg_idx = (~plant_mask).nonzero(as_tuple=True)[0]

        # Cache lengths for fast vectorized sampling
        self.n_total = int(self.rays_o.shape[0])
        self.n_plant = int(self.plant_idx.shape[0])
        self.n_bg = int(self.bg_idx.shape[0])

        print(f"  Total rays : {self.n_total:>12,}")
        print(f"  Plant rays : {self.n_plant:>12,} ({100*self.n_plant/max(1,self.n_total):.1f}%)")
        print(f"  BG rays    : {self.n_bg:>12,} ({100*self.n_bg/max(1,self.n_total):.1f}%)")
        print(f"  Sampling   : {int(100*self.plant_bias)}% plant / {int(100*(1-self.plant_bias))}% background")

    def _move_if_needed(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        return x if x.device == device else x.to(device, non_blocking=True)

    def get_batch(self, batch_size: int, device: Optional[torch.device] = None):
        device = device if device is not None else self.device
        index_device = self.plant_idx.device

        num_plant = int(batch_size * self.plant_bias)
        num_bg = batch_size - num_plant

        if self.n_bg == 0:
            num_plant, num_bg = batch_size, 0
        elif self.n_plant == 0:
            num_plant, num_bg = 0, batch_size

        parts = []
        if num_plant > 0:
            r = torch.randint(0, self.n_plant, (num_plant,), device=index_device)
            parts.append(self.plant_idx[r])
        if num_bg > 0:
            r = torch.randint(0, self.n_bg, (num_bg,), device=index_device)
            parts.append(self.bg_idx[r])

        batch_idx = torch.cat(parts, dim=0)
        # Shuffle to avoid plant-first/bg-second structure inside the batch.
        batch_idx = batch_idx[torch.randperm(batch_idx.shape[0], device=index_device)]

        return (
            self._move_if_needed(self.rays_o[batch_idx], device),
            self._move_if_needed(self.rays_d[batch_idx], device),
            self._move_if_needed(self.ray_view_ids[batch_idx], device),
            self._move_if_needed(self.gt_rgb[batch_idx], device),
            self._move_if_needed(self.gt_mask[batch_idx], device),
            self._move_if_needed(self.gt_sem[batch_idx], device),
        )