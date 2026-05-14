import json
import numpy as np
from typing import List

from src.config.project import Config

def save_transforms_json(views: List[str], cfg: Config, poses_dict, out_path):
    frames = []
    for view_key in views:
        if view_key not in poses_dict:
            continue
        frames.append({
            'file_path': f'images_resized/{view_key}.png',
            'transform_matrix': poses_dict[view_key].tolist()
        })

    transform_data = {
                          'camera_model': 'OPENCV',
                          **cfg.camera.nerf_intrinsics,
                          'k1': 0.0, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0,
                          'frames': frames
                      }

    with open(out_path, 'w') as f:
        json.dump(transform_data, f, indent=2)