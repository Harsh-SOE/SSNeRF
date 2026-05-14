import cv2
import numpy as np
from pathlib import Path

def extract_silhouette(image_path: Path, threshold: int=15):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    mask = (np.max(img, axis=2) > threshold).astype(np.uint8) * 255
    return mask
