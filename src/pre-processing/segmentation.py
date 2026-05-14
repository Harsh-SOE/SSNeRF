import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
from rembg.sessions import BaseSession 
from rembg import remove as rembg_remove
from segment_anything import SamPredictor

from src.config.sam import SAMConfig

def rembg_foreground(image: Image.Image, session: BaseSession) -> np.ndarray:
    """
    STEP 1: Run rembg to remove the background.
    Returns boolean mask (H, W) — True = foreground (plant + pot).
    """
    result = rembg_remove(
        image,
        session                            = session,
        alpha_matting                      = True,
        alpha_matting_foreground_threshold = 240,
        alpha_matting_background_threshold = 20,
        alpha_matting_erode_size           = 7,
    )
    alpha  = np.array(result)[:, :, 3]
    return alpha > 10

def get_green_mask(img_rgb: np.ndarray, fg_mask: np.ndarray, sam_cfg: SAMConfig) -> np.ndarray:
    """
    Step 2: HSV-based green pixel detection, restricted to rembg foreground.
    Returns boolean mask (H, W).
    """
    hsv   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower = np.array(sam_cfg.hsv_lower, dtype=np.uint8)
    upper = np.array(sam_cfg.hsv_upper, dtype=np.uint8)
    green = cv2.inRange(hsv, lower, upper).astype(bool)
    return green & fg_mask

def sample_points(mask: np.ndarray, n: int) -> np.ndarray:
    """
    Step 3 — Randomly sample n (x, y) coords from a boolean mask.
    Returns array of shape (k, 2) where k ≤ n.
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.empty((0, 2), dtype=int)
    idx = np.random.choice(len(ys), min(n, len(ys)), replace=False)
    return np.column_stack([xs[idx], ys[idx]])

def sample_plant_points(green_mask: np.ndarray,
                        fg_mask: np.ndarray,
                        n: int) -> np.ndarray:
    """
    STEP 3 (a): Sample n positive SAM points from green pixels in the PLANT REGION only.
    Plant zone = top 70% of the foreground bounding box.
    Falls back to all green pixels if the plant zone is too sparse.
    """
    rows = np.where(fg_mask.any(axis=1))[0]
    if len(rows) == 0:
        return sample_points(green_mask, n)

    top, bottom  = rows[0], rows[-1]
    zone_bottom  = top + int((bottom - top) * 0.70)

    plant_zone_mask = np.zeros_like(fg_mask)
    plant_zone_mask[top:zone_bottom, :] = True

    zone_green = green_mask & plant_zone_mask

    if zone_green.sum() < 10:
        return sample_points(green_mask, n)

    return sample_points(zone_green, n)

def sample_non_plant_points(green_mask: np.ndarray,
                             fg_mask: np.ndarray,
                             img_rgb: np.ndarray,
                             n: int
                            ) -> np.ndarray:
    """
    STEP 3 (b): Sample negative SAM points from TWO regions:
      A) Bottom 35% of fg bbox  → pot/soil area
      B) Canopy gap pixels      → dark grey trapped between leaves

    Canopy gaps = pixels inside the plant's bounding box that are:
      - NOT green (not plant-coloured)
      - Dark and unsaturated (the grey backdrop showing between leaves)
      - NOT in the fg_mask (rembg already said they're background)
    """
    H, W = fg_mask.shape
    n_pot    = n // 3
    n_gaps   = n - n_pot

    rows = np.where(fg_mask.any(axis=1))[0]
    if len(rows) == 0:
        return np.empty((0, 2), dtype=int)

    top, bottom = rows[0], rows[-1]
    cols = np.where(fg_mask.any(axis=0))[0]
    left, right = cols[0], cols[-1]

    non_plant_top  = top + int((bottom - top) * 0.65)
    pot_mask       = np.zeros((H, W), dtype=bool)
    pot_mask[non_plant_top:bottom + 1, :] = fg_mask[non_plant_top:bottom + 1, :]
    pot_mask       = pot_mask & ~green_mask

    pot_pts = sample_points(pot_mask, n_pot)

    canopy_bbox = np.zeros((H, W), dtype=bool)
    canopy_bbox[top:non_plant_top, left:right + 1] = True

    canopy_gaps = canopy_bbox & ~fg_mask

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    is_grey_dark = (S < 60) & (V < 190)
    canopy_gaps  = canopy_gaps & is_grey_dark

    gap_pts = sample_points(canopy_gaps, n_gaps)

    parts = [p for p in [pot_pts, gap_pts] if len(p) > 0]

    if len(parts) == 0:
        return sample_points(fg_mask & ~green_mask, n)

    return np.vstack(parts)

def get_plant_bbox(green_mask: np.ndarray,
                   fg_mask: np.ndarray,
                   padding_frac: float = 0.05) -> np.ndarray:
    """
    STEP 4 (a): Computes a bounding box around all green pixels (plant candidates),
    padded slightly outward to catch pale leaves just outside the green zone.

    Falls back to the full foreground bbox if green mask is too sparse.

    Returns: np.ndarray of shape (4,) → [x_min, y_min, x_max, y_max]
             in the format SAM's predict() expects for 'box' argument.
    """
    source = green_mask if green_mask.sum() > 100 else fg_mask

    rows = np.where(source.any(axis=1))[0]
    cols = np.where(source.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        # absolute fallback: full image
        H, W = fg_mask.shape
        return np.array([0, 0, W, H])

    y_min, y_max = rows[0],  rows[-1]
    x_min, x_max = cols[0],  cols[-1]

    # Pad outward by padding_frac of the bbox dimensions
    H, W  = fg_mask.shape
    pad_y = int((y_max - y_min) * padding_frac)
    pad_x = int((x_max - x_min) * padding_frac)

    y_min = max(0,     y_min - pad_y)
    y_max = min(H - 1, y_max + pad_y)
    x_min = max(0,     x_min - pad_x)
    x_max = min(W - 1, x_max + pad_x)

    return np.array([x_min, y_min, x_max, y_max])

def select_best_mask(masks: np.ndarray, scores: np.ndarray, green_mask: np.ndarray) -> np.ndarray:
    """
    Step 4 — Pick the SAM candidate that best covers plant pixels.

    Combined score = SAM_confidence × 0.4 + green_pixel_overlap × 0.6
    Green overlap  = (mask ∩ green_pixels) / mask_area
    This penalises masks that grab a lot of pot without covering the plant.
    """
    best_idx, best_score = 0, -1.0
    for i, (mask, sam_score) in enumerate(zip(masks, scores)):
        area    = mask.sum() + 1e-6
        overlap = (mask & green_mask).sum() / area
        score   = float(sam_score) * 0.4 + overlap * 0.6
        if score > best_score:
            best_score = score
            best_idx   = i
    return masks[best_idx]

def keep_large_components(mask: np.ndarray, min_frac: float = 0.03) -> np.ndarray:
    """
    Keep ALL connected components whose area >= min_frac of the largest.

    Replaces winner-takes-all extract_largest_component().
    Default 3% means: if the main trunk is 10,000px, any blob >=300px is kept.
    This preserves drooping/disconnected leaves that used to get silently dropped.

    Args:
        mask     : boolean or uint8 mask
        min_frac : minimum area as a fraction of the largest component (0.0–1.0)
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return mask.astype(bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    threshold = areas.max() * min_frac

    out = np.zeros(mask.shape, dtype=bool)
    for label_idx, area in enumerate(areas, start=1):
        if area >= threshold:
            out |= (labels == label_idx)

    return out

def erode_mask(mask: np.ndarray, pixels: int = 3) -> np.ndarray:
    """
    Shrink the mask by `pixels` px around the entire boundary.
    This cuts off the alpha-matting blended fringe zone at leaf edges.
    Loses a tiny sliver of leaf edge but gives a perfectly clean boundary.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels*2+1, pixels*2+1))
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

def save_debug_mosaic(image_path: Path,
                      img_rgb: np.ndarray,
                      fg_mask: np.ndarray,
                      green_mask: np.ndarray,
                      pos_pts: np.ndarray,
                      neg_pts: np.ndarray,
                      plant_bbox: np.ndarray,
                      final_mask: np.ndarray,
                      output_dir: Path
                      ) -> None:
    """
    STEP DEBUG: 2×3 mosaic showing every pipeline step.
    SAM pts panel now also draws the green bounding box.
    """
    thumb = lambda arr: cv2.resize(arr, (320, 240))
    H_orig, W_orig = img_rgb.shape[:2]

    orig      = thumb(img_rgb)
    fg_vis    = thumb((fg_mask[:, :, None] * img_rgb).astype(np.uint8))
    green_vis = thumb(
        np.stack([green_mask * 80, green_mask * 200, green_mask * 80], axis=-1).astype(np.uint8)
    )

    pts_img = img_rgb.copy()
    r = max(4, H_orig // 100)
    for x, y in pos_pts: cv2.circle(pts_img, (x, y), r, (0,   255, 0), -1)
    for x, y in neg_pts: cv2.circle(pts_img, (x, y), r, (255, 0,   0), -1)

    bx1, by1, bx2, by2 = plant_bbox.astype(int)
    cv2.rectangle(pts_img, (bx1, by1), (bx2, by2), (0, 255, 255), max(2, H_orig // 300))
    pts_vis = thumb(pts_img)

    mask_vis   = thumb(np.stack([final_mask * 255] * 3, axis=-1).astype(np.uint8))
    result     = img_rgb.copy(); result[~final_mask] = 0
    result_vis = thumb(result)

    mosaic = np.vstack([
        np.hstack([orig,    fg_vis,   green_vis]),
        np.hstack([pts_vis, mask_vis, result_vis]),
    ])

    labels = ["Original", "rembg fg", "Green mask",
              "SAM pts + bbox (cyan)", "Final mask", "Result"]
    for i, lbl in enumerate(labels):
        cv2.putText(mosaic, lbl,
                    ((i % 3) * 320 + 6, (i // 3) * 240 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    Image.fromarray(mosaic).save(
        Path(output_dir) / f"DEBUG_{Path(image_path).stem}.jpg"
    )

def segment_plant_from_raw_images(
        image_path: Path, 
        output_path: Path, 
        debug_path: Path, 
        session: BaseSession, 
        sam_cfg: SAMConfig, 
        predictor: SamPredictor
    ) -> Tuple[bool, Optional[np.ndarray]]:
    image = Image.open(image_path).convert("RGB")
    img_rgb = np.array(image)

    # STEP 1: extract foreground
    fg_mask = rembg_foreground(image, session)
    if fg_mask.sum() == 0:
        print(f"  [WARN] No foreground: {image_path.name}")
        return False, None

    # STEP 2: extract green mask in foreground
    green_mask = get_green_mask(img_rgb, fg_mask, sam_cfg=sam_cfg)

    # STEP 3: Sample SAM prompt points
    # (a): POSITIVE - plant region only (foreground + green mask + top 70%)
    pos_pts = sample_plant_points(green_mask, fg_mask, sam_cfg.positive_points)
    # (b): NEGATIVE: pot & canopy gap region (foreground + non-green mask)
    neg_pts = sample_non_plant_points(green_mask, fg_mask, img_rgb, sam_cfg.negative_points)

    if len(pos_pts) == 0:
        ys, xs  = np.where(fg_mask)
        pos_pts = np.array([[int(xs.mean()), int(ys.mean())]])
        print(f"  [WARN] Centroid fallback: {image_path.name}")

    if len(neg_pts) > 0:
        all_pts    = np.vstack([pos_pts, neg_pts])
        all_labels = np.array([1] * len(pos_pts) + [0] * len(neg_pts))
    else:
        all_pts    = pos_pts
        all_labels = np.ones(len(pos_pts), dtype=int)

    # STEP 4 (a): SAM bounding box from green mask
    plant_bbox = get_plant_bbox(green_mask, fg_mask, padding_frac=0.05)

    # STEP 4 (b): SAM prediction — bbox + prompt points
    predictor.set_image(img_rgb)
    masks, scores, _ = predictor.predict(
        point_coords     = all_pts,
        point_labels     = all_labels,
        box              = plant_bbox,
        multimask_output = True,
    )
    
    masks = np.array(masks) 
    scores = np.array(scores)

    # STEP 5: Pick best mask
    best_mask = select_best_mask(masks, scores, green_mask)

    # STEP 6: Remove any noise
    filtered_mask = keep_large_components(best_mask, min_frac=0.03)
    filtered_mask = erode_mask(filtered_mask, pixels=2)

    # STEP 7: Get output
    output = np.zeros_like(img_rgb)
    output[filtered_mask] = img_rgb[filtered_mask]

    # output = remove_fringe(output)

    Image.fromarray(output).save(Path(output_path) / image_path.name)

    if sam_cfg.save_debug:
        save_debug_mosaic(image_path, img_rgb, fg_mask, green_mask,
                             pos_pts, neg_pts, plant_bbox, filtered_mask, debug_path)

    return True, output