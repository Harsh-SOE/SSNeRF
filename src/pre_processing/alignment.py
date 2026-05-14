import cv2
import numpy as np
from pathlib import Path
from typing import Optional

def detect_pot_axis_x(raw_path: Path, debug_dir: Optional[Path]):
    """
    Detects the projected x-coordinate of the pot / turntable axis.

    This uses the white/cream pot body/rim, not the plant centroid.
    Returns:
        axis_x: float
        debug_info: dict
    """
    img = cv2.imread(raw_path)
    if img is None:
        raise ValueError(f"Could not read image: {raw_path}")

    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Remove green plant pixels.
    green = (
        (hue >= 25) &
        (hue <= 95) &
        (sat > 35) &
        (val > 50)
    )

    # White / cream pot mask.
    pot_candidate = (
        (val > 125) &
        (sat < 115) &
        (~green)
    ).astype(np.uint8) * 255

    # Pot is in lower half of the image.
    # This avoids leaves and upper background.
    roi = np.zeros_like(pot_candidate)
    y0 = int(0.58 * H)
    y1 = int(0.92 * H)
    roi[y0:y1, :] = pot_candidate[y0:y1, :]

    # Clean mask.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=2)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)

    # Connected components.
    n, labels, stats, cents = cv2.connectedComponentsWithStats(image=roi, connectivity=8)

    candidates = []
    for lab in range(1, n):
        x, y, w, h, area = stats[lab]
        cx, cy = cents[lab]

        # Pot should be large and reasonably wide.
        if area < 0.003 * H * W:
            continue
        if w < 0.25 * W:
            continue
        if h < 0.08 * H:
            continue

        # Prefer large central components.
        score = area - abs(cx - W / 2) * 500
        candidates.append((score, lab))

    if not candidates:
        print(f"[WARN] Could not detect pot axis in {raw_path.name}. Using image center.")
        return W / 2.0, {"ok": False, "image": img, "component": None}

    best_lab = max(candidates, key=lambda t: t[0])[1]
    component = (labels == best_lab).astype(np.uint8)

    # Estimate symmetry axis row-by-row.
    centers = []
    widths = []

    for y in range(y0, y1):
        xs = np.where(component[y] > 0)[0]
        if len(xs) < 50:
            continue

        left = int(xs.min())
        right = int(xs.max())
        width = right - left + 1

        centers.append((left + right) / 2.0)
        widths.append(width)

    centers = np.asarray(centers, dtype=np.float32)
    widths = np.asarray(widths, dtype=np.float32)

    if len(centers) == 0:
        print(f"[WARN] Empty pot rows in {raw_path.name}. Using component centroid.")
        return float(cents[best_lab][0]), {"ok": False, "image": img, "component": component}

    # Use only reliable wide rows.
    max_width = np.percentile(widths, 90)
    good = (widths > 0.55 * max_width) & (widths < 1.10 * max_width)

    if good.sum() < 10:
        good = widths > np.median(widths)

    axis_x = float(np.median(centers[good]))

    # Save debug overlay.
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

        vis = img.copy()

        # green: detected pot component
        overlay = vis.copy()
        overlay[component.astype(bool)] = (0, 255, 0)
        vis = cv2.addWeighted(vis, 0.75, overlay, 0.25, 0)

        # blue: image center
        cv2.line(vis, (W // 2, 0), (W // 2, H), (255, 0, 0), 3)

        # red: detected axis
        cv2.line(vis, (int(round(axis_x)), 0), (int(round(axis_x)), H), (0, 0, 255), 4)

        cv2.putText(
            vis,
            f"axis_x={axis_x:.1f}, image_center={W/2:.1f}, dx={axis_x-W/2:+.1f}px",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
            cv2.LINE_AA
        )

        cv2.imwrite(debug_dir / f"axis_{raw_path.stem}.jpg", vis)

    return axis_x, {"ok": True, "image": img, "component": component}

def align_image_horizontally_to_axis(in_path: Path, out_path: Path, axis_x):
    img = cv2.imread(in_path)
    if img is None:
        raise ValueError(f"Could not read image: {in_path}")

    H, W = img.shape[:2]

    target_x = W / 2.0
    dx = target_x - axis_x

    M = np.array([
        [1, 0, dx],
        [0, 1, 0],
    ], dtype=np.float32)

    aligned = cv2.warpAffine(
        img,
        M,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    cv2.imwrite(str(out_path), aligned)
    return dx