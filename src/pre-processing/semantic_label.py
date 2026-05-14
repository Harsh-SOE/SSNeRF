import cv2
import time
import numpy as np
import networkx as nx
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from src.config.project import Config
import matplotlib.patches as mpatches
from segment_anything import SamPredictor
from skimage.morphology import skeletonize
from typing import Optional, List, Any, cast
from skimage.measure   import label as cc_label

# STEP 1: MORPHOLOGY-BASED PSEUDO-LABEL GENERATION
def get_plant_info(mask: np.ndarray):
    rows, cols = np.where(mask > 127)
    H, W = mask.shape
    if len(rows) == 0:
        return dict(found=False, cx=W//2, cy=H//2,
                    rmin=0, rmax=H, cmin=0, cmax=W,
                    pH=H, pW=W, area=0)
    return dict(
        found=True,
        cx=int(cols.mean()), cy=int(rows.mean()),
        rmin=int(rows.min()), rmax=int(rows.max()),
        cmin=int(cols.min()), cmax=int(cols.max()),
        pH=int(rows.max()-rows.min())+1,
        pW=int(cols.max()-cols.min())+1,
        area=len(rows),
    )

def build_skeleton_graph(skel_binary: np.ndarray):
    pts_list = np.argwhere(skel_binary).tolist()
    pt_set = set(map(tuple, pts_list))
    
    G = nx.Graph()
    G.add_nodes_from(map(tuple, pts_list))

    for pt in pts_list:
        r: int = int(pt[0])
        c: int = int(pt[1])
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in pt_set:
                    G.add_edge((r, c), nb)
                    
    degree = dict(G.degree())
    return G, degree

def find_stem_path(skel_binary, plant, mask_dist):
    H, W = skel_binary.shape
    G, degree = build_skeleton_graph(skel_binary)

    if len(G.nodes) < 4:
        return []

    # Central band to prevent the stem from wandering sideways
    c_half = max(10, int(plant['pW'] * 0.15))
    c_lo   = max(0, plant['cx'] - c_half)
    c_hi   = min(W, plant['cx'] + c_half)

    central_pts = [n for n in G.nodes if c_lo <= n[1] <= c_hi]
    if not central_pts:
        return []

    # Source: Bottom-most central point
    bottom_thresh = plant['rmax'] - int(plant['pH'] * 0.20)
    bottom_pts = [n for n in central_pts if n[0] >= bottom_thresh]
    if not bottom_pts:
        bottom_pts = [max(central_pts, key=lambda p: p[0])]
    src = min(bottom_pts, key=lambda p: abs(p[1] - plant['cx']))

    # Target (Shoot Apex): The thickest branching node in the lower/middle section.
    # We ignore the top 30% of the plant so it doesn't accidentally target a thick upper leaf.
    mid_thresh = plant['rmin'] + int(plant['pH'] * 0.30)
    valid_targets = [n for n in central_pts if n[0] >= mid_thresh]
    if not valid_targets:
        valid_targets = central_pts

    # The true apex/hub is almost always the thickest part of the core plant
    dst = max(valid_targets, key=lambda p: mask_dist[p[0], p[1]])

    try:
        # Heavily penalize horizontal movement to keep the main stem straight
        for u, v in G.edges:
            penalty = abs(u[1] - plant['cx']) + abs(v[1] - plant['cx'])
            G[u][v]['weight'] = 1.0 + penalty * 0.1
        path = nx.shortest_path(G, src, dst, weight='weight')
        return path
    except nx.NetworkXNoPath:
        return []

def find_petiole_paths(skel_binary, stem_path_set, plant, mask_dist):
    H, W = skel_binary.shape

    branch_skel = skel_binary.copy()
    for r, c in stem_path_set:
        if 0 <= r < H and 0 <= c < W:
            branch_skel[r, c] = 0

    label_result = cc_label(
    branch_skel.astype(bool),
    connectivity=2,
    return_num=False
)

    labeled = cast(NDArray[Any], label_result)
    n_comp = int(np.max(labeled))

    petiole_masks = []
    leaf_tip_pts = []

    for lbl in range(1, n_comp + 1):
        comp_mask = labeled == lbl
        pts = np.argwhere(comp_mask)

        if len(pts) < 5:
            continue

        mean_thickness = np.mean([mask_dist[int(r), int(c)] for r, c in pts])

        is_petiole = (
            mean_thickness < mask_dist.max() * 0.40
            or len(pts) < plant['area'] * 0.05
        )

        if is_petiole:
            petiole_masks.append(comp_mask)

            tip = max(
                pts,
                key=lambda p: (p[0] - plant['cy']) ** 2 + (p[1] - plant['cx']) ** 2
            )
            leaf_tip_pts.append((int(tip[0]), int(tip[1])))

    return petiole_masks, leaf_tip_pts

def generate_topview_label(img_path: Path, mask_path: Path):
    """
    Morphology-based pseudo-label for a TOP-DOWN plant image.

    Strategy
    ────────
    1. APEX   (class 4): small circle at the plant centroid — the youngest
                         meristem sits at the rosette centre.
    2. STEM   (class 2): disc of radius ≈ 15% of plant half-width, centred
                         at the centroid — the basal stem hub.
    3. PETIOLE(class 3): skeletonise the remaining (non-stem) plant region;
                         keep only skeleton pixels whose local distance-
                         transform value is < 35% of the max (i.e. thin);
                         dilate slightly.
    4. LEAF   (class 1): all remaining foreground pixels.
    5. BACKGROUND (0)  : black pixels.
    """
    img_full = cv2.imread(img_path)
    raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img_full is None or raw_mask is None:
        raise ValueError(f"Could not read image or mask at {img_path.name}")

    mask = (raw_mask > 10).astype(np.uint8) * 255
    H, W = img_full.shape[:2]

    label_map = np.zeros((H, W), dtype=np.uint8)
    plant = get_plant_info(mask)

    if not plant['found'] or plant['area'] < 50:
        return label_map

    mask_dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_dist = float(mask_dist.max()) + 1e-6

    cx, cy = plant['cx'], plant['cy']
    plant_radius = max(plant['pH'], plant['pW']) / 2.0

    # ── STEM: central disc (≈15 % of plant radius) ──────────────────────────
    stem_r = max(8, int(plant_radius * 0.15))
    stem_canvas = np.zeros((H, W), np.uint8)
    cv2.circle(stem_canvas, (cx, cy), stem_r, 255, -1)
    stem_region = (stem_canvas > 0) & (mask > 0)
    label_map[stem_region] = 2

    # ── APEX: very centre (≈5 % of plant radius, overwrites stem centre) ────
    apex_r = max(5, int(plant_radius * 0.05))
    apex_canvas = np.zeros((H, W), np.uint8)
    cv2.circle(apex_canvas, (cx, cy), apex_r, 255, -1)
    label_map[(apex_canvas > 0) & (mask > 0)] = 4

    # ── PETIOLE: thin skeleton branches in the non-stem plant area ───────────
    non_stem_plant = (mask > 0) & ~stem_region
    skel = skeletonize(non_stem_plant.astype(np.uint8)).astype(np.uint8)

    # Retain only thin skeleton pixels (petioles, not thick leaf midribs)
    petiole_skel = np.zeros((H, W), np.uint8)
    ys, xs = np.where(skel > 0)
    for y, x in zip(ys, xs):
        if mask_dist[y, x] < max_dist * 0.35:
            petiole_skel[y, x] = 255

    pet_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    petiole_dil = cv2.dilate(petiole_skel, pet_kernel) > 0
    label_map[petiole_dil & (mask > 0) & (label_map == 0)] = 3

    # ── LEAF: everything else that is plant ──────────────────────────────────
    label_map[(mask > 0) & (label_map == 0)] = 1

    return label_map

def generate_morphology_label(img_path: Path, mask_path: Path):
    img_full  = cv2.imread(str(img_path))
    raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img_full is None or raw_mask is None:
        raise ValueError(f"Could not read image or mask at {img_path.name}")

    mask = (raw_mask > 10).astype(np.uint8) * 255
    H, W      = img_full.shape[:2]

    plant     = get_plant_info(mask)
    label_map = np.zeros((H, W), dtype=np.uint8)

    if not plant['found'] or plant['area'] < 50:
        return label_map, []

    mask_dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    skel_binary = skeletonize((mask > 0).astype(np.uint8))

    stem_path = find_stem_path(skel_binary, plant, mask_dist)
    stem_path_set = set(map(tuple, stem_path))

    stem_skel_img = np.zeros((H, W), np.uint8)
    for r, c in stem_path:
        if 0 <= r < H and 0 <= c < W:
            stem_skel_img[r, c] = 255

    # stem_thick_px = max(5, int(mask_dist.max() * 0.35))
    stem_thick_px = max(4, min(12, int(mask_dist.max() * 0.15)))
    k_stem = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (stem_thick_px, stem_thick_px))
    stem_region = cv2.dilate(stem_skel_img, k_stem) > 0
    stem_region = stem_region & (mask > 0)
    label_map[stem_region] = 2

    stem_px = np.argwhere(stem_region)
    if len(stem_px) > 0:
        top_idx  = np.argmin(stem_px[:, 0])
        apex_y, apex_x = int(stem_px[top_idx, 0]), int(stem_px[top_idx, 1])
    else:
        apex_y, apex_x = plant['rmin'], plant['cx']
    cv2.circle(label_map, (apex_x, apex_y), radius=6, color=4, thickness=-1)

    petiole_masks, leaf_tip_pts = find_petiole_paths(
        skel_binary, stem_path_set, plant, mask_dist)

    for pm in petiole_masks:
        pm_img = (pm.astype(np.uint8)) * 255
        k_pet  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        pm_dil = cv2.dilate(pm_img, k_pet) > 0
        pm_dil = pm_dil & (mask > 0) & (label_map != 2) & (label_map != 4)
        label_map[pm_dil] = 3

    # All remaining plant pixels → leaf (guaranteed non-empty result)
    label_map[(mask > 0) & (label_map == 0)] = 1

    cv2.circle(label_map, (apex_x, apex_y), radius=6, color=4, thickness=-1)
    label_map[mask == 0] = 0

    return label_map, leaf_tip_pts


# STEP 2: SAM POINT-PROMPTED WITH LEAF BLADE REFINEMENT
def load_sam_predictor(checkpoint_path, device):
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry['vit_h'](checkpoint=checkpoint_path)
    sam.to(device)
    print("SAM-H loaded.")
    from segment_anything import SamPredictor
    return SamPredictor(sam)


def refine_leaves_with_sam(img_rgb, label_map, mask, leaf_tip_pts, predictor):
    if not leaf_tip_pts:
        return label_map

    predictor.set_image(img_rgb)
    H, W = label_map.shape

    for tip_r, tip_c in leaf_tip_pts:
        walk_r = max(0, tip_r - 15)
        walk_c = np.clip(tip_c, 0, W-1)
        point_coords = np.array([[walk_c, walk_r]])
        point_labels = np.array([1])

        try:
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            best = masks[np.argmax(scores)]
            # paintable = best & (mask > 0) & \
            #             (label_map != 2) & (label_map != 4)
            paintable = best & (mask > 0) & (label_map != 2) & (label_map != 3) & (label_map != 4)
            label_map[paintable] = 1
        except Exception:
            continue

    label_map[(mask > 0) & (label_map == 0)] = 1
    return label_map

#  PIPELINE (STEP 1 and 2)
def label_to_color(label_map, cfg: Config):
    H, W = label_map.shape
    out  = np.zeros((H, W, 3), dtype=np.uint8)
    for cid, bgr in cfg.semantic.class_colors_bgr.items():
        out[label_map == cid] = bgr
    return out

def run_pseudo_label_pipeline(
        views: List[str], 
        resized_path: Path, 
        mask_path: Path, 
        label_path: Path,
        label_vis_path: Path,
        cfg: Config,
        use_sam: bool=True, 
        predictor: Optional[SamPredictor]=None
        ):
    if not use_sam:
        predictor = None

    print(f"\n{'='*60}")
    print(f"PSEUDO-LABEL GENERATION")
    print(f"  SAM:  {'Yes (point-prompted)' if use_sam else 'No (morphology only)'}")
    print(f"{'='*60}\n")

    t0 = time.time()
    stats = {}

    for view_key in tqdm(views, desc="Pseudo-labels"):
        fname     = f"{view_key}.png"
        img_path  = resized_path / fname
        mask_path = mask_path / fname

        img_full = cv2.imread(str(img_path))
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img_full is None or raw_mask is None:
            print(f"  WARNING: missing file for view '{view_key}', skipping")
            continue

        mask    = (raw_mask > 10).astype(np.uint8) * 255
        img_rgb = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)

        if view_key == 'top':
            label_map    = generate_topview_label(img_path, mask_path)
            leaf_tip_pts = []
        else:
            label_map, leaf_tip_pts = generate_morphology_label(img_path, mask_path)
            if use_sam and predictor is not None:
                label_map = refine_leaves_with_sam(
                    img_rgb, label_map, mask, leaf_tip_pts, predictor)

        # Save
        np.save(str(label_path / f"{view_key}.npy"), label_map)
        cv2.imwrite(str(label_vis_path / fname), label_to_color(label_map, cfg))

        classes, counts = np.unique(label_map, return_counts=True)
        stats[view_key] = {cfg.semantic.class_names[c]: int(n)
                           for c, n in zip(classes, counts)}

    elapsed = (time.time() - t0) / 60
    print(f"\nDone in {elapsed:.1f} min\n")

    print(f"{'view':>5}  {'bg':>8} {'leaf':>8} {'stem':>7} {'pet':>8} {'apex':>7}")
    print('-' * 50)
    for vk in views:
        if vk not in stats:
            continue
        s = stats[vk]
        print(f" {str(vk):>5}  "
              f"{s.get('background',0):>8,} {s.get('leaf',0):>8,} "
              f"{s.get('stem',0):>7,} {s.get('petiole',0):>8,} "
              f"{s.get('apex',0):>7,}")

    return stats

def visualize_pipeline_output(
        views: List[str], 
        resized_path: Path, 
        mask_path: Path, 
        label_path: Path, 
        checkpoint_path: Path,
        cfg: Config, 
        angles_to_show=None,
        ):
    if angles_to_show is None:
        angles_to_show = views[:5]

    fig, axes = plt.subplots(4, len(angles_to_show),
                             figsize=(4 * len(angles_to_show), 14))
    row_labels = ['RGB', 'Silhouette', 'Labels', 'Overlay']

    for i, view_key in enumerate(angles_to_show):
        fname   = f"{view_key}.png"
        img = cv2.imread(resized_path / fname)

        orig_bgr = cv2.imread(str(resized_path / fname))
        maskv_raw = cv2.imread(str(mask_path / fname), cv2.IMREAD_GRAYSCALE)

        if orig_bgr is None or maskv_raw is None:
            print(f"[WARN] Missing image or mask for visualization: {fname}")
            continue

        orig    = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        maskv   = cv2.imread(mask_path / fname, cv2.IMREAD_GRAYSCALE)
        lmap    = np.load(label_path / f"{view_key}.npy")
        lvis    = cv2.cvtColor(label_to_color(lmap, cfg=cfg), cv2.COLOR_BGR2RGB)
        overlay = np.clip(orig.astype(float) * 0.5 +
                          lvis.astype(float) * 0.5, 0, 255).astype(np.uint8)

        for row, (data, cm) in enumerate(
                [(orig, None), (maskv, 'gray'), (lvis, None), (overlay, None)]):
            axes[row][i].imshow(data, cmap=cm)
            axes[row][i].axis('off')
            if i == 0:
                axes[row][i].set_ylabel(row_labels[row], fontsize=9)
            if row == 0:
                axes[row][i].set_title(view_key, fontsize=10)

    handles = [
        mpatches.Patch(
            color=tuple((np.array(cfg.semantic.class_colors_bgr[c][::-1]) / 255.0).tolist()),
            label=cfg.semantic.class_names[c])
        for c in range(cfg.semantic.num_classes)]
    
    fig.legend(handles=handles, loc='lower center',
               ncol=cfg.semantic.num_classes, fontsize=10, bbox_to_anchor=(0.5, 0.01))
    plt.suptitle('Pipeline output', fontsize=12)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    out_path = checkpoint_path / "pipeline_output.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {out_path}")

def visualize_top_view(
        resized_path: Path, 
        mask_path: Path, 
        label_path: Path, 
        checkpoint_path: Path, 
        cfg: Config
        ):
    view_key = "top"

    fname = f"{view_key}.png"

    # Safety check
    if not (resized_path / fname).exists():
        print("Top image not found.")
        return
    if not (mask_path / fname).exists():
        print("Top mask not found.")
        return
    if not (label_path / f"{view_key}.npy").exists():
        print("Top label not found.")
        return

    orig_bgr = cv2.imread(str(resized_path / fname))
    maskv_raw = cv2.imread(str(mask_path / fname), cv2.IMREAD_GRAYSCALE)
    lmap = np.load(label_path / f"{view_key}.npy")
    lvis = cv2.cvtColor(label_to_color(lmap, cfg), cv2.COLOR_BGR2RGB)

    if orig_bgr is None or maskv_raw is None:
            print(f"[WARN] Missing image or mask for visualization: {fname}")
            return


    overlay = np.clip(
        orig_bgr.astype(float) * 0.5 +
        lvis.astype(float) * 0.5,
        0, 255
    ).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    row_labels = ["RGB", "Silhouette", "Labels", "Overlay"]

    for ax, data, label in zip(
        axes,
        [orig_bgr, maskv_raw, lvis, overlay],
        row_labels
    ):
        if label == "Silhouette":
            ax.imshow(data, cmap="gray")
        else:
            ax.imshow(data)
        ax.axis("off")
        ax.set_title(label, fontsize=10)

    plt.suptitle("Top View Output", fontsize=12)
    plt.tight_layout()
    out_path = checkpoint_path / "top_view_output.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Saved → {out_path}")
