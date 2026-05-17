import cv2
import time
import numpy as np
import networkx as nx
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from typing import Any, Optional, List, Tuple, cast
from numpy.typing import NDArray

from skimage.morphology import skeletonize
from skimage.measure import label as cc_label
from segment_anything import SamPredictor

from src.config.project import Config

BoolArray = NDArray[np.bool_]
UInt8Array = NDArray[np.uint8]
IntArray = NDArray[np.int32]
FloatArray = NDArray[np.float32]


def as_bool_array(x: Any) -> BoolArray:
    return np.asarray(x, dtype=np.bool_)


def as_uint8_array(x: Any) -> UInt8Array:
    return np.asarray(x, dtype=np.uint8)


def as_int_array(x: Any) -> IntArray:
    return np.asarray(x, dtype=np.int32)


def as_float32_array(x: Any) -> FloatArray:
    return np.asarray(x, dtype=np.float32)


def safe_cc_label(binary: Any) -> IntArray:
    """
    skimage.measure.label has broad/ambiguous type hints.
    This wrapper forces the output to be an integer ndarray.
    """
    labeled = cc_label(
        np.asarray(binary, dtype=np.bool_),
        connectivity=2,
        return_num=False,
    )
    return np.asarray(labeled, dtype=np.int32)


def safe_skeletonize(binary: Any) -> BoolArray:
    """
    skimage.skeletonize also has weak typing for Pylance.
    This wrapper always returns a bool ndarray.
    """
    skel = skeletonize(np.asarray(binary, dtype=np.uint8))
    return np.asarray(skel, dtype=np.bool_)


BG = 0
LEAF = 1
STEM = 2
PETIOLE = 3
APEX = 4
IGNORE = 255


def read_image_and_mask(img_path: Path, mask_path: Path):
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img_bgr is None:
        raise ValueError(f"Could not read image: {img_path}")

    if raw_mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    mask = (raw_mask > 10).astype(np.uint8) * 255
    return img_bgr, mask


def get_plant_info(mask: np.ndarray):
    rows, cols = np.where(mask > 127)
    H, W = mask.shape

    if len(rows) == 0:
        return dict(
            found=False,
            cx=W // 2,
            cy=H // 2,
            rmin=0,
            rmax=H - 1,
            cmin=0,
            cmax=W - 1,
            pH=H,
            pW=W,
            area=0,
        )

    return dict(
        found=True,
        cx=int(cols.mean()),
        cy=int(rows.mean()),
        rmin=int(rows.min()),
        rmax=int(rows.max()),
        cmin=int(cols.min()),
        cmax=int(cols.max()),
        pH=int(rows.max() - rows.min()) + 1,
        pW=int(cols.max() - cols.min()) + 1,
        area=int(len(rows)),
    )


def build_skeleton_graph(skel_binary: np.ndarray):
    pts = np.argwhere(skel_binary > 0)
    pts_list = [tuple(map(int, p)) for p in pts]
    pt_set = set(pts_list)

    G = nx.Graph()
    G.add_nodes_from(pts_list)

    for r, c in pts_list:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue

                nb = (r + dr, c + dc)

                if nb in pt_set:
                    G.add_edge((r, c), nb)

    degree = dict(G.degree())
    return G, degree


def remove_tiny_components(binary: np.ndarray, min_area: int = 20) -> BoolArray:
    labeled: IntArray = safe_cc_label(binary)
    out: BoolArray = np.zeros(binary.shape, dtype=np.bool_)

    n_labels = int(np.max(labeled))

    for lbl in range(1, n_labels + 1):
        comp: BoolArray = labeled == lbl

        if int(np.sum(comp)) >= min_area:
            out |= comp

    return out

def find_visible_stem_path(
    skel_binary: np.ndarray,
    plant: dict,
    mask_dist: np.ndarray,
) -> Tuple[List[Tuple[int, int]], float]:
    """
    Conservative visible-stem detector.

    The old logic selected the thickest central skeleton node as the stem target.
    That fails when leaves occlude the stem or create thick central blobs.

    This version:
    - finds a lower/root hub,
    - tries multiple upward paths,
    - scores them by verticality, centrality, thickness, and low wandering,
    - returns no stem if confidence is poor.
    """

    H, W = skel_binary.shape
    G, degree = build_skeleton_graph(skel_binary)

    if len(G.nodes) < 8:
        return [], 0.0

    nodes = list(G.nodes)

    pH = max(1, int(plant["pH"]))
    pW = max(1, int(plant["pW"]))
    cx = int(plant["cx"])

    max_dist = float(mask_dist.max()) + 1e-6

    lower_y = int(plant["rmin"] + 0.45 * pH)
    lower_nodes = [n for n in nodes if n[0] >= lower_y]

    if not lower_nodes:
        lower_nodes = nodes

    def root_score(n):
        r, c = n

        centrality = 1.0 - min(1.0, abs(c - cx) / (0.5 * pW + 1e-6))
        thickness = float(mask_dist[r, c]) / max_dist
        deg = int(degree.get(n, 0))

        branch_bonus = 1.0 if deg >= 3 else 0.0
        lower_bonus = (r - plant["rmin"]) / pH

        return (
            2.2 * thickness +
            1.8 * centrality +
            1.0 * branch_bonus +
            0.5 * lower_bonus
        )

    root = max(lower_nodes, key=root_score)
    root_r, root_c = root

    for u, v in G.edges:
        ur, uc = u
        vr, vc = v

        x_dev = (abs(uc - root_c) + abs(vc - root_c)) / (pW + 1e-6)
        thin_penalty = 1.0 / (1.0 + float(mask_dist[vr, vc]))
        downward_penalty = 1.0 if vr > ur else 0.0

        G[u][v]["weight"] = (
            1.0 +
            4.0 * x_dev +
            2.0 * thin_penalty +
            0.3 * downward_penalty
        )

    min_vertical_gain = max(5, int(0.12 * pH))
    max_allowed_xdev = max(12, int(0.30 * pW))

    candidates = []

    for n in nodes:
        r, c = n
        vertical_gain = root_r - r

        if vertical_gain < min_vertical_gain:
            continue

        if abs(c - root_c) > max_allowed_xdev:
            continue

        # Prefer endpoints or branch nodes as possible visible stem targets.
        deg = int(degree.get(n, 0))

        if deg == 1 or deg >= 3:
            candidates.append(n)

    if not candidates:
        return [], 0.0

    best_path: List[Tuple[int, int]] = []
    best_score = -1e9

    for target in candidates:
        try:
            path = nx.shortest_path(G, root, target, weight="weight")
        except nx.NetworkXNoPath:
            continue

        if len(path) < 5:
            continue

        rows = np.array([p[0] for p in path], dtype=np.float32)
        cols = np.array([p[1] for p in path], dtype=np.float32)

        vertical_gain = float(root_r - rows.min())

        if vertical_gain <= 0:
            continue

        path_len = float(len(path))
        horizontal_wander = float(np.max(np.abs(cols - root_c)))
        mean_xdev = float(np.mean(np.abs(cols - root_c)))
        mean_thick = float(np.mean([mask_dist[r, c] for r, c in path]))

        straightness = vertical_gain / (path_len + 1e-6)
        centrality = 1.0 - min(1.0, mean_xdev / (0.35 * pW + 1e-6))
        max_wander_frac = horizontal_wander / (pW + 1e-6)
        thickness_score = mean_thick / max_dist

        # Reject obvious leaf-like paths.
        if max_wander_frac > 0.35:
            continue

        score = (
            3.0 * (vertical_gain / pH) +
            2.0 * straightness +
            1.5 * centrality +
            1.0 * thickness_score -
            2.0 * max_wander_frac
        )

        if score > best_score:
            best_score = score
            best_path = path

    # Do not force stem when uncertain.
    if best_score < 1.0:
        return [], float(best_score)

    return best_path, float(best_score)


def path_to_region(
    path: List[Tuple[int, int]],
    shape: Tuple[int, int],
    mask: np.ndarray,
    mask_dist: np.ndarray,
) -> np.ndarray:
    H, W = shape
    region = np.zeros((H, W), dtype=np.uint8)

    for r, c in path:
        if 0 <= r < H and 0 <= c < W:
            region[r, c] = 255

    if region.sum() == 0:
        return np.zeros((H, W), dtype=bool)

    # Conservative thickness.
    stem_thick_px = max(3, min(7, int(float(mask_dist.max()) * 0.10)))

    if stem_thick_px % 2 == 0:
        stem_thick_px += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (stem_thick_px, stem_thick_px),
    )

    stem_region = cv2.dilate(region, kernel, iterations=1) > 0
    stem_region = stem_region & (mask > 0)

    return stem_region

def find_petiole_paths_v2(
    skel_binary: np.ndarray,
    stem_region: np.ndarray,
    plant: dict,
    mask_dist: np.ndarray,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Conservative petiole detector.

    Old logic labeled many non-stem skeleton components as petiole.
    This version only keeps thin skeleton fragments that touch the visible stem
    and keeps only their proximal near-stem part.
    """

    H, W = skel_binary.shape

    if stem_region.sum() == 0:
        return [], []

    stem_dil: BoolArray = as_bool_array(
            cv2.dilate(
                stem_region.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=1,
            )
        )
    branch_skel = skel_binary.astype(bool).copy()
    branch_skel[stem_dil] = False

    labeled: IntArray = safe_cc_label(branch_skel)
    n_comp = int(np.max(labeled))

    inv_stem = (~stem_dil).astype(np.uint8)
    dist_to_stem = cv2.distanceTransform(inv_stem, cv2.DIST_L2, 5)

    petiole_masks: List[np.ndarray] = []
    leaf_tip_pts: List[Tuple[int, int]] = []

    max_dist = float(mask_dist.max()) + 1e-6
    max_petiole_len_from_stem = max(8, int(plant["pH"] * 0.16))

    for lbl in range(1, n_comp + 1):
        comp = labeled == lbl
        pts = np.argwhere(comp)

        if len(pts) < 5:
            continue

        comp_dil = cv2.dilate(
            comp.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ).astype(bool)

        touches_stem = bool(np.any(comp_dil & stem_dil))

        if not touches_stem:
            continue

        mean_thickness = float(np.mean([mask_dist[int(r), int(c)] for r, c in pts]))

        # Thick component is usually leaf blade/midrib, not petiole.
        if mean_thickness > max_dist * 0.45:
            continue

        # Keep only proximal section close to stem.
        near_stem = comp & (dist_to_stem <= max_petiole_len_from_stem)

        if near_stem.sum() < 4:
            continue

        petiole_masks.append(near_stem)

        # Tip is used only for optional SAM leaf refinement.
        tip = max(
            pts,
            key=lambda p: (int(p[0]) - plant["cy"]) ** 2 +
                          (int(p[1]) - plant["cx"]) ** 2,
        )

        leaf_tip_pts.append((int(tip[0]), int(tip[1])))

    return petiole_masks, leaf_tip_pts

def generate_morphology_label(
    img_path: Path,
    mask_path: Path,
    label_side_apex: bool = False,
):
    """
    Safer side-view semantic pseudo-labeling.

    Classes:
        0 background
        1 leaf
        2 visible high-confidence stem
        3 conservative near-stem petiole
        4 apex only if label_side_apex=True

    Strong recommendation:
        label_side_apex=False for training.
    """

    img_bgr, mask = read_image_and_mask(img_path, mask_path)
    H, W = img_bgr.shape[:2]

    plant = get_plant_info(mask)
    label_map = np.zeros((H, W), dtype=np.uint8)

    if not plant["found"] or plant["area"] < 50:
        return label_map, []

    # Default foreground = leaf.
    label_map[mask > 0] = LEAF

    # Clean tiny isolated mask bits before skeleton.
    clean_mask = remove_tiny_components(mask > 0, min_area=max(10, plant["area"] // 500))
    clean_mask_u8 = clean_mask.astype(np.uint8) * 255

    mask_dist = cv2.distanceTransform(clean_mask_u8, cv2.DIST_L2, 5)
    skel_binary: BoolArray = safe_skeletonize(clean_mask)

    # Visible conservative stem.
    stem_path, stem_conf = find_visible_stem_path(
        skel_binary=skel_binary,
        plant=plant,
        mask_dist=mask_dist,
    )

    stem_region = np.zeros((H, W), dtype=bool)

    if len(stem_path) > 0:
        stem_region = path_to_region(
            path=stem_path,
            shape=(H, W),
            mask=clean_mask_u8,
            mask_dist=mask_dist,
        )

        label_map[stem_region] = STEM

    # Conservative petiole.
    petiole_masks, leaf_tip_pts = find_petiole_paths_v2(
        skel_binary=skel_binary,
        stem_region=stem_region,
        plant=plant,
        mask_dist=mask_dist,
    )

    for pm in petiole_masks:
        pm_img = pm.astype(np.uint8) * 255

        k_pet = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        pm_dil = cv2.dilate(pm_img, k_pet, iterations=1) > 0

        pm_dil = (
            pm_dil &
            (mask > 0) &
            (label_map != STEM)
        )

        label_map[pm_dil] = PETIOLE

    # Apex is unstable in side views.
    # Only draw it if explicitly enabled.
    if label_side_apex and stem_region.sum() > 0:
        stem_px = np.argwhere(stem_region)

        top_idx = int(np.argmin(stem_px[:, 0]))
        apex_y, apex_x = int(stem_px[top_idx, 0]), int(stem_px[top_idx, 1])

        cv2.circle(label_map, (apex_x, apex_y), radius=4, color=APEX, thickness=-1)

    label_map[mask == 0] = BG

    return label_map, leaf_tip_pts

def generate_topview_label_v2(
    img_path: Path,
    mask_path: Path,
    label_apex: bool = True,
):
    """
    Top-view pseudo-labeling.

    Top-view apex is more reasonable than side-view apex because the rosette/hub
    is visible from above. Still, use it cautiously for training.
    """

    img_bgr, mask = read_image_and_mask(img_path, mask_path)
    H, W = img_bgr.shape[:2]

    label_map = np.zeros((H, W), dtype=np.uint8)
    plant = get_plant_info(mask)

    if not plant["found"] or plant["area"] < 50:
        return label_map

    label_map[mask > 0] = LEAF

    mask_dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_dist = float(mask_dist.max()) + 1e-6

    cx, cy = plant["cx"], plant["cy"]
    plant_radius = max(plant["pH"], plant["pW"]) / 2.0

    stem_r = max(6, int(plant_radius * 0.12))
    stem_canvas = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(stem_canvas, (cx, cy), stem_r, 255, -1)

    stem_region = (stem_canvas > 0) & (mask > 0)
    label_map[stem_region] = STEM

    # Petiole skeleton outside central hub.
    non_stem = (mask > 0) & ~stem_region
    skel: BoolArray = safe_skeletonize(non_stem)

    petiole_skel = np.zeros((H, W), dtype=np.uint8)
    ys, xs = np.where(skel)

    for y, x in zip(ys, xs):
        # Thin skeleton pixels only.
        if mask_dist[y, x] < max_dist * 0.30:
            petiole_skel[y, x] = 255

    pet_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    petiole_dil = cv2.dilate(petiole_skel, pet_kernel, iterations=1) > 0

    label_map[
        petiole_dil &
        (mask > 0) &
        (label_map != STEM)
    ] = PETIOLE

    if label_apex:
        apex_r = max(3, int(plant_radius * 0.035))
        cv2.circle(label_map, (cx, cy), apex_r, APEX, -1)

    label_map[mask == 0] = BG

    return label_map


def refine_leaves_with_sam(
    img_rgb: np.ndarray,
    label_map: np.ndarray,
    mask: np.ndarray,
    leaf_tip_pts: List[Tuple[int, int]],
    predictor: SamPredictor,
):
    """
    Refines leaf regions only.

    Safety changes:
    - uses the real tip point, not tip_r - 15,
    - rejects SAM masks that cover too much of the plant,
    - never overwrites stem/petiole/apex.
    """

    if not leaf_tip_pts:
        return label_map

    predictor.set_image(img_rgb)

    H, W = label_map.shape
    plant_mask = mask > 0
    plant_area = float(plant_mask.sum()) + 1e-6

    for tip_r, tip_c in leaf_tip_pts:
        point_coords = np.array([
            [
                np.clip(tip_c, 0, W - 1),
                np.clip(tip_r, 0, H - 1),
            ]
        ])

        point_labels = np.array([1])

        try:
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

            best = None
            best_score = -1.0

            for m, s in zip(masks, scores):
                inside = m & plant_mask

                if inside.sum() == 0:
                    continue

                frac = inside.sum() / plant_area

                # Reject masks that grab nearly the entire plant.
                if frac > 0.45:
                    continue

                # Prefer high SAM score, but also useful leaf-sized region.
                score = float(s) + 0.15 * float(frac)

                if score > best_score:
                    best_score = score
                    best = inside

            if best is None:
                continue

            paintable = (
                best &
                plant_mask &
                (label_map != STEM) &
                (label_map != PETIOLE) &
                (label_map != APEX)
            )

            label_map[paintable] = LEAF

        except Exception:
            continue

    label_map[(mask > 0) & (label_map == BG)] = LEAF
    label_map[mask == 0] = BG

    return label_map


# ============================================================
# TRAINING-SAFE LABEL MAP
# ============================================================

def make_training_safe_label_map(
    label_map: np.ndarray,
    mask: np.ndarray,
    keep_petiole: bool = False,
    keep_apex: bool = False,
):
    """
    Creates a safer label map for NeRF semantic loss.

    Recommended:
        keep_petiole=False
        keep_apex=False

    Because:
        - petiole is often noisy in 2D,
        - apex is a keypoint, not a stable region class,
        - boundaries are uncertain.
    """

    mask_bool = mask > 0

    safe = np.full(label_map.shape, IGNORE, dtype=np.uint8)

    # Background remains valid background.
    safe[~mask_bool] = BG

    # Foreground defaults to leaf.
    safe[mask_bool] = LEAF

    # Keep visible stem.
    safe[(label_map == STEM) & mask_bool] = STEM

    if keep_petiole:
        safe[(label_map == PETIOLE) & mask_bool] = PETIOLE
    else:
        safe[(label_map == PETIOLE) & mask_bool] = IGNORE

    if keep_apex:
        safe[(label_map == APEX) & mask_bool] = APEX
    else:
        safe[(label_map == APEX) & mask_bool] = IGNORE

    # Ignore mask boundaries because they are noisy and anti-aliased.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    boundary = cv2.morphologyEx(
        mask_bool.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        kernel,
    ).astype(bool)

    safe[boundary] = IGNORE
    safe[~mask_bool] = BG

    return safe

def label_to_color(label_map: np.ndarray, cfg: Config):
    H, W = label_map.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)

    for cid, bgr in cfg.semantic.class_colors_bgr.items():
        out[label_map == cid] = bgr

    # Ignore label shown as gray.
    out[label_map == IGNORE] = np.array([128, 128, 128], dtype=np.uint8)

    return out


def get_class_name(cid: int, cfg: Config) -> str:
    if cid == IGNORE:
        return "ignore"

    try:
        return cfg.semantic.class_names[cid]
    except Exception:
        return f"class_{cid}"


def visualize_single_label_v2(
    image_path: Path,
    mask_path: Path,
    label_map: np.ndarray,
    cfg: Config,
    title: str = "",
):
    img_bgr, mask = read_image_and_mask(image_path, mask_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    lvis = cv2.cvtColor(label_to_color(label_map, cfg), cv2.COLOR_BGR2RGB)

    overlay = np.clip(
        img_rgb.astype(np.float32) * 0.55 +
        lvis.astype(np.float32) * 0.45,
        0,
        255,
    ).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    axes[0].imshow(img_rgb)
    axes[0].set_title("RGB")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask")

    axes[2].imshow(lvis)
    axes[2].set_title("Labels")

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()

def run_pseudo_label_pipeline_v2(
    image_names: List[str],
    resized_path: Path,
    mask_path: Path,
    label_path: Path,
    label_vis_path: Path,
    cfg: Config,
    use_sam: bool = True,
    predictor: Optional[SamPredictor] = None,
    save_training_safe: bool = True,
    label_side_apex: bool = False,
    keep_petiole_in_train: bool = False,
    keep_apex_in_train: bool = False,
):
    """
    Main semantic pseudo-label pipeline.

    Outputs:
        label_path/view.npy              full visualization pseudo-labels
        label_path/view_train.npy        safer labels for NeRF training
        label_vis_path/view.png          full label visualization
        label_vis_path/view_train.png    safe training-label visualization

    Recommended settings:
        label_side_apex=False
        keep_petiole_in_train=False
        keep_apex_in_train=False
    """

    if not use_sam:
        predictor = None

    label_path.mkdir(parents=True, exist_ok=True)
    label_vis_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("PSEUDO-LABEL GENERATION")
    print(f"  SAM leaf refinement:       {'Yes' if use_sam and predictor is not None else 'No'}")
    print(f"  Side-view apex labeling:   {'Yes' if label_side_apex else 'No'}")
    print(f"  Save train-safe labels:    {'Yes' if save_training_safe else 'No'}")
    print(f"  Keep petiole in training:  {'Yes' if keep_petiole_in_train else 'No'}")
    print(f"  Keep apex in training:     {'Yes' if keep_apex_in_train else 'No'}")
    print(f"{'=' * 70}\n")

    t0 = time.time()
    stats = {}

    for image_name in tqdm(image_names, desc="Pseudo-labels"):
        image_path = resized_path / image_name
        image_mask_path = mask_path / image_name

        if not image_path.exists() or not image_mask_path.exists():
            print(f"WARNING: missing file for view '{image_name}', skipping")
            continue

        img_bgr, mask = read_image_and_mask(image_path, image_mask_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if image_name == "top.png":
            label_map = generate_topview_label_v2(
                img_path=image_path,
                mask_path=image_mask_path,
                label_apex=True,
            )
            leaf_tip_pts = []
        else:
            label_map, leaf_tip_pts = generate_morphology_label(
                img_path=image_path,
                mask_path=image_mask_path,
                label_side_apex=label_side_apex,
            )

            if use_sam and predictor is not None:
                label_map = refine_leaves_with_sam(
                    img_rgb=img_rgb,
                    label_map=label_map,
                    mask=mask,
                    leaf_tip_pts=leaf_tip_pts,
                    predictor=predictor,
                )

        view = image_name.split(".")[0]

        # Full pseudo-label for visualization / thesis figure.
        np.save(str(label_path / f"{view}.npy"), label_map)
        cv2.imwrite(str(label_vis_path / image_name), label_to_color(label_map, cfg))

        # Safer map for NeRF semantic training.
        if save_training_safe:
            train_label_map = make_training_safe_label_map(
                label_map=label_map,
                mask=mask,
                keep_petiole=keep_petiole_in_train,
                keep_apex=keep_apex_in_train,
            )

            np.save(str(label_path / f"{view}_train.npy"), train_label_map)

            train_vis_name = f"{view}_train.png"
            cv2.imwrite(
                str(label_vis_path / train_vis_name),
                label_to_color(train_label_map, cfg),
            )

        classes, counts = np.unique(label_map, return_counts=True)
        stats[image_name] = {
            get_class_name(int(c), cfg): int(n)
            for c, n in zip(classes, counts)
        }

    elapsed = (time.time() - t0) / 60.0

    print(f"\nDone in {elapsed:.1f} min\n")

    print(
        f"{'view':>12} "
        f"{'bg':>10} "
        f"{'leaf':>10} "
        f"{'stem':>10} "
        f"{'petiole':>10} "
        f"{'apex':>10}"
    )
    print("-" * 68)

    for vk in image_names:
        if vk not in stats:
            continue

        s = stats[vk]

        print(
            f"{str(vk):>12} "
            f"{s.get('background', 0):>10,} "
            f"{s.get('leaf', 0):>10,} "
            f"{s.get('stem', 0):>10,} "
            f"{s.get('petiole', 0):>10,} "
            f"{s.get('apex', 0):>10,}"
        )

    return stats

def visualize_pipeline_output_v2(
    image_names: List[str],
    resized_path: Path,
    mask_path: Path,
    label_path: Path,
    checkpoint_path: Path,
    cfg: Config,
    use_train_safe: bool = False,
):
    """
    Visualizes RGB, silhouette, labels, overlay.

    Set use_train_safe=True to inspect *_train.npy labels.
    """

    suffix = "_train" if use_train_safe else ""

    fig, axes = plt.subplots(
        4,
        len(image_names),
        figsize=(4 * len(image_names), 14),
    )

    if len(image_names) == 1:
        axes = np.expand_dims(axes, axis=1)

    row_labels = ["RGB", "Silhouette", "Labels", "Overlay"]

    for i, image_name in enumerate(image_names):
        image_path = resized_path / image_name
        image_mask_path = mask_path / image_name

        orig_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        maskv = cv2.imread(str(image_mask_path), cv2.IMREAD_GRAYSCALE)

        if orig_bgr is None or maskv is None:
            print(f"[WARN] Missing image or mask for visualization: {image_name}")
            continue

        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

        view = image_name.split(".")[0]
        lmap_path = label_path / f"{view}{suffix}.npy"

        if not lmap_path.exists():
            print(f"[WARN] Missing label map: {lmap_path}")
            continue

        lmap = np.load(str(lmap_path))
        lvis = cv2.cvtColor(label_to_color(lmap, cfg), cv2.COLOR_BGR2RGB)

        overlay = np.clip(
            orig_rgb.astype(np.float32) * 0.55 +
            lvis.astype(np.float32) * 0.45,
            0,
            255,
        ).astype(np.uint8)

        data_items = [
            (orig_rgb, None),
            (maskv, "gray"),
            (lvis, None),
            (overlay, None),
        ]

        for row, (data, cm) in enumerate(data_items):
            axes[row][i].imshow(data, cmap=cm)
            axes[row][i].axis("off")

            if i == 0:
                axes[row][i].set_ylabel(row_labels[row], fontsize=9)

            if row == 0:
                axes[row][i].set_title(image_name, fontsize=10)

    handles = []

    for cid in range(cfg.semantic.num_classes):
        color = tuple(
            (
                np.array(cfg.semantic.class_colors_bgr[cid][::-1]) / 255.0
            ).tolist()
        )

        handles.append(
            mpatches.Patch(
                color=color,
                label=cfg.semantic.class_names[cid],
            )
        )

    handles.append(
        mpatches.Patch(
            color=(0.5, 0.5, 0.5),
            label="ignore",
        )
    )

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=cfg.semantic.num_classes + 1,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
    )

    title = "Training-safe semantic labels" if use_train_safe else "Full pseudo-labels"
    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=(0, 0.05, 1, 1))

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    out_name = "pipeline_output_train_safe.png" if use_train_safe else "pipeline_output.png"
    out_path = checkpoint_path / out_name

    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Saved → {out_path}")


def visualize_top_view_v2(
    resized_path: Path,
    mask_path: Path,
    label_path: Path,
    checkpoint_path: Path,
    cfg: Config,
    use_train_safe: bool = False,
):
    view_key = "top"
    view = f"{view_key}.png"

    image_path = resized_path / view
    image_mask_path = mask_path / view

    if not image_path.exists():
        print("Top image not found.")
        return

    if not image_mask_path.exists():
        print("Top mask not found.")
        return

    suffix = "_train" if use_train_safe else ""
    lmap_path = label_path / f"{view_key}{suffix}.npy"

    if not lmap_path.exists():
        print(f"Top label not found: {lmap_path}")
        return

    orig_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    maskv = cv2.imread(str(image_mask_path), cv2.IMREAD_GRAYSCALE)
    lmap = np.load(str(lmap_path))

    if orig_bgr is None or maskv is None:
        print("[WARN] Missing top image or mask.")
        return

    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    lvis = cv2.cvtColor(label_to_color(lmap, cfg), cv2.COLOR_BGR2RGB)

    overlay = np.asarray(
        np.clip(
            np.asarray(orig_rgb, dtype=np.float32) * 0.55 +
            np.asarray(lvis, dtype=np.float32) * 0.45,
            0,
            255,
        ),
        dtype=np.uint8,
    )

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    labels = ["RGB", "Silhouette", "Labels", "Overlay"]
    datas = [orig_rgb, maskv, lvis, overlay]
    cms = [None, "gray", None, None]

    for ax, data, label, cm in zip(axes, datas, labels, cms):
        ax.imshow(data, cmap=cm)
        ax.axis("off")
        ax.set_title(label, fontsize=10)

    title = "Top View Training-safe Output" if use_train_safe else "Top View Output"
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    out_name = "top_view_output_train_safe.png" if use_train_safe else "top_view_output.png"
    out_path = checkpoint_path / out_name

    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Saved → {out_path}")