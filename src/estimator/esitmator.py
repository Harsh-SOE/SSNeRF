"""
Clean protocol-aware 3D plant trait extractor.

This file is intentionally simple and self-contained.
It does NOT use manual/ground-truth trait values to compute predictions.
Manual/destructive values should only be used later for evaluation.

Expected input:
    - Open3D triangle mesh or point-cloud-like mesh with vertices
    - optional per-vertex semantic labels
    - optional independent unit scale, e.g. cm_per_model_unit from a ruler/pot marker
    - optional ground area for strict LAI

Traits reported:
    - observed vertical height
    - straightened/stretched height proxy
    - stem radius/diameter/perimeter near base/soil-emergence point
    - estimated opened leaf count
    - estimated leaf area and LAI proxy/strict LAI
    - occupied canopy volume and convex-hull canopy envelope volume
    - branch/petiole angles using proximal stem-leaf geometry

Coordinate convention:
    Internally points are converted to canonical coordinates:
        x, y = horizontal axes
        z    = vertical/up axis
    Set config.up_axis=1 for your NeRF Y-up convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import warnings

import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull, KDTree


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class TraitExtractionConfig:
    # Coordinate convention
    up_axis: int = 1

    # Scale. Keep 1.0 for model units. Use only an independent calibration object
    # for real units. Do not use measured plant height when height is evaluated.
    unit_scale: float = 1.0
    units: str = "model_unit"

    # Sampling and denoising
    sample_points: int = 250_000
    remove_statistical_outliers: bool = True
    outlier_nb_neighbors: int = 24
    outlier_std_ratio: float = 2.5
    robust_lower_percentile: float = 1.0
    robust_upper_percentile: float = 99.0

    # Strict LAI denominator. If unavailable, strict LAI is not computed.
    ground_area: Optional[float] = None

    # Base/stem estimation
    stem_lower_band_frac_height: float = 0.10
    stem_measure_offset_frac_height: float = 0.020
    stem_slice_half_thickness_frac_height: float = 0.006
    stem_core_percentile_lower_band: float = 8.0
    stem_slice_core_percentile_geometry: float = 8.0
    stem_radius_percentile_geometry: float = 55.0
    stem_radius_percentile_semantic: float = 80.0
    stem_min_points_in_slice: int = 20

    # Leaf candidate filtering
    leaf_min_radius_frac_height: float = 0.035
    leaf_exclude_lowest_frac_height: float = 0.020
    leaf_hist_bins: int = 360
    leaf_min_points: int = 25
    leaf_max_reasonable_count: int = 30
    leaf_open_min_area_frac_of_largest: float = 0.05
    leaf_open_min_area_frac_height2: float = 0.00045
    leaf_open_min_radial_extent_frac_height: float = 0.035

    # Multi-scale leaf-lobe detection. These are not versions; they are
    # simultaneous candidate settings used to reduce sensitivity to mesh noise.
    leaf_profiles: Sequence[Dict[str, float]] = field(default_factory=lambda: (
        {
            "name": "fine",
            "sigma": 0.75,
            "min_sep_deg": 5.0,
            "prominence": 0.004,
            "height": 0.004,
            "keep": 0.010,
            "merge_deg": 7.0,
            "valley": 0.70,
            "window_deg": 24.0,
        },
        {
            "name": "balanced",
            "sigma": 1.00,
            "min_sep_deg": 6.0,
            "prominence": 0.006,
            "height": 0.006,
            "keep": 0.015,
            "merge_deg": 10.0,
            "valley": 0.62,
            "window_deg": 28.0,
        },
        {
            "name": "coarse",
            "sigma": 1.35,
            "min_sep_deg": 8.0,
            "prominence": 0.010,
            "height": 0.010,
            "keep": 0.025,
            "merge_deg": 14.0,
            "valley": 0.50,
            "window_deg": 34.0,
        },
    ))

    # Branch/petiole angle extraction
    branch_proximal_fraction_candidates: Sequence[float] = (0.10, 0.14, 0.18, 0.24, 0.30)
    branch_inner_ignore_frac_of_leaf_extent: float = 0.015
    branch_local_stem_height_frac: float = 0.070
    branch_local_stem_core_percentile: float = 55.0
    branch_min_points: int = 20
    branch_min_valid_fraction_candidates: int = 2
    branch_angle_stability_iqr_deg: float = 22.0
    branch_max_attachment_distance_frac_height: float = 0.08
    branch_reject_low_confidence_from_summary: bool = False

    # Canopy volume
    voxel_size_frac_height: float = 0.018

    # Reporting
    print_report: bool = True


# Backward-compatible name for earlier notebook imports.
ProtocolTraitConfig = TraitExtractionConfig


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _as_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def canonicalize_points(points: np.ndarray, up_axis: int = 1) -> np.ndarray:
    """Return points in canonical coordinates: [horizontal_0, horizontal_1, up]."""
    P = np.asarray(points, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be 0, 1, or 2")
    h_axes = [a for a in (0, 1, 2) if a != up_axis]
    return np.stack([P[:, h_axes[0]], P[:, h_axes[1]], P[:, up_axis]], axis=1)


def _robust_z_bounds(P: np.ndarray, lo: float, hi: float) -> Tuple[float, float, float]:
    z0 = float(np.percentile(P[:, 2], lo))
    z1 = float(np.percentile(P[:, 2], hi))
    return z0, z1, max(z1 - z0, 1e-12)


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _pca_axes(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, eigenvalues(desc), eigenvectors(columns desc)."""
    P = np.asarray(P, dtype=np.float64)
    if len(P) == 0:
        return np.zeros(3), np.zeros(3), np.eye(3)
    mu = P.mean(axis=0)
    if len(P) < 3:
        return mu, np.zeros(3), np.eye(3)
    X = P - mu
    C = np.cov(X.T)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    return mu, vals[order], vecs[:, order]


def _convex_hull_area_2d(P2: np.ndarray) -> float:
    P2 = np.asarray(P2, dtype=np.float64)
    if len(P2) < 3:
        return 0.0
    try:
        return float(ConvexHull(P2).volume)  # in 2D, .volume is area
    except Exception:
        pass
    # Fallback: angular polygon around centroid.
    c = P2.mean(axis=0)
    ang = np.arctan2(P2[:, 1] - c[1], P2[:, 0] - c[0])
    Q = P2[np.argsort(ang)]
    x, y = Q[:, 0], Q[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _convex_hull_volume_3d(P: np.ndarray) -> float:
    P = np.asarray(P, dtype=np.float64)
    if len(P) < 4:
        return 0.0
    try:
        return float(ConvexHull(P).volume)
    except Exception:
        return 0.0


def _project_to_pca_plane(P: np.ndarray) -> np.ndarray:
    mu, _, vecs = _pca_axes(P)
    return (P - mu) @ vecs[:, :2]


def _ellipse_perimeter_ramanujan(a: float, b: float) -> float:
    a, b = abs(float(a)), abs(float(b))
    if a < b:
        a, b = b, a
    if a <= 1e-12:
        return 0.0
    h = ((a - b) ** 2) / ((a + b) ** 2 + 1e-12)
    return float(math.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + math.sqrt(max(4.0 - 3.0 * h, 1e-12)))))


def _circular_angle_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * (np.asarray(a) - b))))


def _circular_mean(angles: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    angles = np.asarray(angles, dtype=np.float64)
    if len(angles) == 0:
        return 0.0
    if weights is None:
        weights = np.ones_like(angles)
    z = np.sum(weights * np.exp(1j * angles))
    return float(np.angle(z) % (2.0 * np.pi))


def _robust_median(values: Sequence[float]) -> Optional[float]:
    vals = np.asarray([v for v in values if np.isfinite(v) and v > 0], dtype=np.float64)
    if len(vals) == 0:
        return None
    if len(vals) <= 2:
        return float(np.median(vals))
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    if mad > 1e-12:
        vals = vals[np.abs(vals - med) <= 3.0 * 1.4826 * mad]
    return float(np.median(vals)) if len(vals) else float(med)


def query_vertex_semantic_labels(model, vertices, cfg, chunk: int = 65536, direction=None) -> np.ndarray:
    """Query semantic class labels at mesh vertices from a NeRF/MLP semantic head."""
    import torch

    model.eval()
    device = cfg.device
    V = torch.as_tensor(vertices, dtype=torch.float32, device=device)

    if direction is None:
        direction = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
    else:
        direction = torch.as_tensor(direction, dtype=torch.float32, device=device)
        direction = direction / (torch.linalg.norm(direction) + 1e-8)

    labels = []
    with torch.no_grad():
        for s in range(0, V.shape[0], chunk):
            e = min(s + chunk, V.shape[0])
            pts = V[s:e]
            dirs = direction[None, :].expand(pts.shape[0], 3)
            _, _, sem_logits = model(pts, dirs)
            labels.append(torch.softmax(sem_logits.float(), dim=-1).argmax(dim=-1).cpu())
    return torch.cat(labels, dim=0).numpy().astype(np.int32)


# -----------------------------------------------------------------------------
# Main extractor
# -----------------------------------------------------------------------------

class PlantTraitExtractor3D:
    def __init__(
        self,
        mesh,
        config: Optional[TraitExtractionConfig] = None,
        vertex_labels: Optional[np.ndarray] = None,
        class_ids: Optional[Dict[str, int]] = None,
    ):
        self.mesh = mesh
        self.cfg = config or TraitExtractionConfig()
        self.vertex_labels = None if vertex_labels is None else np.asarray(vertex_labels)
        self.class_ids = class_ids or {}

        # Use empty arrays instead of Optional arrays. This keeps the runtime
        # behavior simple and also prevents Pylance from reporting optional
        # subscript/member-access errors after run() initializes these fields.
        self.points_raw: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.points: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.labels_sampled: Optional[np.ndarray] = None

        self.z_base = 0.0
        self.z_top = 0.0
        self.height = 0.0
        self.base_center_xy: np.ndarray = np.zeros(2, dtype=np.float64)

        self.stem_points: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.leaf_candidate_points: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.open_leaf_clusters: List[np.ndarray] = []

        self.traits: Dict[str, Any] = {}
        self.diagnostics: Dict[str, Any] = {}
        self.leaf_table: List[Dict[str, Any]] = []
        self.angle_table: List[Dict[str, Any]] = []

    def run(self, print_report: Optional[bool] = None) -> Dict[str, Any]:
        self._prepare_points()
        self._estimate_base_center()
        self._segment_stem_and_leaf_candidates()

        traits: Dict[str, Any] = {}
        traits.update(self._height_traits())
        traits.update(self._stem_traits())
        self._detect_open_leaves()
        traits.update(self._leaf_area_and_lai_traits())
        traits.update(self._canopy_volume_traits())
        traits.update(self._branch_angle_traits())
        traits.update(self._diagnostic_traits())

        self.traits = traits
        do_print = self.cfg.print_report if print_report is None else print_report
        if do_print:
            self.print_report(traits)
        return traits

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------
    def _prepare_points(self) -> None:
        if not self.mesh.has_vertices():
            raise ValueError("mesh has no vertices")

        V = np.asarray(self.mesh.vertices).astype(np.float64)
        labels = None

        if self.mesh.has_triangles() and self.cfg.sample_points > 0:
            try:
                pcd = self.mesh.sample_points_uniformly(number_of_points=int(self.cfg.sample_points))
                P = np.asarray(pcd.points).astype(np.float64)
                if self.vertex_labels is not None and len(self.vertex_labels) == len(V):
                    tree = KDTree(V)
                    _, nn = tree.query(P, k=1)
                    labels = self.vertex_labels[nn]
            except Exception:
                P = V.copy()
                if self.vertex_labels is not None and len(self.vertex_labels) == len(V):
                    labels = self.vertex_labels.copy()
        else:
            P = V.copy()
            if self.vertex_labels is not None and len(self.vertex_labels) == len(V):
                labels = self.vertex_labels.copy()

        if self.cfg.remove_statistical_outliers and len(P) > 100:
            try:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(P)
                _, ind = pcd.remove_statistical_outlier(
                    nb_neighbors=int(self.cfg.outlier_nb_neighbors),
                    std_ratio=float(self.cfg.outlier_std_ratio),
                )
                ind = np.asarray(ind, dtype=np.int64)
                P = P[ind]
                if labels is not None:
                    labels = labels[ind]
            except Exception:
                pass

        Q = canonicalize_points(P, up_axis=self.cfg.up_axis) * float(self.cfg.unit_scale)
        self.points_raw = P
        self.points = Q
        self.labels_sampled = labels
        self.z_base, self.z_top, self.height = _robust_z_bounds(
            Q, self.cfg.robust_lower_percentile, self.cfg.robust_upper_percentile
        )

    def _estimate_base_center(self) -> None:
        P = self.points
        z0, h = self.z_base, self.height
        lower = P[(P[:, 2] >= z0) & (P[:, 2] <= z0 + self.cfg.stem_lower_band_frac_height * h)]
        if len(lower) < 20:
            lower = P[P[:, 2] <= np.percentile(P[:, 2], 15)]
        if len(lower) < 20:
            lower = P

        center = np.median(lower[:, :2], axis=0)
        for _ in range(4):
            d = np.linalg.norm(lower[:, :2] - center[None, :], axis=1)
            radius = np.percentile(d, self.cfg.stem_core_percentile_lower_band)
            core = lower[d <= radius]
            if len(core) >= 10:
                center = np.median(core[:, :2], axis=0)
        self.base_center_xy = center

    def _segment_stem_and_leaf_candidates(self) -> None:
        P = self.points
        z0, h = self.z_base, self.height
        center = self.base_center_xy
        r = np.linalg.norm(P[:, :2] - center[None, :], axis=1)

        labels = self.labels_sampled
        have_labels = labels is not None and len(labels) == len(P)
        stem_id = self.class_ids.get("stem")
        leaf_id = self.class_ids.get("leaf")

        if have_labels and stem_id is not None:
            self.stem_points = P[labels == stem_id]
        else:
            lower = P[(P[:, 2] >= z0) & (P[:, 2] <= z0 + 0.55 * h)]
            if len(lower) < 20:
                lower = P
            dl = np.linalg.norm(lower[:, :2] - center[None, :], axis=1)
            core_r = max(np.percentile(dl, self.cfg.stem_core_percentile_lower_band), 0.008 * h)
            mask = (r <= 1.8 * core_r) & (P[:, 2] <= z0 + 0.70 * h)
            self.stem_points = P[mask]

        if have_labels and leaf_id is not None:
            self.leaf_candidate_points = P[labels == leaf_id]
        else:
            min_r = max(self.cfg.leaf_min_radius_frac_height * h, np.percentile(r, 18))
            z_min = z0 + self.cfg.leaf_exclude_lowest_frac_height * h
            self.leaf_candidate_points = P[(r >= min_r) & (P[:, 2] >= z_min)]

    # ------------------------------------------------------------------
    # Height
    # ------------------------------------------------------------------
    def _height_traits(self) -> Dict[str, Any]:
        u = self.cfg.units
        straight = self._estimate_straightened_height()
        observed = float(self.height)
        centerline = max(float(straight["centerline_length"]), observed)
        base_to_tip = max(float(straight["base_to_tip_distance"]), observed)
        recommended = centerline

        self.diagnostics["height_centerline_points"] = straight["centerline_points"]

        return {
            f"plant_height_observed_vertical_{u}": observed,
            f"plant_height_straightened_centerline_{u}": centerline,
            f"plant_height_base_to_tip_distance_{u}": base_to_tip,
            f"plant_height_stretched_protocol_recommended_{u}": recommended,
            f"plant_height_{u}": observed,
            f"plant_base_z_{u}": float(self.z_base),
            f"plant_top_z_{u}": float(self.z_top),
            "plant_height_observed_definition": "robust vertical extent of reconstructed plant pose",
            "plant_height_stretched_definition": "model-only centerline length proxy for manual straightened-height protocol",
        }

    def _estimate_straightened_height(self) -> Dict[str, Any]:
        P = self.points
        z0, z1, h = self.z_base, self.z_top, self.height
        if h <= 1e-12:
            return {"centerline_length": 0.0, "base_to_tip_distance": 0.0, "centerline_points": []}

        base_xy = self.base_center_xy
        base_pt = np.array([base_xy[0], base_xy[1], z0], dtype=np.float64)

        n_slices = 32
        min_pts = max(12, int(0.00035 * len(P)))
        edges = np.linspace(z0, z1, n_slices + 1)
        centers = [base_pt]
        last_xy = base_xy.copy()

        for i in range(n_slices):
            a, b = edges[i], edges[i + 1]
            S = P[(P[:, 2] >= a) & (P[:, 2] <= b if i == n_slices - 1 else P[:, 2] < b)]
            if len(S) < min_pts:
                continue
            d = np.linalg.norm(S[:, :2] - last_xy[None, :], axis=1)
            core = S[d <= np.percentile(d, 45.0)]
            if len(core) < max(8, min_pts // 3):
                core = S
            cxy = np.median(core[:, :2], axis=0)
            c = np.array([cxy[0], cxy[1], float(np.median(core[:, 2]))])
            if np.linalg.norm(c - centers[-1]) > 0.0025 * h:
                centers.append(c)
                last_xy = cxy

        upper = P[P[:, 2] >= z0 + 0.92 * h]
        if len(upper) >= 10:
            d = np.linalg.norm(upper[:, :2] - last_xy[None, :], axis=1)
            core = upper[d <= np.percentile(d, 55.0)]
            if len(core) < 6:
                core = upper
            top = np.array([
                float(np.median(core[:, 0])),
                float(np.median(core[:, 1])),
                float(np.percentile(core[:, 2], 85.0)),
            ])
            if np.linalg.norm(top - centers[-1]) > 0.0025 * h:
                centers.append(top)

        C = np.vstack(centers)
        centerline_len = float(np.sum(np.linalg.norm(np.diff(C, axis=0), axis=1))) if len(C) > 1 else float(h)

        tip_pool = P[P[:, 2] >= z0 + 0.65 * h]
        if len(tip_pool) < 20:
            tip_pool = P
        dist = np.linalg.norm(tip_pool - base_pt[None, :], axis=1)
        tip = np.median(tip_pool[dist >= np.percentile(dist, 98.0)], axis=0)
        base_to_tip = float(np.linalg.norm(tip - base_pt))

        return {
            "centerline_length": max(centerline_len, h),
            "base_to_tip_distance": max(base_to_tip, h),
            "centerline_points": C.tolist(),
        }

    # ------------------------------------------------------------------
    # Stem perimeter
    # ------------------------------------------------------------------
    def _stem_traits(self) -> Dict[str, Any]:
        u = self.cfg.units
        P = self.points
        z0, h = self.z_base, self.height
        z_m = z0 + self.cfg.stem_measure_offset_frac_height * h
        half = self.cfg.stem_slice_half_thickness_frac_height * h
        center = self.base_center_xy

        labels = self.labels_sampled
        stem_id = self.class_ids.get("stem")
        have_stem_labels = labels is not None and len(labels) == len(P) and stem_id is not None
        candidate = P[labels == stem_id] if have_stem_labels else P
        candidate_mode = "semantic_stem" if have_stem_labels else "geometry_central_core"

        S = candidate[np.abs(candidate[:, 2] - z_m) <= half]
        if len(S) < self.cfg.stem_min_points_in_slice:
            S = candidate[np.abs(candidate[:, 2] - z_m) <= 2.0 * half]
        if len(S) < self.cfg.stem_min_points_in_slice:
            return {
                "stem_measurement_status": "not_computed_too_few_points",
                "stem_points_in_measurement_slice": int(len(S)),
            }

        if not have_stem_labels:
            d_all = np.linalg.norm(S[:, :2] - center[None, :], axis=1)
            core = S[d_all <= np.percentile(d_all, self.cfg.stem_slice_core_percentile_geometry)]
            if len(core) >= self.cfg.stem_min_points_in_slice:
                S = core
                center = np.median(S[:, :2], axis=0)

        d = np.linalg.norm(S[:, :2] - center[None, :], axis=1)
        radius_percentile = self.cfg.stem_radius_percentile_semantic if have_stem_labels else self.cfg.stem_radius_percentile_geometry
        radius = float(np.percentile(d, radius_percentile))
        circular_perim = 2.0 * math.pi * radius

        XY = S[:, :2] - np.median(S[:, :2], axis=0)
        if len(XY) >= 5:
            C = np.cov(XY.T)
            vals, vecs = np.linalg.eigh(C)
            order = np.argsort(vals)[::-1]
            proj = XY @ vecs[:, order]
            # Percentile semiaxes are robust against isolated petiole/noise points.
            a = float(np.percentile(np.abs(proj[:, 0]), 85.0))
            b = float(np.percentile(np.abs(proj[:, 1]), 85.0))
        else:
            a = b = radius
        ellipse_perim = _ellipse_perimeter_ramanujan(a, b)

        lower = min(circular_perim, ellipse_perim)
        upper = max(circular_perim, ellipse_perim)
        if have_stem_labels:
            recommended = _robust_median([circular_perim, ellipse_perim]) or circular_perim
            basis = "semantic stem slice; robust median of circular and ellipse estimates"
        else:
            # Geometry-only cross-section is vulnerable to petiole/base contamination.
            # Geometric mean is a conservative compromise between lower/upper estimates.
            recommended = math.sqrt(max(lower, 1e-12) * max(upper, 1e-12))
            basis = "geometry-only central-core slice; geometric mean of circular and ellipse estimates"

        spread_ratio = float(np.percentile(d, 90) / max(np.percentile(d, 25), 1e-12)) if len(d) else float("nan")
        if have_stem_labels and spread_ratio < 3.0:
            confidence = "medium_semantic_slice"
        elif spread_ratio < 3.0:
            confidence = "medium_geometry_core_slice"
        else:
            confidence = "low_slice_contains_outward_structures"

        return {
            "stem_measurement_status": "ok",
            "stem_measurement_protocol": "horizontal cross-section near base/soil emergence point",
            f"stem_measurement_height_above_base_{u}": float(z_m - z0),
            f"stem_radius_base_circular_{u}": radius,
            f"stem_diameter_base_circular_{u}": 2.0 * radius,
            f"stem_perimeter_base_circular_{u}": circular_perim,
            f"stem_cross_section_area_base_circular_{u}2": math.pi * radius * radius,
            f"stem_ellipse_semimajor_base_{u}": max(a, b),
            f"stem_ellipse_semiminor_base_{u}": min(a, b),
            f"stem_perimeter_base_ellipse_{u}": ellipse_perim,
            f"stem_cross_section_area_base_ellipse_{u}2": math.pi * a * b,
            f"stem_perimeter_base_lower_bound_{u}": lower,
            f"stem_perimeter_base_upper_bound_{u}": upper,
            f"stem_perimeter_base_recommended_{u}": recommended,
            f"stem_equivalent_radius_recommended_{u}": recommended / (2.0 * math.pi),
            f"stem_equivalent_diameter_recommended_{u}": recommended / math.pi,
            f"plant_height_to_stem_perimeter_recommended_ratio": float(self.height / max(recommended, 1e-12)),
            "stem_perimeter_recommended_basis": basis,
            "stem_points_in_measurement_slice": int(len(S)),
            "stem_candidate_mode": candidate_mode,
            "stem_measurement_confidence": confidence,
            "stem_slice_radial_spread_ratio_p90_p25": spread_ratio,
        }

    # ------------------------------------------------------------------
    # Leaf detection and leaf area
    # ------------------------------------------------------------------
    def _detect_open_leaves(self) -> None:
        self.open_leaf_clusters = []
        P = self.leaf_candidate_points
        if len(P) < self.cfg.leaf_min_points:
            self.diagnostics["leaf_detection_status"] = "not_computed_insufficient_leaf_candidates"
            return

        profile_results = []
        for profile in self.cfg.leaf_profiles:
            res = self._run_leaf_profile(P, profile)
            if 0 < len(res["clusters"]) <= self.cfg.leaf_max_reasonable_count:
                profile_results.append(res)

        if not profile_results:
            self.diagnostics["leaf_detection_status"] = "not_computed_no_valid_profile"
            return

        counts = np.array([len(r["clusters"]) for r in profile_results], dtype=np.float64)
        target = float(np.median(counts))
        # Select the profile closest to the median count. This is a simple consensus
        # rule: avoid both very fine over-splitting and very coarse over-merging.
        selected = min(profile_results, key=lambda r: (abs(len(r["clusters"]) - target), r["profile_index"]))
        self.open_leaf_clusters = selected["clusters"]

        self.diagnostics["leaf_detection_status"] = "ok"
        self.diagnostics["leaf_multiscale_counts"] = [
            {
                "profile": r["profile_name"],
                "raw_peaks": r["raw_peak_count"],
                "merged_peaks": r["merged_peak_count"],
                "candidate_count": r["candidate_count_before_filter"],
                "opened_count": len(r["clusters"]),
            }
            for r in profile_results
        ]
        self.diagnostics["leaf_selected_profile"] = selected["profile_name"]
        self.diagnostics["leaf_histogram"] = selected.get("histogram")
        self.diagnostics["leaf_histogram_smooth"] = selected.get("histogram_smooth")
        self.diagnostics["leaf_peak_angles_rad"] = selected.get("peak_angles")

    def _run_leaf_profile(self, P: np.ndarray, profile: Dict[str, float]) -> Dict[str, Any]:
        assert self.base_center_xy is not None
        h = self.height
        center = self.base_center_xy
        dxy = P[:, :2] - center[None, :]
        r = np.linalg.norm(dxy, axis=1)
        min_r = max(self.cfg.leaf_min_radius_frac_height * h, np.percentile(r, 10))
        valid = r > min_r
        P2, r2, dxy2 = P[valid], r[valid], dxy[valid]
        if len(P2) < self.cfg.leaf_min_points:
            return self._empty_leaf_profile_result(profile)

        theta = (np.arctan2(dxy2[:, 1], dxy2[:, 0]) + 2.0 * np.pi) % (2.0 * np.pi)
        n_bins = int(self.cfg.leaf_hist_bins)
        bin_ids = np.floor(theta / (2.0 * np.pi) * n_bins).astype(int)
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)

        hist = np.zeros(n_bins, dtype=np.float64)
        weights = np.maximum(r2, 1e-12) ** 1.25
        np.add.at(hist, bin_ids, weights)

        sigma = float(profile.get("sigma", 1.0))
        hist_s = gaussian_filter1d(hist, sigma=sigma, mode="wrap")
        if hist_s.max() <= 1e-12:
            return self._empty_leaf_profile_result(profile, hist, hist_s)

        min_sep = max(1, int(n_bins * float(profile.get("min_sep_deg", 6.0)) / 360.0))
        peaks, _ = find_peaks(
            hist_s,
            distance=min_sep,
            prominence=float(profile.get("prominence", 0.006)) * hist_s.max(),
            height=float(profile.get("height", 0.006)) * hist_s.max(),
        )
        if len(peaks) == 0:
            return self._empty_leaf_profile_result(profile, hist, hist_s)

        scores = hist_s[peaks]
        keep = scores >= float(profile.get("keep", 0.015)) * scores.max()
        peaks = peaks[keep]
        scores = scores[keep]
        if len(peaks) == 0:
            return self._empty_leaf_profile_result(profile, hist, hist_s)

        peak_angles = peaks / n_bins * 2.0 * np.pi
        merged_angles = self._merge_leaf_peak_angles(
            peak_angles, scores, hist_s,
            merge_deg=float(profile.get("merge_deg", 10.0)),
            valley_ratio=float(profile.get("valley", 0.62)),
        )

        clusters = self._assign_points_to_leaf_angles(
            P2, theta, r2, merged_angles,
            window_deg=float(profile.get("window_deg", 28.0)),
        )

        return {
            "profile_name": str(profile.get("name", "profile")),
            "profile_index": list(self.cfg.leaf_profiles).index(profile) if profile in self.cfg.leaf_profiles else 0,
            "raw_peak_count": int(len(peaks)),
            "merged_peak_count": int(len(merged_angles)),
            "candidate_count_before_filter": int(len(clusters["all_candidates"])),
            "clusters": clusters["opened_clusters"],
            "histogram": hist,
            "histogram_smooth": hist_s,
            "peak_angles": merged_angles,
        }

    def _empty_leaf_profile_result(self, profile: Dict[str, float], hist=None, hist_s=None) -> Dict[str, Any]:
        return {
            "profile_name": str(profile.get("name", "profile")),
            "profile_index": 0,
            "raw_peak_count": 0,
            "merged_peak_count": 0,
            "candidate_count_before_filter": 0,
            "clusters": [],
            "histogram": hist,
            "histogram_smooth": hist_s,
            "peak_angles": np.array([], dtype=np.float64),
        }

    def _merge_leaf_peak_angles(
        self,
        angles: np.ndarray,
        scores: np.ndarray,
        hist_smooth: np.ndarray,
        merge_deg: float,
        valley_ratio: float,
    ) -> np.ndarray:
        if len(angles) <= 1:
            return np.asarray(angles, dtype=np.float64)

        order = np.argsort(angles)
        angles = angles[order]
        scores = scores[order]
        n = len(angles)
        bins = len(hist_smooth)
        groups: List[List[int]] = []
        current = [0]
        merge_rad = np.deg2rad(merge_deg)

        for k in range(n):
            i = k
            j = (k + 1) % n
            a_i = angles[i]
            a_j = angles[j] + (2.0 * np.pi if j == 0 else 0.0)
            gap = a_j - a_i

            bi = int((angles[i] % (2.0 * np.pi)) / (2.0 * np.pi) * bins) % bins
            bj = int((angles[j] % (2.0 * np.pi)) / (2.0 * np.pi) * bins) % bins
            if bi <= bj:
                valley = float(np.min(hist_smooth[bi:bj + 1]))
            else:
                valley = float(np.min(np.concatenate([hist_smooth[bi:], hist_smooth[:bj + 1]])))
            weak_peak = min(scores[i], scores[j])
            shallow_valley = valley >= valley_ratio * max(weak_peak, 1e-12)

            should_merge = gap < merge_rad or shallow_valley
            if k == n - 1:
                # circular last-first merge is handled after initial grouping
                if should_merge and groups:
                    current.extend(groups[0])
                    groups[0] = current
                else:
                    groups.append(current)
            else:
                if should_merge:
                    current.append(j)
                else:
                    groups.append(current)
                    current = [j]

        merged = []
        for g in groups:
            aa = angles[np.asarray(g) % n]
            ss = scores[np.asarray(g) % n]
            merged.append(_circular_mean(aa, ss))
        merged = np.asarray(sorted(merged), dtype=np.float64)
        return merged

    def _assign_points_to_leaf_angles(
        self,
        P: np.ndarray,
        theta: np.ndarray,
        r: np.ndarray,
        peak_angles: np.ndarray,
        window_deg: float,
    ) -> Dict[str, List[np.ndarray]]:
        if len(peak_angles) == 0:
            return {"all_candidates": [], "opened_clusters": []}

        h = self.height
        window = np.deg2rad(window_deg)
        diff = np.stack([_circular_angle_diff(theta, a) for a in peak_angles], axis=1)
        nearest = np.argmin(diff, axis=1)
        nearest_dist = diff[np.arange(len(P)), nearest]
        valid = nearest_dist <= window

        all_candidates: List[np.ndarray] = []
        candidate_stats: List[Dict[str, float]] = []
        for k in range(len(peak_angles)):
            C = P[(nearest == k) & valid]
            if len(C) < self.cfg.leaf_min_points:
                continue
            flat = _project_to_pca_plane(C)
            area = _convex_hull_area_2d(flat)
            radial_extent = float(np.percentile(np.linalg.norm(C[:, :2] - self.base_center_xy[None, :], axis=1), 95) -
                                  np.percentile(np.linalg.norm(C[:, :2] - self.base_center_xy[None, :], axis=1), 5))
            all_candidates.append(C)
            candidate_stats.append({"area": area, "radial_extent": radial_extent})

        if not all_candidates:
            return {"all_candidates": [], "opened_clusters": []}

        largest_area = max(s["area"] for s in candidate_stats)
        opened: List[np.ndarray] = []
        for C, s in zip(all_candidates, candidate_stats):
            if s["area"] < self.cfg.leaf_open_min_area_frac_of_largest * largest_area:
                continue
            if s["area"] < self.cfg.leaf_open_min_area_frac_height2 * h * h:
                continue
            if s["radial_extent"] < self.cfg.leaf_open_min_radial_extent_frac_height * h:
                continue
            opened.append(C)

        return {"all_candidates": all_candidates, "opened_clusters": opened}

    def _leaf_area_and_lai_traits(self) -> Dict[str, Any]:
        u = self.cfg.units
        self.leaf_table = []
        areas = []
        for i, C in enumerate(self.open_leaf_clusters, start=1):
            area = _convex_hull_area_2d(_project_to_pca_plane(C))
            radial = float(np.percentile(np.linalg.norm(C[:, :2] - self.base_center_xy[None, :], axis=1), 95))
            mean_z = float(np.mean(C[:, 2]))
            areas.append(area)
            self.leaf_table.append({
                "leaf_id": int(i),
                f"leaf_area_estimated_{u}2": float(area),
                f"leaf_radial_extent_{u}": radial,
                f"leaf_mean_z_{u}": mean_z,
                "leaf_points": int(len(C)),
            })

        total_area = float(np.sum(areas)) if areas else 0.0
        mean_area = float(np.mean(areas)) if areas else None
        projected_area = _convex_hull_area_2d(self.points[:, :2]) if len(self.points) >= 3 else 0.0
        lai_strict = None
        lai_status = "not_computed_ground_area_not_provided"
        if self.cfg.ground_area is not None and self.cfg.ground_area > 0:
            lai_strict = float(total_area / self.cfg.ground_area)
            lai_status = "ok"
        lai_proxy = float(total_area / projected_area) if projected_area > 1e-12 else None

        diag = self.diagnostics
        return {
            "leaf_detection_status": diag.get("leaf_detection_status", "ok" if areas else "not_computed"),
            "leaf_count_definition": "estimated fully opened leaves; tiny early-stage lobes filtered by area and radial extent",
            "open_leaf_count_estimated": int(len(self.open_leaf_clusters)),
            "leaf_selected_profile": diag.get("leaf_selected_profile"),
            "leaf_multiscale_counts": diag.get("leaf_multiscale_counts"),
            f"total_leaf_area_estimated_{u}2": total_area,
            f"mean_leaf_area_estimated_{u}2": mean_area,
            f"projected_canopy_area_{u}2": float(projected_area),
            "LAI_strict_total_leaf_area_over_ground_area": lai_strict,
            "LAI_strict_status": lai_status,
            "LAI_proxy_total_leaf_area_over_projected_canopy_area": lai_proxy,
        }

    # ------------------------------------------------------------------
    # Canopy volume
    # ------------------------------------------------------------------
    def _canopy_volume_traits(self) -> Dict[str, Any]:
        u = self.cfg.units
        P = self.points
        voxel = max(self.cfg.voxel_size_frac_height * self.height, 1e-9)
        mins = P.min(axis=0)
        idx = np.floor((P - mins[None, :]) / voxel).astype(np.int64)
        occ = len(np.unique(idx, axis=0))
        occupied_vol = float(occ * voxel ** 3)
        hull_vol = _convex_hull_volume_3d(P)
        return {
            f"canopy_occupied_voxel_volume_{u}3": occupied_vol,
            f"canopy_convex_hull_envelope_volume_{u}3": hull_vol,
            f"voxel_size_used_{u}": float(voxel),
        }

    # ------------------------------------------------------------------
    # Safer branch/petiole angles
    # ------------------------------------------------------------------
    def _branch_angle_traits(self) -> Dict[str, Any]:
        u = self.cfg.units
        self.angle_table = []
        if len(self.stem_points) < 10 or not self.open_leaf_clusters:
            return {
                "branch_angle_status": "not_computed_insufficient_stem_or_leaf_clusters",
                "branch_angle_count": 0,
            }

        S = np.asarray(self.stem_points, dtype=np.float64)
        h = max(self.height, 1e-12)
        tree = KDTree(S)
        raw_vals: List[float] = []
        acute_vals: List[float] = []
        weights: List[float] = []

        for leaf_id, C in enumerate(self.open_leaf_clusters, start=1):
            C = np.asarray(C, dtype=np.float64)
            attach = self._estimate_attachment(C, S, tree)
            if attach is None:
                continue
            attach_leaf, attach_stem, attach_dist = attach
            stem_axis, stem_status = self._local_stem_axis(S, attach_stem, h)
            if stem_axis is None:
                continue
            if stem_axis[2] < 0:
                stem_axis = -stem_axis

            d_leaf = np.linalg.norm(C - attach_stem[None, :], axis=1)
            extent = float(np.percentile(d_leaf, 95)) if len(d_leaf) else 0.0
            if extent <= 1e-12:
                continue

            candidates = []
            meta = []
            for frac in self.cfg.branch_proximal_fraction_candidates:
                axis_info = self._proximal_leaf_axis(C, attach_stem, extent, float(frac))
                if axis_info is None:
                    continue
                leaf_axis = axis_info["axis"]
                away = axis_info["centroid"] - attach_stem
                if np.dot(leaf_axis, away) < 0:
                    leaf_axis = -leaf_axis
                raw, acute = self._angle_between(stem_axis, leaf_axis)
                if np.isfinite(acute):
                    candidates.append(float(acute))
                    meta.append({
                        "proximal_fraction": float(frac),
                        "raw_angle_deg": float(raw),
                        "acute_angle_deg": float(acute),
                        "proximal_points": int(axis_info["n_points"]),
                        "method": axis_info["method"],
                    })

            if len(candidates) < self.cfg.branch_min_valid_fraction_candidates:
                axis_info = self._proximal_leaf_axis(C, attach_stem, extent, 0.35, force_min_points=True)
                if axis_info is not None:
                    leaf_axis = axis_info["axis"]
                    away = axis_info["centroid"] - attach_stem
                    if np.dot(leaf_axis, away) < 0:
                        leaf_axis = -leaf_axis
                    raw, acute = self._angle_between(stem_axis, leaf_axis)
                    if np.isfinite(acute):
                        candidates.append(float(acute))
                        meta.append({
                            "proximal_fraction": 0.35,
                            "raw_angle_deg": float(raw),
                            "acute_angle_deg": float(acute),
                            "proximal_points": int(axis_info["n_points"]),
                            "method": "fallback_" + axis_info["method"],
                        })

            if not candidates:
                continue

            arr = np.asarray(candidates, dtype=np.float64)
            acute_final = float(np.median(arr))
            best = int(np.argmin(np.abs(arr - acute_final)))
            raw_final = float(meta[best]["raw_angle_deg"])
            q25, q75 = np.percentile(arr, [25, 75])
            iqr = float(q75 - q25)

            flags = []
            if attach_dist > self.cfg.branch_max_attachment_distance_frac_height * h:
                flags.append("leaf_stem_gap_large")
            if iqr > self.cfg.branch_angle_stability_iqr_deg:
                flags.append("proximal_angle_unstable")
            if stem_status != "ok":
                flags.append("stem_axis_" + stem_status)
            if len(candidates) < self.cfg.branch_min_valid_fraction_candidates:
                flags.append("few_candidates")
            confidence = "high" if not flags else "low_" + "+".join(flags)
            include = not (self.cfg.branch_reject_low_confidence_from_summary and confidence != "high")

            row = {
                "leaf_id": int(leaf_id),
                "branch_angle_raw_deg": raw_final,
                "branch_angle_acute_deg": acute_final,
                "branch_angle_confidence": confidence,
                "branch_angle_candidate_count": int(len(candidates)),
                "branch_angle_candidate_iqr_deg": iqr,
                f"attachment_leaf_x_{u}": float(attach_leaf[0]),
                f"attachment_leaf_y_{u}": float(attach_leaf[1]),
                f"attachment_leaf_z_{u}": float(attach_leaf[2]),
                f"attachment_stem_x_{u}": float(attach_stem[0]),
                f"attachment_stem_y_{u}": float(attach_stem[1]),
                f"attachment_stem_z_{u}": float(attach_stem[2]),
                f"nearest_leaf_stem_distance_{u}": float(attach_dist),
                "leaf_points_total": int(len(C)),
                "include_in_summary": bool(include),
                "candidate_angles": meta,
            }
            self.angle_table.append(row)
            if include:
                raw_vals.append(raw_final)
                acute_vals.append(acute_final)
                weights.append(max(1.0, math.sqrt(len(C))) / (1.0 + iqr / 20.0))

        if not acute_vals:
            return {
                "branch_angle_status": "not_computed_no_valid_angles",
                "branch_angle_count": 0,
                "branch_angle_definition": "angle between local stem axis and proximal leaf/petiole direction near attachment",
            }

        raw_arr = np.asarray(raw_vals)
        acute_arr = np.asarray(acute_vals)
        w = np.asarray(weights, dtype=np.float64)
        w = w / max(float(w.sum()), 1e-12)
        high = sum(1 for r in self.angle_table if r["branch_angle_confidence"] == "high")
        low = len(self.angle_table) - high
        return {
            "branch_angle_status": "ok",
            "branch_angle_definition": "manual-protocol approximation: local stem axis vs proximal leaf/petiole direction near attachment; bent outer lamina is avoided",
            "branch_angle_count": int(len(acute_arr)),
            "branch_angle_high_confidence_count": int(high),
            "branch_angle_low_confidence_count": int(low),
            "mean_branch_angle_raw_deg": float(np.mean(raw_arr)),
            "median_branch_angle_raw_deg": float(np.median(raw_arr)),
            "mean_branch_angle_acute_deg": float(np.mean(acute_arr)),
            "median_branch_angle_acute_deg": float(np.median(acute_arr)),
            "std_branch_angle_acute_deg": float(np.std(acute_arr, ddof=1)) if len(acute_arr) > 1 else 0.0,
            "weighted_mean_branch_angle_acute_deg": float(np.sum(w * acute_arr)),
        }

    def _estimate_attachment(self, C: np.ndarray, S: np.ndarray, tree: Optional[Any]) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        if len(C) == 0 or len(S) == 0:
            return None
        if tree is not None:
            dists, nn = tree.query(C, k=1)
        else:
            D = np.linalg.norm(C[:, None, :] - S[None, :, :], axis=2)
            nn = np.argmin(D, axis=1)
            dists = D[np.arange(len(C)), nn]
        dists = np.asarray(dists, dtype=np.float64)
        nn = np.asarray(nn, dtype=np.int64)
        q = np.percentile(dists, 6.0)
        mask = dists <= max(q, float(dists.min()) + 1e-12)
        if mask.sum() < 3:
            order = np.argsort(dists)[:min(max(3, len(C) // 20), len(C))]
            mask = np.zeros(len(C), dtype=bool)
            mask[order] = True
        attach_leaf = np.median(C[mask], axis=0)
        attach_stem = np.median(S[nn[mask]], axis=0)
        if tree is not None:
            _, idx = tree.query(attach_stem[None, :], k=1)
            attach_stem = S[int(np.asarray(idx).ravel()[0])]
        else:
            attach_stem = S[int(np.argmin(np.linalg.norm(S - attach_stem[None, :], axis=1)))]
        return attach_leaf, attach_stem, float(np.median(dists[mask]))

    def _local_stem_axis(self, S: np.ndarray, attach_stem: np.ndarray, h: float) -> Tuple[Optional[np.ndarray], str]:
        dz = max(self.cfg.branch_local_stem_height_frac * h, 1e-8)
        z = float(attach_stem[2])
        band = S[np.abs(S[:, 2] - z) <= dz]
        status = "ok"
        if len(band) < 8:
            d = np.linalg.norm(S - attach_stem[None, :], axis=1)
            k = min(max(20, len(S) // 25), len(S))
            band = S[np.argsort(d)[:k]]
            status = "nearest_fallback"
        if len(band) < 5:
            return None, "too_few_points"
        dxy = np.linalg.norm(band[:, :2] - attach_stem[None, :2], axis=1)
        core = band[dxy <= np.percentile(dxy, self.cfg.branch_local_stem_core_percentile)]
        if len(core) >= 5:
            band = core
        _, _, vecs = _pca_axes(band)
        axis = _unit(vecs[:, 0])
        vertical = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(axis, vertical))) < 0.35:
            axis = _unit(0.35 * axis + 0.65 * vertical)
            status = "vertical_blended"
        return axis, status

    def _proximal_leaf_axis(
        self,
        C: np.ndarray,
        attach_stem: np.ndarray,
        leaf_extent: float,
        proximal_fraction: float,
        force_min_points: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if len(C) < 3 or leaf_extent <= 1e-12:
            return None
        d = np.linalg.norm(C - attach_stem[None, :], axis=1)
        inner = max(self.cfg.branch_inner_ignore_frac_of_leaf_extent * leaf_extent, 1e-10)
        outer = max(proximal_fraction * leaf_extent, inner * 2.0)
        proximal = C[(d >= inner) & (d <= outer)]
        min_pts = max(5, min(int(self.cfg.branch_min_points), 18))
        if len(proximal) < min_pts:
            if not force_min_points and len(C) < min_pts:
                return None
            order = np.argsort(d)
            order = order[d[order] >= inner] if np.any(d[order] >= inner) else order
            proximal = C[order[:min(max(min_pts, 8), len(order))]]
        if len(proximal) < 3:
            return None

        centroid = proximal.mean(axis=0)
        _, vals, vecs = _pca_axes(proximal)
        axis_pca = _unit(vecs[:, 0])
        axis_centroid = _unit(centroid - attach_stem)
        linearity = float((vals[0] - vals[1]) / max(vals[0], 1e-12)) if vals[0] > 1e-12 else 0.0
        if linearity >= 0.18 and np.linalg.norm(axis_pca) > 1e-9:
            axis = axis_pca
            method = "pca_proximal"
        else:
            axis = axis_centroid
            method = "centroid_proximal"
        if np.linalg.norm(axis_centroid) > 1e-9:
            if np.dot(axis, axis_centroid) < 0:
                axis = -axis
            axis = _unit(0.70 * axis + 0.30 * axis_centroid)
        return {"axis": axis, "centroid": centroid, "n_points": len(proximal), "method": method, "linearity": linearity}

    @staticmethod
    def _angle_between(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        a = _unit(a)
        b = _unit(b)
        if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
            return float("nan"), float("nan")
        raw = float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))
        acute = min(raw, 180.0 - raw)
        return raw, acute

    # ------------------------------------------------------------------
    # Diagnostics and printing
    # ------------------------------------------------------------------
    def _diagnostic_traits(self) -> Dict[str, Any]:
        return {
            "trait_extraction_mode": "semantic_assisted" if self.labels_sampled is not None else "geometry_only",
            "n_points_used": int(len(self.points)),
            "n_stem_points": int(len(self.stem_points)),
            "n_leaf_candidate_points": int(len(self.leaf_candidate_points)),
            "reconstruction_noise_note": "organ-level traits may be biased by mesh holes, ghost geometry, fused leaves, or weak stem/leaf separation",
        }

    def print_report(self, traits: Optional[Dict[str, Any]] = None) -> None:
        traits = self.traits if traits is None else traits
        print("\n" + "=" * 78)
        print("PROTOCOL-AWARE 3D PLANT PHENOTYPIC TRAIT REPORT")
        print("=" * 78)
        for k, v in traits.items():
            if isinstance(v, float):
                print(f"{k:<64}: {v:.6f}")
            else:
                print(f"{k:<64}: {v}")
        print("=" * 78)
        print(f"Linear unit: {self.cfg.units}  |  unit_scale={self.cfg.unit_scale}")
        print("No manual trait values are used by this extractor.")
        print("Observed and straightened height are reported separately.")
        print("Organ-level traits are estimates unless reliable organ semantics are available.")
        print("=" * 78 + "\n")

    def print_angle_table(self) -> None:
        if not self.angle_table:
            print("No angle table. Run extractor.run() first.")
            return
        print("\n" + "=" * 92)
        print("PROXIMAL BRANCH/PETIOLE ANGLE TABLE")
        print("=" * 92)
        print(f"{'leaf':>4} | {'acute':>8} | {'raw':>8} | {'confidence':>28} | {'cand':>4} | {'IQR':>7}")
        print("-" * 92)
        for r in self.angle_table:
            print(
                f"{int(r['leaf_id']):4d} | "
                f"{float(r['branch_angle_acute_deg']):8.2f} | "
                f"{float(r['branch_angle_raw_deg']):8.2f} | "
                f"{str(r['branch_angle_confidence'])[:28]:>28} | "
                f"{int(r['branch_angle_candidate_count']):4d} | "
                f"{float(r['branch_angle_candidate_iqr_deg']):7.2f}"
            )
        print("=" * 92 + "\n")

    def plot_leaf_debug(self) -> None:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 7))
        for i, C in enumerate(self.open_leaf_clusters, start=1):
            plt.scatter(C[:, 0], C[:, 1], s=2, label=f"leaf {i}")
        if len(self.stem_points):
            S = self.stem_points
            plt.scatter(S[:, 0], S[:, 1], s=2, c="black", label="stem/core")
        plt.axis("equal")
        plt.title(f"Estimated opened leaves: {len(self.open_leaf_clusters)}")
        plt.legend(markerscale=4, fontsize=8)
        plt.show()

    def plot_leaf_histogram_debug(self) -> None:
        import matplotlib.pyplot as plt
        hist = self.diagnostics.get("leaf_histogram_smooth")
        if hist is None:
            print("No leaf histogram available. Run extractor.run() first.")
            return
        theta = np.linspace(0, 360, len(hist), endpoint=False)
        plt.figure(figsize=(9, 3))
        plt.plot(theta, hist)
        peaks = self.diagnostics.get("leaf_peak_angles_rad")
        if peaks is not None:
            for a in peaks:
                plt.axvline(np.degrees(a), linestyle="--", alpha=0.4)
        plt.xlabel("Angular direction around stem (deg)")
        plt.ylabel("Smoothed radial leaf evidence")
        plt.title(f"Leaf angular histogram | selected profile: {self.diagnostics.get('leaf_selected_profile')}")
        plt.tight_layout()
        plt.show()

    def plot_stem_slice_debug(self) -> None:
        import matplotlib.pyplot as plt
        if len(self.points) == 0:
            print("No points available. Run extractor.run() first.")
            return
        z_m = self.z_base + self.cfg.stem_measure_offset_frac_height * self.height
        half = self.cfg.stem_slice_half_thickness_frac_height * self.height
        S = self.points[np.abs(self.points[:, 2] - z_m) <= max(half, 1e-12)]
        plt.figure(figsize=(6, 6))
        if len(S):
            plt.scatter(S[:, 0], S[:, 1], s=3, alpha=0.6)
        plt.scatter([self.base_center_xy[0]], [self.base_center_xy[1]], c="red", s=60, marker="x", label="base center")
        plt.axis("equal")
        plt.legend()
        plt.title("Stem/base measurement slice")
        plt.show()


# Backward-compatible class name for your previous notebook imports.
ProtocolAwarePlantTraitExtractor = PlantTraitExtractor3D


# -----------------------------------------------------------------------------
# Evaluation helper — use only after extraction
# -----------------------------------------------------------------------------

def evaluate_against_reference(
    predicted_traits: Dict[str, Any],
    reference_traits: Dict[str, float],
    key_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Compare predictions to manual/destructive references after extraction."""
    default_map = {
        "plant_height_observed": "plant_height_observed_vertical_cm",
        "plant_height_stretched": "plant_height_stretched_protocol_recommended_cm",
        "plant_height": "plant_height_stretched_protocol_recommended_cm",
        "stem_perimeter": "stem_perimeter_base_recommended_cm",
        "open_leaf_count": "open_leaf_count_estimated",
        "leaf_count": "open_leaf_count_estimated",
        "mean_branch_angle": "mean_branch_angle_acute_deg",
        "median_branch_angle": "median_branch_angle_acute_deg",
        "total_leaf_area": "total_leaf_area_estimated_cm2",
        "canopy_volume": "canopy_occupied_voxel_volume_cm3",
        "LAI": "LAI_strict_total_leaf_area_over_ground_area",
    }
    if key_map:
        default_map.update(key_map)

    rows = []
    for ref_key, ref_val in reference_traits.items():
        pred_key = default_map.get(ref_key, ref_key)
        if pred_key not in predicted_traits:
            suffix_candidates = [k for k in predicted_traits if k.startswith(pred_key)]
            if suffix_candidates:
                pred_key = suffix_candidates[0]
        pred_val = predicted_traits.get(pred_key)
        row = {
            "reference_key": ref_key,
            "predicted_key": pred_key,
            "reference_value": ref_val,
            "predicted_value": pred_val,
            "absolute_error": None,
            "percentage_error": None,
            "status": "ok",
        }
        if pred_val is None:
            row["status"] = "prediction_missing"
        else:
            try:
                p, r = float(pred_val), float(ref_val)
                row["absolute_error"] = abs(p - r)
                row["percentage_error"] = None if abs(r) < 1e-12 else 100.0 * abs(p - r) / abs(r)
            except Exception:
                row["status"] = "non_numeric"
        rows.append(row)
    return rows
