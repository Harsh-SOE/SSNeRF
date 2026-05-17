"""
Protocol-aware 3D plant phenotypic trait extractor.

The extractor does not use manual or ground-truth trait values during prediction.
Manual measurements should only be used later for evaluation.

Inputs:
    - reconstructed Open3D mesh
    - optional per-vertex semantic labels
    - optional independent unit scale, such as cm_per_model_unit from a ruler,
      pot diameter, calibration marker, or other object that is not an evaluated trait
    - optional independent ground area for strict LAI

Outputs:
    - observed vertical plant height
    - straightened/stretched height proxy
    - near-base stem perimeter with robust bounds
    - fully-opened leaf-count estimate
    - leaf area and LAI proxy/strict LAI
    - occupied canopy volume and convex-hull canopy-envelope volume
    - branch/petiole angle using local stem axis and proximal leaf/petiole direction

Geometry-only organ-level traits remain estimates. Reliable stem/leaf/petiole
semantic labels improve stem perimeter, leaf count, leaf area, and branch angles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
import warnings

import numpy as np

try:
    import open3d as o3d
except Exception as exc:  # pragma: no cover
    o3d = None
    warnings.warn(f"Open3D import failed: {exc}")

try:
    from scipy.ndimage import gaussian_filter1d as _scipy_gaussian_filter1d
    from scipy.signal import find_peaks as _scipy_find_peaks
    from scipy.spatial import ConvexHull as _SciPyConvexHull
    from scipy.spatial import KDTree as _SciPyKDTree
except Exception as exc:  # pragma: no cover
    _scipy_gaussian_filter1d = None
    _scipy_find_peaks = None
    _SciPyConvexHull = None
    _SciPyKDTree = None
    warnings.warn(f"SciPy import failed: {exc}")


# Keep the rest of the code simple while making Pylance happy:
# these wrappers are always callable. At runtime they gracefully degrade if SciPy
# is unavailable.
def gaussian_filter1d(input: np.ndarray, sigma: float, mode: str = "reflect") -> np.ndarray:
    arr = np.asarray(input, dtype=np.float64)
    if _scipy_gaussian_filter1d is None:
        return arr
    return np.asarray(_scipy_gaussian_filter1d(arr, sigma=sigma, mode=mode), dtype=np.float64)


def find_peaks(x: np.ndarray, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    if _scipy_find_peaks is None:
        return np.asarray([], dtype=np.int64), {}
    peaks, props = _scipy_find_peaks(x, **kwargs)
    return np.asarray(peaks, dtype=np.int64), dict(props)


ConvexHull: Any = _SciPyConvexHull
KDTree: Any = _SciPyKDTree


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class TraitExtractionConfig:
    up_axis: int = 1
    unit_scale: float = 1.0
    units: str = 'model_unit'
    sample_points: int = 250000
    remove_statistical_outliers: bool = True
    outlier_nb_neighbors: int = 24
    outlier_std_ratio: float = 2.5
    robust_lower_percentile: float = 1.0
    robust_upper_percentile: float = 99.0
    ground_area: Optional[float] = None
    stem_measure_offset_frac_height: float = 0.02
    stem_slice_half_thickness_frac_height: float = 0.006
    stem_lower_band_frac_height: float = 0.1
    stem_core_percentile_lower_band: float = 8.0
    stem_slice_core_percentile_geometry: float = 8.0
    stem_radius_percentile_geometry: float = 55.0
    stem_radius_percentile_semantic: float = 80.0
    stem_min_points_in_slice: int = 20
    leaf_min_radius_frac_height: float = 0.035
    leaf_exclude_lowest_frac_height: float = 0.02
    leaf_hist_bins: int = 360
    leaf_hist_sigma: float = 1.0
    leaf_peak_prominence_ratio: float = 0.006
    leaf_peak_height_ratio: float = 0.006
    leaf_keep_peak_ratio: float = 0.015
    leaf_min_peak_sep_deg: float = 6.0
    leaf_fragment_merge_angle_deg: float = 10.0
    leaf_valley_merge_ratio: float = 0.62
    leaf_assignment_window_deg: float = 28.0
    leaf_min_points: int = 25
    leaf_open_min_area_frac_of_largest: float = 0.05
    leaf_open_min_area_frac_height2: float = 0.00045
    leaf_open_min_radial_extent_frac_height: float = 0.035
    leaf_max_reasonable_count: int = 30
    leaf_area_method: str = 'pca_hull'
    voxel_size_frac_height: float = 0.018
    branch_attachment_search_frac_height: float = 0.12
    branch_proximal_leaf_frac: float = 0.25
    branch_min_points: int = 20
    print_report: bool = True
    branch_angle_mode: str = 'proximal_consensus'
    branch_proximal_fraction_candidates: Tuple[float, ...] = (0.1, 0.14, 0.18, 0.24, 0.3)
    branch_inner_ignore_frac_of_leaf_extent: float = 0.015
    branch_local_stem_height_frac: float = 0.07
    branch_local_stem_core_percentile: float = 55.0
    branch_angle_stability_iqr_deg: float = 22.0
    branch_max_attachment_distance_frac_height: float = 0.08
    branch_min_valid_fraction_candidates: int = 2
    branch_reject_low_confidence_from_summary: bool = False
    enable_robust_ensemble: bool = True
    ensemble_mad_k: float = 3.0
    ensemble_trim_fraction: float = 0.15
    stem_ensemble_offset_fracs: Tuple[float, ...] = (0.014, 0.018, 0.022, 0.026)
    stem_ensemble_half_thickness_fracs: Tuple[float, ...] = (0.004, 0.006, 0.008)
    stem_ensemble_core_percentiles: Tuple[float, ...] = (4.0, 6.0, 8.0, 10.0, 12.0)
    stem_ensemble_radius_percentiles_geometry: Tuple[float, ...] = (45.0, 50.0, 55.0, 60.0, 65.0)
    stem_ensemble_radius_percentiles_semantic: Tuple[float, ...] = (65.0, 70.0, 75.0, 80.0)
    stem_ensemble_max_estimators_reported: int = 80
    leaf_use_ensemble_profiles: bool = True
    leaf_consensus_profile_preference_strength: float = 0.12
    leaf_consensus_area_cv_penalty: float = 0.22
    leaf_consensus_small_leaf_penalty: float = 0.45
    leaf_count_stability_high_iqr_frac: float = 0.2
    leaf_count_stability_medium_iqr_frac: float = 0.45
    leaf_area_ensemble: bool = True
    leaf_area_trim_percentiles: Tuple[float, ...] = (90.0, 95.0, 98.0)
    leaf_area_ellipse_percentiles: Tuple[float, ...] = (80.0, 85.0, 90.0)
    branch_angle_use_candidate_outlier_rejection: bool = True
    branch_angle_summary_use_robust_mean: bool = True
    branch_angle_leaf_candidate_mad_k: float = 2.8
    branch_angle_summary_mad_k: float = 3.2



# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def _as_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def canonicalize_points(points: np.ndarray, up_axis: int = 1) -> np.ndarray:
    """
    Convert arbitrary XYZ points to canonical coordinates:
        Q[:, 0] = horizontal axis 1
        Q[:, 1] = horizontal axis 2
        Q[:, 2] = vertical/up axis
    """
    P = np.asarray(points, dtype=np.float64)
    axes = [0, 1, 2]
    if up_axis not in axes:
        raise ValueError("up_axis must be 0, 1, or 2")
    h_axes = [a for a in axes if a != up_axis]
    return np.stack([P[:, h_axes[0]], P[:, h_axes[1]], P[:, up_axis]], axis=1)


def _robust_bounds_z(P: np.ndarray, lo: float, hi: float) -> Tuple[float, float, float]:
    z0 = float(np.percentile(P[:, 2], lo))
    z1 = float(np.percentile(P[:, 2], hi))
    h = max(z1 - z0, 1e-12)
    return z0, z1, h


def _safe_norm(v: np.ndarray, eps: float = 1e-12) -> float:
    return float(max(np.linalg.norm(v), eps))


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _pca_axes(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, eigenvalues(desc), eigenvectors(columns desc)."""
    P = np.asarray(P, dtype=np.float64)
    mu = P.mean(axis=0)
    X = P - mu
    if len(P) < 3:
        return mu, np.zeros(3), np.eye(3)
    C = np.cov(X.T)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    return mu, vals, vecs


def _convex_hull_area_2d(P2: np.ndarray) -> float:
    P2 = np.asarray(P2, dtype=np.float64)
    if len(P2) < 3:
        return 0.0
    if ConvexHull is None:
        # Fallback: polygon area after angular sorting around centroid.
        c = P2.mean(axis=0)
        ang = np.arctan2(P2[:, 1] - c[1], P2[:, 0] - c[0])
        Q = P2[np.argsort(ang)]
        x, y = Q[:, 0], Q[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    try:
        hull = ConvexHull(P2)
        return float(hull.volume)  # in 2D scipy stores area as .volume
    except Exception:
        return 0.0


def _project_leaf_to_pca_plane(P: np.ndarray) -> np.ndarray:
    """Project a 3D leaf patch to its best-fit 2D PCA plane."""
    mu, vals, vecs = _pca_axes(P)
    X = P - mu
    # First two PCA axes span leaf plane.
    return X @ vecs[:, :2]


def _ellipse_perimeter_ramanujan(a: float, b: float) -> float:
    a, b = float(abs(a)), float(abs(b))
    if a < b:
        a, b = b, a
    if a <= 1e-12:
        return 0.0
    h = ((a - b) ** 2) / ((a + b) ** 2 + 1e-12)
    return float(math.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + math.sqrt(max(4.0 - 3.0 * h, 1e-12)))))


def _circular_angle_diff(a: np.ndarray, b: float | np.ndarray) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * (a - b))))


def _circular_mean(angles: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    if len(angles) == 0:
        return 0.0
    if weights is None:
        weights = np.ones_like(angles)
    z = np.sum(weights * np.exp(1j * angles))
    return float(np.angle(z) % (2 * np.pi))


def _find_circular_interval_min(hist: np.ndarray, i: int, j: int) -> float:
    """Minimum histogram value traveling forward from index i to j circularly."""
    n = len(hist)
    i = int(i) % n
    j = int(j) % n
    if i <= j:
        vals = hist[i:j + 1]
    else:
        vals = np.concatenate([hist[i:], hist[:j + 1]])
    if len(vals) == 0:
        return float(hist[min(i, j)])
    return float(np.min(vals))


def query_vertex_semantic_labels(model, vertices, cfg, chunk: int = 65536, direction=None):
    """
    Query a semantic NeRF/MLP at mesh vertices.

    Assumes model(points, dirs) returns (density, color, semantic_logits).
    Semantic labels are obtained by argmax over softmax(logits).
    """
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
            sem_cls = torch.softmax(sem_logits.float(), dim=-1).argmax(dim=-1)
            labels.append(sem_cls.detach().cpu())
    return torch.cat(labels, dim=0).numpy().astype(np.int32)




# -----------------------------------------------------------------------------
# Trait extraction pipeline
# -----------------------------------------------------------------------------

class _BaseTraitExtractor:
    def __init__(
        self,
        mesh,
        config: Optional[TraitExtractionConfig] = None,
        vertex_labels: Optional[np.ndarray] = None,
        class_ids: Optional[Dict[str, int]] = None,
    ):
        if o3d is None:
            raise ImportError("open3d is required for this extractor")
        self.mesh = mesh
        self.cfg = config or TraitExtractionConfig()
        self.vertex_labels = None if vertex_labels is None else np.asarray(vertex_labels)
        self.class_ids = class_ids or {}

        self.points_raw: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.labels_sampled: Optional[np.ndarray] = None
        self.points: np.ndarray = np.empty((0, 3), dtype=np.float64)  # canonical + scaled
        self.z_base: float = 0.0
        self.z_top: float = 0.0
        self.height: float = 0.0
        self.base_center_xy: np.ndarray = np.zeros(2, dtype=np.float64)

        self.stem_points: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.leaf_candidate_points: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.open_leaf_clusters: List[np.ndarray] = []
        self.leaf_table: List[Dict[str, Any]] = []
        self.angle_table: List[Dict[str, Any]] = []
        self.traits: Dict[str, Any] = {}
        self.diagnostics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, print_report: Optional[bool] = None) -> Dict[str, Any]:
        self._prepare_points()
        self._estimate_base_center()
        self._segment_stem_and_leaf_candidates()

        traits: Dict[str, Any] = {}
        traits.update(self._height_traits())
        traits.update(self._stem_traits())
        self._detect_open_leaf_clusters()
        traits.update(self._leaf_area_and_lai_traits())
        traits.update(self._canopy_volume_traits())
        traits.update(self._branch_angle_traits())
        traits.update(self._diagnostic_traits())

        self.traits = traits
        do_print = self.cfg.print_report if print_report is None else print_report
        if do_print:
            self.print_report(traits)
        return traits

    def print_report(self, traits: Optional[Dict[str, Any]] = None) -> None:
        traits = self.traits if traits is None else traits
        print("\n" + "=" * 72)
        print("PROTOCOL-AWARE 3D PLANT PHENOTYPIC TRAIT REPORT")
        print("=" * 72)
        for k, v in traits.items():
            if isinstance(v, float):
                print(f"{k:<58}: {v:.6f}")
            else:
                print(f"{k:<58}: {v}")
        print("=" * 72)
        print(f"Linear unit: {self.cfg.units}  |  unit_scale={self.cfg.unit_scale}")
        print("No manual trait values are used by this extractor.")
        print("Geometry-only organ traits remain estimates; semantics improve reliability.")
        print("=" * 72 + "\n")

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------
    def _prepare_points(self) -> None:
        mesh = self.mesh
        if not mesh.has_vertices():
            raise ValueError("Mesh has no vertices")

        V = np.asarray(mesh.vertices).astype(np.float64)
        labels = None

        # Prefer mesh surface sampling when available; assign nearest vertex labels.
        if mesh.has_triangles() and self.cfg.sample_points and self.cfg.sample_points > 0:
            try:
                pcd = mesh.sample_points_uniformly(number_of_points=int(self.cfg.sample_points))
                P = np.asarray(pcd.points).astype(np.float64)

                if self.vertex_labels is not None and len(self.vertex_labels) == len(V) and KDTree is not None:
                    tree = KDTree(V)
                    _, nn = tree.query(P, k=1)
                    labels = self.vertex_labels[nn]
                else:
                    labels = None
            except Exception:
                P = V.copy()
                labels = self.vertex_labels.copy() if self.vertex_labels is not None and len(self.vertex_labels) == len(V) else None
        else:
            P = V.copy()
            labels = self.vertex_labels.copy() if self.vertex_labels is not None and len(self.vertex_labels) == len(V) else None

        # Optional statistical outlier removal in raw coordinates.
        if self.cfg.remove_statistical_outliers and len(P) > 100 and o3d is not None:
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
        self.z_base, self.z_top, self.height = _robust_bounds_z(
            Q,
            self.cfg.robust_lower_percentile,
            self.cfg.robust_upper_percentile,
        )

    def _estimate_base_center(self) -> None:
        P = self.points
        z0, _, h = self.z_base, self.z_top, self.height
        lower = P[(P[:, 2] >= z0) & (P[:, 2] <= z0 + self.cfg.stem_lower_band_frac_height * h)]
        if len(lower) < 20:
            lower = P[P[:, 2] <= np.percentile(P[:, 2], 15)]
        if len(lower) < 20:
            lower = P

        center = np.median(lower[:, :2], axis=0)
        # Iteratively keep only the central lower core to avoid leaves/petioles.
        for _ in range(4):
            d = np.linalg.norm(lower[:, :2] - center[None, :], axis=1)
            r = np.percentile(d, self.cfg.stem_core_percentile_lower_band)
            core = lower[d <= r]
            if len(core) >= 10:
                center = np.median(core[:, :2], axis=0)
        self.base_center_xy = center

    def _segment_stem_and_leaf_candidates(self) -> None:
        P = self.points
        z0, _, h = self.z_base, self.z_top, self.height
        center = self.base_center_xy
        r = np.linalg.norm(P[:, :2] - center[None, :], axis=1)

        labels = self.labels_sampled
        have_labels = labels is not None and len(labels) == len(P)
        stem_id = self.class_ids.get("stem", None)
        leaf_id = self.class_ids.get("leaf", None)
        petiole_id = self.class_ids.get("petiole", None)

        if have_labels and stem_id is not None:
            stem_mask = labels == stem_id
            self.stem_points = P[stem_mask]
        else:
            # Geometry-only stem candidate: central lower core. Conservative.
            lower = P[(P[:, 2] >= z0) & (P[:, 2] <= z0 + 0.55 * h)]
            if len(lower) < 20:
                lower = P
            dl = np.linalg.norm(lower[:, :2] - center[None, :], axis=1)
            core_r = max(np.percentile(dl, self.cfg.stem_core_percentile_lower_band), 0.008 * h)
            mask = (r <= 1.8 * core_r) & (P[:, 2] <= z0 + 0.70 * h)
            self.stem_points = P[mask]

        if have_labels and leaf_id is not None:
            leaf_mask = labels == leaf_id
            if petiole_id is not None:
                # Petiole points can help branch-angle attachment, but not leaf area.
                pass
            self.leaf_candidate_points = P[leaf_mask]
        else:
            # Geometry-only leaf candidates: points away from central core and not
            # extremely close to the base. This intentionally over-includes; later
            # opened-leaf filtering removes small/noisy lobes.
            min_r = max(self.cfg.leaf_min_radius_frac_height * h, np.percentile(r, 18))
            z_min = z0 + self.cfg.leaf_exclude_lowest_frac_height * h
            leaf_mask = (r >= min_r) & (P[:, 2] >= z_min)
            self.leaf_candidate_points = P[leaf_mask]

    # ------------------------------------------------------------------
    # Traits
    # ------------------------------------------------------------------
    def _height_traits(self) -> Dict[str, Any]:
        u = self.cfg.units
        return {
            f"plant_height_{u}": float(self.height),
            f"plant_base_z_{u}": float(self.z_base),
            f"plant_top_z_{u}": float(self.z_top),
        }

    def _stem_traits(self) -> Dict[str, Any]:
        u = self.cfg.units
        P = self.points
        z0, _, h = self.z_base, self.z_top, self.height
        center = self.base_center_xy

        offset = self.cfg.stem_measure_offset_frac_height * h
        half = self.cfg.stem_slice_half_thickness_frac_height * h
        z_m = z0 + offset

        labels = self.labels_sampled
        have_labels = labels is not None and len(labels) == len(P) and self.class_ids.get("stem", None) is not None
        stem_id = self.class_ids.get("stem", None)

        if have_labels:
            candidate = P[labels == stem_id]
            mode = "semantic_stem"
        else:
            candidate = P
            mode = "geometry_central_core"

        S = candidate[np.abs(candidate[:, 2] - z_m) <= half]
        if len(S) < self.cfg.stem_min_points_in_slice:
            S = candidate[np.abs(candidate[:, 2] - z_m) <= 2.0 * half]
        if len(S) < self.cfg.stem_min_points_in_slice:
            return {
                "stem_measurement_status": "failed_too_few_points",
                "stem_measurement_protocol": "horizontal cross-section near base/soil emergence point",
                f"stem_measurement_height_above_base_{u}": float(offset),
                "stem_points_in_measurement_slice": int(len(S)),
                "stem_candidate_mode": mode,
            }

        d0 = np.linalg.norm(S[:, :2] - center[None, :], axis=1)

        if not have_labels:
            # Important: do not estimate stem radius from the entire slice.
            # The slice often contains petiole/leaf bases. Keep only the central
            # core first, then estimate radius from that core.
            gate = np.percentile(d0, self.cfg.stem_slice_core_percentile_geometry)
            S_core = S[d0 <= gate]
            if len(S_core) >= self.cfg.stem_min_points_in_slice:
                S = S_core
                center = np.median(S[:, :2], axis=0)
                d = np.linalg.norm(S[:, :2] - center[None, :], axis=1)
                radius_q = self.cfg.stem_radius_percentile_geometry
            else:
                d = d0
                radius_q = min(self.cfg.stem_radius_percentile_geometry, 50.0)
        else:
            center = np.median(S[:, :2], axis=0)
            d = np.linalg.norm(S[:, :2] - center[None, :], axis=1)
            radius_q = self.cfg.stem_radius_percentile_semantic

        radius = float(np.percentile(d, radius_q))
        radius = max(radius, 0.0)
        diameter = 2.0 * radius
        perim_circ = 2.0 * math.pi * radius
        area_circ = math.pi * radius * radius

        # Ellipse fit from robust covariance of slice/core.
        X = S[:, :2] - np.median(S[:, :2], axis=0)
        if len(X) >= 5:
            # Robustly trim farthest 10% before covariance.
            dd = np.linalg.norm(X, axis=1)
            trim = dd <= np.percentile(dd, 90)
            Xt = X[trim] if np.sum(trim) >= 5 else X
            _, vals, vecs = _pca_axes(np.column_stack([Xt, np.zeros(len(Xt))]))
            # More direct 2D covariance eigenvalues.
            C = np.cov(Xt.T)
            ev, _ = np.linalg.eigh(C)
            ev = np.sort(np.maximum(ev, 0.0))[::-1]
            # 80th percentile radial extent along principal axes.
            # The sqrt covariance alone underestimates boundary; use projected percentiles.
            evals2, evecs2 = np.linalg.eigh(C)
            order = np.argsort(evals2)[::-1]
            evecs2 = evecs2[:, order]
            proj = Xt @ evecs2
            a = float(np.percentile(np.abs(proj[:, 0]), radius_q))
            b = float(np.percentile(np.abs(proj[:, 1]), radius_q))
        else:
            a = b = radius
        if a < b:
            a, b = b, a
        perim_ellipse = _ellipse_perimeter_ramanujan(a, b)
        area_ellipse = math.pi * a * b

        # Confidence diagnostics
        d_full = np.linalg.norm(candidate[np.abs(candidate[:, 2] - z_m) <= max(half, 1e-12)][:, :2] - self.base_center_xy[None, :], axis=1)
        if len(d_full) >= 10:
            spread_ratio = float((np.percentile(d_full, 90) + 1e-12) / (np.percentile(d_full, 25) + 1e-12))
        else:
            spread_ratio = None
        confidence = "medium" if have_labels else "low_geometry_only"
        if spread_ratio is not None and spread_ratio > 4.0:
            confidence = "low_slice_contains_outward_structures"

        return {
            "stem_measurement_status": "ok",
            "stem_measurement_protocol": "horizontal cross-section near base/soil emergence point",
            f"stem_measurement_height_above_base_{u}": float(offset),
            f"stem_radius_base_circular_{u}": float(radius),
            f"stem_diameter_base_circular_{u}": float(diameter),
            f"stem_perimeter_base_circular_{u}": float(perim_circ),
            f"stem_cross_section_area_base_circular_{u}2": float(area_circ),
            f"stem_ellipse_semimajor_base_{u}": float(a),
            f"stem_ellipse_semiminor_base_{u}": float(b),
            f"stem_perimeter_base_ellipse_{u}": float(perim_ellipse),
            f"stem_cross_section_area_base_ellipse_{u}2": float(area_ellipse),
            f"stem_perimeter_base_recommended_{u}": float(perim_ellipse),
            "stem_perimeter_recommended_basis": "robust ellipse/core slice; circular value is also reported",
            "stem_points_in_measurement_slice": int(len(S)),
            "stem_candidate_mode": mode,
            "stem_measurement_confidence": confidence,
            "stem_slice_radial_spread_ratio_p90_p25": spread_ratio,
        }

    # ------------------------------------------------------------------
    # Leaf detection: exclusive assignment + fragment peak merging
    # ------------------------------------------------------------------
    def _detect_open_leaf_clusters(self) -> None:
        self.open_leaf_clusters = []
        self.leaf_table = []

        P = self.leaf_candidate_points
        if P is None or len(P) < max(50, self.cfg.leaf_min_points):
            self.diagnostics["leaf_detection_status"] = "failed_too_few_leaf_candidates"
            return
        if gaussian_filter1d is None or find_peaks is None:
            self.diagnostics["leaf_detection_status"] = "failed_scipy_required"
            return

        center = self.base_center_xy
        z0, _, h = self.z_base, self.z_top, self.height
        dxy = P[:, :2] - center[None, :]
        r = np.linalg.norm(dxy, axis=1)
        theta = (np.arctan2(dxy[:, 1], dxy[:, 0]) + 2.0 * np.pi) % (2.0 * np.pi)

        # Remove central/very low points before leaf-lobe detection.
        min_r = max(self.cfg.leaf_min_radius_frac_height * h, np.percentile(r, 10))
        valid = (r > min_r) & (P[:, 2] > z0 + self.cfg.leaf_exclude_lowest_frac_height * h)
        P = P[valid]
        r = r[valid]
        theta = theta[valid]
        if len(P) < max(50, self.cfg.leaf_min_points):
            self.diagnostics["leaf_detection_status"] = "failed_after_central_filter"
            return

        n_bins = int(self.cfg.leaf_hist_bins)
        bins = np.floor(theta / (2.0 * np.pi) * n_bins).astype(int)
        bins = np.clip(bins, 0, n_bins - 1)
        hist = np.zeros(n_bins, dtype=np.float64)

        # Weight outer lamina points more than central points, but avoid huge
        # dominance by one long/noisy leaf.
        weights = np.sqrt(np.maximum(r, 1e-12))
        np.add.at(hist, bins, weights)
        hist_s = gaussian_filter1d(hist, sigma=float(self.cfg.leaf_hist_sigma), mode="wrap")

        if hist_s.max() <= 1e-12:
            self.diagnostics["leaf_detection_status"] = "failed_empty_histogram"
            return

        min_dist = max(1, int(n_bins * self.cfg.leaf_min_peak_sep_deg / 360.0))
        peaks, props = find_peaks(
            hist_s,
            distance=min_dist,
            prominence=self.cfg.leaf_peak_prominence_ratio * hist_s.max(),
            height=self.cfg.leaf_peak_height_ratio * hist_s.max(),
        )
        raw_peak_count = int(len(peaks))
        if len(peaks) == 0:
            self.diagnostics["leaf_detection_status"] = "failed_no_peaks"
            self.diagnostics["leaf_raw_angular_peak_count"] = 0
            return

        peak_scores = hist_s[peaks]
        keep = peak_scores >= self.cfg.leaf_keep_peak_ratio * peak_scores.max()
        peaks = peaks[keep]
        peak_scores = peak_scores[keep]
        if len(peaks) == 0:
            self.diagnostics["leaf_detection_status"] = "failed_all_peaks_filtered"
            self.diagnostics["leaf_raw_angular_peak_count"] = raw_peak_count
            return

        # Merge nearby/shallow-valley peaks so one real leaf is not counted as fragments.
        merged_peaks = self._merge_fragment_peaks(peaks, peak_scores, hist_s)
        if len(merged_peaks) > self.cfg.leaf_max_reasonable_count:
            # Keep strongest peaks if reconstruction noise creates excessive lobes.
            scores = np.array([hist_s[int(p) % n_bins] for p in merged_peaks])
            order = np.argsort(scores)[::-1][: self.cfg.leaf_max_reasonable_count]
            merged_peaks = [merged_peaks[i] for i in order]

        peak_angles = np.array([(p % n_bins) / n_bins * 2.0 * np.pi for p in merged_peaks])

        # Exclusive assignment: each point goes to at most one nearest lobe.
        D = np.stack([_circular_angle_diff(theta, a) for a in peak_angles], axis=1)
        nearest = np.argmin(D, axis=1)
        nearest_dist = D[np.arange(len(theta)), nearest]
        max_window = math.radians(self.cfg.leaf_assignment_window_deg)
        assigned_mask = nearest_dist <= max_window

        candidate_clusters: List[np.ndarray] = []
        candidate_infos: List[Dict[str, Any]] = []
        for i, pa in enumerate(peak_angles):
            C = P[assigned_mask & (nearest == i)]
            if len(C) < self.cfg.leaf_min_points:
                continue
            area = self._leaf_area(C)
            radial_extent = float(np.percentile(np.linalg.norm(C[:, :2] - center[None, :], axis=1), 95) -
                                  np.percentile(np.linalg.norm(C[:, :2] - center[None, :], axis=1), 5))
            bbox_extent = float(np.linalg.norm(C.max(axis=0) - C.min(axis=0)))
            candidate_clusters.append(C)
            candidate_infos.append({
                "area": area,
                "radial_extent": radial_extent,
                "bbox_extent": bbox_extent,
                "angle_deg": float(math.degrees(pa)),
                "n_points": int(len(C)),
            })

        if len(candidate_clusters) == 0:
            self.diagnostics["leaf_detection_status"] = "failed_no_clusters_after_assignment"
            self.diagnostics["leaf_raw_angular_peak_count"] = raw_peak_count
            self.diagnostics["leaf_merged_peak_count"] = int(len(merged_peaks))
            return

        areas = np.array([info["area"] for info in candidate_infos])
        max_area = float(max(np.max(areas), 1e-12))
        kept_clusters: List[np.ndarray] = []
        kept_infos: List[Dict[str, Any]] = []
        rejected = 0
        for C, info in zip(candidate_clusters, candidate_infos):
            area = info["area"]
            radial_extent = info["radial_extent"]
            area_ok_rel = area >= self.cfg.leaf_open_min_area_frac_of_largest * max_area
            area_ok_abs = area >= self.cfg.leaf_open_min_area_frac_height2 * (h ** 2)
            extent_ok = radial_extent >= self.cfg.leaf_open_min_radial_extent_frac_height * h
            if area_ok_rel and area_ok_abs and extent_ok:
                kept_clusters.append(C)
                kept_infos.append(info)
            else:
                rejected += 1

        # Sort around plant for stable leaf IDs.
        def _cluster_angle(C):
            c = C[:, :2].mean(axis=0) - center
            return float((math.atan2(c[1], c[0]) + 2 * math.pi) % (2 * math.pi))

        order = np.argsort([_cluster_angle(C) for C in kept_clusters])
        self.open_leaf_clusters = [kept_clusters[i] for i in order]

        self.diagnostics.update({
            "leaf_detection_status": "ok" if self.open_leaf_clusters else "failed_all_open_leaf_filters",
            "leaf_raw_angular_peak_count": raw_peak_count,
            "leaf_kept_raw_peak_count": int(len(peaks)),
            "leaf_merged_peak_count": int(len(merged_peaks)),
            "leaf_candidate_cluster_count_before_open_filter": int(len(candidate_clusters)),
            "leaf_rejected_small_or_early_count": int(rejected),
            "leaf_exclusive_assignment": True,
            "leaf_no_duplicate_points_between_clusters": True,
        })

    def _merge_fragment_peaks(self, peaks: np.ndarray, scores: np.ndarray, hist_s: np.ndarray) -> List[int]:
        """
        Merge angular peaks that are likely fragments of the same biological leaf.

        Earlier merging was too aggressive because it merged peaks whenever they were within
        a fixed angular threshold OR the valley was shallow. In noisy NeRF meshes,
        neighboring real leaves can be closer than 18 degrees, so this caused
        under-counting.

        This method uses a staged rule:
            - very close peaks are merged directly;
            - moderately close peaks are merged only when the valley between them
              is shallow;
            - wider peaks are kept separate unless the valley is extremely shallow.

        This does not use manual/ground-truth leaf count.
        """
        n = len(hist_s)
        if len(peaks) == 0:
            return []

        # Sort peaks circularly by angle/index.
        order = np.argsort(peaks)
        groups = [[int(peaks[i]) % n] for i in order]

        close_deg = max(2.0, 0.55 * float(self.cfg.leaf_fragment_merge_angle_deg))
        mid_deg = float(self.cfg.leaf_fragment_merge_angle_deg)
        wide_deg = 1.65 * float(self.cfg.leaf_fragment_merge_angle_deg)
        mid_valley_ratio = float(self.cfg.leaf_valley_merge_ratio)
        wide_valley_ratio = min(0.92, mid_valley_ratio + 0.18)

        def rep_of_group(g: List[int]) -> int:
            idx = np.array(g, dtype=int) % n
            ang = idx / n * 2.0 * np.pi
            w = hist_s[idx]
            return int(round(_circular_mean(ang, w) / (2.0 * np.pi) * n)) % n

        def should_merge_groups(g1: List[int], g2: List[int]) -> bool:
            p1 = rep_of_group(g1)
            p2 = rep_of_group(g2)
            a1 = p1 / n * 2.0 * np.pi
            a2 = p2 / n * 2.0 * np.pi
            sep_deg = math.degrees(float(_circular_angle_diff(np.array([a1]), a2)[0]))

            # Evaluate both circular directions and use the lower valley.
            valley_12 = _find_circular_interval_min(hist_s, p1, p2)
            valley_21 = _find_circular_interval_min(hist_s, p2, p1)
            valley = max(valley_12, valley_21) if sep_deg < 180.0 else min(valley_12, valley_21)
            low_peak = min(float(hist_s[p1]), float(hist_s[p2])) + 1e-12
            valley_ratio = float(valley / low_peak)

            if sep_deg <= close_deg:
                return True
            if sep_deg <= mid_deg and valley_ratio >= mid_valley_ratio:
                return True
            if sep_deg <= wide_deg and valley_ratio >= wide_valley_ratio:
                return True
            return False

        changed = True
        while changed and len(groups) > 1:
            changed = False
            new_groups: List[List[int]] = []
            used = [False] * len(groups)
            m = len(groups)

            for idx in range(m):
                if used[idx]:
                    continue
                j = (idx + 1) % m

                # For the last group, allow circular last-first merge only if
                # the first group has not already been used.
                if idx == m - 1 and used[0]:
                    new_groups.append(groups[idx])
                    used[idx] = True
                    continue

                if m > 1 and not used[j] and should_merge_groups(groups[idx], groups[j]):
                    merged = groups[idx] + groups[j]
                    used[idx] = True
                    used[j] = True
                    new_groups.append(merged)
                    changed = True
                else:
                    new_groups.append(groups[idx])
                    used[idx] = True

            # Sort groups again by representative angle after each pass.
            groups = sorted(new_groups, key=rep_of_group)

        representatives = []
        for g in groups:
            representatives.append(rep_of_group(g))
        return representatives

    def _leaf_area(self, C: np.ndarray) -> float:
        if len(C) < 3:
            return 0.0
        if self.cfg.leaf_area_method == "pca_hull":
            P2 = _project_leaf_to_pca_plane(C)
            return _convex_hull_area_2d(P2)
        # fallback: horizontal projected hull
        return _convex_hull_area_2d(C[:, :2])

    def _leaf_area_and_lai_traits(self) -> Dict[str, Any]:
        u = self.cfg.units
        total_area = 0.0
        self.leaf_table = []

        for i, C in enumerate(self.open_leaf_clusters, start=1):
            area = self._leaf_area(C)
            center = C.mean(axis=0)
            radial_extent = float(np.percentile(np.linalg.norm(C[:, :2] - self.base_center_xy[None, :], axis=1), 95) -
                                  np.percentile(np.linalg.norm(C[:, :2] - self.base_center_xy[None, :], axis=1), 5))
            total_area += area
            self.leaf_table.append({
                "leaf_id": i,
                f"leaf_area_{u}2": float(area),
                f"leaf_center_x_{u}": float(center[0]),
                f"leaf_center_y_{u}": float(center[1]),
                f"leaf_center_z_{u}": float(center[2]),
                f"leaf_radial_extent_{u}": float(radial_extent),
                "n_points": int(len(C)),
            })

        proj_area = _convex_hull_area_2d(self.points[:, :2]) if self.points is not None and len(self.points) >= 3 else 0.0
        lai_proxy = float(total_area / proj_area) if proj_area > 1e-12 else None
        if self.cfg.ground_area is not None and self.cfg.ground_area > 0:
            lai_strict = float(total_area / float(self.cfg.ground_area))
            lai_status = "computed_using_provided_independent_ground_area"
        else:
            lai_strict = None
            lai_status = "not_computed_ground_area_not_provided"

        return {
            "leaf_detection_status": self.diagnostics.get("leaf_detection_status", None),
            "leaf_count_definition": "estimated fully opened leaves; tiny early-stage lobes filtered by area and radial extent",
            "open_leaf_count_estimated": int(len(self.open_leaf_clusters)),
            "leaf_raw_angular_peak_count": self.diagnostics.get("leaf_raw_angular_peak_count", None),
            "leaf_merged_peak_count": self.diagnostics.get("leaf_merged_peak_count", None),
            "leaf_candidate_cluster_count_before_open_filter": self.diagnostics.get("leaf_candidate_cluster_count_before_open_filter", None),
            "leaf_rejected_small_or_early_count": self.diagnostics.get("leaf_rejected_small_or_early_count", None),
            "leaf_exclusive_assignment": self.diagnostics.get("leaf_exclusive_assignment", None),
            f"total_leaf_area_estimated_{u}2": float(total_area),
            f"mean_leaf_area_estimated_{u}2": float(total_area / len(self.open_leaf_clusters)) if self.open_leaf_clusters else None,
            f"projected_canopy_area_{u}2": float(proj_area),
            "LAI_strict_total_leaf_area_over_ground_area": lai_strict,
            "LAI_strict_status": lai_status,
            "LAI_proxy_total_leaf_area_over_projected_canopy_area": lai_proxy,
        }

    def _canopy_volume_traits(self) -> Dict[str, Any]:
        u = self.cfg.units
        P = self.points
        h = self.height
        voxel = max(self.cfg.voxel_size_frac_height * h, 1e-8)
        idx = np.floor((P - P.min(axis=0)[None, :]) / voxel).astype(np.int64)
        occupied = len(np.unique(idx, axis=0)) * (voxel ** 3)

        hull_vol = None
        if ConvexHull is not None and len(P) >= 4:
            try:
                hull_vol = float(ConvexHull(P).volume)
            except Exception:
                hull_vol = None

        return {
            f"canopy_occupied_voxel_volume_{u}3": float(occupied),
            f"canopy_convex_hull_envelope_volume_{u}3": hull_vol,
            f"voxel_size_used_{u}": float(voxel),
        }

    def _branch_angle_traits(self) -> Dict[str, Any]:
        u = self.cfg.units
        self.angle_table = []
        if self.stem_points is None or len(self.stem_points) < 10 or not self.open_leaf_clusters:
            return {
                "branch_angle_status": "not_computed_insufficient_stem_or_leaf_clusters",
                "branch_angle_count": 0,
            }

        S = self.stem_points
        _, _, h = self.z_base, self.z_top, self.height
        angles_raw = []
        angles_acute = []

        for i, C in enumerate(self.open_leaf_clusters, start=1):
            if len(C) < self.cfg.branch_min_points:
                continue
            # Attachment is nearest leaf point to stem centerline/core.
            # Find leaf points closest to stem points.
            if KDTree is not None:
                tree = KDTree(S)
                dist, nn = tree.query(C, k=1)
                j = int(np.argmin(dist))
                attach_leaf = C[j]
                attach_stem = S[int(nn[j])]
            else:
                # slower fallback
                D = np.linalg.norm(C[:, None, :] - S[None, :, :], axis=2)
                j, k = np.unravel_index(np.argmin(D), D.shape)
                attach_leaf = C[j]
                attach_stem = S[k]

            # Local stem axis from nearby stem points.
            rad = self.cfg.branch_attachment_search_frac_height * h
            near_stem = S[np.linalg.norm(S - attach_stem[None, :], axis=1) <= rad]
            if len(near_stem) < 5:
                near_stem = S
            _, vals, vecs = _pca_axes(near_stem)
            stem_axis = _unit(vecs[:, 0])
            # Prefer vertical direction sign.
            if stem_axis[2] < 0:
                stem_axis = -stem_axis

            # Proximal leaf/petiole direction: use only the part of the leaf near
            # attachment, not the curved outer lamina.
            d = np.linalg.norm(C - attach_leaf[None, :], axis=1)
            thresh = np.percentile(d, max(5.0, min(60.0, self.cfg.branch_proximal_leaf_frac * 100.0)))
            proximal = C[d <= thresh]
            if len(proximal) < 5:
                proximal = C
            leaf_vec = proximal.mean(axis=0) - attach_stem
            if np.linalg.norm(leaf_vec) < 1e-9:
                # fallback to cluster center
                leaf_vec = C.mean(axis=0) - attach_stem
            leaf_axis = _unit(leaf_vec)
            if np.linalg.norm(leaf_axis) < 1e-9:
                continue

            cosv = float(np.clip(np.dot(stem_axis, leaf_axis), -1.0, 1.0))
            angle = math.degrees(math.acos(cosv))
            angle_acute = min(angle, 180.0 - angle)
            angles_raw.append(angle)
            angles_acute.append(angle_acute)
            self.angle_table.append({
                "leaf_id": i,
                "branch_angle_raw_deg": float(angle),
                "branch_angle_acute_deg": float(angle_acute),
                f"attachment_x_{u}": float(attach_stem[0]),
                f"attachment_y_{u}": float(attach_stem[1]),
                f"attachment_z_{u}": float(attach_stem[2]),
            })

        if not angles_raw:
            return {
                "branch_angle_status": "not_computed_no_valid_angles",
                "branch_angle_count": 0,
            }
        return {
            "branch_angle_status": "ok",
            "branch_angle_definition": "angle between local stem axis and proximal leaf/petiole direction near attachment",
            "branch_angle_count": int(len(angles_raw)),
            "mean_branch_angle_raw_deg": float(np.mean(angles_raw)),
            "median_branch_angle_raw_deg": float(np.median(angles_raw)),
            "mean_branch_angle_acute_deg": float(np.mean(angles_acute)),
            "median_branch_angle_acute_deg": float(np.median(angles_acute)),
        }

    def _diagnostic_traits(self) -> Dict[str, Any]:
        mode = "semantic_assisted" if self.labels_sampled is not None and ("leaf" in self.class_ids or "stem" in self.class_ids) else "geometry_only"
        return {
            "trait_extraction_mode": mode,
            "n_points_used": int(len(self.points)) if self.points is not None else 0,
            "n_stem_points": int(len(self.stem_points)) if self.stem_points is not None else 0,
            "n_leaf_candidate_points": int(len(self.leaf_candidate_points)) if self.leaf_candidate_points is not None else 0,
            "reconstruction_noise_note": "organ-level traits may be biased by mesh holes, ghost geometry, fused leaves, or weak stem/leaf separation",
        }

    # ------------------------------------------------------------------
    # Debug plots
    # ------------------------------------------------------------------
    def plot_leaf_debug(self, ax=None, show: bool = True):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
        if self.leaf_candidate_points is not None:
            P = self.leaf_candidate_points
            ax.scatter(P[:, 0], P[:, 1], s=0.4, alpha=0.15, label="leaf candidates")
        for i, C in enumerate(self.open_leaf_clusters, start=1):
            ax.scatter(C[:, 0], C[:, 1], s=2, label=f"leaf {i}")
        if self.stem_points is not None and len(self.stem_points):
            S = self.stem_points
            ax.scatter(S[:, 0], S[:, 1], s=1, c="black", alpha=0.35, label="stem/core")
        ax.scatter([self.base_center_xy[0]], [self.base_center_xy[1]], marker="x", s=80, c="black", label="base center")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Opened leaf clusters: {len(self.open_leaf_clusters)}")
        ax.legend(markerscale=4, fontsize=8, loc="best")
        if show:
            plt.show()
        return ax

    def plot_leaf_histogram_debug(self, show: bool = True):
        import matplotlib.pyplot as plt
        P = self.leaf_candidate_points
        if P is None or len(P) == 0:
            print("No leaf candidate points.")
            return None
        center = self.base_center_xy
        dxy = P[:, :2] - center[None, :]
        r = np.linalg.norm(dxy, axis=1)
        theta = (np.arctan2(dxy[:, 1], dxy[:, 0]) + 2 * np.pi) % (2 * np.pi)
        n_bins = int(self.cfg.leaf_hist_bins)
        bins = np.floor(theta / (2 * np.pi) * n_bins).astype(int)
        bins = np.clip(bins, 0, n_bins - 1)
        hist = np.zeros(n_bins)
        np.add.at(hist, bins, np.sqrt(np.maximum(r, 1e-12)))
        hist_s = gaussian_filter1d(hist, sigma=float(self.cfg.leaf_hist_sigma), mode="wrap") if gaussian_filter1d is not None else hist
        deg = np.linspace(0, 360, n_bins, endpoint=False)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(deg, hist, alpha=0.35, label="raw angular density")
        ax.plot(deg, hist_s, label="smoothed angular density")
        ax.set_xlabel("angle around stem/base (deg)")
        ax.set_ylabel("weighted point density")
        ax.set_title("Leaf angular-density debug")
        ax.legend()
        if show:
            plt.show()
        return ax

    def plot_stem_slice_debug(self, show: bool = True):
        import matplotlib.pyplot as plt
        P = self.points
        z0, _, h = self.z_base, self.z_top, self.height
        offset = self.cfg.stem_measure_offset_frac_height * h
        half = self.cfg.stem_slice_half_thickness_frac_height * h
        z_m = z0 + offset
        S = P[np.abs(P[:, 2] - z_m) <= half]
        fig, ax = plt.subplots(figsize=(6, 6))
        if len(S):
            ax.scatter(S[:, 0], S[:, 1], s=3, alpha=0.5, label="measurement slice")
        ax.scatter([self.base_center_xy[0]], [self.base_center_xy[1]], marker="x", s=80, c="black", label="base center")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Stem/base measurement slice debug")
        ax.legend()
        if show:
            plt.show()
        return ax

class _StemLeafConsensusExtractor(_BaseTraitExtractor):
    """Protocol-aware 3D trait extractor.

    This subclass does not use any manual/ground-truth trait value. Manual values
    should only be used after extraction for validation/error analysis.
    """

    def print_report(self, traits: Optional[Dict[str, Any]] = None) -> None:
        traits = self.traits if traits is None else traits
        print("\n" + "=" * 72)
        print("PROTOCOL-AWARE 3D PLANT PHENOTYPIC TRAIT REPORT")
        print("=" * 72)
        for k, v in traits.items():
            if isinstance(v, float):
                print(f"{k:<58}: {v:.6f}")
            else:
                print(f"{k:<58}: {v}")
        print("=" * 72)
        print(f"Linear unit: {self.cfg.units}  |  unit_scale={self.cfg.unit_scale}")
        print("No manual trait values are used by this extractor.")
        print("Geometry-only organ traits remain estimates; semantics improve reliability.")
        print("=" * 72 + "\n")

    def _stem_traits(self) -> Dict[str, Any]:
        out = super()._stem_traits()
        u = self.cfg.units

        circ_key = f"stem_perimeter_base_circular_{u}"
        ell_key = f"stem_perimeter_base_ellipse_{u}"
        rec_key = f"stem_perimeter_base_recommended_{u}"

        circ = out.get(circ_key, None)
        ell = out.get(ell_key, None)
        conf = str(out.get("stem_measurement_confidence", ""))

        if isinstance(circ, (float, int)) and isinstance(ell, (float, int)) and circ > 0 and ell > 0:
            lo = float(min(circ, ell))
            hi = float(max(circ, ell))
            mid = float(math.sqrt(lo * hi))

            out[f"stem_perimeter_base_lower_bound_{u}"] = lo
            out[f"stem_perimeter_base_upper_bound_{u}"] = hi

            # In geometry-only mode the ellipse can underestimate when the slice
            # is incomplete/asymmetric, while the circular perimeter can
            # overestimate when petiole/base fragments are included. A geometric
            # mean is a neutral compromise and does not use ground truth.
            if "low" in conf or out.get("stem_candidate_mode") == "geometry_central_core":
                out[rec_key] = mid
                out[f"stem_equivalent_radius_recommended_{u}"] = float(mid / (2.0 * math.pi))
                out[f"stem_equivalent_diameter_recommended_{u}"] = float(mid / math.pi)
                out["stem_perimeter_recommended_basis"] = (
                    "geometry-only low-confidence slice: recommended value is the geometric mean "
                    "of ellipse/core and circular/core estimates; lower/upper bounds are reported"
                )
            else:
                out[rec_key] = float(ell)
                out[f"stem_equivalent_radius_recommended_{u}"] = float(ell / (2.0 * math.pi))
                out[f"stem_equivalent_diameter_recommended_{u}"] = float(ell / math.pi)

            if self.height > 1e-12:
                out["plant_height_to_stem_perimeter_recommended_ratio"] = float(self.height / out[rec_key])
                out["plant_height_to_stem_perimeter_circular_ratio"] = float(self.height / circ)
                out["plant_height_to_stem_perimeter_ellipse_ratio"] = float(self.height / ell)

        return out

    # ------------------------------------------------------------------
    # Leaf detection consensus: multi-scale consensus
    # ------------------------------------------------------------------
    def _detect_open_leaf_clusters(self) -> None:
        self.open_leaf_clusters = []
        self.leaf_table = []

        P = self.leaf_candidate_points
        if P is None or len(P) < max(50, self.cfg.leaf_min_points):
            self.diagnostics["leaf_detection_status"] = "failed_too_few_leaf_candidates"
            return
        if gaussian_filter1d is None or find_peaks is None:
            self.diagnostics["leaf_detection_status"] = "failed_scipy_required"
            return

        profiles = self._leaf_multiscale_profiles()
        results = []
        for profile in profiles:
            result = self._detect_leaf_clusters_single_profile(profile)
            if result is not None and len(result["clusters"]) > 0:
                results.append(result)

        if not results:
            self.diagnostics["leaf_detection_status"] = "failed_no_valid_multiscale_leaf_solution"
            self.diagnostics["leaf_multiscale_counts"] = []
            return

        counts = np.array([len(r["clusters"]) for r in results], dtype=np.float64)
        median_count = float(np.median(counts))

        # Prefer a median-scale solution, but avoid pathological solutions with
        # extremely high area imbalance. This is model-based, not manual-count based.
        best_idx = 0
        best_score = float("inf")
        for idx, r in enumerate(results):
            count = len(r["clusters"])
            count_score = abs(count - median_count)

            areas = np.array([max(self._leaf_area(C), 1e-12) for C in r["clusters"]], dtype=np.float64)
            area_cv = float(np.std(areas) / (np.mean(areas) + 1e-12)) if len(areas) > 1 else 0.0

            # Penalize extreme over-fragmentation where many tiny pieces survive.
            small_frac = float(np.mean(areas < 0.12 * np.max(areas))) if len(areas) > 0 else 0.0

            # A small preference for the named balanced profile if scores tie.
            balance_bonus = -0.05 if r["profile_name"] == "balanced" else 0.0

            score = count_score + 0.25 * area_cv + 0.50 * small_frac + balance_bonus
            if score < best_score:
                best_score = score
                best_idx = idx

        selected = results[best_idx]
        self.open_leaf_clusters = selected["clusters"]

        self.diagnostics.update(selected["diagnostics"])
        self.diagnostics["leaf_detection_status"] = "ok"
        self.diagnostics["leaf_detection_method"] = "multiscale_consensus"
        self.diagnostics["leaf_selected_profile"] = selected["profile_name"]
        self.diagnostics["leaf_multiscale_counts"] = [
            {"profile": r["profile_name"], "count": int(len(r["clusters"]))}
            for r in results
        ]
        self.diagnostics["leaf_multiscale_median_count"] = median_count
        self.diagnostics["leaf_exclusive_assignment"] = True
        self.diagnostics["leaf_no_duplicate_points_between_clusters"] = True

    def _leaf_multiscale_profiles(self) -> List[Dict[str, float | str]]:
        """A priori profiles from fine to coarse. No manual leaf count is used."""
        return [
            {
                "name": "very_fine",
                "sigma": 0.85,
                "min_sep_deg": 5.5,
                "prom": 0.0045,
                "height": 0.0045,
                "keep": 0.010,
                "merge_angle_deg": 7.5,
                "valley": 0.68,
                "assign_window_deg": 25.0,
                "min_points": max(18, int(0.75 * self.cfg.leaf_min_points)),
                "area_rel": 0.040,
                "area_h2": 0.00035,
                "extent_h": 0.030,
            },
            {
                "name": "fine",
                "sigma": 1.00,
                "min_sep_deg": 6.0,
                "prom": 0.0060,
                "height": 0.0060,
                "keep": 0.015,
                "merge_angle_deg": 10.0,
                "valley": 0.62,
                "assign_window_deg": 28.0,
                "min_points": max(20, int(0.85 * self.cfg.leaf_min_points)),
                "area_rel": 0.050,
                "area_h2": 0.00045,
                "extent_h": 0.035,
            },
            {
                "name": "balanced",
                "sigma": 1.18,
                "min_sep_deg": 7.0,
                "prom": 0.0080,
                "height": 0.0080,
                "keep": 0.020,
                "merge_angle_deg": 14.0,
                "valley": 0.54,
                "assign_window_deg": 31.0,
                "min_points": max(22, int(self.cfg.leaf_min_points)),
                "area_rel": 0.060,
                "area_h2": 0.00055,
                "extent_h": 0.040,
            },
            {
                "name": "coarse",
                "sigma": 1.38,
                "min_sep_deg": 8.0,
                "prom": 0.0100,
                "height": 0.0100,
                "keep": 0.025,
                "merge_angle_deg": 16.0,
                "valley": 0.49,
                "assign_window_deg": 34.0,
                "min_points": max(25, int(1.1 * self.cfg.leaf_min_points)),
                "area_rel": 0.070,
                "area_h2": 0.00065,
                "extent_h": 0.045,
            },
            {
                "name": "very_coarse",
                "sigma": 1.65,
                "min_sep_deg": 10.0,
                "prom": 0.0120,
                "height": 0.0120,
                "keep": 0.030,
                "merge_angle_deg": 18.0,
                "valley": 0.44,
                "assign_window_deg": 38.0,
                "min_points": max(30, int(1.2 * self.cfg.leaf_min_points)),
                "area_rel": 0.080,
                "area_h2": 0.00080,
                "extent_h": 0.050,
            },
        ]

    def _detect_leaf_clusters_single_profile(self, profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        P0 = self.leaf_candidate_points
        center = self.base_center_xy
        z0, _, h = self.z_base, self.z_top, self.height

        dxy0 = P0[:, :2] - center[None, :]
        r0 = np.linalg.norm(dxy0, axis=1)
        theta0 = (np.arctan2(dxy0[:, 1], dxy0[:, 0]) + 2.0 * np.pi) % (2.0 * np.pi)

        min_r = max(self.cfg.leaf_min_radius_frac_height * h, np.percentile(r0, 10))
        valid = (r0 > min_r) & (P0[:, 2] > z0 + self.cfg.leaf_exclude_lowest_frac_height * h)
        P = P0[valid]
        r = r0[valid]
        theta = theta0[valid]

        if len(P) < max(50, int(profile["min_points"])):
            return None

        n_bins = int(self.cfg.leaf_hist_bins)
        bins = np.floor(theta / (2.0 * np.pi) * n_bins).astype(int)
        bins = np.clip(bins, 0, n_bins - 1)
        hist = np.zeros(n_bins, dtype=np.float64)

        # Radial weight emphasizes lamina tips but does not let long noisy arms dominate.
        weights = np.sqrt(np.maximum(r, 1e-12))
        np.add.at(hist, bins, weights)
        hist_s = gaussian_filter1d(hist, sigma=float(profile["sigma"]), mode="wrap")

        if hist_s.max() <= 1e-12:
            return None

        min_dist = max(1, int(n_bins * float(profile["min_sep_deg"]) / 360.0))
        peaks, props = find_peaks(
            hist_s,
            distance=min_dist,
            prominence=float(profile["prom"]) * hist_s.max(),
            height=float(profile["height"]) * hist_s.max(),
        )
        raw_peak_count = int(len(peaks))
        if len(peaks) == 0:
            return None

        peak_scores = hist_s[peaks]
        keep = peak_scores >= float(profile["keep"]) * peak_scores.max()
        peaks = peaks[keep]
        peak_scores = peak_scores[keep]
        if len(peaks) == 0:
            return None

        merged_peaks = self._merge_fragment_peaks_consensus(
            peaks=peaks,
            hist_s=hist_s,
            merge_angle_deg=float(profile["merge_angle_deg"]),
            valley_ratio=float(profile["valley"]),
        )

        if len(merged_peaks) > self.cfg.leaf_max_reasonable_count:
            scores = np.array([hist_s[int(p) % n_bins] for p in merged_peaks])
            order = np.argsort(scores)[::-1][: self.cfg.leaf_max_reasonable_count]
            merged_peaks = [merged_peaks[i] for i in order]

        if not merged_peaks:
            return None

        peak_angles = np.array([(p % n_bins) / n_bins * 2.0 * np.pi for p in merged_peaks])

        # Exclusive assignment: each point belongs to at most one leaf lobe.
        D = np.stack([_circular_angle_diff(theta, a) for a in peak_angles], axis=1)
        nearest = np.argmin(D, axis=1)
        nearest_dist = D[np.arange(len(theta)), nearest]
        assigned_mask = nearest_dist <= math.radians(float(profile["assign_window_deg"]))

        candidate_clusters: List[np.ndarray] = []
        candidate_infos: List[Dict[str, Any]] = []
        for i, pa in enumerate(peak_angles):
            C = P[assigned_mask & (nearest == i)]
            if len(C) < int(profile["min_points"]):
                continue

            area = self._leaf_area(C)
            rr = np.linalg.norm(C[:, :2] - center[None, :], axis=1)
            radial_extent = float(np.percentile(rr, 95) - np.percentile(rr, 5))
            bbox_extent = float(np.linalg.norm(C.max(axis=0) - C.min(axis=0)))

            candidate_clusters.append(C)
            candidate_infos.append({
                "area": float(area),
                "radial_extent": radial_extent,
                "bbox_extent": bbox_extent,
                "angle_deg": float(math.degrees(pa)),
                "n_points": int(len(C)),
            })

        if not candidate_clusters:
            return None

        areas = np.array([info["area"] for info in candidate_infos], dtype=np.float64)
        max_area = float(max(np.max(areas), 1e-12))

        kept_clusters: List[np.ndarray] = []
        rejected = 0
        for C, info in zip(candidate_clusters, candidate_infos):
            area = info["area"]
            radial_extent = info["radial_extent"]
            area_ok_rel = area >= float(profile["area_rel"]) * max_area
            area_ok_abs = area >= float(profile["area_h2"]) * (h ** 2)
            extent_ok = radial_extent >= float(profile["extent_h"]) * h
            if area_ok_rel and area_ok_abs and extent_ok:
                kept_clusters.append(C)
            else:
                rejected += 1

        if not kept_clusters:
            return None

        # Sort around plant for stable IDs.
        def _cluster_angle(C):
            c = C[:, :2].mean(axis=0) - center
            return float((math.atan2(c[1], c[0]) + 2 * math.pi) % (2 * math.pi))

        order = np.argsort([_cluster_angle(C) for C in kept_clusters])
        kept_clusters = [kept_clusters[i] for i in order]

        diagnostics = {
            "leaf_raw_angular_peak_count": raw_peak_count,
            "leaf_kept_raw_peak_count": int(len(peaks)),
            "leaf_merged_peak_count": int(len(merged_peaks)),
            "leaf_candidate_cluster_count_before_open_filter": int(len(candidate_clusters)),
            "leaf_rejected_small_or_early_count": int(rejected),
            "leaf_selected_profile_raw": dict(profile),
        }

        return {
            "profile_name": str(profile["name"]),
            "clusters": kept_clusters,
            "diagnostics": diagnostics,
        }

    def _merge_fragment_peaks_consensus(
        self,
        peaks: np.ndarray,
        hist_s: np.ndarray,
        merge_angle_deg: float,
        valley_ratio: float,
    ) -> List[int]:
        n = len(hist_s)
        if len(peaks) == 0:
            return []

        order = np.argsort(peaks)
        groups = [[int(peaks[i]) % n] for i in order]

        close_deg = max(2.0, 0.55 * float(merge_angle_deg))
        mid_deg = float(merge_angle_deg)
        wide_deg = 1.45 * float(merge_angle_deg)
        mid_valley_ratio = float(valley_ratio)
        wide_valley_ratio = min(0.90, mid_valley_ratio + 0.16)

        def rep_of_group(g: List[int]) -> int:
            idx = np.array(g, dtype=int) % n
            ang = idx / n * 2.0 * np.pi
            w = hist_s[idx]
            return int(round(_circular_mean(ang, w) / (2.0 * np.pi) * n)) % n

        def should_merge(g1: List[int], g2: List[int]) -> bool:
            p1 = rep_of_group(g1)
            p2 = rep_of_group(g2)
            a1 = p1 / n * 2.0 * np.pi
            a2 = p2 / n * 2.0 * np.pi
            sep_deg = math.degrees(float(_circular_angle_diff(np.array([a1]), a2)[0]))

            # Merge based on the shallow valley along the shorter circular interval.
            valley_12 = _find_circular_interval_min(hist_s, p1, p2)
            valley_21 = _find_circular_interval_min(hist_s, p2, p1)
            valley = max(valley_12, valley_21) if sep_deg < 180.0 else min(valley_12, valley_21)
            low_peak = min(float(hist_s[p1]), float(hist_s[p2])) + 1e-12
            vr = float(valley / low_peak)

            if sep_deg <= close_deg:
                return True
            if sep_deg <= mid_deg and vr >= mid_valley_ratio:
                return True
            if sep_deg <= wide_deg and vr >= wide_valley_ratio:
                return True
            return False

        changed = True
        while changed and len(groups) > 1:
            changed = False
            new_groups: List[List[int]] = []
            used = [False] * len(groups)
            m = len(groups)
            for idx in range(m):
                if used[idx]:
                    continue
                j = (idx + 1) % m
                if idx == m - 1 and used[0]:
                    new_groups.append(groups[idx])
                    used[idx] = True
                    continue
                if m > 1 and not used[j] and should_merge(groups[idx], groups[j]):
                    new_groups.append(groups[idx] + groups[j])
                    used[idx] = True
                    used[j] = True
                    changed = True
                else:
                    new_groups.append(groups[idx])
                    used[idx] = True
            groups = sorted(new_groups, key=rep_of_group)

        return [rep_of_group(g) for g in groups]

class _HeightAwareTraitExtractor(_StemLeafConsensusExtractor):
    """Protocol-aware 3D trait extractor.

    Adds both height definitions required by the measurement protocol:
        1. observed vertical height of the reconstructed pose,
        2. straightened/stretched height proxy estimated from the 3D model.

    The straightened height is NOT calibrated using manual plant height. It is a
    geometric estimate from the reconstructed model only.
    """

    def print_report(self, traits: Optional[Dict[str, Any]] = None) -> None:
        traits = self.traits if traits is None else traits
        print("\n" + "=" * 72)
        print("PROTOCOL-AWARE 3D PLANT PHENOTYPIC TRAIT REPORT")
        print("=" * 72)
        for k, v in traits.items():
            if isinstance(v, float):
                print(f"{k:<58}: {v:.6f}")
            else:
                print(f"{k:<58}: {v}")
        print("=" * 72)
        print(f"Linear unit: {self.cfg.units}  |  unit_scale={self.cfg.unit_scale}")
        print("No manual trait values are used by this extractor.")
        print("Both observed vertical height and model-estimated straightened height are reported.")
        print("Geometry-only organ traits remain estimates; semantics improve reliability.")
        print("=" * 72 + "\n")

    def _height_traits(self) -> Dict[str, Any]:
        """Return both observed and straightened plant-height definitions.

        observed_vertical_height:
            Robust vertical height in the reconstructed/bent pose.

        straightened_centerline_height:
            Approximate length obtained by following a robust slice-wise plant
            centerline from the base/soil emergence point to the top region. It
            is intended to mimic the manual protocol where a bent plant is
            gently straightened before measurement.

        base_to_tip_distance:
            Secondary proxy: robust 3D distance from the estimated base point to
            an upper/farthest plant point. This can overestimate if a horizontal
            leaf tip is the farthest point, so it is reported as diagnostic.
        """
        u = self.cfg.units
        straight = self._estimate_straightened_height_geometry()

        vertical = float(self.height)
        straight_centerline = float(straight["centerline_arc_length"])
        base_to_tip = float(straight["base_to_tip_distance"])

        # For the manual stretched-height protocol, prefer the centerline arc
        # length, but never allow it to be smaller than the observed vertical
        # height due to sampling/slice failures.
        stretched_recommended = max(vertical, straight_centerline)

        self.diagnostics.update({
            "height_centerline_points": straight.get("centerline_points", []),
            "height_centerline_slice_count": straight.get("slice_count", 0),
            "height_upper_tip_candidate": straight.get("upper_tip_candidate", None),
        })

        return {
            f"plant_height_observed_vertical_{u}": vertical,
            f"plant_height_straightened_centerline_{u}": straight_centerline,
            f"plant_height_base_to_tip_distance_{u}": base_to_tip,
            f"plant_height_stretched_protocol_recommended_{u}": float(stretched_recommended),
            "plant_height_observed_definition": "robust vertical extent of reconstructed/bent plant pose",
            "plant_height_stretched_definition": "model-only straightened-height proxy from robust centerline arc length; no manual height used",
            f"plant_base_z_{u}": float(self.z_base),
            f"plant_top_z_{u}": float(self.z_top),

            # Backward-compatible key. Kept as observed vertical height because
            # existing downstream thresholds and old reports used this meaning.
            f"plant_height_{u}": vertical,
        }

    def _estimate_straightened_height_geometry(self) -> Dict[str, Any]:
        P = self.points
        if P is None or len(P) < 10:
            return {
                "centerline_arc_length": float(self.height),
                "base_to_tip_distance": float(self.height),
                "slice_count": 0,
                "centerline_points": [],
                "upper_tip_candidate": None,
            }

        z0, z1, h = self.z_base, self.z_top, self.height
        if h <= 1e-12:
            return {
                "centerline_arc_length": 0.0,
                "base_to_tip_distance": 0.0,
                "slice_count": 0,
                "centerline_points": [],
                "upper_tip_candidate": None,
            }

        if self.base_center_xy is None:
            lower = P[P[:, 2] <= np.percentile(P[:, 2], 10)]
            base_xy = np.median(lower[:, :2], axis=0) if len(lower) else np.median(P[:, :2], axis=0)
        else:
            base_xy = np.asarray(self.base_center_xy, dtype=np.float64)

        base_pt = np.array([base_xy[0], base_xy[1], z0], dtype=np.float64)

        # ------------------------------------------------------------------
        # Centerline arc-length estimate.
        # ------------------------------------------------------------------
        # We divide the plant into vertical slices and track the robust center
        # of each occupied slice. To avoid leaf tips/petioles pulling the
        # centerline too strongly, each slice center is computed from the
        # central portion of points relative to the previous center.
        n_slices = 32
        min_pts = max(12, int(0.00035 * len(P)))
        edges = np.linspace(z0, z1, n_slices + 1)

        centers: List[np.ndarray] = [base_pt]
        last_xy = base_xy.copy()

        for i in range(n_slices):
            a, b = edges[i], edges[i + 1]
            if i == n_slices - 1:
                mask = (P[:, 2] >= a) & (P[:, 2] <= b)
            else:
                mask = (P[:, 2] >= a) & (P[:, 2] < b)
            S = P[mask]
            if len(S) < min_pts:
                continue

            d = np.linalg.norm(S[:, :2] - last_xy[None, :], axis=1)

            # Keep the central 45% of the slice. If the plant bends, the center
            # can still move gradually because last_xy is updated every slice.
            q = np.percentile(d, 45.0)
            core = S[d <= q]
            if len(core) < max(8, min_pts // 3):
                core = S

            cxy = np.median(core[:, :2], axis=0)
            cz = float(np.median(core[:, 2]))
            c = np.array([cxy[0], cxy[1], cz], dtype=np.float64)

            # Avoid near-duplicate consecutive centers.
            if np.linalg.norm(c - centers[-1]) > 0.0025 * h:
                centers.append(c)
                last_xy = cxy

        # Ensure the centerline reaches the upper plant region. The top point is
        # estimated from robust upper-slice central points, not from a single
        # outlier vertex.
        upper = P[P[:, 2] >= z0 + 0.92 * h]
        if len(upper) >= 10:
            d_upper = np.linalg.norm(upper[:, :2] - last_xy[None, :], axis=1)
            q_upper = np.percentile(d_upper, 55.0)
            upper_core = upper[d_upper <= q_upper]
            if len(upper_core) < 6:
                upper_core = upper
            top_center = np.array([
                float(np.median(upper_core[:, 0])),
                float(np.median(upper_core[:, 1])),
                float(np.percentile(upper_core[:, 2], 85.0)),
            ], dtype=np.float64)
            if np.linalg.norm(top_center - centers[-1]) > 0.0025 * h:
                centers.append(top_center)

        if len(centers) >= 2:
            C = np.vstack(centers)
            seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
            centerline_len = float(np.sum(seg))
        else:
            C = np.vstack(centers)
            centerline_len = float(h)

        # ------------------------------------------------------------------
        # Base-to-tip straight-line diagnostic.
        # ------------------------------------------------------------------
        # Candidate points must be in the upper part to avoid horizontal lower
        # leaves becoming the "tip". This is diagnostic only.
        upper_tip_pool = P[P[:, 2] >= z0 + 0.65 * h]
        if len(upper_tip_pool) < 20:
            upper_tip_pool = P[P[:, 2] >= z0 + 0.50 * h]
        if len(upper_tip_pool) < 20:
            upper_tip_pool = P

        dist = np.linalg.norm(upper_tip_pool - base_pt[None, :], axis=1)
        # Robust tip: median of the farthest 2% points instead of a single max.
        cutoff = np.percentile(dist, 98.0)
        tip_candidates = upper_tip_pool[dist >= cutoff]
        if len(tip_candidates) == 0:
            tip = upper_tip_pool[int(np.argmax(dist))]
            base_to_tip = float(np.max(dist))
        else:
            tip = np.median(tip_candidates, axis=0)
            base_to_tip = float(np.linalg.norm(tip - base_pt))

        centerline_len = max(centerline_len, float(h))
        base_to_tip = max(base_to_tip, float(h))

        return {
            "centerline_arc_length": centerline_len,
            "base_to_tip_distance": base_to_tip,
            "slice_count": int(max(len(C) - 1, 0)),
            "centerline_points": C.tolist(),
            "upper_tip_candidate": tip.tolist(),
        }

    def _stem_traits(self) -> Dict[str, Any]:
        out = super()._stem_traits()
        u = self.cfg.units

        rec_key = f"stem_perimeter_base_recommended_{u}"
        if rec_key in out and isinstance(out[rec_key], (float, int)) and float(out[rec_key]) > 0:
            perim = float(out[rec_key])
            straight = self._estimate_straightened_height_geometry()
            vertical_h = float(self.height)
            stretched_h = max(vertical_h, float(straight["centerline_arc_length"]))
            base_to_tip_h = max(vertical_h, float(straight["base_to_tip_distance"]))

            out["plant_height_observed_vertical_to_stem_perimeter_recommended_ratio"] = float(vertical_h / perim)
            out["plant_height_stretched_centerline_to_stem_perimeter_recommended_ratio"] = float(stretched_h / perim)
            out["plant_height_base_to_tip_to_stem_perimeter_recommended_ratio"] = float(base_to_tip_h / perim)

        return out

class _SafeAngleTraitExtractor(_HeightAwareTraitExtractor):
    """Protocol-aware 3D trait extractor.

    Complete trait extractor with safer branch-angle extraction.

    Important methodological rule:
        Manual/real trait values are NOT used inside this extractor. Manual
        values should only be used later with evaluate_against_reference().
    """

    def print_report(self, traits: Optional[Dict[str, Any]] = None) -> None:
        traits = self.traits if traits is None else traits
        print("\n" + "=" * 76)
        print("PROTOCOL-AWARE 3D PLANT PHENOTYPIC TRAIT REPORT")
        print("=" * 76)
        for k, v in traits.items():
            if isinstance(v, float):
                print(f"{k:<62}: {v:.6f}")
            else:
                print(f"{k:<62}: {v}")
        print("=" * 76)
        print(f"Linear unit: {self.cfg.units}  |  unit_scale={self.cfg.unit_scale}")
        print("No manual trait values are used by this extractor.")
        print("Height is reported as both observed vertical and straightened/stretched proxy.")
        print("Branch angles use proximal leaf/petiole consensus near the attachment point.")
        print("Geometry-only organ traits remain estimates; semantics improve reliability.")
        print("=" * 76 + "\n")

    # ------------------------------------------------------------------
    # Safer proximal/protractor-like branch-angle extraction
    # ------------------------------------------------------------------
    def _branch_angle_traits(self) -> Dict[str, Any]:
        u = self.cfg.units
        self.angle_table = []

        if self.stem_points is None or len(self.stem_points) < 10 or not self.open_leaf_clusters:
            return {
                "branch_angle_status": "not_computed_insufficient_stem_or_leaf_clusters",
                "branch_angle_count": 0,
                "branch_angle_definition": "proximal petiole/stem angle; not computed",
            }

        S = np.asarray(self.stem_points, dtype=np.float64)
        h = max(float(self.height), 1e-12)

        if KDTree is not None:
            stem_tree = KDTree(S)
        else:
            stem_tree = None

        summary_raw: List[float] = []
        summary_acute: List[float] = []
        summary_weights: List[float] = []

        for leaf_id, C in enumerate(self.open_leaf_clusters, start=1):
            C = np.asarray(C, dtype=np.float64)
            if len(C) < max(5, int(self.cfg.branch_min_points * 0.5)):
                continue

            attach_info = self._estimate_leaf_stem_attachment(C, S, stem_tree)
            if attach_info is None:
                continue

            attach_leaf = attach_info["attach_leaf"]
            attach_stem = attach_info["attach_stem"]
            attach_distance = float(attach_info["attach_distance"])

            stem_axis, stem_axis_conf = self._estimate_local_stem_axis_safe(S, attach_stem, h)
            if stem_axis is None or np.linalg.norm(stem_axis) < 1e-9:
                continue
            if stem_axis[2] < 0:
                stem_axis = -stem_axis

            angle_candidates = []
            candidate_meta = []

            # leaf extent from attachment. Used for proximal fractions and inner ignore.
            d_from_attach = np.linalg.norm(C - attach_stem[None, :], axis=1)
            leaf_extent = float(np.percentile(d_from_attach, 95.0)) if len(d_from_attach) else 0.0
            if leaf_extent <= 1e-9:
                continue

            for frac in self.cfg.branch_proximal_fraction_candidates:
                axis_info = self._estimate_proximal_leaf_axis_safe(
                    C=C,
                    attach_stem=attach_stem,
                    leaf_extent=leaf_extent,
                    proximal_fraction=float(frac),
                )
                if axis_info is None:
                    continue

                leaf_axis = axis_info["axis"]
                if leaf_axis is None or np.linalg.norm(leaf_axis) < 1e-9:
                    continue

                # Orient away from stem/attachment.
                away = axis_info["centroid"] - attach_stem
                if np.dot(leaf_axis, away) < 0:
                    leaf_axis = -leaf_axis

                raw, acute = self._angle_between_axes(stem_axis, leaf_axis)
                if not np.isfinite(acute):
                    continue

                angle_candidates.append(float(acute))
                candidate_meta.append({
                    "proximal_fraction": float(frac),
                    "raw_angle_deg": float(raw),
                    "acute_angle_deg": float(acute),
                    "proximal_points": int(axis_info["n_points"]),
                    "axis_method": axis_info["method"],
                })

            if len(angle_candidates) < int(self.cfg.branch_min_valid_fraction_candidates):
                # Fallback: use the closest proximal points directly, still not full leaf centroid.
                fallback = self._estimate_proximal_leaf_axis_safe(
                    C=C,
                    attach_stem=attach_stem,
                    leaf_extent=leaf_extent,
                    proximal_fraction=0.35,
                    force_min_points=True,
                )
                if fallback is not None:
                    leaf_axis = fallback["axis"]
                    away = fallback["centroid"] - attach_stem
                    if np.dot(leaf_axis, away) < 0:
                        leaf_axis = -leaf_axis
                    raw, acute = self._angle_between_axes(stem_axis, leaf_axis)
                    angle_candidates.append(float(acute))
                    candidate_meta.append({
                        "proximal_fraction": 0.35,
                        "raw_angle_deg": float(raw),
                        "acute_angle_deg": float(acute),
                        "proximal_points": int(fallback["n_points"]),
                        "axis_method": "fallback_" + fallback["method"],
                    })

            if not angle_candidates:
                continue

            angle_arr = np.asarray(angle_candidates, dtype=np.float64)
            # Robust final angle: median across proximal scales.
            acute_final = float(np.median(angle_arr))

            # Raw final corresponding to the candidate closest to final acute.
            best_idx = int(np.argmin(np.abs(angle_arr - acute_final)))
            raw_final = float(candidate_meta[best_idx]["raw_angle_deg"])

            q25, q75 = np.percentile(angle_arr, [25, 75])
            iqr = float(q75 - q25)
            amin, amax = float(np.min(angle_arr)), float(np.max(angle_arr))

            max_attach = float(self.cfg.branch_max_attachment_distance_frac_height) * h
            confidence_flags = []
            if attach_distance > max_attach:
                confidence_flags.append("leaf_stem_gap_large")
            if iqr > float(self.cfg.branch_angle_stability_iqr_deg):
                confidence_flags.append("proximal_angle_unstable")
            if stem_axis_conf != "ok":
                confidence_flags.append("local_stem_axis_" + stem_axis_conf)
            if len(angle_candidates) < int(self.cfg.branch_min_valid_fraction_candidates):
                confidence_flags.append("few_angle_candidates")

            confidence = "high" if not confidence_flags else "low_" + "+".join(confidence_flags)
            include_in_summary = True
            if self.cfg.branch_reject_low_confidence_from_summary and confidence != "high":
                include_in_summary = False

            # Weight is not used for the main unweighted mean/median, but is useful
            # if you want weighted summaries later.
            weight = max(1.0, math.sqrt(len(C))) / (1.0 + iqr / 20.0)

            row = {
                "leaf_id": int(leaf_id),
                "branch_angle_raw_deg": raw_final,
                "branch_angle_acute_deg": acute_final,
                "branch_angle_protocol": "proximal_consensus_near_attachment",
                "branch_angle_confidence": confidence,
                "branch_angle_candidate_count": int(len(angle_candidates)),
                "branch_angle_candidate_min_deg": amin,
                "branch_angle_candidate_max_deg": amax,
                "branch_angle_candidate_iqr_deg": iqr,
                f"attachment_leaf_x_{u}": float(attach_leaf[0]),
                f"attachment_leaf_y_{u}": float(attach_leaf[1]),
                f"attachment_leaf_z_{u}": float(attach_leaf[2]),
                f"attachment_stem_x_{u}": float(attach_stem[0]),
                f"attachment_stem_y_{u}": float(attach_stem[1]),
                f"attachment_stem_z_{u}": float(attach_stem[2]),
                f"nearest_leaf_stem_distance_{u}": attach_distance,
                "leaf_points_total": int(len(C)),
                "include_in_summary": bool(include_in_summary),
                "candidate_angles": candidate_meta,
            }
            self.angle_table.append(row)

            if include_in_summary:
                summary_raw.append(raw_final)
                summary_acute.append(acute_final)
                summary_weights.append(weight)

        if not summary_acute:
            return {
                "branch_angle_status": "not_computed_no_valid_angles",
                "branch_angle_count": 0,
                "branch_angle_definition": "angle between local stem axis and proximal leaf/petiole direction near attachment",
                "branch_angle_protocol": "proximal_consensus_near_attachment",
            }

        raw_arr = np.asarray(summary_raw, dtype=np.float64)
        acute_arr = np.asarray(summary_acute, dtype=np.float64)
        w_arr = np.asarray(summary_weights, dtype=np.float64)
        w_arr = w_arr / max(float(w_arr.sum()), 1e-12)

        low_conf = sum(1 for r in self.angle_table if str(r.get("branch_angle_confidence", "")).startswith("low"))
        high_conf = sum(1 for r in self.angle_table if r.get("branch_angle_confidence") == "high")

        self.diagnostics["branch_angle_rows"] = self.angle_table
        self.diagnostics["branch_angle_low_confidence_count"] = int(low_conf)
        self.diagnostics["branch_angle_high_confidence_count"] = int(high_conf)

        return {
            "branch_angle_status": "ok",
            "branch_angle_definition": "manual-protocol approximation: angle between local stem axis and proximal leaf/petiole direction near attachment; bent outer lamina is avoided",
            "branch_angle_protocol": "proximal_consensus_near_attachment",
            "branch_angle_count": int(len(acute_arr)),
            "branch_angle_total_rows": int(len(self.angle_table)),
            "branch_angle_high_confidence_count": int(high_conf),
            "branch_angle_low_confidence_count": int(low_conf),
            "mean_branch_angle_raw_deg": float(np.mean(raw_arr)),
            "median_branch_angle_raw_deg": float(np.median(raw_arr)),
            "mean_branch_angle_acute_deg": float(np.mean(acute_arr)),
            "median_branch_angle_acute_deg": float(np.median(acute_arr)),
            "std_branch_angle_acute_deg": float(np.std(acute_arr, ddof=1)) if len(acute_arr) > 1 else 0.0,
            "weighted_mean_branch_angle_acute_deg": float(np.sum(w_arr * acute_arr)),
        }

    def _estimate_leaf_stem_attachment(
        self,
        C: np.ndarray,
        S: np.ndarray,
        stem_tree: Optional[Any],
    ) -> Optional[Dict[str, Any]]:
        """Robust leaf-stem attachment estimate.

        Instead of taking a single closest point pair, take the small set of leaf
        points nearest to the stem and use their median. This avoids one noisy
        vertex deciding the attachment.
        """
        if len(C) == 0 or len(S) == 0:
            return None

        if stem_tree is not None:
            dists, nn = stem_tree.query(C, k=1)
        else:
            D = np.linalg.norm(C[:, None, :] - S[None, :, :], axis=2)
            nn = np.argmin(D, axis=1)
            dists = D[np.arange(len(C)), nn]

        dists = np.asarray(dists, dtype=np.float64)
        nn = np.asarray(nn)

        # Use closest 3-8% of leaf points to define attachment zone.
        q = np.percentile(dists, 6.0)
        mask = dists <= max(q, np.min(dists) + 1e-12)
        if mask.sum() < 3:
            order = np.argsort(dists)[:min(max(3, len(C) // 20), len(C))]
            mask = np.zeros(len(C), dtype=bool)
            mask[order] = True

        leaf_attach_zone = C[mask]
        stem_attach_zone = S[nn[mask].astype(int)]

        attach_leaf = np.median(leaf_attach_zone, axis=0)
        attach_stem = np.median(stem_attach_zone, axis=0)

        # Snap attach_stem back to nearest actual stem point for local stem lookup.
        if stem_tree is not None:
            d_snap, idx_snap = stem_tree.query(attach_stem[None, :], k=1)
            attach_stem = S[int(np.asarray(idx_snap).ravel()[0])]
            attach_distance = float(np.median(dists[mask]))
        else:
            dd = np.linalg.norm(S - attach_stem[None, :], axis=1)
            attach_stem = S[int(np.argmin(dd))]
            attach_distance = float(np.median(dists[mask]))

        return {
            "attach_leaf": attach_leaf,
            "attach_stem": attach_stem,
            "attach_distance": attach_distance,
        }

    def _estimate_local_stem_axis_safe(
        self,
        S: np.ndarray,
        attach_stem: np.ndarray,
        h: float,
    ) -> Tuple[Optional[np.ndarray], str]:
        """Estimate local stem axis near attachment using a vertical band/core."""
        dz = max(float(self.cfg.branch_local_stem_height_frac) * h, 1e-8)
        z = float(attach_stem[2])

        band = S[np.abs(S[:, 2] - z) <= dz]
        status = "ok"

        if len(band) < 8:
            # nearest stem points fallback
            d = np.linalg.norm(S - attach_stem[None, :], axis=1)
            k = min(max(20, len(S) // 25), len(S))
            band = S[np.argsort(d)[:k]]
            status = "nearest_fallback"

        if len(band) < 5:
            return None, "too_few_points"

        # Keep local core around attachment xy to reduce petiole contamination.
        dxy = np.linalg.norm(band[:, :2] - attach_stem[None, :2], axis=1)
        q = np.percentile(dxy, np.clip(float(self.cfg.branch_local_stem_core_percentile), 10.0, 100.0))
        core = band[dxy <= q]
        if len(core) >= 5:
            band = core
        else:
            status = "wide_band"

        _, vals, vecs = _pca_axes(band)
        axis = _unit(vecs[:, 0])

        # PCA can accidentally choose a horizontal spread if the local stem band is
        # contaminated. If the main axis is too horizontal, blend with vertical.
        vertical = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(axis, vertical))) < 0.35:
            axis = _unit(0.35 * axis + 0.65 * vertical)
            status = "vertical_blended"

        return axis, status

    def _estimate_proximal_leaf_axis_safe(
        self,
        C: np.ndarray,
        attach_stem: np.ndarray,
        leaf_extent: float,
        proximal_fraction: float,
        force_min_points: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Estimate initial petiole/leaf direction near attachment.

        Uses PCA on proximal points when possible, with centroid-vector fallback.
        Crucially, it does NOT use the whole bent leaf/lobe.
        """
        if len(C) < 3 or leaf_extent <= 1e-9:
            return None

        d = np.linalg.norm(C - attach_stem[None, :], axis=1)
        inner = max(float(self.cfg.branch_inner_ignore_frac_of_leaf_extent) * leaf_extent, 1e-10)
        outer = max(float(proximal_fraction) * leaf_extent, inner * 2.0)

        mask = (d >= inner) & (d <= outer)
        proximal = C[mask]

        min_pts = max(5, min(int(self.cfg.branch_min_points), 18))
        if len(proximal) < min_pts:
            if force_min_points or len(C) >= min_pts:
                order = np.argsort(d)
                # skip extremely closest collapsed points if possible
                order = order[d[order] >= inner] if np.any(d[order] >= inner) else order
                proximal = C[order[:min(max(min_pts, 8), len(order))]]
            else:
                return None

        if len(proximal) < 3:
            return None

        centroid = np.mean(proximal, axis=0)

        # PCA direction along initial petiole/leaf segment.
        _, vals, vecs = _pca_axes(proximal)
        axis_pca = _unit(vecs[:, 0])
        axis_centroid = _unit(centroid - attach_stem)

        if np.linalg.norm(axis_centroid) < 1e-9 and np.linalg.norm(axis_pca) < 1e-9:
            return None

        # Use PCA if the proximal points are line-like; otherwise centroid vector
        # is often more stable for flat noisy patches.
        linearity = 0.0
        if vals[0] > 1e-12:
            linearity = float((vals[0] - vals[1]) / max(vals[0], 1e-12))

        if linearity >= 0.18 and np.linalg.norm(axis_pca) > 1e-9:
            axis = axis_pca
            method = "pca_proximal"
        else:
            axis = axis_centroid
            method = "centroid_proximal"

        # Blend slightly toward centroid direction to make orientation local and
        # avoid PCA choosing cross-leaf direction on flat lamina fragments.
        if np.linalg.norm(axis_centroid) > 1e-9 and np.linalg.norm(axis) > 1e-9:
            if np.dot(axis, axis_centroid) < 0:
                axis = -axis
            axis = _unit(0.70 * axis + 0.30 * axis_centroid)

        return {
            "axis": axis,
            "centroid": centroid,
            "n_points": int(len(proximal)),
            "method": method,
            "linearity": float(linearity),
        }

    @staticmethod
    def _angle_between_axes(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        a = _unit(np.asarray(a, dtype=np.float64))
        b = _unit(np.asarray(b, dtype=np.float64))
        if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
            return float("nan"), float("nan")
        cosv = float(np.clip(np.dot(a, b), -1.0, 1.0))
        raw = float(np.degrees(np.arccos(cosv)))
        acute = float(min(raw, 180.0 - raw))
        return raw, acute

    def print_angle_table(self) -> None:
        """Convenience printer for checking the safer angle extraction."""
        if not self.angle_table:
            print("No angle table. Run extractor.run() first.")
            return
        print("\n" + "=" * 92)
        print("SAFER PROXIMAL BRANCH ANGLE TABLE")
        print("=" * 92)
        print(f"{'leaf':>4} | {'acute':>8} | {'raw':>8} | {'conf':>28} | {'cand':>4} | {'IQR':>7}")
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

class PlantTraitExtractor3D(_SafeAngleTraitExtractor):
    """Protocol-aware 3D plant phenotypic trait extractor.

    Main design principles:
        1. No manual/real trait values are used during prediction.
        2. Noisy organ-level traits are estimated by robust consensus across
           multiple comparable geometry estimators.
        3. Incompatible biological definitions are not averaged.
        4. Disagreement is returned as uncertainty/confidence diagnostics.
    """

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def print_report(self, traits: Optional[Dict[str, Any]] = None) -> None:
        traits = self.traits if traits is None else traits
        print("\n" + "=" * 84)
        print("PROTOCOL-AWARE 3D PLANT PHENOTYPIC TRAIT REPORT")
        print("=" * 84)
        for k, v in traits.items():
            if isinstance(v, float):
                print(f"{k:<70}: {v:.6f}")
            else:
                print(f"{k:<70}: {v}")
        print("=" * 84)
        print(f"Linear unit: {self.cfg.units}  |  unit_scale={self.cfg.unit_scale}")
        print("No manual trait values are used by this extractor.")
        print("Organ-level traits use robust multi-estimator consensus with confidence diagnostics.")
        print("Observed height, stretched height, LAI definitions, and volume definitions are not mixed.")
        print("=" * 84 + "\n")

    # ------------------------------------------------------------------
    # Robust aggregation helpers
    # ------------------------------------------------------------------
    def _robust_scalar_consensus(
        self,
        values: List[float],
        labels: Optional[List[str]] = None,
        *,
        positive: bool = True,
        use_log: bool = False,
        mad_k: Optional[float] = None,
        trim_fraction: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Robustly combine comparable scalar estimates.

        Returns recommended, median, trimmed mean, bounds, inlier mask, and
        dispersion. For positive measurements such as perimeter, log-space
        trimming can reduce the influence of multiplicative outliers.
        """
        labels = labels or [f"estimator_{i}" for i in range(len(values))]
        arr0 = np.asarray(values, dtype=np.float64)
        valid = np.isfinite(arr0)
        if positive:
            valid &= arr0 > 0
        arr = arr0[valid]
        valid_labels = [lab for lab, ok in zip(labels, valid) if ok]

        if len(arr) == 0:
            return {
                "status": "failed_no_valid_values",
                "recommended": None,
                "values": [],
                "labels": [],
                "inlier_values": [],
                "inlier_labels": [],
                "n_total": 0,
                "n_inliers": 0,
            }

        mad_k = float(self.cfg.ensemble_mad_k if mad_k is None else mad_k)
        trim_fraction = float(self.cfg.ensemble_trim_fraction if trim_fraction is None else trim_fraction)

        work = np.log(arr) if (use_log and positive) else arr.copy()
        med = float(np.median(work))
        abs_dev = np.abs(work - med)
        mad = float(np.median(abs_dev))

        if len(work) >= 4 and mad > 1e-12:
            robust_sigma = 1.4826 * mad
            inlier_mask = abs_dev <= mad_k * robust_sigma
        elif len(work) >= 4:
            q1, q3 = np.percentile(work, [25, 75])
            iqr = float(q3 - q1)
            if iqr > 1e-12:
                inlier_mask = (work >= q1 - 1.5 * iqr) & (work <= q3 + 1.5 * iqr)
            else:
                inlier_mask = np.ones(len(work), dtype=bool)
        else:
            inlier_mask = np.ones(len(work), dtype=bool)

        # Do not allow an over-aggressive mask to remove too much evidence.
        if np.sum(inlier_mask) < max(2, int(0.45 * len(work))):
            inlier_mask = np.ones(len(work), dtype=bool)

        inliers = arr[inlier_mask]
        inlier_work = work[inlier_mask]
        inlier_labels = [lab for lab, ok in zip(valid_labels, inlier_mask) if ok]

        sorted_work = np.sort(inlier_work)
        n = len(sorted_work)
        k = int(math.floor(trim_fraction * n))
        if n - 2 * k >= 2:
            trimmed_work = sorted_work[k:n-k]
        else:
            trimmed_work = sorted_work

        rec_work = float(np.mean(trimmed_work))
        rec = float(math.exp(rec_work)) if (use_log and positive) else rec_work

        q10, q25, q50, q75, q90 = np.percentile(inliers, [10, 25, 50, 75, 90])
        iqr_val = float(q75 - q25)
        rel_iqr = float(iqr_val / (abs(rec) + 1e-12))

        rejected = [lab for lab, ok in zip(valid_labels, inlier_mask) if not ok]

        if rel_iqr < 0.15 and len(inliers) >= 5:
            conf = "high"
        elif rel_iqr < 0.35 and len(inliers) >= 3:
            conf = "medium"
        else:
            conf = "low_high_disagreement"
        if rejected:
            conf = conf + f"_outliers_rejected_{len(rejected)}"

        return {
            "status": "ok",
            "recommended": rec,
            "median": float(q50),
            "mean": float(np.mean(inliers)),
            "trimmed_mean": rec,
            "lower_bound_p10": float(q10),
            "lower_bound_p25": float(q25),
            "upper_bound_p75": float(q75),
            "upper_bound_p90": float(q90),
            "iqr": iqr_val,
            "relative_iqr": rel_iqr,
            "confidence": conf,
            "values": [float(x) for x in arr],
            "labels": valid_labels,
            "inlier_values": [float(x) for x in inliers],
            "inlier_labels": inlier_labels,
            "rejected_labels": rejected,
            "n_total": int(len(arr)),
            "n_inliers": int(len(inliers)),
        }

    def _weighted_median(self, values: np.ndarray, weights: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if len(values) == 0:
            return float("nan")
        order = np.argsort(values)
        v = values[order]
        w = weights[order]
        w = w / max(float(w.sum()), 1e-12)
        c = np.cumsum(w)
        return float(v[min(int(np.searchsorted(c, 0.5, side="left")), len(v) - 1)])

    # ------------------------------------------------------------------
    # Height: report separate definitions, add robust vertical-height ensemble
    # ------------------------------------------------------------------
    def _height_traits(self) -> Dict[str, Any]:
        out = super()._height_traits()
        u = self.cfg.units
        P = self.points
        if P is not None and len(P) >= 20:
            vals = []
            labels = []
            for lo, hi in [(0.3, 99.7), (0.5, 99.5), (1.0, 99.0), (1.5, 98.5), (2.0, 98.0)]:
                try:
                    z0, z1 = np.percentile(P[:, 2], [lo, hi])
                    h = float(max(z1 - z0, 0.0))
                    if h > 0:
                        vals.append(h)
                        labels.append(f"vertical_p{lo}_p{hi}")
                except Exception:
                    pass
            cons = self._robust_scalar_consensus(vals, labels, positive=True, use_log=False)
            if cons["status"] == "ok":
                out[f"plant_height_observed_vertical_ensemble_{u}"] = float(cons["recommended"])
                out[f"plant_height_observed_vertical_ensemble_lower_{u}"] = float(cons["lower_bound_p25"])
                out[f"plant_height_observed_vertical_ensemble_upper_{u}"] = float(cons["upper_bound_p75"])
                out["plant_height_observed_vertical_ensemble_confidence"] = cons["confidence"]
                self.diagnostics["height_vertical_ensemble"] = cons
        return out

    # ------------------------------------------------------------------
    # Stem perimeter: robust consensus across many base-slice estimators
    # ------------------------------------------------------------------
    def _ellipse_from_xy(self, XY: np.ndarray, percentile: float) -> Tuple[float, float, float, float]:
        if len(XY) < 5:
            return 0.0, 0.0, 0.0, 0.0
        C = XY - np.median(XY, axis=0, keepdims=True)
        d = np.linalg.norm(C, axis=1)
        keep = d <= np.percentile(d, 92.0)
        X = C[keep] if np.sum(keep) >= 5 else C
        if len(X) < 5:
            return 0.0, 0.0, 0.0, 0.0
        cov = np.cov(X.T)
        ev, evec = np.linalg.eigh(cov)
        order = np.argsort(ev)[::-1]
        evec = evec[:, order]
        proj = X @ evec
        a = float(np.percentile(np.abs(proj[:, 0]), percentile))
        b = float(np.percentile(np.abs(proj[:, 1]), percentile))
        if a < b:
            a, b = b, a
        perim = _ellipse_perimeter_ramanujan(a, b)
        area = math.pi * a * b
        return float(a), float(b), float(perim), float(area)

    def _stem_perimeter_ensemble(self) -> Dict[str, Any]:
        u = self.cfg.units
        P = self.points
        if P is None or len(P) < 20:
            return {"status": "failed_no_points"}

        z0, _, h = self.z_base, self.z_top, max(float(self.height), 1e-12)
        center0 = self.base_center_xy

        labels = self.labels_sampled
        stem_id = self.class_ids.get("stem", None)
        have_labels = labels is not None and len(labels) == len(P) and stem_id is not None
        candidate = P[labels == stem_id] if have_labels else P
        mode = "semantic_stem" if have_labels else "geometry_central_core"

        estimates = []
        values = []
        labels_out = []

        for off_frac in self.cfg.stem_ensemble_offset_fracs:
            z_m = z0 + float(off_frac) * h
            for half_frac in self.cfg.stem_ensemble_half_thickness_fracs:
                half = max(float(half_frac) * h, 1e-12)
                S0 = candidate[np.abs(candidate[:, 2] - z_m) <= half]
                if len(S0) < self.cfg.stem_min_points_in_slice:
                    S0 = candidate[np.abs(candidate[:, 2] - z_m) <= 2.0 * half]
                if len(S0) < self.cfg.stem_min_points_in_slice:
                    continue

                if have_labels:
                    core_percentiles = (100.0,)
                    radius_percentiles = self.cfg.stem_ensemble_radius_percentiles_semantic
                else:
                    core_percentiles = self.cfg.stem_ensemble_core_percentiles
                    radius_percentiles = self.cfg.stem_ensemble_radius_percentiles_geometry

                d0 = np.linalg.norm(S0[:, :2] - center0[None, :], axis=1)
                for core_q in core_percentiles:
                    if have_labels or core_q >= 99.0:
                        S = S0
                    else:
                        gate = np.percentile(d0, float(core_q))
                        S = S0[d0 <= gate]
                        if len(S) < self.cfg.stem_min_points_in_slice:
                            continue
                    c = np.median(S[:, :2], axis=0)
                    d = np.linalg.norm(S[:, :2] - c[None, :], axis=1)
                    for rq in radius_percentiles:
                        radius = float(np.percentile(d, float(rq)))
                        if radius <= 0 or not np.isfinite(radius):
                            continue
                        per_circ = float(2.0 * math.pi * radius)
                        lab_c = f"circ_off{off_frac:.3f}_half{half_frac:.3f}_core{core_q:.1f}_r{rq:.1f}"
                        values.append(per_circ)
                        labels_out.append(lab_c)
                        estimates.append({
                            "method": lab_c,
                            "type": "circular_radius_percentile",
                            f"perimeter_{u}": per_circ,
                            f"radius_{u}": radius,
                            "offset_frac_height": float(off_frac),
                            "half_thickness_frac_height": float(half_frac),
                            "core_percentile": float(core_q),
                            "radius_percentile": float(rq),
                            "n_points": int(len(S)),
                        })

                        a, b, per_ell, area_ell = self._ellipse_from_xy(S[:, :2], percentile=float(rq))
                        if per_ell > 0 and np.isfinite(per_ell):
                            lab_e = f"ellipse_off{off_frac:.3f}_half{half_frac:.3f}_core{core_q:.1f}_r{rq:.1f}"
                            values.append(float(per_ell))
                            labels_out.append(lab_e)
                            estimates.append({
                                "method": lab_e,
                                "type": "ellipse_axis_percentile",
                                f"perimeter_{u}": float(per_ell),
                                f"ellipse_a_{u}": float(a),
                                f"ellipse_b_{u}": float(b),
                                f"area_{u}2": float(area_ell),
                                "offset_frac_height": float(off_frac),
                                "half_thickness_frac_height": float(half_frac),
                                "core_percentile": float(core_q),
                                "axis_percentile": float(rq),
                                "n_points": int(len(S)),
                            })

        cons = self._robust_scalar_consensus(values, labels_out, positive=True, use_log=True)
        cons["estimates_reported"] = estimates[: int(self.cfg.stem_ensemble_max_estimators_reported)]
        cons["candidate_mode"] = mode
        return cons

    def _stem_traits(self) -> Dict[str, Any]:
        out = super()._stem_traits()
        u = self.cfg.units

        if not self.cfg.enable_robust_ensemble:
            return out

        ens = self._stem_perimeter_ensemble()
        self.diagnostics["stem_perimeter_ensemble"] = ens
        if ens.get("status") != "ok" or ens.get("recommended") is None:
            out["stem_perimeter_ensemble_status"] = ens.get("status", "failed")
            return out

        rec = float(ens["recommended"])
        low25 = float(ens["lower_bound_p25"])
        up75 = float(ens["upper_bound_p75"])
        low10 = float(ens["lower_bound_p10"])
        up90 = float(ens["upper_bound_p90"])

        out[f"stem_perimeter_base_recommended_{u}"] = rec
        out[f"stem_perimeter_base_consensus_median_{u}"] = float(ens["median"])
        out[f"stem_perimeter_base_lower_bound_{u}"] = low25
        out[f"stem_perimeter_base_upper_bound_{u}"] = up75
        out[f"stem_perimeter_base_uncertainty_p10_{u}"] = low10
        out[f"stem_perimeter_base_uncertainty_p90_{u}"] = up90
        out[f"stem_equivalent_radius_recommended_{u}"] = float(rec / (2.0 * math.pi))
        out[f"stem_equivalent_diameter_recommended_{u}"] = float(rec / math.pi)
        out["stem_perimeter_recommended_basis"] = "robust consensus of circular and ellipse base-slice estimators after outlier rejection"
        out["stem_perimeter_ensemble_status"] = "ok"
        out["stem_perimeter_ensemble_n_estimators"] = int(ens["n_total"])
        out["stem_perimeter_ensemble_n_inliers"] = int(ens["n_inliers"])
        out["stem_perimeter_ensemble_relative_iqr"] = float(ens["relative_iqr"])
        out["stem_perimeter_ensemble_confidence"] = ens["confidence"]
        out["stem_perimeter_ensemble_rejected_estimators"] = ens.get("rejected_labels", [])[:20]

        # Update height/perimeter ratios using the consensus perimeter.
        if rec > 0:
            vertical = float(self.height)
            stretched = out.get(f"plant_height_stretched_protocol_recommended_{u}", vertical)
            base_to_tip = out.get(f"plant_height_base_to_tip_distance_{u}", vertical)
            out["plant_height_to_stem_perimeter_recommended_ratio"] = float(vertical / rec)
            out["plant_height_observed_vertical_to_stem_perimeter_recommended_ratio"] = float(vertical / rec)
            out["plant_height_stretched_centerline_to_stem_perimeter_recommended_ratio"] = float(float(stretched) / rec)
            out["plant_height_base_to_tip_to_stem_perimeter_recommended_ratio"] = float(float(base_to_tip) / rec)

        return out

    # ------------------------------------------------------------------
    # Leaf detection: multi-profile robust consensus, no manual leaf count
    # ------------------------------------------------------------------
    def _leaf_multiscale_profiles(self) -> List[Dict[str, float | str]]:
        if not self.cfg.leaf_use_ensemble_profiles:
            return super()._leaf_multiscale_profiles()

        return [
            {"name": "ultra_fine", "sigma": 0.70, "min_sep_deg": 4.5, "prom": 0.0035, "height": 0.0035, "keep": 0.008, "merge_angle_deg": 6.0, "valley": 0.74, "assign_window_deg": 22.0, "min_points": max(15, int(0.65 * self.cfg.leaf_min_points)), "area_rel": 0.035, "area_h2": 0.00030, "extent_h": 0.026},
            {"name": "very_fine", "sigma": 0.85, "min_sep_deg": 5.5, "prom": 0.0045, "height": 0.0045, "keep": 0.010, "merge_angle_deg": 7.5, "valley": 0.68, "assign_window_deg": 25.0, "min_points": max(18, int(0.75 * self.cfg.leaf_min_points)), "area_rel": 0.040, "area_h2": 0.00035, "extent_h": 0.030},
            {"name": "fine", "sigma": 1.00, "min_sep_deg": 6.0, "prom": 0.0060, "height": 0.0060, "keep": 0.015, "merge_angle_deg": 10.0, "valley": 0.62, "assign_window_deg": 28.0, "min_points": max(20, int(0.85 * self.cfg.leaf_min_points)), "area_rel": 0.050, "area_h2": 0.00045, "extent_h": 0.035},
            {"name": "balanced", "sigma": 1.18, "min_sep_deg": 7.0, "prom": 0.0080, "height": 0.0080, "keep": 0.020, "merge_angle_deg": 14.0, "valley": 0.54, "assign_window_deg": 31.0, "min_points": max(22, int(self.cfg.leaf_min_points)), "area_rel": 0.060, "area_h2": 0.00055, "extent_h": 0.040},
            {"name": "coarse", "sigma": 1.38, "min_sep_deg": 8.0, "prom": 0.0100, "height": 0.0100, "keep": 0.025, "merge_angle_deg": 16.0, "valley": 0.49, "assign_window_deg": 34.0, "min_points": max(25, int(1.1 * self.cfg.leaf_min_points)), "area_rel": 0.070, "area_h2": 0.00065, "extent_h": 0.045},
            {"name": "very_coarse", "sigma": 1.65, "min_sep_deg": 10.0, "prom": 0.0120, "height": 0.0120, "keep": 0.030, "merge_angle_deg": 18.0, "valley": 0.44, "assign_window_deg": 38.0, "min_points": max(30, int(1.2 * self.cfg.leaf_min_points)), "area_rel": 0.080, "area_h2": 0.00080, "extent_h": 0.050},
            {"name": "ultra_coarse", "sigma": 1.95, "min_sep_deg": 12.0, "prom": 0.0160, "height": 0.0160, "keep": 0.040, "merge_angle_deg": 22.0, "valley": 0.38, "assign_window_deg": 42.0, "min_points": max(35, int(1.35 * self.cfg.leaf_min_points)), "area_rel": 0.095, "area_h2": 0.00100, "extent_h": 0.060},
        ]

    def _profile_weight_for_leaf_consensus(self, profile_name: str) -> float:
        # Avoid letting the intentionally extreme profiles dominate. The central
        # profiles represent the a priori expected operating region.
        weights = {
            "ultra_fine": 0.45,
            "very_fine": 0.70,
            "fine": 0.95,
            "balanced": 1.00,
            "coarse": 0.95,
            "very_coarse": 0.70,
            "ultra_coarse": 0.45,
        }
        return float(weights.get(profile_name, 0.75))

    def _detect_open_leaf_clusters(self) -> None:
        self.open_leaf_clusters = []
        self.leaf_table = []

        P = self.leaf_candidate_points
        if P is None or len(P) < max(50, self.cfg.leaf_min_points):
            self.diagnostics["leaf_detection_status"] = "failed_too_few_leaf_candidates"
            return
        if gaussian_filter1d is None or find_peaks is None:
            self.diagnostics["leaf_detection_status"] = "failed_scipy_required"
            return

        profiles = self._leaf_multiscale_profiles()
        results = []
        for profile in profiles:
            result = self._detect_leaf_clusters_single_profile(profile)
            if result is not None and len(result.get("clusters", [])) > 0:
                results.append(result)

        if not results:
            self.diagnostics["leaf_detection_status"] = "failed_no_valid_multiscale_leaf_solution"
            self.diagnostics["leaf_multiscale_counts"] = []
            return

        counts = np.array([len(r["clusters"]) for r in results], dtype=np.float64)
        weights = np.array([self._profile_weight_for_leaf_consensus(r["profile_name"]) for r in results], dtype=np.float64)
        consensus_count = self._weighted_median(counts, weights)

        q25, q75 = np.percentile(counts, [25, 75])
        count_iqr = float(q75 - q25)
        rel_iqr = float(count_iqr / max(consensus_count, 1.0))
        if rel_iqr <= self.cfg.leaf_count_stability_high_iqr_frac:
            leaf_conf = "high"
        elif rel_iqr <= self.cfg.leaf_count_stability_medium_iqr_frac:
            leaf_conf = "medium"
        else:
            leaf_conf = "low_multiscale_disagreement"

        best_idx = 0
        best_score = float("inf")
        for idx, r in enumerate(results):
            count = len(r["clusters"])
            count_score = abs(count - consensus_count) / max(consensus_count, 1.0)

            areas = np.array([max(self._leaf_area(C), 1e-12) for C in r["clusters"]], dtype=np.float64)
            area_cv = float(np.std(areas) / (np.mean(areas) + 1e-12)) if len(areas) > 1 else 0.0
            small_frac = float(np.mean(areas < 0.10 * np.max(areas))) if len(areas) > 0 else 0.0

            profile_pref = -self.cfg.leaf_consensus_profile_preference_strength * self._profile_weight_for_leaf_consensus(r["profile_name"])
            score = count_score + self.cfg.leaf_consensus_area_cv_penalty * area_cv + self.cfg.leaf_consensus_small_leaf_penalty * small_frac + profile_pref
            r["selection_score"] = float(score)
            r["area_cv"] = float(area_cv)
            r["small_leaf_fraction"] = float(small_frac)
            if score < best_score:
                best_score = score
                best_idx = idx

        selected = results[best_idx]
        self.open_leaf_clusters = selected["clusters"]

        self.diagnostics.update(selected["diagnostics"])
        self.diagnostics["leaf_detection_status"] = "ok"
        self.diagnostics["leaf_detection_method"] = "multiscale_robust_consensus"
        self.diagnostics["leaf_selected_profile"] = selected["profile_name"]
        self.diagnostics["leaf_selected_profile_score"] = float(selected.get("selection_score", best_score))
        self.diagnostics["leaf_multiscale_counts"] = [
            {
                "profile": r["profile_name"],
                "count": int(len(r["clusters"])),
                "weight": float(self._profile_weight_for_leaf_consensus(r["profile_name"])),
                "selection_score": float(r.get("selection_score", 0.0)),
                "area_cv": float(r.get("area_cv", 0.0)),
                "small_leaf_fraction": float(r.get("small_leaf_fraction", 0.0)),
            }
            for r in results
        ]
        self.diagnostics["leaf_consensus_count_weighted_median"] = float(consensus_count)
        self.diagnostics["leaf_multiscale_count_iqr"] = count_iqr
        self.diagnostics["leaf_multiscale_count_relative_iqr"] = rel_iqr
        self.diagnostics["leaf_count_confidence"] = leaf_conf
        self.diagnostics["leaf_exclusive_assignment"] = True
        self.diagnostics["leaf_no_duplicate_points_between_clusters"] = True

    # ------------------------------------------------------------------
    # Leaf area: robust consensus across local-area estimators
    # ------------------------------------------------------------------
    def _leaf_area_estimates(self, C: np.ndarray) -> Dict[str, Any]:
        estimates = []
        values = []
        labels = []
        if len(C) < 3:
            return {"status": "failed_too_few_points", "recommended": 0.0, "estimates": []}

        P2 = _project_leaf_to_pca_plane(C)
        if len(P2) >= 3:
            a = _convex_hull_area_2d(P2)
            if a > 0:
                values.append(float(a)); labels.append("pca_hull_full"); estimates.append({"method": "pca_hull_full", "area": float(a)})

            center = np.median(P2, axis=0)
            d = np.linalg.norm(P2 - center[None, :], axis=1)
            for pct in self.cfg.leaf_area_trim_percentiles:
                keep = d <= np.percentile(d, float(pct))
                if np.sum(keep) >= 3:
                    aa = _convex_hull_area_2d(P2[keep])
                    if aa > 0:
                        lab = f"pca_hull_trim_p{pct:.0f}"
                        values.append(float(aa)); labels.append(lab); estimates.append({"method": lab, "area": float(aa)})

            # Robust ellipse-like planar area. This is not the final area alone,
            # but helps stabilize when convex hull is inflated by ghost vertices.
            X = P2 - np.median(P2, axis=0, keepdims=True)
            if len(X) >= 5:
                cov = np.cov(X.T)
                ev, evec = np.linalg.eigh(cov)
                order = np.argsort(ev)[::-1]
                evec = evec[:, order]
                proj = X @ evec
                for pct in self.cfg.leaf_area_ellipse_percentiles:
                    aa1 = float(np.percentile(np.abs(proj[:, 0]), float(pct)))
                    bb1 = float(np.percentile(np.abs(proj[:, 1]), float(pct)))
                    area_e = math.pi * aa1 * bb1
                    if area_e > 0:
                        lab = f"pca_ellipse_area_p{pct:.0f}"
                        values.append(float(area_e)); labels.append(lab); estimates.append({"method": lab, "area": float(area_e)})

        # Horizontal projection is a lower-ish diagnostic for inclined leaves;
        # include it with robust consensus, but outlier rejection can remove it.
        ah = _convex_hull_area_2d(C[:, :2])
        if ah > 0:
            values.append(float(ah)); labels.append("horizontal_projected_hull"); estimates.append({"method": "horizontal_projected_hull", "area": float(ah)})

        cons = self._robust_scalar_consensus(values, labels, positive=True, use_log=True)
        cons["estimates_detail"] = estimates
        return cons

    def _leaf_area(self, C: np.ndarray) -> float:
        if not getattr(self.cfg, "leaf_area_ensemble", True):
            return super()._leaf_area(C)
        cons = self._leaf_area_estimates(C)
        if cons.get("status") == "ok" and cons.get("recommended") is not None:
            return float(cons["recommended"])
        return super()._leaf_area(C)

    def _leaf_area_and_lai_traits(self) -> Dict[str, Any]:
        u = self.cfg.units
        total_area = 0.0
        self.leaf_table = []
        area_confidences = []
        area_rel_iqrs = []

        for i, C in enumerate(self.open_leaf_clusters, start=1):
            cons = self._leaf_area_estimates(C) if self.cfg.leaf_area_ensemble else None
            if cons is not None and cons.get("status") == "ok" and cons.get("recommended") is not None:
                area = float(cons["recommended"])
                area_conf = cons.get("confidence", None)
                rel_iqr = cons.get("relative_iqr", None)
            else:
                area = float(super()._leaf_area(C))
                area_conf = "fallback_single_method"
                rel_iqr = None

            center = C.mean(axis=0)
            rr = np.linalg.norm(C[:, :2] - self.base_center_xy[None, :], axis=1)
            radial_extent = float(np.percentile(rr, 95) - np.percentile(rr, 5))
            total_area += area
            area_confidences.append(area_conf)
            if rel_iqr is not None:
                area_rel_iqrs.append(float(rel_iqr))

            row = {
                "leaf_id": i,
                f"leaf_area_{u}2": float(area),
                f"leaf_center_x_{u}": float(center[0]),
                f"leaf_center_y_{u}": float(center[1]),
                f"leaf_center_z_{u}": float(center[2]),
                f"leaf_radial_extent_{u}": float(radial_extent),
                "n_points": int(len(C)),
                "leaf_area_confidence": area_conf,
            }
            if cons is not None and cons.get("status") == "ok":
                row["leaf_area_relative_iqr"] = cons.get("relative_iqr", None)
                row["leaf_area_method_count"] = cons.get("n_inliers", None)
                row["leaf_area_rejected_methods"] = cons.get("rejected_labels", [])
                row["leaf_area_estimates"] = cons.get("estimates_detail", [])
            self.leaf_table.append(row)

        proj_area = _convex_hull_area_2d(self.points[:, :2]) if self.points is not None and len(self.points) >= 3 else 0.0
        lai_proxy = float(total_area / proj_area) if proj_area > 1e-12 else None
        if self.cfg.ground_area is not None and self.cfg.ground_area > 0:
            lai_strict = float(total_area / float(self.cfg.ground_area))
            lai_status = "computed_using_provided_independent_ground_area"
        else:
            lai_strict = None
            lai_status = "not_computed_ground_area_not_provided"

        # Aggregate leaf-area confidence.
        if not area_rel_iqrs:
            area_conf = "unknown"
        else:
            med_rel = float(np.median(area_rel_iqrs))
            if med_rel < 0.20:
                area_conf = "high"
            elif med_rel < 0.45:
                area_conf = "medium"
            else:
                area_conf = "low_area_estimator_disagreement"

        return {
            "leaf_detection_status": self.diagnostics.get("leaf_detection_status", None),
            "leaf_count_definition": "estimated fully opened leaves from robust multi-scale lobe consensus; tiny early-stage lobes filtered by area and radial extent",
            "open_leaf_count_estimated": int(len(self.open_leaf_clusters)),
            "open_leaf_count_confidence": self.diagnostics.get("leaf_count_confidence", None),
            "leaf_consensus_count_weighted_median": self.diagnostics.get("leaf_consensus_count_weighted_median", None),
            "leaf_multiscale_count_iqr": self.diagnostics.get("leaf_multiscale_count_iqr", None),
            "leaf_selected_profile": self.diagnostics.get("leaf_selected_profile", None),
            "leaf_multiscale_counts": self.diagnostics.get("leaf_multiscale_counts", None),
            "leaf_raw_angular_peak_count": self.diagnostics.get("leaf_raw_angular_peak_count", None),
            "leaf_merged_peak_count": self.diagnostics.get("leaf_merged_peak_count", None),
            "leaf_candidate_cluster_count_before_open_filter": self.diagnostics.get("leaf_candidate_cluster_count_before_open_filter", None),
            "leaf_rejected_small_or_early_count": self.diagnostics.get("leaf_rejected_small_or_early_count", None),
            "leaf_exclusive_assignment": self.diagnostics.get("leaf_exclusive_assignment", None),
            f"total_leaf_area_estimated_{u}2": float(total_area),
            f"mean_leaf_area_estimated_{u}2": float(total_area / len(self.open_leaf_clusters)) if self.open_leaf_clusters else None,
            "leaf_area_estimation_basis": "robust consensus of PCA hull, trimmed hull, ellipse-like area, and horizontal projected hull diagnostics",
            "leaf_area_overall_confidence": area_conf,
            f"projected_canopy_area_{u}2": float(proj_area),
            "LAI_strict_total_leaf_area_over_ground_area": lai_strict,
            "LAI_strict_status": lai_status,
            "LAI_proxy_total_leaf_area_over_projected_canopy_area": lai_proxy,
        }

    # ------------------------------------------------------------------
    # Branch angles: safe-angle proximal consensus + candidate/outlier robustification
    # ------------------------------------------------------------------
    def _branch_angle_traits(self) -> Dict[str, Any]:
        out = super()._branch_angle_traits()
        if out.get("branch_angle_status") != "ok" or not self.angle_table:
            return out

        acute_values = []
        raw_values = []
        per_leaf_iqrs = []
        low_conf = 0
        high_conf = 0

        for row in self.angle_table:
            metas = row.get("candidate_angles", []) or []
            cand_acute = [float(m.get("acute_angle_deg")) for m in metas if m.get("acute_angle_deg") is not None and np.isfinite(float(m.get("acute_angle_deg")))]
            cand_raw = [float(m.get("raw_angle_deg")) for m in metas if m.get("raw_angle_deg") is not None and np.isfinite(float(m.get("raw_angle_deg")))]

            base_acute = row.get("branch_angle_acute_deg")
            if base_acute is None:
                continue
            final_acute = float(base_acute)

            if self.cfg.branch_angle_use_candidate_outlier_rejection and len(cand_acute) >= 3:
                cons = self._robust_scalar_consensus(
                    cand_acute,
                    [f"prox_{i}" for i in range(len(cand_acute))],
                    positive=False,
                    use_log=False,
                    mad_k=float(self.cfg.branch_angle_leaf_candidate_mad_k),
                    trim_fraction=0.0,
                )
                if cons.get("status") == "ok" and cons.get("recommended") is not None:
                    final_acute = float(cons["median"])
                    row["branch_angle_acute_deg"] = final_acute
                    row["branch_angle_candidate_consensus_confidence"] = cons.get("confidence")
                    row["branch_angle_candidate_inlier_count"] = cons.get("n_inliers")
                    row["branch_angle_candidate_rejected"] = cons.get("rejected_labels", [])
                    row["branch_angle_candidate_iqr_deg"] = cons.get("iqr", row.get("branch_angle_candidate_iqr_deg"))

            # Keep raw angle as the candidate whose acute value is closest to final acute.
            if metas and np.isfinite(final_acute):
                diffs = [abs(float(m.get("acute_angle_deg", final_acute)) - final_acute) for m in metas]
                best = int(np.argmin(diffs))
                if best < len(metas) and metas[best].get("raw_angle_deg") is not None:
                    row["branch_angle_raw_deg"] = float(metas[best]["raw_angle_deg"])

            # Update confidence using candidate stability.
            iqr = row.get("branch_angle_candidate_iqr_deg", None)
            if iqr is not None and np.isfinite(float(iqr)):
                per_leaf_iqrs.append(float(iqr))
                if float(iqr) > float(self.cfg.branch_angle_stability_iqr_deg):
                    conf = str(row.get("branch_angle_confidence", "high"))
                    if "proximal_angle_unstable" not in conf:
                        row["branch_angle_confidence"] = "low_proximal_angle_unstable"

            if row.get("include_in_summary", True):
                acute_values.append(float(row["branch_angle_acute_deg"]))
                raw_values.append(float(row["branch_angle_raw_deg"]))

            if str(row.get("branch_angle_confidence", "")).startswith("low"):
                low_conf += 1
            else:
                high_conf += 1

        if not acute_values:
            return out

        acute_arr = np.asarray(acute_values, dtype=np.float64)
        raw_arr = np.asarray(raw_values, dtype=np.float64)

        if self.cfg.branch_angle_summary_use_robust_mean and len(acute_arr) >= 3:
            cons_summary = self._robust_scalar_consensus(
                acute_arr.tolist(),
                [f"leaf_{i+1}" for i in range(len(acute_arr))],
                positive=False,
                use_log=False,
                mad_k=float(self.cfg.branch_angle_summary_mad_k),
                trim_fraction=float(self.cfg.ensemble_trim_fraction),
            )
            robust_mean = cons_summary.get("recommended", float(np.mean(acute_arr)))
            summary_conf = cons_summary.get("confidence", "unknown")
        else:
            cons_summary = None
            robust_mean = float(np.mean(acute_arr))
            summary_conf = "limited_leaf_count"

        out.update({
            "branch_angle_status": "ok",
            "branch_angle_definition": "manual-protocol approximation: robust consensus angle between local stem axis and proximal leaf/petiole direction near attachment; bent outer lamina is avoided",
            "branch_angle_protocol": "proximal_multi_fraction_robust_consensus",
            "branch_angle_count": int(len(acute_arr)),
            "branch_angle_high_confidence_count": int(high_conf),
            "branch_angle_low_confidence_count": int(low_conf),
            "mean_branch_angle_raw_deg": float(np.mean(raw_arr)),
            "median_branch_angle_raw_deg": float(np.median(raw_arr)),
            "mean_branch_angle_acute_deg": float(np.mean(acute_arr)),
            "median_branch_angle_acute_deg": float(np.median(acute_arr)),
            "robust_mean_branch_angle_acute_deg": float(robust_mean),
            "branch_angle_summary_confidence": summary_conf,
            "median_per_leaf_angle_candidate_iqr_deg": float(np.median(per_leaf_iqrs)) if per_leaf_iqrs else None,
        })
        if cons_summary is not None:
            out["branch_angle_summary_rejected_leaf_angles"] = cons_summary.get("rejected_labels", [])
            out["branch_angle_summary_relative_iqr"] = cons_summary.get("relative_iqr", None)

        self.diagnostics["branch_angle_rows"] = self.angle_table
        self.diagnostics["branch_angle_summary_consensus"] = cons_summary
        return out



# -----------------------------------------------------------------------------
# Evaluation helper
# -----------------------------------------------------------------------------

def evaluate_against_reference(
    predicted_traits: Dict[str, Any],
    reference_traits: Dict[str, float],
    key_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Compare extracted traits to manual/destructive reference values.

    This function is for AFTER extraction only. It does not affect predictions.

    Args:
        predicted_traits:
            Output dictionary from extractor.run().
        reference_traits:
            Manual/destructive values, e.g.
                {
                    "plant_height_cm": 16.5,
                    "stem_perimeter_cm": 2.4,
                    "open_leaf_count": 7,
                    "mean_branch_angle_deg": 38.0,
                }
        key_map:
            Optional mapping from reference key -> predicted key. If omitted,
            this function tries a useful default map for common fields.

    Returns:
        List of rows with absolute and percentage error.
    """
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

    if key_map is not None:
        default_map.update(key_map)

    rows: List[Dict[str, Any]] = []
    for ref_key, ref_val in reference_traits.items():
        pred_key = default_map.get(ref_key, ref_key)

        # If exact predicted key not present, try model-unit variant or common suffixes.
        if pred_key not in predicted_traits:
            candidates = [k for k in predicted_traits.keys() if k.startswith(pred_key)]
            if candidates:
                pred_key = candidates[0]

        pred_val = predicted_traits.get(pred_key, None)
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
                p = float(pred_val)
                r = float(ref_val)
                ae = abs(p - r)
                pe = None if abs(r) < 1e-12 else 100.0 * ae / abs(r)
                row["absolute_error"] = ae
                row["percentage_error"] = pe
            except Exception:
                row["status"] = "non_numeric"

        rows.append(row)

    return rows




# Backward-compatible aliases for older notebooks.
ProtocolTraitConfig = TraitExtractionConfig
ProtocolAwarePlantTraitExtractor = PlantTraitExtractor3D
