import torch
import numpy as np
import gc
from tqdm import tqdm
from pathlib import Path
from skimage import measure
import open3d as o3d

def unwrap_model(model):
    """
    Handles torch.compile(model), which wraps the original module.
    """
    return model._orig_mod if hasattr(model, "_orig_mod") else model


@torch.no_grad()
def query_density(model, pts):
    """
    Query only density from your SemanticNeRF.
    Assumes model has:
        hash_enc
        trunk
        density_head
        plant_bound
    """
    nerf = unwrap_model(model)

    pos_normalized = pts / nerf.plant_bound

    inside = (pos_normalized.abs() <= 1.0).all(dim=-1, keepdim=True)

    x = nerf.trunk(nerf.hash_enc(pos_normalized))

    density = torch.nn.functional.softplus(nerf.density_head(x))
    density = density * inside.float()

    return density

@torch.no_grad()
def build_density_volume(
    model,
    cfg,
    grid_res=None,
    bound=None,
    chunk=131072,
):
    if grid_res is None:
        grid_res = cfg.grid_res

    if bound is None:
        bound = cfg.plant_bound

    device = cfg.device

    model.eval()

    nerf = unwrap_model(model)
    if hasattr(nerf, "update_step"):
        nerf.update_step(cfg.training.num_iters)

    print("Querying density on 3D voxel grid...")

    xs = torch.linspace(-bound, bound, grid_res, device=device)

    gx, gy, gz = torch.meshgrid(xs, xs, xs, indexing="ij")

    pts_flat = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    density_vol = np.zeros(pts_flat.shape[0], dtype=np.float32)

    for s in tqdm(range(0, pts_flat.shape[0], chunk), desc="Density query"):
        e = min(s + chunk, pts_flat.shape[0])

        p = pts_flat[s:e]

        den = query_density(model, p)

        density_vol[s:e] = den.squeeze(-1).detach().cpu().numpy()

    density_vol = density_vol.reshape(grid_res, grid_res, grid_res)

    del pts_flat, gx, gy, gz, xs
    torch.cuda.empty_cache()
    gc.collect()

    print(f"Density: min={density_vol.min():.6f}, max={density_vol.max():.6f}")

    return density_vol

def print_density_stats(density_vol):
    flat = density_vol.reshape(-1)
    nz = flat[flat > 1e-8]

    print("\nDensity stats:")
    print(f"min: {flat.min():.6f}")
    print(f"max: {flat.max():.6f}")
    print(f"nonzero voxels: {len(nz):,}/{len(flat):,}")

    for p in [50, 75, 90, 92, 94, 95, 96, 97, 98, 99, 99.5, 99.9]:
        print(f"p{p:>5}: {np.percentile(nz, p):.6f}")

def extract_mesh_from_density(
    density_vol,
    cfg,
    percentile=95,
    bound=None,
):
    if bound is None:
        bound = cfg.plant_bound

    grid_res = density_vol.shape[0]

    flat = density_vol.reshape(-1)
    nz = flat[flat > 1e-8]

    iso_level = float(np.percentile(nz, percentile))

    voxel_size = 2.0 * bound / (grid_res - 1)

    print(f"\nExtracting mesh:")
    print(f"  percentile = p{percentile}")
    print(f"  iso_level  = {iso_level:.6f}")
    print(f"  voxel_size = {voxel_size:.6f}")

    verts, faces, normals, values = measure.marching_cubes(
        density_vol,
        level=iso_level,
        spacing=(voxel_size, voxel_size, voxel_size)
    )

    verts = verts + np.array([-bound, -bound, -bound], dtype=np.float32)

    print(f"Mesh: {len(verts):,} vertices, {len(faces):,} faces")

    return verts, faces, normals, values, iso_level

def save_ply_ascii(path, verts, faces, colors=None):
    path = Path(path)

    if colors is None:
        colors = np.ones((len(verts), 3), dtype=np.float32) * np.array([0.45, 0.8, 0.25])

    colors = np.clip(colors, 0.0, 1.0)
    colors_u8 = (colors * 255).astype(np.uint8)

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for v, c in zip(verts, colors_u8):
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    print(f"Saved PLY: {path}")


def clean_mesh_open3d(
    verts,
    faces,
    min_triangles=300,
    smooth=True,
    smoothing_type="taubin",
):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    mesh.compute_vertex_normals()

    print("\nConnected-component filtering...")

    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    print(f"Found {len(cluster_n_triangles)} triangle clusters")

    keep_clusters = np.where(cluster_n_triangles >= min_triangles)[0]

    if len(keep_clusters) == 0:
        print("[WARN] No clusters survived. Keeping largest cluster.")
        keep_clusters = [int(cluster_n_triangles.argmax())]

    remove_mask = ~np.isin(triangle_clusters, keep_clusters)

    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    print(
        f"After component filtering: "
        f"{len(mesh.vertices):,} vertices, {len(mesh.triangles):,} faces"
    )

    if smooth:
        print("Applying light smoothing...")

        if smoothing_type == "taubin":
            mesh = mesh.filter_smooth_taubin(
                number_of_iterations=5
            )
        elif smoothing_type == "laplacian":
            mesh = mesh.filter_smooth_laplacian(
                number_of_iterations=2,
                lambda_filter=0.3
            )
        else:
            raise ValueError(f"Unknown smoothing_type: {smoothing_type}")

        mesh.compute_vertex_normals()

    return mesh