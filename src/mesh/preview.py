import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def set_axes_equal(ax, verts):
    """
    Make 3D plot axes have equal scale.
    Otherwise the plant may look stretched/compressed.
    """
    x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    z_min, z_max = verts[:, 2].min(), verts[:, 2].max()

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)

    radius = 0.5 * max(
        x_max - x_min,
        y_max - y_min,
        z_max - z_min
    )

    ax.set_xlim(x_mid - radius, x_mid + radius)
    ax.set_ylim(y_mid - radius, y_mid + radius)
    ax.set_zlim(z_mid - radius, z_mid + radius)


def preview_mesh(
    verts,
    faces,
    vcol=None,
    output_path=None,
    title="3D Reconstruction Preview",
    max_faces=6000,
    use_semantic_colors=False,
):
    """
    Preview extracted mesh from multiple camera angles.

    Args:
        verts: [V, 3]
        faces: [F, 3]
        vcol: optional [V, 3] colors in [0, 1]
        output_path: optional save path
        use_semantic_colors: only True if semantic head was trained
    """

    fig = plt.figure(figsize=(14, 5))

    views = [
        (25, 30),
        (25, 120),
        (25, 210),
    ]

    # Face subsampling for speed
    if len(faces) > max_faces:
        stride = max(1, len(faces) // max_faces)
        fs = faces[::stride][:max_faces]
    else:
        fs = faces

    # Default neutral plant color
    if vcol is None or not use_semantic_colors:
        face_colors = np.tile(
            np.array([[0.45, 0.75, 0.25, 0.75]]),
            (len(fs), 1)
        )
    else:
        face_colors = np.array([
            [vcol[f[0], 0], vcol[f[0], 1], vcol[f[0], 2], 0.75]
            for f in fs
        ])

    for idx, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")

        poly = Poly3DCollection(
            [verts[f] for f in fs],
            alpha=0.75,
            linewidths=0.02
        )

        poly.set_facecolor(face_colors)
        poly.set_edgecolor((0, 0, 0, 0.05))

        ax.add_collection3d(poly)

        set_axes_equal(ax, verts)

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"View {idx + 1}", fontsize=10)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(str(output_path), dpi=180, bbox_inches="tight")
        print(f"Saved mesh preview to: {output_path}")

    plt.show()