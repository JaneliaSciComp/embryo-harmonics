import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import pyvista as pv

from embryoharmonics.celegans import OvoidModel
from embryoharmonics.celegans.ovoid_horn import compute_sphere_map_harmonics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute the lowest sphere-map modes of the ovoid "
        "for each azimuthal branch m = 0..k_max and plot them as a grid indexed "
        "by their quantum numbers (p, k), one figure per m."
    )
    parser.add_argument("path", help="Path to the csv file of outline Fourier coefficients")
    parser.add_argument(
        "--n-modes",
        type=int,
        default=30,
        help="Number of modes per azimuthal branch (default: %(default)s)",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=3,
        help="Highest azimuthal order to compute (default: %(default)s)",
    )
    parser.add_argument(
        "--mesh-size",
        type=float,
        default=5,
        help="Mesh size of the meridian mesh (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getcwd(),
        help="Directory to save the figures to (default: current directory)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ovoid = OvoidModel.from_csv(args.path)
    mesh = ovoid.generate_meridian_mesh(mesh_size=args.mesh_size)
    triangulation = mtri.Triangulation(
        mesh.points[:, 0], mesh.points[:, 2], mesh.cells_dict[pv.CellType.TRIANGLE]
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for k in range(args.k_max + 1):
        harmonics, labels = compute_sphere_map_harmonics(
            mesh, ovoid, n=args.n_modes, angular_order=k
        )

        # Rows: radial zero p; columns: number of polar nodes k - m
        n_rows = max(p for _, _, p in labels)
        n_cols = max(order - k for _, order, _ in labels) + 1
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(1.0 * n_cols + 0.5, 2.0 * n_rows + 0.8),
            squeeze=False,
        )
        for ax in axes.flat:
            ax.set_axis_off()

        for i, (_, order, p) in enumerate(labels):
            ax = axes[p - 1, order - k]
            ax.tripcolor(
                triangulation, harmonics[i].data, cmap="turbo", shading="gouraud"
            )
            ax.set_aspect("equal")
            ax.set_title(
                f"$p$={p}, $k$={order}\n"
                f"$\\lambda$={harmonics.eigenvalues[i]:.2e}",
                fontsize=6,
            )

        fig.suptitle(f"Ovoid, m = {k} (sphere map)")
        fig.tight_layout()
        file_name = os.path.join(
            args.output_dir, f"ovoid_sphere_map_quantum_grid_k{k}.png"
        )
        fig.savefig(file_name, dpi=300)
        print(f"Saved {file_name}")

    plt.show()


if __name__ == "__main__":
    main()
