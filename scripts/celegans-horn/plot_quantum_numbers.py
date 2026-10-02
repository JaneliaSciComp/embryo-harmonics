import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import pyvista as pv

from embryoharmonics import celegans
from embryoharmonics.celegans import compute_webster_horn_meridian_harmonics


def parse_args():
    parser = argparse.ArgumentParser(
        description="For one time point, compute the lowest Webster-Horn modes "
        "of each azimuthal branch m = 0..k_max and plot them as a grid indexed "
        "by their exact quantum numbers (l, n), one figure per m."
    )
    parser.add_argument("path", help="Path to celegans_models.h5")
    parser.add_argument("time_step", type=int, help="Time step to process")
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
    model = celegans.EmbryoModelLoader(args.path).load(args.time_step, symmetric=True)
    mesh = model.generate_meridian_mesh(mesh_size=args.mesh_size)
    triangulation = mtri.Triangulation(
        mesh.points[:, 0], mesh.points[:, 2], mesh.cells_dict[pv.CellType.TRIANGLE]
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for k in range(args.k_max + 1):
        harmonics, labels = compute_webster_horn_meridian_harmonics(
            mesh, model, n=args.n_modes, angular_order=k
        )

        n_rows = max(l for _, l, _ in labels)
        n_cols = max(n for _, _, n in labels) + 1
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(1.0 * n_cols + 0.5, 14 * n_rows + 0.8),
            squeeze=False,
        )
        for ax in axes.flat:
            ax.set_axis_off()

        for i, (_, l, n) in enumerate(labels):
            ax = axes[l - 1, n]
            ax.tripcolor(
                triangulation, harmonics[i].data, cmap="turbo", shading="gouraud"
            )
            ax.set_aspect("equal")
            ax.set_title(
                f"$l$={l}, $n$={n}\n"
                f"$\\lambda$={harmonics.eigenvalues[i]:.2e}",
                fontsize=6,
            )

        fig.suptitle(f"t = {args.time_step}, m = {k} (Webster-Horn)")
        fig.tight_layout()
        file_name = os.path.join(
            args.output_dir, f"horn_quantum_grid_t{args.time_step:03d}_k{k}.png"
        )
        fig.savefig(file_name, dpi=300)
        print(f"Saved {file_name}")

    plt.show()


if __name__ == "__main__":
    main()
