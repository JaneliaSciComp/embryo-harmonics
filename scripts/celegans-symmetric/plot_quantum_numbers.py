import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import pyvista as pv

from embryoharmonics import celegans
from embryoharmonics.celegans import AxisymmetricHarmonics


def parse_args():
    parser = argparse.ArgumentParser(
        description="For one time point, compute the lowest meridian modes of "
        "each angular branch k = 0..k_max and plot them as a grid indexed by "
        "the nodal quantum numbers (n_r, n_z), one figure per k."
    )
    parser.add_argument("path", help="Path to celegans_models.h5")
    parser.add_argument("time_step", type=int, help="Time step to process")
    parser.add_argument(
        "--n-modes",
        type=int,
        default=30,
        help="Number of meridian modes per angular branch (default: %(default)s)",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=3,
        help="Highest angular order to compute (default: %(default)s)",
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
        harmonics = AxisymmetricHarmonics.compute(
            mesh, n=args.n_modes, angular_order=k
        )
        n_radial, n_axial = harmonics.nodal_counts()

        n_rows, n_cols = n_radial.max() + 1, n_axial.max() + 1
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(1.0 * n_cols + 0.5, 14 * n_rows + 0.8),
            squeeze=False,
        )
        for ax in axes.flat:
            ax.set_axis_off()

        occupied = {}
        for i, _ in enumerate(harmonics):
            cell = (n_radial[i], n_axial[i])
            if cell in occupied:
                print(
                    f"k={k}: modes {occupied[cell]} and {i} both labeled "
                    f"(n_r, n_z)={cell}; plotting only the first"
                )
                continue
            occupied[cell] = i
            ax = axes[cell]
            ax.tripcolor(
                triangulation, harmonics[i].data, cmap="turbo", shading="gouraud"
            )
            ax.set_aspect("equal")
            ax.set_title(
                f"$n_r$={cell[0]}, $n_z$={cell[1]}\n"
                f"$\\lambda$={harmonics.eigenvalues[i]:.2e}",
                fontsize=6,
            )

        fig.suptitle(f"t = {args.time_step}, k = {k}")
        fig.tight_layout()
        file_name = os.path.join(
            args.output_dir, f"quantum_grid_t{args.time_step:03d}_k{k}.png"
        )
        fig.savefig(file_name, dpi=300)
        print(f"Saved {file_name}")

    plt.show()


if __name__ == "__main__":
    main()
