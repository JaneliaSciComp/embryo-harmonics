"""Find how much the geometries have to be expanded in radial (x, y) and axial
(z) direction so that the gene locations of all time points lie inside them.
The geometry is approximated by its axial profile r_max(z) from the seam cell
splines; a location (r, z) is inside the geometry expanded by (buf_r, buf_z) if
r < (1 + buf_r) * r_max(z / (1 + buf_z)). Plots the fraction of locations
inside on a grid of expansion factors.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from embryoharmonics import celegans


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gene_path", help="Path to the gene data: an HDF5 file or a directory of parquet files")
    parser.add_argument("model_path", help="Path to the embryo models (avg_models HDF5 file)")
    parser.add_argument("--time-offset", type=int, default=380,
                        help="Gene data at time t is matched to the model at t - offset (default: %(default)s)")
    parser.add_argument("--max-expansion", type=int, default=20,
                        help="Largest expansion in percent to test on both axes (default: %(default)s)")
    parser.add_argument("--output", default="minimal_expansion.png", help="Plot file name")
    return parser.parse_args()


def main():
    args = parse_args()
    gdl = celegans.open_gene_data_loader(args.gene_path)
    models = celegans.EmbryoModelLoader(args.model_path)
    time_steps = [t for t in gdl.time_steps if t - args.time_offset in models.time_steps]
    if not time_steps:
        raise ValueError("No matching time steps found between models and gene data!")

    expansion = np.arange(args.max_expansion + 1)
    factor = 1 + expansion / 100
    inside = np.zeros((len(expansion), len(expansion)), dtype=int)  # (radial, axial)
    n_points = 0
    for t in tqdm(time_steps):
        _, r_max, _, z_profile = celegans.axial_profile(models.load(t - args.time_offset), 1000)
        locations, _ = gdl.load_all(t)
        r = np.hypot(locations[:, 0], locations[:, 1])
        z = locations[:, 2]
        n_points += len(z)
        # Radius of the axially expanded geometry at each location, 0 beyond its ends
        r_geometry = np.stack([np.interp(z / f, z_profile, r_max, left=0, right=0) for f in factor])
        inside += np.sum(r[None, None, :] < factor[:, None, None] * r_geometry[None, :, :], axis=2)

    fraction = 100 * inside / n_points
    print(f"{n_points} gene locations over {len(time_steps)} time steps")
    print("rows: radial expansion %, columns: axial expansion %, values: locations inside %")
    print("     " + "".join(f"{e:7d}" for e in expansion))
    for e, row in zip(expansion, fraction):
        print(f"{e:4d} " + "".join(f"{v:7.2f}" for v in row))

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.pcolormesh(expansion, expansion, fraction.T, shading="nearest")
    fig.colorbar(image, ax=ax, label="gene locations inside (%)")
    contours = ax.contour(expansion, expansion, fraction.T, levels=[95, 98, 99, 99.5, 99.9, 100],
                          colors="w", linewidths=0.8)
    ax.clabel(contours, fmt="%g%%", fontsize=7)
    ax.set(xlabel="radial expansion (%)", ylabel="axial expansion (%)",
           title=os.path.basename(args.gene_path.rstrip("/")))
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
