"""Plot, for every time step with both model and gene data, half of the
geometry's axial profile r_max(z) in the z-r plane together with the (z, r)
coordinates of all gene locations. A second profile (blue) shows the model
expanded radially (relative or by an absolute offset of about half a cell
diameter) with its tail continued along the end tangent, see
``celegans.expand_and_extend_tail``. Locations inside the model are green,
locations only inside the expansion orange, and locations outside both red
and labeled with their lineage name. One PNG per time step, axes limits
shared across all time steps.
"""
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

from embryoharmonics import celegans
from embryoharmonics.celegans.gene_data import VOXEL_SIZE_UM

matplotlib.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gene_path", help="Path to the gene data: a directory of parquet files")
    parser.add_argument("model_path", help="Path to the embryo models (avg_models HDF5 file)")
    parser.add_argument("--time-offset", type=int, default=380,
                        help="Gene data at time t is matched to the model at t - offset (default: %(default)s)")
    parser.add_argument("--output-dir", default="profile_locations", help="Directory for the PNGs")
    parser.add_argument("--expansion", choices=["none", "relative", "absolute"], default="absolute",
                        help="How to expand the model for the blue profile (default: %(default)s)")
    parser.add_argument("--amount", type=float,
                        help="Expansion amount: percent for relative (default 5), µm for absolute "
                             "(default 1.9, about half a cell diameter)")
    parser.add_argument("--linear-fraction", type=float, default=0.8,
                        help="Fraction of the end radius covered by the linear tail continuation of "
                             "the blue profile (default: %(default)s)")
    parser.add_argument("--tolerance", type=float, default=0.5,
                        help="Radial/axial tolerance in percent; seam cell nuclei lie on the profile and would "
                             "otherwise be flagged by rounding errors (default: %(default)s)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.amount is None:
        args.amount = {"none": 0.0, "relative": 5.0, "absolute": 1.9}[args.expansion]
    expansion = dict(radial_expansion=args.amount / 100) if args.expansion == "relative" else \
        dict(radial_offset=args.amount / VOXEL_SIZE_UM) if args.expansion == "absolute" else {}
    gdl = celegans.open_gene_data_loader(args.gene_path)
    models = celegans.EmbryoModelLoader(args.model_path)
    time_steps = [t for t in gdl.time_steps if t - args.time_offset in models.time_steps]
    if not time_steps:
        raise ValueError("No matching time steps found between models and gene data!")

    lineages = pq.read_table(os.path.join(args.gene_path, "xyz-lineages.parquet")).to_pydict()
    name_of = dict(zip(lineages["i"], lineages["LI"]))

    # Collect everything first to get global axes limits
    data = []
    for t in tqdm(time_steps, desc="Loading"):
        model = models.load(t - args.time_offset, symmetric=True)
        _, r_max, _, z_profile = celegans.axial_profile(model, 1000)
        extended = celegans.expand_and_extend_tail(model, linear_fraction=args.linear_fraction, **expansion)
        _, r_ext, _, z_ext = celegans.axial_profile(extended, 1000)
        locations, _ = gdl.load_all(t)
        xyz = gdl._xyz.filter(pc.field("TI") == t)  # same row order as locations
        names = [name_of[i] for i in xyz["iLI"].to_pylist()]
        r = np.hypot(locations[:, 0], locations[:, 1]) * VOXEL_SIZE_UM
        z = locations[:, 2] * VOXEL_SIZE_UM
        data.append((t, z_profile * VOXEL_SIZE_UM, r_max * VOXEL_SIZE_UM,
                     z_ext * VOXEL_SIZE_UM, r_ext * VOXEL_SIZE_UM, z, r, names))

    z_min = min(min(0, z.min()) for *_, z, _, _ in data)
    z_max = max(max(ze.max(), z.max()) for _, _, _, ze, _, z, _, _ in data)
    r_lim = max(max(re.max(), r.max()) for _, _, _, _, re, _, r, _ in data)
    pad_z, pad_r = 0.03 * (z_max - z_min), 0.1 * r_lim

    os.makedirs(args.output_dir, exist_ok=True)
    for t, zp, rp, ze, re, z, r, names in tqdm(data, desc="Plotting"):
        tol = 1 + args.tolerance / 100

        def is_outside(zp, rp):
            return (z < zp.min() - (tol - 1) * zp.max()) | (z > tol * zp.max()) | (r > tol * np.interp(z, zp, rp))

        outside_model = is_outside(zp, rp)
        outside = is_outside(ze, re) if args.expansion != "none" else outside_model
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(np.r_[zp[0], zp, zp[-1]], np.r_[0, rp, 0], "k-", lw=1.2)
        if args.expansion != "none":
            ax.plot(np.r_[ze[0], ze, ze[-1]], np.r_[0, re, 0], "-", color="tab:blue", lw=1.2)
        ax.plot(z[~outside_model], r[~outside_model], ".", color="tab:green", ms=4)
        ax.plot(z[outside_model & ~outside], r[outside_model & ~outside], ".", color="tab:orange", ms=6)
        ax.plot(z[outside], r[outside], ".", color="tab:red", ms=6)
        for k, (zi, ri, name) in enumerate(zip(z[outside], r[outside], np.array(names)[outside])):
            ax.annotate(name, (zi, ri), xytext=(3, 3 + 8 * (k % 3)), textcoords="offset points",
                        fontsize=7, color="tab:red")
        ax.set(xlim=(z_min - pad_z, z_max + pad_z), ylim=(-pad_r / 3, r_lim + pad_r),
               xlabel="z (µm)", ylabel="r (µm)", title=f"TI {t} (model {t - args.time_offset}), {outside.sum()} outside")
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f"profile_locations_{t:04d}.png"), dpi=150)
        plt.close(fig)
        print(f"TI {t}: outside = {', '.join(np.array(names)[outside]) or '-'}")
    print(f"Saved {len(data)} plots to {args.output_dir}")


if __name__ == "__main__":
    main()
