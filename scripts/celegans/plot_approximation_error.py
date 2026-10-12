"""How fast does each harmonic basis capture the gene expression at one time step?

Top row: for every gene, the relative series remainder after the first n
harmonics, sum_{k>=n} c_k^2 / sum_{k>=0} c_k^2, computed from the stored
projection coefficients (see process_all_genes.py).

Bottom row: the cross-validated fraction of variance explained, R^2(n), of a
least-squares fit of the first n harmonics to the expression at the cells.
Cells are split into folds; each fold is predicted from a fit to the others and
R^2 = 1 - sum of held-out squared errors / sum of squares about the mean.

One heatmap row per gene. The first column shows the first basis; the other
columns show the difference of each other basis to it. Genes without expression
are dropped; within each row of panels, genes share the order of the first basis
(sorted by area under its curve).
"""
import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm, PowerNorm
from scipy.linalg import solve_triangular

from embryoharmonics import celegans, io

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coefficients", nargs="+", metavar="LABEL=PATH", default=[
        f"3D={RESULTS_DIR}/harmonic_coefficients_smoothed_3D_20260924.h5",
        f"2D={RESULTS_DIR}/harmonic_coefficients_smoothed_2D_20261002.h5",
        f"WH={RESULTS_DIR}/harmonic_coefficients_smoothed_WH_20261002.h5",
    ], help="Coefficient files to compare; the first one sets the gene order")
    parser.add_argument("--harmonics", nargs="+", metavar="LABEL=PATH", default=[
        f"3D={RESULTS_DIR}/embryo-harmonics_smoothed-3D_20260923.h5",
        f"2D={RESULTS_DIR}/embryo-harmonics_smoothed-2D_20261002.h5",
        f"WH={RESULTS_DIR}/embryo-harmonics_smoothed-WH_20261002.h5",
    ], help="Harmonics the coefficients were computed with, in the same order")
    parser.add_argument("--gene-path", default=os.path.join(DATA_DIR, "cpm-20260917-all010"))
    parser.add_argument("--time-step", type=int, default=452, help="(default: %(default)s)")
    parser.add_argument("--time-offset", type=int, default=380, help="(default: %(default)s)")
    parser.add_argument("--buffer", type=float, nargs="+", default=[4.0, 6.0],
                        help="As in process_all_genes.py (default: %(default)s)")
    parser.add_argument("--folds", type=int, default=10, help="(default: %(default)s)")
    parser.add_argument("--output", default=os.path.join(RESULTS_DIR, "approximation_error.png"))
    return parser.parse_args()


def split_labels(entries):
    return zip(*(entry.split("=", 1) for entry in entries))


def load_harmonics(path, time):
    """Load 3D or axisymmetric harmonics, whichever the file holds."""
    with h5py.File(path, "r") as file:
        axisymmetric = "angular_orders" in file[f"/harmonics/{time:03d}"]
    if axisymmetric:
        return celegans.load_axisymmetric_harmonics(path, time)
    return io.load_harmonics(path, time)


def series_remainder(coefficients):
    """Relative remainder sum_{k>=n} c_k^2 / sum_k c_k^2 for n = 0, ..., n_harmonics - 1."""
    energy = coefficients**2
    tail = np.cumsum(energy[:, ::-1], axis=1)[:, ::-1]
    return tail / tail[:, :1]


def cross_validated_r2(samples, values, folds, n_max):
    """Cross-validated R^2 of least-squares fits with the first n = 1, ..., n_max harmonics.

    :param samples: The harmonics at the cells (n_cells, n_harmonics)
    :param values: The expression at the cells (n_cells, n_genes)
    :return: R^2 (n_genes, n_max); NaN for genes without variance
    """
    fold = np.random.default_rng(0).permutation(len(values)) % folds
    squared_error = np.zeros((n_max, values.shape[1]))
    for f in range(folds):
        train, test = fold != f, fold == f
        # The fit with the first n columns uses the first n columns of the QR
        # decomposition, so the held-out prediction grows by one rank-one term
        # per harmonic: A_test R^-1 picks up column j, Q^T y row j.
        q, r = np.linalg.qr(samples[train, :n_max])
        test_basis = solve_triangular(r, samples[test, :n_max].T, trans="T").T
        projection = q.T @ values[train]
        residual = values[test].copy()
        for j in range(n_max):
            residual -= np.outer(test_basis[:, j], projection[j])
            squared_error[j] += (residual**2).sum(axis=0)
    total = ((values - values.mean(axis=0))**2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1 - squared_error.T / total[:, None]


def main():
    args = parse_args()
    labels, coefficient_paths = split_labels(args.coefficients)
    harmonic_labels, harmonic_paths = split_labels(args.harmonics)
    if labels != harmonic_labels:
        raise ValueError(f"Labels of coefficients {labels} and harmonics {harmonic_labels} differ")

    coefficients, gene_names = [], []
    for path in coefficient_paths:
        with h5py.File(path, "r") as file:
            time_points = file["time_points"][:]
            if args.time_step not in time_points:
                raise ValueError(f"{path} has no time step {args.time_step}")
            i = int(np.flatnonzero(time_points == args.time_step)[0])
            coefficients.append(file["gene_coefficients"][i])
            gene_names.append(file["gene_names"][:])

    if any(not np.array_equal(names, gene_names[0]) for names in gene_names):
        raise ValueError("Coefficient files have different genes")
    n_harmonics = min(c.shape[1] for c in coefficients)
    coefficients = [c[:, :n_harmonics] for c in coefficients]

    expressed = np.all([np.any(c != 0, axis=1) for c in coefficients], axis=0)
    remainders = [series_remainder(c[expressed]) for c in coefficients]
    print(f"{expressed.sum()} of {len(expressed)} genes expressed at time step {args.time_step}")

    # Expression at the cells that have data and lie inside every geometry
    gene_data = celegans.open_gene_data_loader(
        args.gene_path, location_scale=celegans.gene_data.buffered_location_scale(args.buffer))
    locations, activities = gene_data.load_all(args.time_step)
    samples = [load_harmonics(path, args.time_step - args.time_offset).sample(locations)[:n_harmonics].T
               for path in harmonic_paths]
    cells = ~np.all(np.isnan(activities), axis=0)
    cells &= np.all([np.any(s != 0, axis=1) for s in samples], axis=0)
    values = activities[expressed][:, cells].T.astype(np.float64)
    # The data is missing per cell, never per gene, so dropping cells without data suffices
    if np.isnan(values).any():
        raise ValueError(f"Some genes lack data in cells where others have it at time step {args.time_step}")
    n_max = min(n_harmonics, len(values) - int(np.ceil(len(values) / args.folds)))
    print(f"Fitting {cells.sum()} of {len(cells)} cells with up to {n_max} harmonics, {args.folds}-fold")
    r2 = [cross_validated_r2(s[cells], values, args.folds, n_max) for s in samples]

    for label, r in zip(labels, r2):
        median = np.nanmedian(r, axis=0)
        print(f"{label}: median R^2 peaks at {median.max():.3f} with n = {median.argmax() + 1}; "
              + ", ".join(f"n={n}: {median[n - 1]:.3f}" for n in (1, 5, 10, 20, 50, 100) if n <= n_max))

    # Negative R^2 (worse than the mean) is clipped to 0, as in the color scale
    r2 = [np.nan_to_num(r, nan=0).clip(0) for r in r2]
    rows = [
        (remainders, np.argsort(remainders[0].sum(axis=1)), PowerNorm(0.5, vmin=0, vmax=1),
         "series remainder", r"$\sum_{k\geq n} c_k^2 \,/\, \sum_{k\geq 0} c_k^2$", (-0.5, n_harmonics - 0.5)),
        (r2, np.argsort(-r2[0].sum(axis=1)), PowerNorm(0.5, vmin=0, vmax=1),
         "cross-validated $R^2$", rf"$R^2$ ({args.folds}-fold cross-validated)", (0.5, n_max + 0.5)),
    ]
    fig, axes = plt.subplots(2, len(labels), figsize=(5 * len(labels), 11), squeeze=False, layout="constrained")
    for row, (curves, order, norm, title, quantity, (first, last)) in zip(axes, rows):
        extent = (first, last, len(order), 0)
        image = row[0].imshow(curves[0][order], aspect="auto", norm=norm, cmap="magma",
                              interpolation="nearest", extent=extent)
        row[0].set(title=f"{labels[0]}: {title}", xlabel="number of harmonics n",
                   ylabel=f"gene (sorted by {labels[0]} area under curve)")
        fig.colorbar(image, ax=row[0], label=quantity)
        differences = [c[order] - curves[0][order] for c in curves[1:]]
        # Symmetric scale that saturates only the most extreme 0.1% of differences
        diff_norm = CenteredNorm(halfrange=max(np.percentile(np.abs(differences), 99.9), 1e-12))
        for ax, label, difference in zip(row[1:], labels[1:], differences):
            image = ax.imshow(difference, aspect="auto", norm=diff_norm, cmap="RdBu_r",
                              interpolation="nearest", extent=extent)
            ax.set(title=f"{label} $-$ {labels[0]}: {title}", xlabel="number of harmonics n")
        if len(labels) > 1:
            fig.colorbar(image, ax=row[1:], label=f"difference in {quantity}")

    for row in axes:
        for ax in row[1:]:
            ax.sharey(row[0])
            ax.tick_params(labelleft=False)
    fig.suptitle(f"Approximation of gene expression per gene, time step {args.time_step}")
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
