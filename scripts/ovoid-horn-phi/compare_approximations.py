import argparse
import csv
import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

from embryoharmonics.celegans import load_axisymmetric_harmonics
from embryoharmonics.fem import compute_mass_and_stiffness_axisymmetric


EXACT = "exact"
CAPTURE_SIZES = (10, 30, 100, 300)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Webster-Horn approximations of the ovoid harmonics "
        "with the exact ones. Expects a directory with <variant>_harmonics.h5 "
        "and (optionally) <variant>_coefficients.h5 files, one variant named "
        f"'{EXACT}', all computed on the same meridian mesh."
    )
    parser.add_argument("directory", help="Directory with the results of all variants")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write the figure and metrics to (default: the input directory)",
    )
    return parser.parse_args()


def fields_of(harmonics):
    """The meridian fields of all modes, one per row."""
    return np.array([mode.data for mode in harmonics])


def best_match(exact, approximate, mass):
    """For every exact mode, the largest normalized scalar product with an
    approximate mode of the same angular order and kind. Degenerate cos/sin
    partners share their meridian field, so each pair is matched as one.
    """
    e, a = fields_of(exact), fields_of(approximate)
    e = e / np.sqrt(np.einsum("ij,ij->i", e, (mass @ e.T).T))[:, None]
    a = a / np.sqrt(np.einsum("ij,ij->i", a, (mass @ a.T).T))[:, None]
    products = np.abs(a @ (mass @ e.T))
    same = (approximate.angular_orders[:, None] == exact.angular_orders[None, :]) \
        & (approximate.trig_kinds[:, None] == exact.trig_kinds[None, :])
    return np.where(same, products, 0).max(axis=0)


def subspace_capture(exact, approximate, mass, n):
    """The mean squared norm of the projection of the first n exact modes onto
    the span of the first n approximate modes (1 means the same subspace).
    The 3D modes of different angular order or kind are orthogonal, so the
    projection splits into one block per (order, kind).
    """
    e, a = fields_of(exact)[:n], fields_of(approximate)[:n]
    e = e / np.sqrt(np.einsum("ij,ij->i", e, (mass @ e.T).T))[:, None]
    captured = 0.0
    for j in range(len(e)):
        block = (approximate.angular_orders[:n] == exact.angular_orders[j]) \
            & (approximate.trig_kinds[:n] == exact.trig_kinds[j])
        if not block.any():
            continue
        f = a[block]
        gram = f @ (mass @ f.T)
        b = f @ (mass @ e[j])
        captured += b @ np.linalg.lstsq(gram, b, rcond=None)[0]
    return captured / len(e)


def energy_fraction(coefficient_file):
    """The fraction of the coefficient energy in the first N modes, averaged
    over all genes and time steps with any expression.
    """
    with h5py.File(coefficient_file, "r") as file:
        coefficients = file["gene_coefficients"][:]
    energy = np.cumsum(coefficients ** 2, axis=-1).reshape(-1, coefficients.shape[-1])
    energy = energy[energy[:, -1] > 0]
    return (energy / energy[:, -1:]).mean(axis=0)


def main():
    args = parse_args()
    output_dir = args.output_dir or args.directory
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.directory, "*_harmonics.h5")))
    variants = {os.path.basename(f)[:-len("_harmonics.h5")]: f for f in files}
    if EXACT not in variants:
        raise ValueError(f"No {EXACT}_harmonics.h5 in {args.directory}")

    exact = load_axisymmetric_harmonics(variants[EXACT], 0)
    mass = compute_mass_and_stiffness_axisymmetric(exact.mesh, compute_stiffness=False).mass

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    rows = []
    for name, file in variants.items():
        harmonics = load_axisymmetric_harmonics(file, 0)
        if not np.allclose(harmonics.mesh.points, exact.mesh.points):
            raise ValueError(f"{name} was computed on a different mesh than {EXACT}")
        n = min(len(harmonics), len(exact))
        row = {"variant": name}

        error = harmonics.eigenvalues[1:n] / exact.eigenvalues[1:n] - 1
        row["mean_eigenvalue_error_first_30"] = np.mean(np.abs(error[:29]))
        row["mean_eigenvalue_error_all"] = np.mean(np.abs(error))
        match = best_match(exact.subset(np.arange(n)), harmonics, mass)
        row["mean_best_match_first_30"] = match[:30].mean()
        row["mean_best_match_all"] = match.mean()
        for size in CAPTURE_SIZES:
            if size <= n:
                row[f"subspace_capture_{size}"] = subspace_capture(exact, harmonics, mass, size)

        coefficient_file = os.path.join(args.directory, f"{name}_coefficients.h5")
        if os.path.exists(coefficient_file):
            fraction = energy_fraction(coefficient_file)
            for size in CAPTURE_SIZES:
                if size <= len(fraction):
                    row[f"gene_energy_fraction_{size}"] = fraction[size - 1]
            axes[1, 1].plot(np.arange(1, len(fraction) + 1), fraction, label=name)
        rows.append(row)
        print(", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in row.items()))

        if name == EXACT:
            continue
        axes[0, 0].plot(np.arange(1, n), 100 * error, label=name, lw=0.8)
        axes[0, 1].plot(match, label=name, lw=0.8)
        sizes = np.unique(np.geomspace(1, n, 25).astype(int))
        axes[1, 0].plot(
            sizes, [subspace_capture(exact, harmonics, mass, s) for s in sizes],
            label=name, marker=".",
        )

    axes[0, 0].axhline(0, color="k", lw=0.5)
    axes[0, 0].set(title="Eigenvalue error", xlabel="# harmonic", ylabel="relative error (%)")
    axes[0, 1].set(title="Best match of every exact mode", xlabel="# exact harmonic",
                   ylabel="|scalar product|", ylim=(0, 1.02))
    axes[1, 0].set(title="Captured subspace of the first N exact modes", xlabel="N",
                   ylabel="mean squared projection", xscale="log", ylim=(0, 1.02))
    axes[1, 1].set(title="Gene coefficient energy in the first N modes", xlabel="N",
                   ylabel="fraction (mean over genes and time steps)", xscale="log")
    for ax in axes.flat:
        ax.legend(fontsize=8)
    fig.tight_layout()
    figure_file = os.path.join(output_dir, "approximation_comparison.png")
    fig.savefig(figure_file, dpi=200)
    print(f"Saved {figure_file}")

    metrics_file = os.path.join(output_dir, "approximation_metrics.csv")
    with open(metrics_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {metrics_file}")


if __name__ == "__main__":
    main()
