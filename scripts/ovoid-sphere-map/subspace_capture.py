import argparse
import csv
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

from embryoharmonics.celegans import load_axisymmetric_harmonics
from embryoharmonics.fem import compute_mass_and_stiffness_axisymmetric


REFERENCE = "sphere_map"
CANDIDATES = ("exact", "horn_z_natural", "horn_z_flat", "horn_z_corrected")
SIZES = (10, 30, 100, 300)
CURVE_SIZES = (10, 30, 100)
STYLES = ("-", "--", ":")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assess how well the exact (symmetric) and z-horn harmonics "
        f"capture the subspace spanned by the first N sphere-map harmonics. "
        f"Expects {REFERENCE}_harmonics.h5, {REFERENCE}_coefficients.h5 and "
        "<variant>_harmonics.h5 files in one directory, all on the same "
        "meridian mesh. Writes subspace_capture.png and subspace_capture.csv."
    )
    parser.add_argument("directory", help="Directory with the results of all variants")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write the figure and metrics to (default: the input directory)",
    )
    return parser.parse_args()


class Blocks:
    """The Gram matrices of the reference and a candidate basis and their
    cross products, one block per (angular order, kind). 3D modes of
    different blocks are orthogonal, and within a block the angular factor
    cancels, so the meridian fields with the weight r suffice. Every mode is
    normalized like the coefficients (unit 3D norm).
    """

    def __init__(self, reference, candidate, mass):
        s, c = normalized_fields(reference, mass), normalized_fields(candidate, mass)
        self.reference_labels = labels(reference)
        self.candidate_labels = labels(candidate)
        self.blocks = []
        for label in sorted(set(self.reference_labels)):
            i = np.flatnonzero(self.reference_labels == label)
            j = np.flatnonzero(self.candidate_labels == label)
            ms, mc = mass @ s[i].T, mass @ c[j].T
            self.blocks.append((i, j, s[i] @ ms, c[j] @ mc, c[j] @ ms))

    def restricted(self, n, m):
        """Per block: the reference indices among the first n modes, and the
        Gram matrices and cross products restricted to the first n reference
        and m candidate modes.
        """
        for i, j, gs, gc, b in self.blocks:
            ks, kc = np.sum(i < n), np.sum(j < m)
            if ks > 0:
                yield i[:ks], gs[:ks, :ks], gc[:kc, :kc], b[:kc, :ks]

    def principal_cosines(self, n, m):
        """The squared cosines of the principal angles between the span of the
        first n reference modes and the span of the first m candidate modes.
        """
        cosines = []
        for _, gs, gc, b in self.restricted(n, m):
            if len(gc) == 0:
                cosines.append(np.zeros(len(gs)))
                continue
            captured = b.T @ np.linalg.solve(gc, b)
            cosines.append(scipy.linalg.eigh(captured, gs, eigvals_only=True))
        return np.clip(np.concatenate(cosines), 0, 1)

    def gene_capture(self, coefficients, n, m):
        """For every row of reference coefficients c, the squared norm of the
        projection of the data onto the first n reference modes, c^T Gs^-1 c,
        and of its projection onto the first m candidate modes.
        """
        total = np.zeros(len(coefficients))
        captured = np.zeros(len(coefficients))
        for i, gs, gc, b in self.restricted(n, m):
            c = coefficients[:, i]
            a = np.linalg.inv(gs)
            total += np.sum((c @ a) * c, axis=1)
            if len(gc) > 0:
                k = a @ b.T @ np.linalg.solve(gc, b) @ a
                captured += np.sum((c @ k) * c, axis=1)
        return total, captured


def normalized_fields(harmonics, mass):
    fields = np.array([mode.data for mode in harmonics])
    return fields / np.sqrt(np.einsum("ij,ij->i", fields, (mass @ fields.T).T))[:, None]


def labels(harmonics):
    return np.array([f"{m}{k}" for m, k in zip(harmonics.angular_orders, harmonics.trig_kinds)])


def main():
    args = parse_args()
    output_dir = args.output_dir or args.directory
    os.makedirs(output_dir, exist_ok=True)

    load = lambda name: load_axisymmetric_harmonics(
        os.path.join(args.directory, f"{name}_harmonics.h5"), 0
    )
    reference = load(REFERENCE)
    mass = compute_mass_and_stiffness_axisymmetric(reference.mesh, compute_stiffness=False).mass
    with h5py.File(os.path.join(args.directory, f"{REFERENCE}_coefficients.h5"), "r") as file:
        coefficients = file["gene_coefficients"][:]
    coefficients = coefficients.reshape(-1, coefficients.shape[-1])
    coefficients = coefficients[np.any(coefficients != 0, axis=1)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    rows = []
    for name in CANDIDATES:
        if not os.path.exists(os.path.join(args.directory, f"{name}_harmonics.h5")):
            continue
        candidate = load(name)
        if not np.allclose(candidate.mesh.points, reference.mesh.points):
            raise ValueError(f"{name} was computed on a different mesh than {REFERENCE}")
        blocks = Blocks(reference, candidate, mass)
        n_max = min(len(reference), len(candidate))

        sizes = np.unique(np.geomspace(1, n_max, 30).astype(int))
        cosines = [blocks.principal_cosines(n, n) for n in sizes]
        gene = [blocks.gene_capture(coefficients, n, n) for n in sizes]

        row = {"variant": name}
        for size in SIZES:
            if size > n_max:
                continue
            row[f"mean_capture_{size}"] = blocks.principal_cosines(size, size).mean()
            row[f"capture_by_all_{size}"] = blocks.principal_cosines(size, len(candidate)).mean()
            total, captured = blocks.gene_capture(coefficients, size, size)
            row[f"gene_capture_pooled_{size}"] = captured.sum() / total.sum()
            row[f"gene_capture_median_{size}"] = np.median(captured[total > 0] / total[total > 0])
            total, captured = blocks.gene_capture(coefficients, size, len(candidate))
            row[f"gene_capture_by_all_{size}"] = captured.sum() / total.sum()
        rows.append(row)
        print(", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in row.items()))

        line, = axes[0, 0].plot(sizes, [c.mean() for c in cosines], marker=".", label=name)
        color = line.get_color()
        candidate_sizes = np.unique(np.geomspace(1, len(candidate), 40).astype(int))
        for size, style in zip(CURVE_SIZES, STYLES):
            capture = [blocks.principal_cosines(size, m).mean() for m in candidate_sizes]
            axes[0, 1].plot(candidate_sizes, capture, ls=style, color=color,
                            label=name if size == CURVE_SIZES[0] else None)
        axes[1, 0].plot(sizes, [c.sum() / t.sum() for t, c in gene], marker=".",
                        color=color, label=name)
        axes[1, 0].plot(sizes, [np.median(c[t > 0] / t[t > 0]) for t, c in gene],
                        ls=":", color=color)

        # The part of every reference mode outside the span of all candidate
        # modes, to see which modes are missed
        if name == "exact":
            missed = np.zeros(n_max)
            for indices, gs, gc, b in blocks.restricted(n_max, n_max):
                projection = np.einsum("ij,ij->j", b, np.linalg.solve(gc, b))
                missed[indices] = 1 - projection / np.diag(gs)
            scatter = axes[1, 1].scatter(np.arange(n_max), missed, s=6,
                                         c=reference.angular_orders[:n_max], cmap="viridis")
            fig.colorbar(scatter, ax=axes[1, 1], label="angular order m")

    axes[0, 0].set(title="Captured subspace of the first N sphere-map modes\nby the first N modes",
                   xlabel="N", ylabel="mean squared cosine of the principal angles", xscale="log",
                   ylim=(0, 1.02))
    for size, style in zip(CURVE_SIZES, STYLES):
        axes[0, 1].plot([], [], ls=style, color="k", label=f"N = {size}")
    axes[0, 1].set(title="Captured subspace of the first N sphere-map modes\nby the first M modes",
                   xlabel="M", ylabel="mean squared cosine of the principal angles",
                   xscale="log", ylim=(0, 1.02))
    axes[1, 0].plot([], [], ls=":", color="k", label="median over genes and time steps")
    axes[1, 0].set(title="Captured sphere-map projection of the gene data\nby the first N modes",
                   xlabel="N", ylabel="fraction of the squared norm", xscale="log",
                   ylim=(0, 1.02))
    axes[1, 1].set(title=f"Part of every sphere-map mode outside the first {n_max} exact modes",
                   xlabel="# sphere-map harmonic", ylabel="1 - squared projection",
                   yscale="log", ylim=(1e-6, 2))
    for ax in axes.flat[:3]:
        ax.legend(fontsize=8)
    fig.tight_layout()
    figure_file = os.path.join(output_dir, "subspace_capture.png")
    fig.savefig(figure_file, dpi=200)
    print(f"Saved {figure_file}")

    metrics_file = os.path.join(output_dir, "subspace_capture.csv")
    with open(metrics_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {metrics_file}")


if __name__ == "__main__":
    main()
