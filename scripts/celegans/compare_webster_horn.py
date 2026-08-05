"""How closely does the Webster-Horn approximation reproduce the natural basis
of an embryo geometry?

Two errors are separated, both measured against the Webster-Horn basis of the
rotationally symmetric geometry:

* adiabatic error -- the cost of the 1D reduction, on a geometry where the
  approximation is exactly applicable. This is the headline number.
* symmetrization error -- the additional cost of treating the real,
  non-symmetric embryo as a body of revolution.

Each error has a mode-shape part and an eigenvalue part. The mode-shape part is
a one-to-one pairing: every FEM eigenfunction is matched with the single
Webster-Horn eigenfunction closest to it, and the scalar product between the two
is reported. Where the Webster-Horn eigenvalue is degenerate (m > 0, giving the
pair cos(m*theta), sin(m*theta)) the matched eigenfunction is the rotation
cos(m*(theta - phi)) that fits best, since the phase of a degenerate pair is
gauge rather than part of the approximation.

A third, Webster-Horn-free comparison isolates the symmetrization error alone:
the symmetric geometry's own FEM basis, resampled onto the raw mesh, against the
raw geometry's FEM basis. Since resampling destroys mass-normalization and
within-cluster orthogonality, it also carries interpolation error and should be
read as a lower bound on agreement -- see
scripts/celegans/compare_webster_horn_timeline.py for the same comparison
across the whole time course.
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from embryoharmonics import Harmonics, celegans
from embryoharmonics.celegans.webster_horn_harmonics import (
    compute_webster_horn_harmonics,
    degenerate_clusters,
    match_modes,
    orthonormalize_clusters,
)
from embryoharmonics.fem import FemMatrices

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--plot-harmonics", action="store_true",
    help="for each mode, render the raw/symmetric/Webster-Horn triplet from three "
         "orthogonal views into results/harmonic_plots_<time step>/harmonic_<k>.png "
         "(slow: one render per mode)"
)
args = parser.parse_args()

CWD = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CWD, "..", "..", "data", "avg_models_n371.h5")
RESULTS_DIR = os.path.join(CWD, "..", "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TIME_STEP = 370  # middle of the 371 averaged models
N = 100  # below ~15 modes the spectrum is purely axial (m = 0) and degeneracy-free
# Every basis used for one-to-one matching (Webster-Horn and, for the
# symmetrization comparison, the symmetric geometry's own FEM basis) is
# computed with this many modes -- deliberately more than N -- so even the
# last of the N target modes still has its correct partner available, instead
# of being forced onto the wrong one because its partner was cut off by N.
N_POOL = N + 20
MESH_SIZE = 5
MAX_TICKS = 25  # thin out tick labels so they stay readable

model_loader = celegans.EmbryoModelLoader(MODEL_PATH)

# The rotationally symmetric geometry, with both bases on the same mesh
# (so nothing has to be resampled).
embryo_model_sym = model_loader.load(TIME_STEP, symmetric=True)
mesh_sym = embryo_model_sym.generate_mesh(mesh_size=MESH_SIZE)
webster, labels = compute_webster_horn_harmonics(mesh_sym, embryo_model_sym, n=N_POOL)
harmonics_sym = Harmonics.compute(mesh_sym, n=N_POOL)
print(f"symmetric mesh: {mesh_sym.n_cells} cells, "
      f"{N} FEM modes vs a pool of {len(webster)} Webster-Horn modes "
      f"(FEM-sym also computed with an {N_POOL}-mode pool, for the "
      f"symmetrization comparison below)")

# The raw, non-symmetric geometry. Its mesh is stretched onto the symmetric one
# (as in scripts/celegans/correlate_harmonics.py); since the Webster-Horn modes
# are analytic they can be evaluated directly on the stretched mesh, which
# avoids the interpolation error that resampling the harmonics would introduce.
embryo_model_raw = model_loader.load(TIME_STEP)
mesh_raw = embryo_model_raw.generate_mesh(mesh_size=MESH_SIZE)
mesh_raw_stretched = celegans.transform_mesh(mesh_raw, mesh_sym)
webster_on_raw, _ = compute_webster_horn_harmonics(
    mesh_raw_stretched, embryo_model_sym, n=N_POOL
)
harmonics_raw_on_stretched = Harmonics.compute(mesh_raw_stretched, n=N)
print(f"non-symmetric mesh: {mesh_raw.n_cells} cells")

clusters = degenerate_clusters(labels)
scale = harmonics_sym.eigenvalues[N - 1]


def coefficient_matrix(basis, targets):
    """Stack decomposition coefficients column-wise: entry (i, j) is the
    coefficient of target mode j with respect to basis mode i.
    """
    coefficients = basis.decompose(targets)
    return np.column_stack([coefficients[data.name] for data in targets])


# Adiabatic error: Webster-Horn vs the true harmonics of the same (symmetric)
# geometry. Total error: the same basis against the true harmonics of the raw
# geometry, stretched onto the symmetric one.
#
# targets_sym_pool decomposes the full N_POOL FEM-sym pool against the
# Webster-Horn basis -- not just the first N -- so the symmetrization
# comparison further down can reuse that pairing as a padded basis of its own.
# match_modes scores each target column independently, so this doesn't change
# the matches for the first N columns, only adds pairings for the rest.
targets_sym_pool = [harmonics_sym[k] for k in range(N_POOL)]
targets_sym = targets_sym_pool[:N]
targets_raw = [harmonics_raw_on_stretched[k] for k in range(N)]
matrix_adiabatic_pool = coefficient_matrix(webster, targets_sym_pool)
matrix_total = coefficient_matrix(webster_on_raw, targets_raw)

# One-to-one pairing: for each FEM eigenfunction, the Webster-Horn
# eigenfunction it matches best, and the scalar product with it.
matched_adiabatic_pool, product_adiabatic_pool = match_modes(matrix_adiabatic_pool, clusters)
matched_adiabatic, product_adiabatic = matched_adiabatic_pool[:N], product_adiabatic_pool[:N]
matched_total, product_total = match_modes(matrix_total, clusters)

# The pairing is only one-to-one if no cluster is over-subscribed: a cluster of
# dimension d can host at most d mutually orthogonal eigenfunctions. Clusters at
# the top of the pool going unclaimed is expected and harmless; two FEM modes
# competing for the same one is not, and means the approximation has failed to
# separate them. Checked over the first N modes only -- that's the actual
# target set; the rest of the pool exists only to give them room.
claims = np.bincount(matched_adiabatic, minlength=len(clusters))
dimensions = np.array([len(indices) for indices in clusters])
oversubscribed = claims > dimensions
is_one_to_one = not oversubscribed.any()

# Symmetrization alone, no Webster-Horn involved: the symmetric geometry's own
# FEM basis, resampled onto the raw mesh, against the raw geometry's FEM basis.
# The degenerate structure of a FEM basis is not known from mode labels, so it
# is taken from how the adiabatic pairing above matched the symmetric basis.
# Grouping the full N_POOL pool (not just the first N) gives this basis the
# same headroom the Webster-Horn bases have, so a raw FEM mode near the N
# cutoff can still land on its correct (possibly beyond-N) symmetric partner
# instead of being forced onto the wrong cluster.
fem_clusters = [
    list(np.flatnonzero(matched_adiabatic_pool == cluster))
    for cluster in np.unique(matched_adiabatic_pool)
]
resampled = np.array([
    data.resample(mesh_raw_stretched, project_outside_data=True).data
    for data in targets_sym_pool
])
mass_raw = FemMatrices.compute_for(mesh_raw_stretched, stiffness=False).mass
orthonormalize_clusters(resampled, fem_clusters, mass_raw)
harmonics_sym_on_raw = Harmonics(
    mesh_raw_stretched, resampled, harmonics_sym.eigenvalues
)
_, product_sym_raw = match_modes(
    coefficient_matrix(harmonics_sym_on_raw, targets_raw), fem_clusters
)

# Eigenvalue errors, relative to the spectral range so the zero (constant) mode
# does not blow up. Mode 0 is exactly zero for both bases by construction.
eigenvalues_sym = harmonics_sym.eigenvalues[:N]
eigenvalue_error_adiabatic = np.abs(webster.eigenvalues[:N] - eigenvalues_sym) / scale
eigenvalue_error_total = np.abs(
    webster.eigenvalues[:N] - harmonics_raw_on_stretched.eigenvalues
) / scale
eigenvalue_error_sym = np.abs(
    harmonics_raw_on_stretched.eigenvalues - eigenvalues_sym
) / scale


def block_label(indices):
    """Name a cluster by its (m, l, n) label; a degenerate cos/sin pair shares
    one label.
    """
    m, l, n = labels[indices[0]]
    return f"({m},{l},{n})"


block_labels = [block_label(indices) for indices in clusters]
modes = np.arange(N)

print(f"\n{'FEM mode':>9} {'matched (m,l,n)':>16} {'dim':>4} {'lambda_FEM':>11} "
      f"{'<u,w>':>8} {'<u,w>_raw':>10} {'<u,w>_sym':>10}")
for k in modes:
    cluster = matched_adiabatic[k]
    flag = "  (contested)" if oversubscribed[cluster] else ""
    print(f"{k:>9} {block_labels[cluster]:>16} {len(clusters[cluster]):>4} "
          f"{harmonics_sym.eigenvalues[k]:>11.4g} "
          f"{product_adiabatic[k]:>8.4f} {product_total[k]:>10.4f} "
          f"{product_sym_raw[k]:>10.4f}{flag}")

print(f"\npairing is one-to-one: {is_one_to_one}"
      f" ({oversubscribed.sum()} clusters contested by more FEM modes than "
      f"their dimension)")
print(f"\nscalar product with the matched Webster-Horn eigenfunction:")
print(f"  adiabatic (symmetric geometry): worst {product_adiabatic.min():.4f}, "
      f"median {np.median(product_adiabatic):.4f}")
print(f"  total (raw geometry):           worst {product_total.min():.4f}, "
      f"median {np.median(product_total):.4f}")
print(f"  symmetrization (FEM sym vs FEM raw, no WH): "
      f"worst {product_sym_raw.min():.4f}, "
      f"median {np.median(product_sym_raw):.4f}")
print(f"\neigenvalue error, max over modes:")
print(f"  adiabatic:      {eigenvalue_error_adiabatic.max():.2e}")
print(f"  total:          {eigenvalue_error_total.max():.2e}")
print(f"  symmetrization: {eigenvalue_error_sym.max():.2e}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(modes, product_adiabatic, "o-", ms=4,
             label="adiabatic (WH vs FEM, symmetric)")
axes[0].plot(modes, product_total, "s-", ms=4,
             label="total (WH vs FEM, raw)")
axes[0].plot(modes, product_sym_raw, "^-", ms=4,
             label="symmetrization (FEM sym vs FEM raw)")
axes[0].axhline(1.0, color="gray", linestyle=":", linewidth=1)
axes[0].set_xlabel("FEM harmonic index")
axes[0].set_ylabel(r"$\langle u_{FEM},\, w_{WH}\rangle$")
axes[0].set_title("Scalar product with the matched Webster-Horn eigenfunction")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].semilogy(modes[1:], eigenvalue_error_adiabatic[1:], "o-", ms=3,
                 label="adiabatic (WH vs FEM, symmetric)")
axes[1].semilogy(modes[1:], eigenvalue_error_total[1:], "^-", ms=3,
                 label="total (WH vs FEM, raw)")
axes[1].semilogy(modes[1:], eigenvalue_error_sym[1:], "s-", ms=3,
                 label="symmetrization (FEM raw vs symmetric)")
axes[1].set_xlabel("mode index")
axes[1].set_ylabel(r"$|\Delta\lambda| / \lambda_{max}$")
axes[1].set_title("Eigenvalue error by source")
axes[1].grid(True, which="both", alpha=0.3)
axes[1].legend()

fig.suptitle(
    f"Error budget: adiabatic vs symmetrization "
    f"(time step {TIME_STEP}, {N} modes)"
)
fig.tight_layout()
budget_path = os.path.join(RESULTS_DIR, "webster_horn_error_budget.png")
fig.savefig(budget_path, dpi=150)

print(f"\nSaved {budget_path}")

if args.plot_harmonics:
    # One figure per mode: raw / symmetric / Webster-Horn side by side (columns),
    # each from three axis-aligned views stacked top to bottom (rows). Webster-Horn
    # is a member of the cluster matched to the symmetric FEM mode at this index
    # (see the adiabatic matching above) -- for a degenerate (m > 0) cluster, which
    # member is not the phase-fitted rotation match_modes scores against, but a
    # cluster has as many members as there are FEM modes matched to it, so cycling
    # through them by occurrence at least shows two distinct cos/sin-like shapes
    # instead of plotting the same array twice.
    plot_dir = os.path.join(RESULTS_DIR, f"harmonic_plots_{TIME_STEP}")
    os.makedirs(plot_dir, exist_ok=True)
    views = ("view_yz", "view_xz", "view_xy")
    cluster_occurrence = {}

    for k in modes:
        k = int(k)
        cluster = matched_adiabatic[k]
        occurrence = cluster_occurrence.get(cluster, 0)
        cluster_occurrence[cluster] = occurrence + 1
        webster_index = int(clusters[cluster][occurrence % len(clusters[cluster])])
        columns = (
            ("raw", harmonics_raw_on_stretched[k]),
            ("symmetric", harmonics_sym[k]),
            ("Webster-Horn", webster[webster_index]),
        )
        clim = max(np.abs(data.data).max() for _, data in columns)

        plotter = pv.Plotter(shape=(3, 3), off_screen=True)
        for col, (title, data) in enumerate(columns):
            for row, view in enumerate(views):
                plotter.subplot(row, col)
                plotter.add_mesh(data.mesh, scalars=data.data, cmap="RdBu",
                                  clim=(-clim, clim), show_scalar_bar=False)
                if row == 0:
                    plotter.add_text(title, font_size=10)
                getattr(plotter, view)()
        plot_path = os.path.join(plot_dir, f"harmonic_{k:02d}.png")
        plotter.screenshot(plot_path)
        plotter.close()

    print(f"Saved {len(modes)} harmonic triplet plots to {plot_dir}")
