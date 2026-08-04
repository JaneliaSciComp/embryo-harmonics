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
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from embryoharmonics import Harmonics, celegans
from embryoharmonics.celegans.webster_horn_harmonics import (
    compute_webster_horn_harmonics,
    degenerate_clusters,
    match_modes,
)

CWD = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CWD, "..", "..", "data", "avg_models_n371.h5")
RESULTS_DIR = os.path.join(CWD, "..", "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TIME_STEP = 186  # middle of the 371 averaged models
N = 100  # below ~15 modes the spectrum is purely axial (m = 0) and degeneracy-free
# The Webster-Horn pool is deliberately larger than the FEM basis, so that even
# the last FEM mode still has its partner available to be matched with.
N_POOL = N + 20
MESH_SIZE = 5
MAX_TICKS = 25  # thin out tick labels so they stay readable

model_loader = celegans.EmbryoModelLoader(MODEL_PATH)

# The rotationally symmetric geometry, with both bases on the same mesh
# (so nothing has to be resampled).
embryo_model_sym = model_loader.load(TIME_STEP, symmetric=True)
mesh_sym = embryo_model_sym.generate_mesh(mesh_size=MESH_SIZE)
webster, labels = compute_webster_horn_harmonics(mesh_sym, embryo_model_sym, n=N_POOL)
harmonics_sym = Harmonics.compute(mesh_sym, n=N)
print(f"symmetric mesh: {mesh_sym.n_cells} cells, "
      f"{N} FEM modes vs a pool of {len(webster)} Webster-Horn modes")

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
scale = harmonics_sym.eigenvalues[-1]


def coefficient_matrix(basis, targets):
    """Stack decomposition coefficients column-wise: entry (i, j) is the
    coefficient of target mode j with respect to basis mode i.
    """
    coefficients = basis.decompose(targets)
    return np.column_stack([coefficients[data.name] for data in targets])


# Adiabatic error: Webster-Horn vs the true harmonics of the same (symmetric)
# geometry. Total error: the same basis against the true harmonics of the raw
# geometry, stretched onto the symmetric one.
matrix_adiabatic = coefficient_matrix(
    webster, [harmonics_sym[k] for k in range(N)]
)
matrix_total = coefficient_matrix(
    webster_on_raw, [harmonics_raw_on_stretched[k] for k in range(N)]
)

# One-to-one pairing: for each FEM eigenfunction, the Webster-Horn
# eigenfunction it matches best, and the scalar product with it.
matched_adiabatic, product_adiabatic = match_modes(matrix_adiabatic, clusters)
matched_total, product_total = match_modes(matrix_total, clusters)

# The pairing is only one-to-one if no cluster is over-subscribed: a cluster of
# dimension d can host at most d mutually orthogonal eigenfunctions. Clusters at
# the top of the pool going unclaimed is expected and harmless; two FEM modes
# competing for the same one is not, and means the approximation has failed to
# separate them.
claims = np.bincount(matched_adiabatic, minlength=len(clusters))
dimensions = np.array([len(indices) for indices in clusters])
oversubscribed = claims > dimensions
is_one_to_one = not oversubscribed.any()

# Eigenvalue errors, relative to the spectral range so the zero (constant) mode
# does not blow up. Mode 0 is exactly zero for both bases by construction.
eigenvalue_error_adiabatic = np.abs(
    webster.eigenvalues[:N] - harmonics_sym.eigenvalues
) / scale
eigenvalue_error_total = np.abs(
    webster.eigenvalues[:N] - harmonics_raw_on_stretched.eigenvalues
) / scale
eigenvalue_error_sym = np.abs(
    harmonics_raw_on_stretched.eigenvalues - harmonics_sym.eigenvalues
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
      f"{'<u,w>':>8} {'<u,w>_raw':>10}")
for k in modes:
    cluster = matched_adiabatic[k]
    flag = "  (contested)" if oversubscribed[cluster] else ""
    print(f"{k:>9} {block_labels[cluster]:>16} {len(clusters[cluster]):>4} "
          f"{harmonics_sym.eigenvalues[k]:>11.4g} "
          f"{product_adiabatic[k]:>8.4f} {product_total[k]:>10.4f}{flag}")

print(f"\npairing is one-to-one: {is_one_to_one}"
      f" ({oversubscribed.sum()} clusters contested by more FEM modes than "
      f"their dimension)")
print(f"\nscalar product with the matched Webster-Horn eigenfunction:")
print(f"  adiabatic (symmetric geometry): worst {product_adiabatic.min():.4f}, "
      f"median {np.median(product_adiabatic):.4f}")
print(f"  total (raw geometry):           worst {product_total.min():.4f}, "
      f"median {np.median(product_total):.4f}")
print(f"\neigenvalue error, max over modes:")
print(f"  adiabatic:      {eigenvalue_error_adiabatic.max():.2e}")
print(f"  total:          {eigenvalue_error_total.max():.2e}")
print(f"  symmetrization: {eigenvalue_error_sym.max():.2e}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(modes, product_adiabatic, "o-", ms=4,
             label="adiabatic (WH vs FEM, symmetric)")
axes[0].plot(modes, product_total, "s-", ms=4,
             label="total (WH vs FEM, raw)")
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
