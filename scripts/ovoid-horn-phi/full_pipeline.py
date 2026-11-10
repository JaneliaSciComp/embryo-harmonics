# %% [markdown]
# Instead of the radius R(z) along the axis, the ovoid can be parametrized by
# its polar outline r(phi) around the centroid, which stays smooth at the tips.
# Every mode is approximated as A(phi) * j_k(kappa_{k,p} * rho / r(phi)) *
# cos/sin(m * theta): the radial shape is the radial mode of a ball of the
# local radius r(phi), and the envelope A(phi) solves a cheap 1D problem. The
# approximation is exact for a ball. This pipeline compares it with the exact
# harmonics and decomposes gene expression data against it.

# %%
import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from embryoharmonics import celegans
from embryoharmonics.celegans import AxisymmetricHarmonics, OvoidModel
from embryoharmonics.celegans.ovoid_horn import compute_horn_phi_harmonics

# %%
ROOT = os.path.join(os.getcwd(), 'tests', 'resources')
COEFFICIENT_PATH = os.path.join(ROOT, 'ovoid_coefficients.csv')
# The full data set is 'data/embryo-20261009.zarr'
GENE_PATH = os.path.join(ROOT, 'embryo_pre_twitch.zarr')
pv.set_jupyter_backend('client')

# %%
ovoid = OvoidModel.from_csv(COEFFICIENT_PATH, flip=True, radial_extension=0.5)
meridian_mesh = ovoid.generate_meridian_mesh(mesh_size=5)
print(f"Number of elements: {meridian_mesh.n_cells}")

# %%
# Compute the harmonics; each mode carries an (m, k, p) label: azimuthal
# order, order of the radial shape, radial zero
N = 60
horn, labels = compute_horn_phi_harmonics(meridian_mesh, ovoid, n=N)
exact = AxisymmetricHarmonics.compute(meridian_mesh, n=N)

# %%
# Visualize the meridian part of the first few modes with their labels
for i in range(6):
    plotter = pv.Plotter()
    plotter.add_mesh(meridian_mesh.copy(), scalars=horn[i].data, cmap='turbo')
    plotter.add_text(
        f"eigenvalue {horn.eigenvalues[i]:.2e}, "
        f"(m, k, p)={labels[i]} ({horn.trig_kinds[i]})",
        font_size=12,
    )
    plotter.view_xz()
    plotter.show()

# %%
# Compare the spectrum with the exact one
error = horn.eigenvalues[1:N] / exact.eigenvalues[1:N] - 1
plt.plot(range(1, N), 100 * error)
plt.axhline(0, color='k', lw=0.5)
plt.title("Relative eigenvalue error of the phi-horn")
plt.xlabel("# harmonic")
plt.ylabel("error (%)")
plt.show()

# %%
# One translation fits the cells of all time points into the ovoid
gene_data_loader = celegans.open_gene_data_loader(GENE_PATH)
all_locations = np.vstack([
    gene_data_loader.load_all(t)[0] for t in gene_data_loader.time_steps
])
center = ovoid.fit_center(all_locations)
print(f"Maximal relative radius: {ovoid.containment(all_locations, center).max():.3f}")

# %%
# The genes are decomposed by sampling the 3D modes at the cell positions
TIME_STEP = 202
GENE_NAME = "pal-1"
gene = gene_data_loader.gene_names.index(GENE_NAME)
locations, activities = gene_data_loader.load_all(TIME_STEP)
for name, harmonics in (("phi-horn", horn), ("exact", exact)):
    coefficients = harmonics.decompose_point_data(locations - center, activities)[gene]
    plt.scatter(range(len(harmonics)), np.abs(coefficients), label=name, s=8)
plt.gca().set_yscale('log')
plt.legend()
plt.title(f"Eigen coefficients of {GENE_NAME} at t = {TIME_STEP}")
plt.xlabel("# harmonic")
plt.ylabel("coefficient")
plt.show()

# %%
