# %% [markdown]
# The Webster-Horn equation reduces the Laplace-Neumann eigenproblem on a body
# of revolution to cheap 1D problems for the envelopes A(z) of the local disk
# modes J_m(j'_{m,l} * r / R(z)). On the ovoid, R'(z) -> infinity at the tips,
# which the plain equation (derived for slowly varying R) does not cover. This
# pipeline compares three cap treatments with the exact harmonics and
# decomposes gene expression data against the approximate harmonics.

# %%
import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from embryoharmonics import celegans
from embryoharmonics.celegans import AxisymmetricHarmonics, OvoidModel
from embryoharmonics.celegans.ovoid_horn import CAP_TREATMENTS, compute_horn_z_harmonics

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
# Compute the harmonics with all three cap treatments: natural (whole axis,
# the weight R^2 vanishes at the tips), flat (cut at a wall angle of 60
# degrees) and corrected (keep the wall-slope terms)
N = 60
horns = {
    caps: compute_horn_z_harmonics(meridian_mesh, ovoid, n=N, caps=caps)
    for caps in CAP_TREATMENTS
}
exact = AxisymmetricHarmonics.compute(meridian_mesh, n=N)

# %%
# Visualize the meridian part of the first few modes with their labels
horn, labels = horns["natural"]
for i in range(6):
    plotter = pv.Plotter()
    plotter.add_mesh(meridian_mesh.copy(), scalars=horn[i].data, cmap='turbo')
    plotter.add_text(
        f"eigenvalue {horn.eigenvalues[i]:.2e}, "
        f"(m, l, n)={labels[i]} ({horn.trig_kinds[i]})",
        font_size=12,
    )
    plotter.view_xz()
    plotter.show()

# %%
# All treatments approximate the exact spectrum to a few percent
for caps, (horn, _) in horns.items():
    error = horn.eigenvalues[1:N] / exact.eigenvalues[1:N] - 1
    plt.plot(range(1, N), 100 * error, label=caps)
plt.axhline(0, color='k', lw=0.5)
plt.legend()
plt.title("Relative eigenvalue error of the z-horn")
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
for caps, (horn, _) in horns.items():
    coefficients = horn.decompose_point_data(locations - center, activities)[gene]
    plt.scatter(range(len(horn)), np.abs(coefficients), label=caps, s=8)
coefficients = exact.decompose_point_data(locations - center, activities)[gene]
plt.scatter(range(len(exact)), np.abs(coefficients), label="exact", marker='x')
plt.gca().set_yscale('log')
plt.legend()
plt.title(f"Eigen coefficients of {GENE_NAME} at t = {TIME_STEP}")
plt.xlabel("# harmonic")
plt.ylabel("coefficient")
plt.show()

# %%
