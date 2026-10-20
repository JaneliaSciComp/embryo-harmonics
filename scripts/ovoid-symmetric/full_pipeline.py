# %% [markdown]
# Before twitching, the embryo is an ovoid: a body of revolution whose outline
# r(theta) is a Fourier series around its centroid. Its Laplace harmonics
# separate exactly into a meridian part v(r, z) and an angular factor
# cos/sin(k*theta). This pipeline computes the harmonics on the 2D meridian
# mesh, fits the cells of all time points into the ovoid with one translation,
# and decomposes gene expression data against the harmonics.

# %%
import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from embryoharmonics import celegans
from embryoharmonics.celegans import AxisymmetricHarmonics, OvoidModel

# %%
ROOT = os.path.join(os.getcwd(), 'tests', 'resources')
COEFFICIENT_PATH = os.path.join(ROOT, 'ovoid_coefficients.csv')
# The full data set is 'data/embryo-20261009.zarr'
GENE_PATH = os.path.join(ROOT, 'embryo_pre_twitch.zarr')
pv.set_jupyter_backend('client')

# %%
ovoid = OvoidModel.from_csv(COEFFICIENT_PATH, flip=True)
meridian_mesh = ovoid.generate_meridian_mesh(mesh_size=5)
print(f"Number of elements: {meridian_mesh.n_cells}")

plotter = pv.Plotter()
plotter.add_mesh(meridian_mesh, show_edges=True)
plotter.view_xz()
plotter.show()

# %%
# Compute the first few harmonics; the angular orders k = 0, 1, ... are solved
# one by one, and modes with k > 0 come in degenerate cos/sin pairs
N = 30
axisymmetric = AxisymmetricHarmonics.compute(meridian_mesh, n=N)

# %%
# Visualize the meridian part of the first few modes; nodal counting yields
# approximate radial/axial "quantum numbers" complementing the exact k
n_radial, n_axial = axisymmetric.nodal_counts()
for i in range(6):
    plotter = pv.Plotter()
    plotter.add_mesh(
        meridian_mesh.copy(), scalars=axisymmetric[i].data, cmap='turbo'
    )
    plotter.add_text(
        f"eigenvalue {axisymmetric.eigenvalues[i]:.2e}, "
        f"k={axisymmetric.angular_orders[i]} ({axisymmetric.trig_kinds[i]}), "
        f"n_r={n_radial[i]}, n_z={n_axial[i]}",
        font_size=12,
    )
    plotter.view_xz()
    plotter.show()

# %%
# One translation fits the cells of all time points into the ovoid; a relative
# radius above 1 means a cell lies outside (use a radial extension then)
gene_data_loader = celegans.open_gene_data_loader(GENE_PATH)
all_locations = np.vstack([
    gene_data_loader.load_all(t)[0] for t in gene_data_loader.time_steps
])
center = ovoid.fit_center(all_locations)
print(f"Center: {center}, maximal relative radius: "
      f"{ovoid.containment(all_locations, center).max():.3f}")

R, z = ovoid.outline(200)
d = all_locations - center
plt.plot(z, R, 'k')
plt.scatter(d[:, 2], np.hypot(d[:, 0], d[:, 1]), s=2)
plt.gca().set_aspect('equal')
plt.title("Cells of all time points in the meridian half-plane")
plt.xlabel("z (voxels)")
plt.ylabel("r (voxels)")
plt.show()

# %%
# The genes are decomposed by sampling the 3D modes v(r, z) * cos/sin(k * theta)
# at the cell positions
TIME_STEP = 202
GENE_NAME = "pal-1"
locations, activities = gene_data_loader.load_all(TIME_STEP)
coefficients = axisymmetric.decompose_point_data(locations - center, activities)
coefficients = coefficients[gene_data_loader.gene_names.index(GENE_NAME)]

# %%
# The angular labels show how much of the gene pattern is rotationally
# symmetric (k = 0) versus angularly structured (k > 0)
plt.scatter(
    range(len(axisymmetric)), np.abs(coefficients),
    c=axisymmetric.angular_orders, cmap='tab10'
)
plt.gca().set_yscale('log')
plt.colorbar(label="angular order k")
plt.title(f"Eigen coefficients of {GENE_NAME} at t = {TIME_STEP}")
plt.xlabel("# harmonic")
plt.ylabel("coefficient")
plt.show()

# %%
