# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import h5py
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from embryoharmonics import *

# %%
path = "/Users/innerbergerm/Data/worm-geometry/celegans_avg_models_421_minutes_samples_2024_12_04.h5"
h5file = h5py.File(path, 'r')
pv.set_jupyter_backend('client')

# %%
time_step = 420
embryo_model_loader = EmbryoModelLoader(h5file)
embryo_model = embryo_model_loader.load(time_step)
mesh = generate_embryo_mesh(embryo_model, mesh_size=5)
print(f"Number of elements: {mesh.ne}")

# %%
# Compute first few eigenvectors
k = 30
pv_data, metrics = compute_harmonics(mesh, k=k, boundary_condition="neumann", store_dirichlet_densities=True)

# %%
# Find type of harmonics (Note: this is not the most robust criterion)
dirichlet_rpz = np.row_stack((metrics['dirichlet_r'], metrics['dirichlet_p'], metrics['dirichlet_z']))
max_dirichlet = np.argmax(dirichlet_rpz, axis=0)
harmonic_type = ['radial', 'angular', 'height']
for i, idx in enumerate(max_dirichlet):
    print(f"harmonic {i:02d}: {harmonic_type[idx]}")

# %%
# Visualize harmonics (use dirichlet="r", "p", "z" to plot components of the dirichlet energy density)
for i in range(k):
    p = plot_harmonic(pv_data, i)
    p.show()

# %%
# It's easy to write and read data in the vtk format:
# pv_data.save("data.vtu")
# same_data = pv.read("data.vtu")

# %%
# Gene data can be loaded and smoothly interpolated
mat_file = h5py.File("/home/innerbergerm@hhmi.org/big-data/worm-geometry/4D_transcriptome.mat")
gene_name = "cwn-1"
gene_data_loader = GeneDataLoader(mat_file)
gene_data = gene_data_loader.load(gene_name, time_step)

# Tissue data can be loaded similarly to gene_data
# gene_data = gene_data_loader.load_tissue("intestine", time_step)

# TODO: there is a mismatch between the scales of the geometry and the gene data (about a factor of 5) - fix this in a general way!
# If the factor is chosen too large, some points are outside the domain
gene_data.locations *= 5

smoothed_coefficients = interpolate_gene_data(mesh, gene_data, pv_data, smoothness=10, compute_eigen_coefficients=True)

# %%
# The gene expression data was added to the pyvista data object
plotter = pv.Plotter()
slices = pv_data.slice_orthogonal()
plotter.add_mesh(slices, scalars=gene_name, cmap="turbo")
plotter.show()

# %%
# If computed during smoothing, the eigen-coefficients of the smoothed gene expression can be plotted
plt.scatter(range(k), np.abs(smoothed_coefficients[gene_name]))
# plt.gca().set_yscale('log')
plt.title(f"Eigen coefficients of smoothed {gene_name}")
plt.xlabel("# harmonic")
plt.ylabel("coefficient")
plt.show()

# %%
# Also, it's possible to compute the eigen-coefficients of the gene expression directly without smoothing
eigen_coefficients = compute_eigen_coefficients(pv_data, gene_data)
plt.scatter(range(k), np.abs(eigen_coefficients[gene_name]))
plt.title(f"Eigen coefficients of {gene_name}")
plt.xlabel("# harmonic")
plt.ylabel("coefficient")
plt.show()
# %%
# The gene expression data reconstructed from the eigen-coefficients
compose_eigen_coefficients(pv_data, eigen_coefficients[gene_name], "composed-gene")
plotter = pv.Plotter()
slices = pv_data.slice_orthogonal()
plotter.add_mesh(slices, scalars="composed-gene", cmap="turbo")
plotter.show()
print(pv_data.array_names)

# %%
