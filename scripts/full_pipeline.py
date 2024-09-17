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
path = "/home/innerbergerm@hhmi.org/big-data/worm-geometry/celegans_avg_models_2024_04_23.h5"
h5file = h5py.File(path, 'r')
pv.set_jupyter_backend('client')

# %%
time_step = 1
embryo_model = load_avg_models(h5file, time_steps=[time_step])[time_step]
mesh = generate_embryo_mesh(embryo_model, mesh_size=5)
print(f"Number of elements: {mesh.ne}")

# %%
# Compute first few eigenvectors
k = 30
pv_data, metrics = compute_harmonics(mesh, k=k, boundary_condition="neumann")

# %%
# Find type of eigenfunctions (Note: this is not the most robust criterion)
dirichlet_rpz = np.row_stack((metrics['dirichlet_r'], metrics['dirichlet_phi'], metrics['dirichlet_z']))
max_dirichlet = np.argmax(dirichlet_rpz, axis=0)
harmonic_type = ['radial', 'angular', 'height']
for i, idx in enumerate(max_dirichlet):
    print(f"harmonic {i:02d}: {harmonic_type[idx]}")

# %%
# Visualize eigenvector
for i in range(k):
    p = plot_eigenfunction(pv_data, i)
    p.show()

# %%
# It's easy to write and read data in the vtk format:
# pv_data.save("data.vtu")
# same_data = pv.read("data.vtu")

# %%
# Gene data can be loaded and smoothly interpolated
mat_file = h5py.File("/home/innerbergerm@hhmi.org/big-data/worm-geometry/cwn_pos_data.mat")
gene_data, time = load_gene_data(mat_file, "cwn", time_steps=[time_step])
gene_data = gene_data[time_step]

# TODO: there is a mismatch between the scales of the geometry and the gene data (about a factor of 5) - fix this in a general way!
# If the factor is chosen too large, some points are outside the domain and the kernel will crash
gene_data.locations *= 5

interpolate_gene_data(mesh, gene_data, pv_data, smoothing_factor=1000)
print(pv_data.array_names)

# %%
# The gene expression data was added to the pyvista data object
plotter = pv.Plotter()
slices = pv_data.slice_orthogonal()
plotter.add_mesh(slices, scalars="cwn", cmap="turbo")
plotter.show()

# %%
# With this the gene data loaded, it's possible to compute the eigen-coefficients of the gene expression
eigen_coefficients = compute_eigen_coefficients(pv_data, gene_data)
plt.scatter(range(k), np.abs(eigen_coefficients["cwn"]))
# plt.gca().set_yscale('log')
plt.xlabel("# harmonic")
plt.ylabel("coefficient")
plt.show()

# %%
