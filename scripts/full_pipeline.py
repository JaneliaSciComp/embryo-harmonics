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
import pyvista as pv
import matplotlib.pyplot as plt

from embryoharmonics.geometry import load_avg_models, assemble_embryo_geometry, load_gene_data
from embryoharmonics.fem import mesh_embryo_geometry, compute_harmonics, interpolate_gene_data, compute_eigen_coefficients
from embryoharmonics.visualization import plot_eigenfunction

# %%
path = "/home/innerbergerm@hhmi.org/big-data/worm-geometry/celegans_avg_models_2024_04_23.h5"
h5file = h5py.File(path, 'r')

# %%
time_step = 1
embryo_model = load_avg_models(h5file, time_steps=[time_step])[time_step]
geometry = assemble_embryo_geometry(embryo_model)
mesh = mesh_embryo_geometry(geometry, mesh_size=5)
print(f"Number of elements: {mesh.ne}")

# %%
# Compute first few eigenvectors
# We want the smallest eigenvalues, so search for the largest in shift-invert mode (i.e., find largest w' = 1 / (w - sigma))
k = 30
pv_data, eigvals = compute_harmonics(mesh, k=k, boundary_condition="neumann")

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
interpolate_gene_data(mesh, gene_data, pv_data)
print(pv_data.array_names)

# %%
# The gene expression data was added to the pyvista data object
plotter = pv.Plotter()
slices = pv_data.slice_orthogonal()
plotter.add_mesh(slices, scalars="cwn", cmap="turbo")
plotter.show()

# %%
# With this smooth interpolation, it's possible to compute the eigen-coefficients of the gene expression
eigen_coefficients = compute_eigen_coefficients(pv_data, "cwn")
plt.scatter(range(k), eigen_coefficients["cwn"])
plt.xlabel("# harmonic")
plt.ylabel("coefficient")
plt.show()

# %%
