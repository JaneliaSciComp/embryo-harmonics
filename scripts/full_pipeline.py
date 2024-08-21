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

from embryoharmonics.geometry import load_avg_models, assemble_embryo_geometry
from embryoharmonics.fem import mesh_embryo_geometry, compute_harmonics
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
k = 10
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
