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
from ngsolve import *
from ngsolve.webgui import Draw

from embryoharmonics.geometry import load_avg_models, assemble_embryo_geometry
from embryoharmonics.fem import mesh_embryo_geometry

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
# Set up lowest-order finite element space...
fes = H1(mesh, order=1, dirichlet="default")
u, v = fes.TnT()

# %%
# ... and stiffness / mass matrices
a = BilinearForm(fes, symmetric=True)
a += grad(u) * grad(v) * dx
a.Assemble()

m = BilinearForm(fes, symmetric=True)
m += u * v * dx
m.Assemble()


# %%
def to_scipy_csr(blf, mask):
    row, col, val = blf.mat.COO()
    # filter free dofs
    sparse = sp.csr_matrix((val, (row, col)))
    return sparse[mask][:, mask]


# %%
# Export stiffness and mass matrix to scipy
import scipy.sparse as sp
mask = np.array([free for free in fes.FreeDofs()])
stiffness = to_scipy_csr(a, mask)
mass = to_scipy_csr(m, mask)
print(f"Shape of stiffness matrix: {stiffness.shape}")
print(f"Shape of mass matrix: {mass.shape}")
print(f"Degrees of freedom in the finite element space: {fes.ndof}")

# %%
# Compute first few eigenvectors (for some reason, mass and stiffness need to be swapped?!) 
eigvals, eigvecs = sp.linalg.eigsh(M=stiffness, A=mass, k=10)
xAx = np.diag(eigvecs.T @ stiffness @ eigvecs)
xMx = np.diag(eigvecs.T @ mass @ eigvecs)
print(f"(x, Ax): {xAx}")
print(f"(x, Mx): {xMx}")
print(f"(x, M^-1 Ax): {xAx / xMx}")
print(f"Eigenvalues: {eigvals}")

# %%
# Visualize eigenvector
u = GridFunction(fes)
u.vec[:] = 0
u.vec.FV().NumPy()[mask] = eigvecs[:, 0]

clipping = {"function": True,  "pnt": (0, 0, 250), "vec": (0, 1, 0)}
settings = {"camera": {"euler_angles": [-90, 0, 0]}}
Draw(u, clipping=clipping, settings=settings)

# %%
import matplotlib.pyplot as plt
plt.spy(stiffness, marker='.', alpha=0.1)
plt.show()

# %%
