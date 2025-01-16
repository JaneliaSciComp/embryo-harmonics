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
from ngsolve import H1, dx, grad, BilinearForm, GridFunction
from ngsolve.webgui import Draw
import scipy.sparse as sp
import matplotlib.pyplot as plt

from embryoharmonics import EmbryoModelLoader, generate_embryo_mesh
from scripts.assemble_embryo_geometry import embryo_model_loader

# %%
PATH = "/Users/innerbergerm/Data/worm-geometry/celegans_avg_models_421_minutes_samples_2024_12_04.h5"
h5file = h5py.File(PATH, 'r')

# %%
TIME_STEP = 420
embryo_model_loader = EmbryoModelLoader(h5file)
embryo_model = embryo_model_loader.load(TIME_STEP)
mesh = generate_embryo_mesh(embryo_model, mesh_size=5)
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
    """
    Convert a NGSolve bilinear form to a scipy sparse matrix in CSR format.
    :param blf: NGSolve bilinear form
    :param mask: Boolean mask for free dofs
    :return: Sparse matrix in CSR format
    """
    row, col, val = blf.mat.COO()
    # filter free dofs
    sparse = sp.csr_matrix((val, (row, col)))
    return sparse[mask][:, mask]


# %%
# Export stiffness and mass matrix to scipy
mask = np.array([free for free in fes.FreeDofs()])
stiffness = to_scipy_csr(a, mask)
mass = to_scipy_csr(m, mask)
print(f"Shape of stiffness matrix: {stiffness.shape}")
print(f"Shape of mass matrix: {mass.shape}")
print(f"Degrees of freedom in the finite element space: {fes.ndof}")

# %%
# Compute first few eigenvectors
# We want the smallest eigenvalues, so search for the largest in shift-invert mode (i.e., find largest w' = 1 / (w - sigma))
eigvals, eigvecs = sp.linalg.eigsh(A=stiffness, M=mass, k=10, which='LM', sigma=0.0)
xAx = np.diag(eigvecs.T @ stiffness @ eigvecs)
xMx = np.diag(eigvecs.T @ mass @ eigvecs)

# %%
# Check if eigenvectors are M-normalized and eigenvalues are correct
print(f"(x, Mx): {xMx}")
print(f"(x, Ax): {xAx}")
print(f"Difference to eigenvalues: {np.abs(xAx - eigvals)}")

# %%
# Visualize eigenvector
u = GridFunction(fes)
u.vec[:] = 0
u.vec.FV().NumPy()[mask] = eigvecs[:, -1]

clipping = {"function": True,  "pnt": (0, 0, 250), "vec": (0, 1, 0)}
settings = {"camera": {"euler_angles": [-90, 0, 0]}}
Draw(u, clipping=clipping, settings=settings)

# %%
plt.spy(stiffness, marker='.', alpha=0.1)
plt.show()

# %%
