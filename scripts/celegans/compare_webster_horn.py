import os

import numpy as np
import matplotlib.pyplot as plt

from embryoharmonics import Harmonics, MeshData, celegans
from embryoharmonics.celegans.webster_horn_harmonics import (
    compute_webster_horn_harmonics,
)

CWD = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CWD, "..", "..", "data", "avg_models_n371.h5")
RESULTS_DIR = os.path.join(CWD, "..", "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TIME_STEP = 186  # middle of the 371 averaged models
N = 10
MESH_SIZE = 5

model_loader = celegans.EmbryoModelLoader(MODEL_PATH)

# i. Direct FEM harmonics on the raw (non-symmetric) geometry
embryo_model_i = model_loader.load(TIME_STEP)
mesh_i = embryo_model_i.generate_mesh(mesh_size=MESH_SIZE)
harmonics_i = Harmonics.compute(mesh_i, n=N)
print(f"mesh_i (non-symmetric): {mesh_i.n_cells} cells")

# ii. Direct FEM harmonics on the rotationally symmetric geometry
embryo_model_ii = model_loader.load(TIME_STEP, symmetric=True)
mesh_ii = embryo_model_ii.generate_mesh(mesh_size=MESH_SIZE)
harmonics_ii = Harmonics.compute(mesh_ii, n=N)
print(f"mesh_ii (symmetric): {mesh_ii.n_cells} cells")

# iii. Webster-Horn approximation on the symmetric geometry
webster_harmonics = compute_webster_horn_harmonics(mesh_ii, embryo_model_ii, n=N)

# Transform the non-symmetric mesh onto the symmetric one (same pattern as
# scripts/celegans/correlate_harmonics.py), so harmonics_i can be resampled
# onto a common mesh with harmonics_ii and the Webster-Horn basis.
mesh_i_transformed = celegans.transform_mesh(mesh_i, mesh_ii)
resampled_harmonics_i = [
    MeshData(mesh_i_transformed, harmonics_i[k].name, harmonics_i[k].data).resample(
        mesh_ii, project_outside_data=True
    )
    for k in range(N)
]

# Coefficients of (i) and (ii) with respect to the Webster-Horn basis (iii)
coefficients_i = webster_harmonics.decompose(resampled_harmonics_i)
coefficients_ii = webster_harmonics.decompose([harmonics_ii[k] for k in range(N)])

matrix_i = np.column_stack(
    [coefficients_i[data.name] for data in resampled_harmonics_i]
)
matrix_ii = np.column_stack([coefficients_ii[harmonics_ii[k].name] for k in range(N)])

# Save a single comparison plot (not shown interactively)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
for ax, matrix, title in zip(
    axes,
    [matrix_i, matrix_ii],
    ["(i) non-symmetric geometry", "(ii) symmetric geometry"],
):
    im = ax.imshow(np.abs(matrix), vmin=0, vmax=1, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("harmonic index (direct FEM)")
axes[0].set_ylabel("Webster-Horn basis index")
fig.colorbar(im, ax=axes, label="|coefficient|")
fig.suptitle(f"Coefficients wrt the Webster-Horn basis (time step {TIME_STEP}, N={N})")

output_path = os.path.join(RESULTS_DIR, "webster_horn_comparison.png")
fig.savefig(output_path, dpi=150)
print(f"Saved comparison plot to {output_path}")
