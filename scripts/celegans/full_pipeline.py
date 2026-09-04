# %%
import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from embryoharmonics import Harmonics, MeshData, HarmonicSmoother, DiffusionSmoother
from embryoharmonics import celegans

# %%
ROOT = os.path.join(os.getcwd(), 'tests', 'resources')
MODEL_PATH = os.path.join(ROOT, 'celegans_models.h5')
# A directory of parquet files (e.g. 'data/cpm-20260828') works here as well
GENE_PATH = os.path.join(ROOT, 'celegans_genedata.h5')
pv.set_jupyter_backend('client')

# %%
TIME_STEP = 420
embryo_model_loader = celegans.EmbryoModelLoader(MODEL_PATH)
embryo_model = embryo_model_loader.load(TIME_STEP)
mesh = embryo_model.generate_mesh(mesh_size=5)
print(f"Number of elements: {mesh.n_cells}")

# %%
# Compute first few eigenvectors
N = 30
harmonics = Harmonics.compute(mesh, n=N)

# %%
# Visualize harmonics
def plot_harmonic(pv_mesh: pv.UnstructuredGrid, harmonic: MeshData):
    """Plot the n-th harmonic of a given mesh."""
    # Add data to the mesh
    scalar_name = "harmonic"
    pv_mesh[scalar_name] = harmonic.data

    # Set up visualization
    p = pv.Plotter()
    camera = pv.Camera()
    camera.position = (-400.0, 400.0, -500.0)
    camera.focal_point = (100.0, 50.0, 5.0)
    p.camera = camera
    axes = pv.Axes(show_actor=True, actor_scale=2.0, line_width=5)
    axes.origin = (3.0, 3.0, 3.0)

    # Add slices in different directions at different positions
    slice1 = pv_mesh.slice(normal='x').translate((400, 0, -200))
    p.add_mesh(slice1, scalars=scalar_name, cmap='turbo')
    slice2 = pv_mesh.slice(normal='y').translate((200, 0, -100))
    p.add_mesh(slice2, scalars=scalar_name, cmap='turbo')
    slice3 = pv_mesh.slice_along_axis(n=10, axis="z")
    p.add_mesh(slice3, scalars=scalar_name, cmap='turbo')
    p.show()

    pv_mesh.clear_data()

for i in range(min(N, 5)):
    plot_harmonic(mesh, harmonics[i])

# %%
# Meshes and harmonics are stored per time point in a single HDF5 file with a
# sibling XDMF file that can be opened in ParaView:
# from embryoharmonics import io
# io.save_time_point("embryo.h5", TIME_STEP, mesh, harmonics)
# same_harmonics = io.load_harmonics("embryo.h5", TIME_STEP)
# same_mesh = same_harmonics.mesh  # or io.load_mesh("embryo.h5", TIME_STEP)

# %%
# Gene data can be loaded and smoothly interpolated
GENE_NAME = "cwn-1"
gene_data_loader = celegans.open_gene_data_loader(GENE_PATH)
gene_data = gene_data_loader.load(GENE_NAME, TIME_STEP)

# Tissue data can be loaded similarly to gene_data
# gene_data = gene_data_loader.load_tissue("intestine", TIME_STEP)

# TODO: there is a mismatch between the scales of the geometry and the gene data
#   (about a factor of 5) - fix this in a general way!
# If the factor is chosen too large, some points are outside the domain
gene_data.locations *= 5

# %%
# The gene expression data can be easily visualized alongside the mesh
plotter = pv.Plotter()
plotter.add_mesh(gene_data.as_point_cloud(), point_size=10)
plotter.add_mesh(mesh, scalars=harmonics[2].data, opacity=0.3)
plotter.show()

# %%
# The gene expression data can also be interpolated on the mesh
mesh_data = gene_data.interpolate(mesh)
mesh["data"] = mesh_data.data
plotter = pv.Plotter()
plotter.add_mesh_slice(mesh, scalars="data", cmap="turbo")
plotter.show()

# Since this yields a very sparse data set, it is advisable to smooth the data
harmonic_smoother = HarmonicSmoother(harmonics)
smoothed_data_harmonics = harmonic_smoother.smooth(mesh_data)

# Another way to smooth the data is to use a diffusion process
diffusion_smoother = DiffusionSmoother(mesh, smoothness=10, n_steps=100)
smoothed_data_diffusion = diffusion_smoother.smooth(mesh_data)

# %% [markdown]
# The difference between the two smoothing methods can be visualized. In general:
# - Harmonic smoothing is faster and more flexible. By using a truncated harmonic series
#   (i.e., taking a subset of harmonics with `harmonics.subset(...)`), arbitrary smoothing
#   kernels can be constructed.
# - Diffusion smoothing is more rigorous. It's mass-preserving and smoothing is isotropic.

# %%
plotter = pv.Plotter(shape=(1, 2))
plotting_kwargs = dict(cmap="turbo", smooth_shading=True)
plotter.subplot(0, 0)
plotter.add_text("Harmonic smoothing", font_size=24)
plotter.add_mesh(mesh.copy(), scalars=smoothed_data_harmonics.data, **plotting_kwargs)
plotter.subplot(0, 1)
plotter.add_text("Diffusion smoothing", font_size=24)
plotter.add_mesh(mesh.copy(), scalars=smoothed_data_diffusion.data, **plotting_kwargs)
plotter.link_views()
plotter.show()

# %%
# Eigen-coefficients can be taken from the original or the smoothed data
coefficients = harmonics.decompose(mesh_data)
plt.scatter(range(N), np.abs(coefficients[GENE_NAME]))
plt.gca().set_yscale('log')
plt.title(f"Eigen coefficients of smoothed {GENE_NAME}")
plt.xlabel("# harmonic")
plt.ylabel("coefficient")
plt.show()

# %%
# The gene expression data reconstructed from the eigen-coefficients
# (this is exactly what the harmonic smoother does)
reconstructed_data = harmonics.compose(coefficients)
plotter = pv.Plotter()
plotter.add_mesh(mesh.copy(), scalars=reconstructed_data.data, **plotting_kwargs)
plotter.show()

# %%
