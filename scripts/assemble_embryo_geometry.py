# %%
import h5py
from netgen import occ
from netgen.webgui import Draw

from embryoharmonics.celegans.embryo_model import _get_spline_surface, EmbryoModelLoader
from embryoharmonics.celegans.meshing import _convert_to_volume_mesh

# %%
PATH = "/Users/innerbergerm/Data/worm-geometry/celegans_avg_models_421_minutes_samples_2024_12_04.h5"
h5file = h5py.File(PATH, 'r')

# %%
# visualize all splines for given time step
TIME_STEP = 420
embryo_model_loader = EmbryoModelLoader(h5file)
embryo_model = embryo_model_loader.load(TIME_STEP)

# %%
# Generate surface splines (using fewer interpolations points seems to yield better results because of numerical instabilities)
N_INTERPOLATION = 32
n_splines = embryo_model.n_transverse_splines
spline_surfaces = [_get_spline_surface(embryo_model, N_INTERPOLATION, i) for i in range(n_splines)]
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # [R, G, B]
for i, surf in enumerate(spline_surfaces):
    surf.col = colors[i % 3]

# Combine spline surfaces (alternatives to Compound are Fuse and Glue, but they don't work in this case)
mantle = occ.Compound(spline_surfaces)

# %%
min_threshold = embryo_model.central_spline(0.0)[2] + 1
max_threshold = embryo_model.central_spline(1.0)[2] - 1
posterior_cap = occ.Face(occ.Wire([e.Reversed() for e in mantle.edges[occ.Z < min_threshold]]))
anterior_cap = occ.Face(occ.Wire([e.Reversed() for e in mantle.edges[occ.Z > max_threshold]]))

# %%
# Combine all surfaces and draw the wireframe
total_surface = occ.Compound([mantle, anterior_cap, posterior_cap])
Draw(occ.Wire(total_surface.edges))

# %%
# Mesh the whole surface
# Reduce local mesh-width of one spline to show that meshes across surfaces are actually linked
spline_surfaces[0].maxh = 1
geo = occ.OCCGeometry(total_surface)
surface_mesh = geo.GenerateMesh(maxh=4)

# %%
# Rotate the view so that the first surface is in the front (the red one bordered by two green ones)
settings = {"camera": {"Light": {"ambient": 1.0, "diffuse": 1.0}, "euler_angles": [180, 90, 0], "transformations": [{"type": "move", "dir": [0, 0, 1]}]}}
Draw(surface_mesh, settings=settings)

# %%
vol_mesh, _ = _convert_to_volume_mesh(surface_mesh, mesh_size=4, max_node_distance=0.2)
print(f"Number of tetrahedra: {vol_mesh.ne}")
Draw(vol_mesh, settings=settings)
