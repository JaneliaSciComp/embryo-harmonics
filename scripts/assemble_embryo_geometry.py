# %%
import h5py
import netgen.occ as occ
from netgen.webgui import Draw

# %%
from embryoharmonics.utils import load_avg_models, get_spline_surface

# %%
path = "/home/innerbergerm@hhmi.org/big-data/worm-geometry/celegans_avg_models_2024_04_23.h5"
h5file = h5py.File(path, 'r')

# %%
# visualize all splines for given time step
time_step = 1
embryo_model = load_avg_models(h5file, time_steps=[time_step])[time_step]

# %%
# Generate surface splines (using less interpolations points seems to yield better results because of numerical instabilities)
N = embryo_model.n_transverse_splines
spline_surfaces = [get_spline_surface(embryo_model, N, i) for i in range(N)]
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # [R, G, B]
for i, surf in enumerate(spline_surfaces):
    surf.col = colors[i % 3]

# Reduce local mesh-width to show that meshes across surfaces are actually linked
spline_surfaces[0].maxh = 1

# %%
# Combine spline surfaces (alternatives to Compound are Fuse and Glue, but they don't work in this case)
total_surface = occ.Compound(spline_surfaces)
geo = occ.OCCGeometry(total_surface)
mesh = geo.GenerateMesh(maxh=4)

# %%
# Rotate the view so that the first surfaces is in the front (the red one bordered by two green ones)
settings = {"camera": {"Light": {"ambient": 1.0, "diffuse": 1.0}, "euler_angles": [180,90,0], "transformations": [{"type": "move", "dir": [0, 0, 1.5]}]}}
Draw(mesh, settings=settings)

# %%
