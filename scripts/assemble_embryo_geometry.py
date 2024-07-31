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
N = embryo_model.n_transverse_splines
spline_surfaces = [get_spline_surface(embryo_model, i) for i in range(N)]
spline_surfaces[0].col = (0, 0, 1)
for surf in spline_surfaces[1::2]:
    surf.col = (1, 0, 0)
total_surface = occ.Compound(spline_surfaces)
geo = occ.OCCGeometry(total_surface)
mesh = geo.GenerateMesh(maxh=4)

# %%
settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -90}]}, "Light": {"ambient": 1.0, "diffuse": 1.0}}
Draw(mesh, settings=settings)

# %%
