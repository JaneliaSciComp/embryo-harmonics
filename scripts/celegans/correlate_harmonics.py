# %%
import os

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from embryoharmonics import celegans
from embryoharmonics import Harmonics, FemMatrices

# %%
CWD = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CWD, '..', '..', 'tests', 'resources', 'celegans_models.h5')
model_loader = celegans.EmbryoModelLoader(MODEL_PATH)
mesh_earlier = model_loader.load(420).generate_mesh(mesh_size=10)
mesh_later = model_loader.load(421).generate_mesh(mesh_size=10)
pv.set_jupyter_backend('client')
print(f"Earlier timestep bounds: {mesh_earlier.bounds}")
print(f"Later timestep bounds: {mesh_later.bounds}")

# %%
# Visualize the meshes if desired
# mesh_earlier.plot()
# mesh_later.plot()

# %%
# Compute some harmonics for the two meshes
N = 100
harmonics_earlier = Harmonics.compute(mesh_earlier, n=N)
harmonics_later = Harmonics.compute(mesh_later, n=N)

# %%
# Transform the earlier mesh to the later mesh
mesh_transformed = celegans.transform_mesh(mesh_earlier, mesh_later)
print(f"Earlier timestep bounds after transformation: {mesh_earlier.bounds}")

# %%
# Resample harmonics from earlier timestep to later timestep
harmonics_earlier = list(harmonics_earlier)
resampled_earlier = [
    h.resample(mesh_later, project_outside_data=True) for h in harmonics_earlier
]
harmonics_later = list(harmonics_later)

# %%
# Compare harmonics numerically by computing the dot product between the two
# sets on the same mesh. The dot product is done using the mass matrix of the
# mesh, so that the dot product is invariant to the local mesh size.
dot_products = np.zeros((len(harmonics_earlier), len(harmonics_later)))
fem = FemMatrices.compute_for(mesh_later, stiffness=False)
for i, hi in enumerate(resampled_earlier):
    for j, hj in enumerate(harmonics_later):
        dot_products[i, j] = hj.data.dot(fem.mass @ hi.data)
plt.imshow(dot_products)
plt.clim(-1, 1)
plt.colorbar()

# %%
# Find rearrangement of harmonics between time steps, e.g., 13 and 14 are swapped
earlier_to_later = np.argmax(np.abs(dot_products), axis=1)
for i, idx in enumerate(earlier_to_later):
    if i != idx:
        sign_str = " (sign flip)" if dot_products[i, idx] < 0 else ""
        print(f"{i:03} -> {idx:03}" + sign_str)

# %%
# Compare harmonics visually (13 is swapped with 14, but 13->14 has a wrong sign)
EARLIER_IDX = 13
later_index = earlier_to_later[EARLIER_IDX]
later_name = f"harmonic_{later_index:03}"
earlier_name = f"harmonic_{EARLIER_IDX:03}"

p = pv.Plotter(shape=(1, 3))
# Later timestep
sign = -1 if dot_products[EARLIER_IDX, later_index] < 0 else 1
p.add_mesh(mesh_later, scalars=sign * harmonics_later[later_index].data)
p.add_text(f"{later_name} (later)")
p.subplot(0, 1)
# Earlier timestep resampled to later
p.add_mesh(mesh_transformed, scalars=harmonics_earlier[EARLIER_IDX].data)
p.add_text(f"{earlier_name} (resampled)")
p.subplot(0, 2)
# Earlier timestep
p.add_mesh(mesh_earlier, scalars=harmonics_earlier[EARLIER_IDX].data)
p.add_text(f"{earlier_name} (earlier)")
p.link_views()
p.show()

# %%
