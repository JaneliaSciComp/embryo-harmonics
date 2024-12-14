# %%
# %matplotlib qt

# %%
import h5py
import numpy as np
from matplotlib import pyplot as plt

from embryoharmonics import EmbryoModelLoader

# %%
# Open HDF5 file
path = "/Users/innerbergerm/Data/worm-geometry/celegans_avg_models_421_minutes_samples_2024_12_04.h5"
h5file = h5py.File(path, 'r')

# %%
# Load all splines for given time step
time_step = 420
embryo_model_loader = EmbryoModelLoader(h5file)
embryo_model = embryo_model_loader.load(time_step)

# %%
# Seam cells are the marker cells on the left and right of the embryo by which the straightening was done
print(f"names: {embryo_model.seam_cells}")

# %%
# Plot all 32 splines that make up the surface of the embryo
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_box_aspect([1, 1, 4])

domain = embryo_model.spline_domain
t = np.linspace(domain[0], domain[-1], 100)

central_spline = embryo_model.central_spline
x = central_spline(t)
ax.plot3D(x[:, 0], x[:, 1], x[:, 2])

for spline in embryo_model.transverse_splines:
    x = spline(t)
    ax.plot3D(x[:, 0], x[:, 1], x[:, 2])

# %%
# If viewed from the bottom, the splines are arranged in clock-wise fashion
# Note: in order to have a right-handed coordinate system, the y-axis is inverted here
fig = plt.figure()
ax = plt.axes()
ax.set_aspect('equal')
ax.invert_yaxis()
plt.xlabel("x")
plt.ylabel("y (inverted!)")

starting_points = np.row_stack([spline(t[0]) for spline in embryo_model.transverse_splines])
ax.scatter(starting_points[:, 0], starting_points[:, 1])
for i in range(32):
    ax.text(starting_points[i, 0], starting_points[i, 1], str(i + 1))

ax.plot(0, 0, marker='$\\bigotimes$', markersize=15)
ax.text(3, 0, "z-axis")
plt.show()

# %%
