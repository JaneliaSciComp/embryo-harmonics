import h5py
import numpy as np
from matplotlib import pyplot as plt

from embryoharmonics.utils import load_avg_models

path = "<your path to the example data>/example_data.h5"
h5file = h5py.File(path, 'r')

# visualize all splines for given time step
time_step = 1
embryo_model = load_avg_models(h5file, time_steps=[time_step])[time_step]

print(f"names: {embryo_model.seam_cells}")

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

plt.show()
