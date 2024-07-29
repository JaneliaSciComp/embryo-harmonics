# %%
import math

import h5py
import numpy as np
import netgen.occ as occ
from netgen.webgui import Draw

# %%
from embryoharmonics.utils import load_avg_models

# %%
path = "/home/innerbergerm@hhmi.org/big-data/worm-geometry/celegans_avg_models_2024_04_23.h5"
h5file = h5py.File(path, 'r')

# %%
# visualize all splines for given time step
time_step = 1
embryo_model = load_avg_models(h5file, time_steps=[time_step])[time_step]


# %%
def project_to_plane_through_z_axis(x, normal):
    dist = np.dot(x, normal)
    projection = x - np.outer(dist, normal)
    tangent2 = np.array([0, 0, 1])
    tangent1 = np.cross(normal, tangent2)
    x_projected = np.dot(projection, tangent1)
    y_projected = np.dot(projection, tangent2)
    z_projected = dist
    return x_projected, y_projected, z_projected


def get_spline_surface(embryo_model, i):
    domain = embryo_model.spline_domain
    n = 100
    t = np.linspace(domain[0], domain[-1], n)

    spline1 = embryo_model.transverse_splines[i]
    neighbor_index = (i + 1) % embryo_model.n_transverse_splines
    spline2 = embryo_model.transverse_splines[neighbor_index]
    x1 = spline1(t)
    x2 = spline2(t)
    direction = (x1[0] + x2[0]) / 2
    normal = direction / np.linalg.norm(direction)

    xp1, yp1, zp1 = project_to_plane_through_z_axis(x1, normal)
    xp2, yp2, zp2 = project_to_plane_through_z_axis(x2, normal)

    points = np.array([[(xp1[i], yp1[i], zp1[i]) for i in range(n)],
                       [(xp2[i], yp2[i], zp2[i]) for i in range(n)]])

    surf = occ.SplineSurfaceInterpolation(points)
    surf = surf.Rotate(occ.Axis((0, 0, 0), occ.X), 90)
    angle = np.arctan2(normal[1], normal[0]) / math.pi * 180
    surf = surf.Rotate(occ.Axis((0, 0, 0), occ.Z), -angle)
    return surf


# %%
N = embryo_model.n_transverse_splines
spline_surfaces = [get_spline_surface(embryo_model, i) for i in range(N)]
total_surface = sum(spline_surfaces)
Draw(total_surface)
