import math
from typing import Dict

import numpy as np
from netgen import occ
from scipy.interpolate import CubicSpline


class EmbryoModel:
    def __init__(self, seam_cells, spline_domain, central_coordinates, transverse_coordinates):
        self.seam_cells = {name: index for index, name in enumerate(seam_cells)}
        self.spline_domain = spline_domain
        self.central_spline = CubicSpline(spline_domain, central_coordinates)
        self.transverse_splines = [CubicSpline(spline_domain, coordinates) for coordinates in transverse_coordinates]

    @property
    def n_seam_cells(self):
        return len(self.seam_cells)

    @property
    def n_transverse_splines(self):
        return len(self.transverse_splines)


def load_measurement(h5file) -> tuple[np.ndarray, np.ndarray]:
    length = h5file["measurements/length"][:]
    volume = h5file["measurements/volume"][:]
    return length, volume


def convert_to_ndarray(data):
    return np.array([item for tup in data for item in tup[0]]).reshape(-1, 3)


def load_single_model(model) -> EmbryoModel:
    names = [s.decode('utf-8') for s in model["names"][:]]

    central_spline = model["central_spline"]
    spline_domain = central_spline["abscissa"][:]
    central_coordinates = convert_to_ndarray(central_spline["ordinate"][:])

    locations = model["transverse_splines"]
    transverse_coordinates = []
    for i in range(32):
        raw_data = locations[f"transverse_spline_{i + 1:02d}/ordinate"][:]
        cleaned_data = convert_to_ndarray(raw_data)
        transverse_coordinates.append(cleaned_data)

    return EmbryoModel(names, spline_domain, central_coordinates, transverse_coordinates)


def load_avg_models(h5file, *, time_steps=None) -> Dict[int, EmbryoModel]:
    if time_steps is None:
        # there is one extra group: measurements
        time_steps = range(1, len(h5file.keys()))

    models = {}
    for time_step in time_steps:
        model = h5file[f"avg_model_{time_step:03d}"]
        models[time_step] = load_single_model(model)

    return models


def _project_to_plane_through_z_axis(x, normal):
    """
    Project x to a local coordinate system on a plane through the z-axis. The normal vector of the plane is assumed to
    be orthogonal to the z-axis.
    :param x: The points to be projected
    :param normal: The normal vector of the plane (the vector must be orthogonal to the z-axis)
    :return: Local x, y, and z coordinates of the projected points
    """
    dist_to_plane = np.dot(x, normal)
    projection = x - np.outer(dist_to_plane, normal)
    local_y = np.array([0, 0, 1])
    local_x = np.cross(normal, local_y)
    x_projected = np.dot(projection, local_x)
    y_projected = np.dot(projection, local_y)
    z_projected = dist_to_plane
    return x_projected, y_projected, z_projected


def get_spline_surface(embryo_model, i):
    """
    Create an NGSolve-OCC surface from two neighboring splines of the embryo geometry.
    The resulting surface is the one between the i-th transverse spline and its clockwise neighbor.
    :param embryo_model: The embryo model containing the splines
    :param i: The index of the transverse spline
    :return: An NGSolve-OCC surface
    """
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

    xp1, yp1, zp1 = _project_to_plane_through_z_axis(x1, normal)
    xp2, yp2, zp2 = _project_to_plane_through_z_axis(x2, normal)

    points = np.array([[(xp1[i], yp1[i], zp1[i]) for i in range(n)],
                       [(xp2[i], yp2[i], zp2[i]) for i in range(n)]])

    surf = occ.SplineSurfaceInterpolation(points)
    surf = surf.Rotate(occ.Axis((0, 0, 0), occ.X), 90)
    angle = np.arctan2(normal[1], normal[0]) / math.pi * 180
    surf = surf.Rotate(occ.Axis((0, 0, 0), occ.Z), -angle)
    return surf
