import math
from typing import Dict, Iterable

import h5py
import numpy as np
from netgen import occ
from scipy.interpolate import CubicSpline


class EmbryoModel:
    """
    Class representing the spline data for an embryo model. The model consists of a central spline (which should be
    straight), and a number of transverse splines that make up the surface of the embryo.
    The splines are defined on a common domain and the data for the splines was collected at certain points (called
    seam cells) on the lateral sides of the embryo.
    """
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


def load_measurement(
        h5file: h5py.File
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the length and volume measurements from the given HDF5 file.
    :param h5file: The HDF5 file containing the measurements
    :return: Length and volume measurements as numpy arrays
    """
    length = h5file["measurements/length"][:]
    volume = h5file["measurements/volume"][:]
    return length, volume


def _convert_to_ndarray(
        data: h5py.Dataset
) -> np.ndarray:
    return np.array([item for tup in data for item in tup[0]]).reshape(-1, 3)


def _load_single_model(
        model: h5py.Group,
) -> EmbryoModel:
    """
    Load a single averaged model from the given HDF5 group.
    :param model: The HDF5 group containing spline data for the averaged model
    :return: Spline data for the averaged model
    """
    names = [s.decode('utf-8') for s in model["names"][:]]

    central_spline = model["central_spline"]
    spline_domain = central_spline["abscissa"][:]
    central_coordinates = _convert_to_ndarray(central_spline["ordinate"][:])

    locations = model["transverse_splines"]
    transverse_coordinates = []
    for i in range(32):
        raw_data = locations[f"transverse_spline_{i + 1:02d}/ordinate"][:]
        cleaned_data = _convert_to_ndarray(raw_data)
        transverse_coordinates.append(cleaned_data)

    return EmbryoModel(names, spline_domain, central_coordinates, transverse_coordinates)


def load_avg_models(
       h5file: h5py.File,
        *,
        time_steps: Iterable[int] = None
) -> Dict[int, EmbryoModel]:
    """
    Load all averaged models from the given HDF5 file.
    :param h5file: The HDF5 file containing spline data for the averaged models
    :param time_steps: Which time steps to load (if None, all time steps are loaded; 1-based)
    :return: A dictionary mapping time steps to the corresponding models
    """
    if time_steps is None:
        # there is one extra group (measurements)
        time_steps = range(1, len(h5file.keys()))

    models = {}
    for time_step in time_steps:
        model = h5file[f"avg_model_{time_step:03d}"]
        models[time_step] = _load_single_model(model)

    return models


def _rotate_around_z_and_x(
        points: np.ndarray,
        angle: float
) -> np.ndarray:
    """
    Rotate a set of points (specifying a spline surface) first around the z-axis by a given angle and then around the
    x-axis by -90 degrees.
    :param points: The points to be rotated
    :param angle: The angle by which to rotate in the xy-plane in deg
    :return: Coordinates of the rotated points
    """
    # rotate in xy-plane
    angle = angle / 180 * math.pi
    s = np.sin(angle)
    c = np.cos(angle)
    rotation_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    points = np.dot(points, rotation_matrix.T)
    # manually apply rotation matrix around x-axis
    return np.column_stack((points[:, 0], points[:, 2], -points[:, 1]))


def _get_spline_surface(
        embryo_model: EmbryoModel,
        n: int,
        i:int
) -> occ.SplineSurfaceInterpolation:
    """
    Create an NGSolve-OCC surface from two neighboring splines of the embryo geometry.
    The resulting surface is the one between the i-th transverse spline and its clockwise neighbor.
    :param embryo_model: The embryo model containing the splines
    :param n: The number of points to use for the interpolation (too many points can lead to numerical instabilities)
    :param i: The index of the transverse spline
    :return: An NGSolve-OCC surface
    :raises ValueError: If the spline index is out of bounds
    """
    if i < 0 or i >= embryo_model.n_transverse_splines:
        raise ValueError(f"Invalid spline index {i} (out of {embryo_model.n_transverse_splines} splines)")

    domain = embryo_model.spline_domain
    t = np.linspace(domain[0], domain[-1], n)

    spline1 = embryo_model.transverse_splines[i]
    neighbor = (i + 1) % embryo_model.n_transverse_splines
    spline2 = embryo_model.transverse_splines[neighbor]
    x1 = spline1(t)
    x2 = spline2(t)

    # Rotate so that the domain of the spline is [-a, a] x [0, b] x {0} and the z values represent its height
    midpoint_angle = (2 * i + 1) / (2 * embryo_model.n_transverse_splines) * 180
    angle = - (midpoint_angle + 90)
    xr1 = _rotate_around_z_and_x(x1, angle)
    xr2 = _rotate_around_z_and_x(x2, angle)

    # Generate the surface and rotate it back to the original orientation
    points = np.array([[tuple(xr1[i]) for i in range(n)], [tuple(xr2[i]) for i in range(n)]])
    surf = occ.SplineSurfaceInterpolation(points)
    surf = surf.Rotate(occ.Axis(occ.Pnt(0, 0, 0), occ.X), 90)
    surf = surf.Rotate(occ.Axis(occ.Pnt(0, 0, 0), occ.Z), -angle)
    return surf


def assemble_embryo_geometry(
        embryo_model: EmbryoModel,
        n_interpolation: int = 32
) -> occ.Compound:
    """
    Assemble the geometry of the embryo from the given embryo model.
    :param embryo_model: The embryo model
    :param n_interpolation: The number of points to use for the interpolation
    :return: The NGSolve-OCC geometry representing the embryo
    """

    # Generate the mantle of the worm
    n_splines = embryo_model.n_transverse_splines
    spline_surfaces = [_get_spline_surface(embryo_model, n_interpolation, i) for i in range(n_splines)]
    mantle = occ.Compound(spline_surfaces)

    # Generate the caps on the anterior and posterior end
    min_z_threshold = embryo_model.central_spline(0.0)[2] + 1
    max_z_threshold = embryo_model.central_spline(1.0)[2] - 1
    posterior_cap = occ.Face(occ.Wire([e.Reversed() for e in mantle.edges[occ.Z < min_z_threshold]]))
    anterior_cap = occ.Face(occ.Wire([e.Reversed() for e in mantle.edges[occ.Z > max_z_threshold]]))

    total_surface = occ.Compound([mantle, anterior_cap, posterior_cap])
    return total_surface
