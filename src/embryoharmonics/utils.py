import math
from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import KDTree
from netgen import occ
from ngsolve import Mesh
from netgen.meshing import FaceDescriptor, Element2D
from netgen.meshing import Mesh as NetgenMesh


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


def _rotate_around_z_and_x(points, angle):
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


def get_spline_surface(embryo_model, n, i):
    """
    Create an NGSolve-OCC surface from two neighboring splines of the embryo geometry.
    The resulting surface is the one between the i-th transverse spline and its clockwise neighbor.
    :param embryo_model: The embryo model containing the splines
    :param n: The number of points to use for the interpolation
    :param i: The index of the transverse spline
    :return: An NGSolve-OCC surface
    """
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
    surf = surf.Rotate(occ.Axis((0, 0, 0), occ.X), 90)
    surf = surf.Rotate(occ.Axis((0, 0, 0), occ.Z), -angle)
    return surf


def convert_to_volume_mesh(surface_mesh, max_node_distance=None) -> Tuple[Mesh, Dict[int, int]]:
    n_nodes = len(surface_mesh.Points())
    node_is_unique = np.ones(n_nodes, dtype=bool)
    new_mesh = NetgenMesh()
    old_to_new = {}

    if max_node_distance is not None:
        # Find all nodes that are too close to each other ("doppelgängers")
        kdtree = KDTree(surface_mesh.Coordinates())
        dist = kdtree.sparse_distance_matrix(kdtree, max_distance=max_node_distance, output_type='ndarray')

        # Remove self-distances and compress i->j / j->i pairs
        dist = dist[dist['i'] != dist['j']]
        pairs = np.vstack([dist['i'], dist['j']])
        pairs = np.sort(pairs, axis=0)
        pairs = np.unique(pairs, axis=1)

        # Mark all nodes that have a doppelgänger
        node_is_unique[pairs[0]] = False
        node_is_unique[pairs[1]] = False

        # Copy non-unique nodes
        for i, j in zip(pairs[0], pairs[1]):
            # Identify doppelgängers in new mesh (node indices are 1-based)
            new_node = new_mesh.Add(surface_mesh[i + 1])
            old_to_new[i + 1] = new_node
            old_to_new[j + 1] = new_node

    # Copy unique nodes
    for i in range(n_nodes):
        if node_is_unique[i]:
            # Node indices are 1-based
            old_to_new[i + 1] = new_mesh.Add(surface_mesh[i + 1])

    # Create a face descriptor that is used for all elements
    # (one single surface with one domain inside and no domain outside)
    face_descriptor = new_mesh.Add(FaceDescriptor(surfnr=1, domin=1, domout=0, bc=1))

    # Copy elements
    for e in surface_mesh.Elements2D():
        new_mesh.Add(Element2D(face_descriptor, [old_to_new[v] for v in e.vertices]))

    # Generate volume mesh from surface
    new_mesh.GenerateVolumeMesh()
    return new_mesh, old_to_new
