import logging
import math

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import KDTree
from netgen import occ
import netgen.libngpy._meshing as ng
import ngsolve as ngs
import pyvista as pv


_logger = logging.getLogger(__name__)


class EmbryoModel:
    """Spline data for an embryo model. The model consists of a central spline
    (which should be straight), and a number of transverse splines that make up
    the surface of the embryo.
    The splines are defined on a common (1D) domain; the data for the splines
    was collected at certain points (called seam cells) on the lateral sides of
    the embryo.
    """
    def __init__(self, seam_cells, spline_domain, central_coordinates, transverse_coordinates):
        self.seam_cells = {name: index for index, name in enumerate(seam_cells)}
        self.spline_domain = spline_domain
        self.central_spline = CubicSpline(spline_domain, central_coordinates)
        self.transverse_splines = [
            CubicSpline(spline_domain, coordinates)
            for coordinates in transverse_coordinates
        ]


    @property
    def n_seam_cells(self):
        """The number of seam cells."""
        return len(self.seam_cells)


    @property
    def n_transverse_splines(self):
        """The number of transverse splines delimiting the embryo surface."""
        return len(self.transverse_splines)


    def generate_mesh(
            self,
            n_interpolation: int = 32,
            mesh_size: float = 5.0
    ) -> pv.UnstructuredGrid:
        """Generate a volume mesh from the embryo model.

        :param n_interpolation: The number of points to use for interpolating
            the transverse splines for generating the mesh
        :param mesh_size: The maximum mesh size
        :return: A volume mesh of the embryo model
        """
        _logger.debug("Generating mesh for embryo model")
        geometry = _assemble_embryo_geometry(self, n_interpolation)
        ngs_mesh = _mesh_embryo_geometry(geometry, mesh_size)
        return _to_vtk(ngs_mesh)


def _mesh_embryo_geometry(
        worm_geometry: occ.Compound,
        mesh_size: float
) -> ngs.Mesh:
    """Mesh the geometry of an embryo.
    
    :param worm_geometry: The geometry of the embryo to mesh
    :param mesh_size: The maximum mesh size
    :return: The volume mesh of the embryo geometry
    """
    geo = occ.OCCGeometry(worm_geometry)
    surface_mesh = geo.GenerateMesh(maxh=mesh_size)
    vol_mesh, _ = _convert_to_volume_mesh(surface_mesh, mesh_size, mesh_size / 10)
    return ngs.Mesh(vol_mesh)


def _assemble_embryo_geometry(
        embryo_model: EmbryoModel,
        n_interpolation: int
) -> occ.Compound:
    """Assemble the geometry of the embryo from the given embryo model.

    :param embryo_model: The embryo model
    :param n_interpolation: The number of points to use for interpolating the
        transverse splines for generating the mesh
    :return: The NGSolve-OCC geometry representing the embryo
    """

    # Generate the mantle of the worm
    n_splines = embryo_model.n_transverse_splines
    spline_surfaces = [
        _get_spline_surface(embryo_model, n_interpolation, i)
        for i in range(n_splines)
    ]
    mantle = occ.Compound(spline_surfaces)

    # Generate the caps on the anterior and posterior end
    min_z_threshold = embryo_model.central_spline(0.0)[2] + 1
    max_z_threshold = embryo_model.central_spline(1.0)[2] - 1
    posterior_cap = occ.Face(occ.Wire([e.Reversed() for e in mantle.edges[occ.Z < min_z_threshold]]))
    anterior_cap = occ.Face(occ.Wire([e.Reversed() for e in mantle.edges[occ.Z > max_z_threshold]]))

    total_surface = occ.Compound([mantle, anterior_cap, posterior_cap])
    return total_surface


def _convert_to_volume_mesh(
        surface_mesh: ng.Mesh,
        mesh_size: float,
        max_node_distance: float = None
) -> tuple[ng.Mesh, dict[int, int]]:
    """Convert a surface mesh to a volume mesh by adding a single domain inside
    the surface and no domain outside. If the surface mesh is not closed (i.e.,
    meshing fails), close nodes can be identified by a maximum distance
    threshold and merged to close it.

    :param surface_mesh: The surface mesh to convert
    :param mesh_size: The maximum mesh size
    :param max_node_distance: Maximum distance between nodes to identify close
        nodes; if None, no nodes are merged
    :return: The volume mesh and a mapping from old node indices to new ones
    """
    n_nodes = len(surface_mesh.Points())
    node_is_unique = np.ones(n_nodes, dtype=bool)
    new_mesh = ng.Mesh()
    old_to_new = {}

    _logger.debug("Converting surface mesh with %d nodes to volume mesh", n_nodes)

    if max_node_distance is not None:
        # Find all nodes that are too close to each other ("doppelgängers")
        kdtree = KDTree(surface_mesh.Coordinates())
        dist = kdtree.sparse_distance_matrix(
            kdtree,
            max_distance=max_node_distance,
            output_type='ndarray'
        )

        # Remove self-distances and compress i->j / j->i pairs
        dist = dist[dist['i'] != dist['j']]
        pairs = np.vstack([dist['i'], dist['j']])
        pairs = np.sort(pairs, axis=0)
        pairs = np.unique(pairs, axis=1)
        _logger.debug("Found %d pairs of close nodes to identify", len(pairs[0]))

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
    face_descriptor = new_mesh.Add(ng.FaceDescriptor(surfnr=1, domin=1, domout=0, bc=1))

    # Copy elements
    for e in surface_mesh.Elements2D():
        new_mesh.Add(ng.Element2D(face_descriptor, [old_to_new[v] for v in e.vertices]))

    # Generate volume mesh from surface
    new_mesh.GenerateVolumeMesh(maxh=mesh_size)
    _logger.debug("Converted surface mesh to volume mesh with %d nodes and %d elements",
                  len(new_mesh.Points()), new_mesh.ne)
    return new_mesh, old_to_new


def _rotate_around_z_and_x(
        points: np.ndarray,
        angle: float
) -> np.ndarray:
    """Rotate a set of points (specifying a spline surface) first around the
    z-axis by a given angle and then around the x-axis by -90 degrees.

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
    """Create an NGSolve-OCC surface from two neighboring splines of the embryo geometry.
    The resulting surface is the one between the i-th transverse spline and its clockwise neighbor.

    :param embryo_model: The embryo model containing the splines
    :param n: The number of points to use for the interpolation (too many points
        can lead to numerical instabilities)
    :param i: The index of the transverse spline
    :return: An NGSolve-OCC surface
    :raises ValueError: If the spline index is out of bounds
    """
    if i < 0 or i >= embryo_model.n_transverse_splines:
        raise ValueError(
            f"Invalid spline index {i} (out of {embryo_model.n_transverse_splines} splines)"
        )

    domain = embryo_model.spline_domain
    t = np.linspace(domain[0], domain[-1], n)

    spline1 = embryo_model.transverse_splines[i]
    neighbor = (i + 1) % embryo_model.n_transverse_splines
    spline2 = embryo_model.transverse_splines[neighbor]
    x1 = spline1(t)
    x2 = spline2(t)

    # Rotate so that the domain of the spline is [-a, a] x [0, b] x {0} and the
    # z values represent its height
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


def _to_vtk(mesh: ngs.Mesh) -> pv.UnstructuredGrid:
    """Convert an NGSolve mesh to a pyvista.UnstructuredGrid object.

    :param mesh: The NGSolve mesh to convert
    :return: A :class:`pyvista.UnstructuredGrid` object containing the mesh
    """
    points = mesh.ngmesh.Coordinates()
    cells = []
    cell_types = []
    for el in mesh.ngmesh.Elements3D():
        # NGSolve uses 1-based indexing for vertices
        cells.append([4] + [el.vertices[i].nr - 1 for i in range(4)])
        cell_types.append(pv.CellType.TETRA)

    return pv.UnstructuredGrid(cells, cell_types, points)
