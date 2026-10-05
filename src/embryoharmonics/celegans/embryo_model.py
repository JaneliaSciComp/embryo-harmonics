import copy
import logging
import math

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline, PPoly
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
    def __init__(
            self,
            seam_cells,
            spline_domain,
            central_coordinates,
            transverse_coordinates,
            *,
            symmetric: bool = False,
            bc_type: str = "not-a-knot",
    ):
        self.seam_cells = {name: index for index, name in enumerate(seam_cells)}
        self.spline_domain = spline_domain
        self.central_spline = CubicSpline(spline_domain, central_coordinates, bc_type=bc_type)

        if symmetric:
            transverse_coordinates = _replicate_and_rotate(
                transverse_coordinates[0], central_coordinates, len(transverse_coordinates)
            )

        self.transverse_splines = [
            CubicSpline(spline_domain, coordinates, bc_type=bc_type)
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


    def generate_meridian_mesh(
            self,
            n_interpolation: int = 32,
            mesh_size: float = 5.0
    ) -> pv.UnstructuredGrid:
        """Generate a 2D triangle mesh of the meridian half-plane domain
        {(z, r): 0 <= r <= R(z)} of the embryo model, with points (r, 0, z).

        This is only meaningful for rotationally symmetric models (loaded with
        ``symmetric=True``), whose central spline lies on the z-axis: the 3D
        embryo is then exactly the body of revolution of this domain.

        :param n_interpolation: The number of points to use for interpolating
            the radius profile for generating the mesh
        :param mesh_size: The maximum mesh size
        :return: A triangle mesh of the meridian domain in the y = 0 plane
        """
        _logger.debug("Generating meridian mesh for embryo model")
        _, R, _, z = axial_profile(self, n_interpolation)

        profile = occ.SplineApproximation([
            occ.Pnt(r_i, 0, z_i) for r_i, z_i in zip(R, z)
        ])
        face = occ.Face(occ.Wire([
            occ.Segment(occ.Pnt(0, 0, z[0]), occ.Pnt(R[0], 0, z[0])),
            profile,
            occ.Segment(occ.Pnt(R[-1], 0, z[-1]), occ.Pnt(0, 0, z[-1])),
            occ.Segment(occ.Pnt(0, 0, z[-1]), occ.Pnt(0, 0, z[0])),
        ]))
        surface_mesh = occ.OCCGeometry(face).GenerateMesh(maxh=mesh_size)
        return _to_vtk_2d(surface_mesh)


def axial_profile(
        embryo_model: EmbryoModel,
        n_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample the axial coordinate (arclength of the central spline) and
    radius profile (distance of the 0-th transverse spline from the central
    axis) of an embryo model.

    :param embryo_model: The embryo model to sample
    :param n_samples: The number of points to sample the profile at
    :return: Arclength ``s``, radius profile ``R``, spline domain samples
        ``t``, and the corresponding z-coordinates of the central spline
    """
    domain = embryo_model.spline_domain
    t_samples = np.linspace(domain[0], domain[-1], n_samples)

    central = embryo_model.central_spline(t_samples)
    reference = embryo_model.transverse_splines[0](t_samples)

    tangent = embryo_model.central_spline.derivative()(t_samples)
    speed = np.linalg.norm(tangent, axis=1)
    s_samples = cumulative_trapezoid(speed, t_samples, initial=0.0)

    R_samples = np.linalg.norm(reference - central, axis=1)
    z_samples = central[:, 2]

    return s_samples, R_samples, t_samples, z_samples



def expand_and_extend_tail(
        embryo_model: EmbryoModel,
        radial_expansion: float = 0.0,
        radial_offset: float = 0.0,
        linear_fraction: float = 0.8
) -> EmbryoModel:
    """Expand an embryo model radially and continue its tail (end of the spline
    domain) along the end tangents, without changing the shape of the existing
    body.

    Every transverse spline is scaled about the central spline by
    ``1 + radial_expansion``, pushed outwards by ``radial_offset`` (in model
    units; the offset curve is re-interpolated at the original knots) and gets
    one linear piece appended (the existing pieces stay unchanged) that
    continues its end tangent until the radius has dropped by
    ``linear_fraction`` of its end value. The central spline is continued
    straight. The extended model still ends with a flat cap.

    :param embryo_model: The embryo model to expand and extend
    :param radial_expansion: Relative radial expansion of the transverse splines
    :param radial_offset: Absolute radial offset of the transverse splines, e.g.
        half a cell diameter when the model was fitted to nucleus positions
    :param linear_fraction: Fraction of the end radius covered by the linear
        continuation; 0 leaves the tail where it is
    :return: A new embryo model with extended spline domain
    """
    central = embryo_model.central_spline
    t_end = embryo_model.spline_domain[-1]
    c_end, c_tangent = central(t_end), central.derivative()(t_end)
    f = 1 + radial_expansion

    def linear_piece(p, d):
        """Coefficients of the cubic piece p + d * t."""
        return np.stack([np.zeros_like(p), np.zeros_like(p), d, p])[:, None]

    def scale_and_measure(spline):
        scaled = PPoly(central.c + f * (spline.c - central.c), spline.x)
        if radial_offset:
            c_knots, p_knots = central(spline.x), scaled(spline.x)
            r_knots = np.linalg.norm((p_knots - c_knots)[:, :2], axis=1, keepdims=True)
            scaled = CubicSpline(spline.x, c_knots + (p_knots - c_knots) * (1 + radial_offset / r_knots))
        p_end, d_end = scaled(t_end), scaled.derivative()(t_end)
        offset = p_end - c_end
        radius = np.hypot(*offset[:2])
        d_radius = d_end[:2] @ offset[:2] / radius
        if d_radius >= 0:
            raise ValueError("Transverse spline does not taper at the end of the domain")
        return scaled, p_end, d_end, linear_fraction * radius / -d_radius

    # All splines must end at the same parameter value: continue every one as
    # far as the one that needs the longest continuation
    tails = [scale_and_measure(spline) for spline in embryo_model.transverse_splines]
    dt = max(dt for *_, dt in tails)
    extended = copy.copy(embryo_model)
    extended.transverse_splines = [scaled for scaled, *_ in tails]
    if dt == 0:
        return extended
    for scaled, p_end, d_end, _ in tails:
        scaled.extend(linear_piece(p_end, d_end), [t_end + dt])
    extended.central_spline = PPoly(central.c, central.x)
    extended.central_spline.extend(linear_piece(c_end, c_tangent), [t_end + dt])
    extended.spline_domain = np.append(embryo_model.spline_domain, t_end + dt)
    return extended


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


def _replicate_and_rotate(
        reference_coordinates: np.ndarray,
        central_coordinates: np.ndarray,
        n_splines: int
) -> list[np.ndarray]:
    """Replicate a single transverse spline's coordinates into ``n_splines``
    copies, evenly rotated about the central axis, to make a rotationally
    symmetric embryo geometry.

    :param reference_coordinates: Coordinates of the spline to replicate
        (e.g. the 0-th transverse spline)
    :param central_coordinates: Coordinates of the central spline, defined on
        the same domain as ``reference_coordinates``
    :param n_splines: The number of rotated copies to generate
    :return: A list of ``n_splines`` coordinate arrays, evenly spaced around
        the central axis (the first entry equals ``reference_coordinates``)
    """
    radial = reference_coordinates - central_coordinates
    angles = np.arange(n_splines) / n_splines * 2 * math.pi
    copies = []
    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        copies.append(radial @ rotation_matrix.T + central_coordinates)
    return copies


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


def _to_vtk_2d(mesh: ng.Mesh) -> pv.UnstructuredGrid:
    """Convert a netgen surface mesh to a pyvista.UnstructuredGrid object.

    :param mesh: The netgen mesh to convert
    :return: A :class:`pyvista.UnstructuredGrid` object containing the mesh
    """
    points = mesh.Coordinates()
    # Netgen uses 1-based indexing for vertices
    cells = np.array([
        [v.nr - 1 for v in el.vertices] for el in mesh.Elements2D()
    ])
    return pv.UnstructuredGrid({int(pv.CellType.TRIANGLE): cells}, points)


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


def transform_mesh(
        source: pv.UnstructuredGrid,
        target: pv.UnstructuredGrid,
        *,
        fudge_factor: float = 1e-4,
) -> pv.UnstructuredGrid:
    """Transform the source mesh to match the outline of a target mesh (both are
    supposed to be embryo geometries). The source mesh is streched in the z and
    radial direction to match the target mesh.

    :param source: a mesh that is transformed in place to match the target geometry.
    :param target: a second mesh with a kind of similar shape.
    :param fudge_factor: a small number to make sure the transformed source
        stays inside the target.
    :return: The transformed source mesh.
    """
    transformed = source.copy(deep=True)
    x, y, z = transformed.points[:, 0], transformed.points[:, 1], transformed.points[:, 2]

    # Stretch points in z direction (shifting by the center to respect the fudge factor)
    _, x_src_max, _, y_src_max, z_src_min, z_src_max = transformed.bounds
    _, x_trg_max, _, y_trg_max, z_trg_min, z_trg_max = target.bounds

    z_src_center = (z_src_min + z_src_max) / 2
    z_trg_center = (z_trg_min + z_trg_max) / 2
    z_factor = (z_trg_max - z_trg_min) * (1 - fudge_factor) / (z_src_max - z_src_min)
    transformed.points[:, 2] = (z - z_src_center) * z_factor + z_trg_center

    # Stretch points in radial direction
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    r_factor = np.ones_like(r)

    surface_src = transformed.extract_surface()
    surface_trg = target.extract_surface()
    surface_indices = transformed.surface_indices()

    r_max = np.max([np.sqrt(x_trg_max**2 + y_trg_max**2), np.sqrt(x_src_max**2 + y_src_max**2)])
    ray_start = np.column_stack([np.zeros_like(x), np.zeros_like(y), z])
    ray_end = np.column_stack([np.cos(theta) * r_max * 1.1, np.sin(theta) * r_max * 1.1, z])

    for i, r_current in enumerate(r):
        if np.isclose(r_current, 0):
            r_factor[i] = 1
            continue

        # Find the distance to the target surface
        point, _ = surface_trg.ray_trace(ray_start[i], ray_end[i])
        r_surf_trg = np.linalg.norm(point[:, :2])

        if i in surface_indices:
            # Easy: just project to the target surface
            r_factor[i] = r_surf_trg / r_current
        else:
            # Harder: find the distance to the source mesh surface and stretch accordingly
            point, _ = surface_src.ray_trace(ray_start[i], ray_end[i])
            r_surf_src = np.linalg.norm(point[:, :2])
            r_factor[i] = r_surf_trg / r_surf_src

    transformed.points[:, :2] *= r_factor[:, None] * (1 - fudge_factor)
    return transformed
