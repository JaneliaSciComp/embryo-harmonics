import logging
from typing import Tuple, Dict

import numpy as np
from scipy.spatial import KDTree
from netgen import occ
from netgen.meshing import FaceDescriptor, Element2D
from netgen.meshing import Mesh as NetgenMesh
from ngsolve import Mesh as NgsMesh

from embryoharmonics.geometry import EmbryoModel, _get_spline_surface


_logger = logging.getLogger(__name__)


def generate_embryo_mesh(
        embryo_model: EmbryoModel,
        n_interpolation: int = 32,
        mesh_size: float = 5.0
) -> NgsMesh:
    """
    Generate a volume mesh for the given embryo model.
    :param embryo_model: The embryo model to mesh
    :param n_interpolation: The number of points to use for the interpolation
    :param mesh_size: The maximum mesh size
    :return: The volume mesh of the embryo model
    """
    _logger.info("Generating mesh for embryo model")
    worm_geometry = _assemble_embryo_geometry(embryo_model, n_interpolation)
    return _mesh_embryo_geometry(worm_geometry, mesh_size)


def _mesh_embryo_geometry(
        worm_geometry: occ.Compound,
        mesh_size: float
) -> NgsMesh:
    """
    Mesh the geometry of an embryo.
    :param worm_geometry: The geometry of the embryo to mesh
    :param mesh_size: The maximum mesh size
    :return: The volume mesh of the embryo geometry
    """
    geo = occ.OCCGeometry(worm_geometry)
    surface_mesh = geo.GenerateMesh(maxh=mesh_size)
    vol_mesh, _ = _convert_to_volume_mesh(surface_mesh, mesh_size, mesh_size / 10)
    return NgsMesh(vol_mesh)


def _assemble_embryo_geometry(
        embryo_model: EmbryoModel,
        n_interpolation: int
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


def _convert_to_volume_mesh(
        surface_mesh: NetgenMesh,
        mesh_size: float,
        max_node_distance: float = None
) -> Tuple[NetgenMesh, Dict[int, int]]:
    """
    Convert a surface mesh to a volume mesh by adding a single domain inside the surface and no domain outside. If the
    surface mesh is not closed (i.e., meshing fails), close nodes can be identified by a maximum distance threshold and
    merged to close it.
    :param surface_mesh: The surface mesh to convert
    :param mesh_size: The maximum mesh size
    :param max_node_distance: Maximum distance between nodes to identify close nodes; if None, no nodes are merged
    :return: The volume mesh and a mapping from old node indices to new ones
    """
    n_nodes = len(surface_mesh.Points())
    node_is_unique = np.ones(n_nodes, dtype=bool)
    new_mesh = NetgenMesh()
    old_to_new = {}

    _logger.debug("Converting surface mesh with %d nodes to volume mesh", n_nodes)

    if max_node_distance is not None:
        # Find all nodes that are too close to each other ("doppelgängers")
        kdtree = KDTree(surface_mesh.Coordinates())
        dist = kdtree.sparse_distance_matrix(kdtree, max_distance=max_node_distance, output_type='ndarray')

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
    face_descriptor = new_mesh.Add(FaceDescriptor(surfnr=1, domin=1, domout=0, bc=1))

    # Copy elements
    for e in surface_mesh.Elements2D():
        new_mesh.Add(Element2D(face_descriptor, [old_to_new[v] for v in e.vertices]))

    # Generate volume mesh from surface
    new_mesh.GenerateVolumeMesh(maxh=mesh_size)
    _logger.debug("Converted surface mesh to volume mesh with %d nodes and %d elements", len(new_mesh.Points()), new_mesh.ne)
    return new_mesh, old_to_new
