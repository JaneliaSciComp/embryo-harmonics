"""Evaluation of finite element functions at arbitrary points."""
import functools
import hashlib
import logging
from typing import Any

import numpy as np
import pyvista as pv
import scipy.sparse as sp


_logger = logging.getLogger(__name__)


def interpolation_matrix(
        mesh: pv.UnstructuredGrid,
        locations: np.ndarray
) -> sp.csr_matrix:
    """Evaluate the lowest-order finite element basis functions of the mesh at
    the given locations. Its product with point values is the right-hand side
    of the L2 projection onto the mesh; its product with nodal values samples
    the finite element function at the locations.

    :param mesh: The (tetrahedral or triangular) mesh whose basis functions to
        evaluate.
    :param locations: The locations to evaluate at (n, 3).
    :return: A sparse (mesh.n_points, n) matrix; columns of locations outside
        the mesh are zero.
    """
    # Make locations, points, and cells hashable
    # This is some effort, but still a lot less than the cost of recomputing the
    # mesh query for each interpolation
    mesh_wrapper = HashableWrapper(mesh, [mesh.points, mesh.cells])
    locations_wrapper = HashableWrapper(locations, [locations])
    barycentric, cell_indices = _compute_barycentric_coordinates(mesh_wrapper, locations_wrapper)

    inside = np.flatnonzero(~np.any(np.isnan(barycentric), axis=1))
    _logger.debug("Interpolating %d points, skipping %d points outside the mesh",
                  len(locations), len(locations) - len(inside))

    n_vertices = barycentric.shape[1]
    cells = mesh.cell_connectivity.reshape(-1, n_vertices)[cell_indices[inside]]
    return sp.csr_matrix(
        (barycentric[inside].ravel(), (cells.ravel(), np.repeat(inside, n_vertices))),
        shape=(mesh.n_points, len(locations))
    )


class HashableWrapper:
    """A wrapper class to make various numpy arrays hashable for caching
    purposes.
    """
    def __init__(self, data: Any, arrays_to_hash: list[np.ndarray]):
        """Set up the wrapper to hash the given data.

        :param data: The data to wrap.
        :param arrays_to_hash: The arrays to compute the hash from.
        """
        self.data = data

        hash_accumulator = hashlib.md5(np.ascontiguousarray(arrays_to_hash[0]).data)
        for array in arrays_to_hash[1:]:
            hash_accumulator.update(np.ascontiguousarray(array).data)
        self.checksum = hash_accumulator.hexdigest()

    def __hash__(self):
        return hash(self.checksum)

    def __eq__(self, other):
        if isinstance(other, HashableWrapper):
            return np.array_equal(self.checksum, other.checksum)
        return False


@functools.lru_cache(maxsize=50)
def _compute_barycentric_coordinates(
        mesh_wrapper: HashableWrapper,
        locations_wrapper: HashableWrapper
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the barycentric coordinates of the given locations with respect
    to the given mesh and the indices of their containing cells. If the
    locations are outside the mesh, the barycentric coordinates are set to NaN.
    """
    mesh = mesh_wrapper.data
    locations = locations_wrapper.data
    n_locs = locations.shape[0]

    # Find containing cells, skip points outside the mesh
    _logger.debug("Finding containing cells for %d points in mesh with %d points and %d cells",
                    len(locations), mesh.n_points, mesh.n_cells)
    containing_cells = mesh.find_containing_cell(locations)

    is_outside: np.ndarray = containing_cells == -1
    _logger.debug("Interpolating %d points, skipping %d points outside the mesh",
                    len(locations), np.sum(is_outside))

    locations = locations[~is_outside]
    cell_indices = containing_cells[~is_outside]

    # Get points of the cells containing the points
    # ponytail: assumes a single cell type (tetrahedra or triangles)
    m = locations.shape[0]
    n_vertices = mesh.cell_connectivity.size // mesh.n_cells
    cells = mesh.cell_connectivity.reshape(-1, n_vertices)[cell_indices]
    cell_points = mesh.points[cells]

    # Compute barycentric coordinates of the points in the cells by solving
    # a linear system of equations for each point; the pseudo-inverse also
    # covers planar triangles embedded in 3D
    element_matrices = np.concatenate((
        np.transpose(cell_points, (0, 2, 1)),
        np.ones((m, 1, n_vertices))
    ), axis=1)
    b = np.concatenate((locations, np.ones((m, 1))), axis=1)
    b = b[:, :, None]

    barycentric = np.full((n_locs, n_vertices), np.nan)
    if m > 0:
        barycentric[~is_outside] = (np.linalg.pinv(element_matrices) @ b).squeeze(-1)
    return barycentric, containing_cells
