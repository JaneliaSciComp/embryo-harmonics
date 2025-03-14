"""Collection of finite element utilities for harmonic analysis"""
from dataclasses import dataclass
import numpy as np
import pyvista as pv
import scipy.sparse as scs


@dataclass
class FemMatrices:
    """
    Class to store the mass and stiffness matrices for a given mesh.
    """
    mass: scs.csr_matrix | None = None
    stiffness: scs.csr_matrix | None = None

    def __post_init__(self):
        mass_is_defined = self.mass is not None
        stiffness_is_defined = self.stiffness is not None

        if not mass_is_defined and not stiffness_is_defined:
            raise ValueError("Either mass or stiffness matrix must be defined")

        if mass_is_defined and stiffness_is_defined and self.mass.shape != self.stiffness.shape:
            raise ValueError("Mass and stiffness matrices must have the same shape")

    @staticmethod
    def compute_for(
            mesh: pv.UnstructuredGrid,
            mass: bool = True,
            stiffness: bool = True
    ) -> 'FemMatrices':
        """
        Compute the mass and stiffness matrices for a given 2D or 3D simplicial
        mesh.

        :param mesh: The mesh to compute the matrices on
        :param mass: Whether to compute the mass matrix
        :param stiffness: Whether to compute the stiffness matrix
        :return: The mass and stiffness matrices as sparse matrices
        """

        # PyVista stores cells as [n_vertices, vertex_1, ..., vertex_n [...]]
        mesh_dimension = mesh.cells[0] - 1
        match mesh_dimension:
            case 3:
                return compute_mass_and_stiffness_3d(mesh, mass, stiffness)
            case 2:
                return compute_mass_and_stiffness_2d(mesh, mass, stiffness)
            case _:
                raise ValueError(f"Invalid mesh dimension {mesh_dimension}")


def compute_mass_and_stiffness_3d(
        mesh: pv.UnstructuredGrid,
        compute_mass: bool = True,
        compute_stiffness: bool = True
) -> FemMatrices:
    """
    Compute the mass and stiffness matrices for a given 3D mesh.

    :param mesh: The mesh to compute the matrices on
    :param mass: Whether to compute the mass matrix
    :param stiffness: Whether to compute the stiffness matrix
    :return: The mass and stiffness matrices as sparse matrices
    """
    # Extract element to vertex mapping
    cell_to_vertex = mesh.cell_connectivity.reshape(-1, 4)
    matrix_shape = (mesh.n_points, mesh.n_points)
    mass, stiffness = None, None

    # Compute the volume of each cell and fix orientation of the cells
    v_0 = mesh.points[cell_to_vertex[:, 0]].astype(np.float64)
    e_10 = mesh.points[cell_to_vertex[:, 1]].astype(np.float64) - v_0
    e_20 = mesh.points[cell_to_vertex[:, 2]].astype(np.float64) - v_0
    e_30 = mesh.points[cell_to_vertex[:, 3]].astype(np.float64) - v_0
    det_element_trafo = np.vecdot(np.cross(e_10, e_20, axis=1), e_30, axis=1)
    swap = det_element_trafo < 0
    if np.any(swap):
        cell_to_vertex[swap, 1], cell_to_vertex[swap, 2] = \
            cell_to_vertex[swap, 2], cell_to_vertex[swap, 1]
        e_10[swap], e_20[swap] = e_20[swap], e_10[swap]
        det_element_trafo[swap] = -det_element_trafo[swap]

    # Set up local to global index mapping
    row_ind = np.repeat(cell_to_vertex, 4, axis=1).flatten()
    col_ind = np.tile(cell_to_vertex, (1, 4)).flatten()

    if compute_mass:
        # Set up lowest-order finite element mass matrix for one simplicial cell
        cell_mass = (np.ones((4, 4)) + np.eye(4)) / 120
        cell_mass = np.outer(det_element_trafo, cell_mass).flatten()

        # Combine element matrices into global matrix
        mass = scs.csr_matrix((cell_mass, (row_ind, col_ind)), shape=matrix_shape)

    if compute_stiffness:
        # Compute the normal vectors (=scaled gradients) on opposite faces
        n_0 = np.cross(e_30 - e_10, e_20 - e_10, axis=1)
        n_1 = np.cross(e_20, e_30, axis=1)
        n_2 = np.cross(e_30, e_10, axis=1)
        n_3 = np.cross(e_10, e_20, axis=1)

        # Compute pairwise dot products of normals
        normals = np.stack([n_0, n_1, n_2, n_3], axis=1)
        cell_stiffness = (np.einsum("...jk,...lk->...jl", normals, normals)
                        / (6 * np.expand_dims(det_element_trafo, axis=(1, 2)))).flatten()

        # Combine element matrices into global matrix
        stiffness = scs.csr_matrix((cell_stiffness, (row_ind, col_ind)), shape=matrix_shape)

    return FemMatrices(mass=mass, stiffness=stiffness)


def compute_mass_and_stiffness_2d(
    mesh: pv.UnstructuredGrid,
    compute_mass: bool = True,
    compute_stiffness: bool = True
) -> FemMatrices:
    """
    Compute the mass and stiffness matrices for a given 2D mesh.
    
    :param mesh: The mesh to compute the matrices on
    :param mass: Whether to compute the mass matrix
    :param stiffness: Whether to compute the stiffness matrix
    :return: The mass and stiffness matrices as sparse matrices
    """
    # Extract element to vertex mapping
    cell_to_vertex = mesh.cell_connectivity.reshape(-1, 3)
    matrix_shape = (mesh.n_points, mesh.n_points)
    mass, stiffness = None, None

    # Compute the area of each cell and fix orientation of the cells
    v_0 = mesh.points[cell_to_vertex[:, 0]].astype(np.float64)
    e_10 = mesh.points[cell_to_vertex[:, 1]].astype(np.float64) - v_0
    e_20 = mesh.points[cell_to_vertex[:, 2]].astype(np.float64) - v_0
    element_normal = np.cross(e_10, e_20)
    det_element_trafo = np.linalg.norm(element_normal, axis=1)

    # Set up local to global index mapping
    row_ind = np.repeat(cell_to_vertex, 3, axis=1).flatten()
    col_ind = np.tile(cell_to_vertex, (1, 3)).flatten()

    if compute_mass:
        # Set up lowest-order finite element mass matrix for one simplicial cell
        cell_mass = (np.ones((3, 3)) + np.eye(3)) / 24
        cell_mass = np.outer(det_element_trafo, cell_mass).flatten()

        # Combine element matrices into global matrix
        mass = scs.csr_matrix((cell_mass, (row_ind, col_ind)), shape=matrix_shape)

    if compute_stiffness:
        # Compute the normal vectors (=scaled gradients) on opposite edges
        element_normal /= det_element_trafo[:, np.newaxis]
        n_0 = np.cross(element_normal, e_20 - e_10)
        n_1 = np.cross(e_20, element_normal)
        n_2 = np.cross(element_normal, e_10)

        # Compute pairwise dot products of normals
        normals = np.stack([n_0, n_1, n_2], axis=1)
        cell_stiffness = (np.einsum("...jk,...lk->...jl", normals, normals)
                        / (2 * np.expand_dims(det_element_trafo, axis=(1, 2)))).flatten()

        # Combine element matrices into global matrix
        stiffness = scs.csr_matrix((cell_stiffness, (row_ind, col_ind)), shape=matrix_shape)

    return FemMatrices(mass=mass, stiffness=stiffness)
