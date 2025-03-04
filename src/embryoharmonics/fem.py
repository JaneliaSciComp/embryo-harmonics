"""Collection of finite element utilities for harmonic analysis"""
import numpy as np
import pyvista as pv
import scipy.sparse as scs


def compute_mass_and_stiffness(
        mesh: pv.UnstructuredGrid
    ) -> tuple[scs.csr_matrix, scs.csr_matrix]:
    """
    Compute the mass and stiffness matrices for a given 2D or 3D simplicial
    mesh.
    :param mesh: The mesh to compute the matrices on
    :return: The mass and stiffness matrices as sparse matrices
    """

    # PyVista stores cells as [n_vertices, vertex_1, ..., vertex_n [...]]
    match mesh.cells[0] - 1:
        case 3:
            return compute_mass_and_stiffness_3d(mesh)
        case 2:
            return compute_mass_and_stiffness_2d(mesh)
        case _:
            raise ValueError(f"Invalid mesh dimension {mesh.points.shape[1]}")


def compute_mass_and_stiffness_3d(mesh: pv.UnstructuredGrid):
    """
    Compute the mass and stiffness matrices for a given 3D mesh.
    :param mesh: The mesh to compute the matrices on
    :return: The mass and stiffness matrices as sparse matrices
    """
    # Extract element to vertex mapping
    cell_to_vertex = mesh.cell_connectivity.reshape(-1, 4)
    matrix_shape = (mesh.n_points, mesh.n_points)

    # Compute the volume of each cell and fix orientation of the cells
    v_0 = mesh.points[cell_to_vertex[:, 0]]
    e_10 = mesh.points[cell_to_vertex[:, 1]] - v_0
    e_20 = mesh.points[cell_to_vertex[:, 2]] - v_0
    e_30 = mesh.points[cell_to_vertex[:, 3]] - v_0
    det_element_trafo = np.vecdot(np.cross(e_10, e_20, axis=1), e_30, axis=1)
    swap = det_element_trafo < 0
    if np.any(swap):
        cell_to_vertex[swap, [1, 2]] = cell_to_vertex[swap, [2, 1]]
        e_10[swap], e_20[swap] = e_20[swap], e_10[swap]
        det_element_trafo[swap] = -det_element_trafo[swap]

    # Set up lowest-order finite element mass matrix for one simplicial cell
    cell_mass = (np.ones((4, 4)) + np.eye(4)) / 120
    cell_mass = np.outer(det_element_trafo, cell_mass).flatten()

    # Combine element matrices into global matrix
    row_ind = np.repeat(cell_to_vertex, 4, axis=1).flatten()
    col_ind = np.tile(cell_to_vertex, (1, 4)).flatten()
    mass = scs.csr_matrix((cell_mass, (row_ind, col_ind)), shape=matrix_shape)

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

    return mass, stiffness


def compute_mass_and_stiffness_2d(mesh: pv.UnstructuredGrid):
    """
    Compute the mass and stiffness matrices for a given 2D mesh.
    :param mesh: The mesh to compute the matrices on
    :return: The mass and stiffness matrices as sparse matrices
    """
    # Extract element to vertex mapping
    cell_to_vertex = mesh.cell_connectivity.reshape(-1, 3)
    matrix_shape = (mesh.n_points, mesh.n_points)

    # Compute the area of each cell and fix orientation of the cells
    v_0 = mesh.points[cell_to_vertex[:, 0]]
    e_10 = mesh.points[cell_to_vertex[:, 1]] - v_0
    e_20 = mesh.points[cell_to_vertex[:, 2]] - v_0
    element_normal = np.cross(e_10, e_20)
    det_element_trafo = np.linalg.norm(element_normal, axis=1)
    element_normal /= det_element_trafo[:, np.newaxis]

    # Set up lowest-order finite element mass matrix for one simplicial cell
    cell_mass = (np.ones((3, 3)) + np.eye(3)) / 24
    cell_mass = np.outer(det_element_trafo, cell_mass).flatten()

    # Combine element matrices into global matrix
    row_ind = np.repeat(cell_to_vertex, 3, axis=1).flatten()
    col_ind = np.tile(cell_to_vertex, (1, 3)).flatten()
    mass = scs.csr_matrix((cell_mass, (row_ind, col_ind)), shape=matrix_shape)

    # Compute the normal vectors (=scaled gradients) on opposite edges
    n_0 = np.cross(element_normal, e_20 - e_10)
    n_1 = np.cross(e_20, element_normal)
    n_2 = np.cross(element_normal, e_10)

    # Compute pairwise dot products of normals
    normals = np.stack([n_0, n_1, n_2], axis=1)
    cell_stiffness = (np.einsum("...jk,...lk->...jl", normals, normals)
                      / (2 * np.expand_dims(det_element_trafo, axis=(1, 2)))).flatten()

    # Combine element matrices into global matrix
    stiffness = scs.csr_matrix((cell_stiffness, (row_ind, col_ind)), shape=matrix_shape)

    return mass, stiffness
