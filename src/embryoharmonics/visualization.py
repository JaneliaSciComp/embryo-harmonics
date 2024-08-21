import numpy as np
import pyvista as pv
from ngsolve import Mesh
from pyvista import UnstructuredGrid


def to_vtk(
        mesh: Mesh,
        eigenfunctions: np.ndarray,
) -> UnstructuredGrid:
    """
    Convert mesh and eigenfunctions to a pyvista.UnstructuredGrid object.
    :param mesh: The underlying mesh
    :param eigenfunctions: The eigenfunctions as a numpy array
    :return: A :class:`pyvista.UnstructuredGrid` object containing mesh and eigenfunctions
    """
    points = mesh.ngmesh.Coordinates()
    cells = []
    cell_types = []
    for el in mesh.ngmesh.Elements3D():
        # NGSolve uses 1-based indexing for vertices
        cells.append([4] + [el.vertices[i].nr - 1 for i in range(4)])
        cell_types.append(pv.CellType.TETRA)

    pv_data = pv.UnstructuredGrid(cells, cell_types, points)

    n_eigenfunctions = eigenfunctions.shape[1]
    for i in range(n_eigenfunctions):
        pv_data[f"Eigenfunction {i}"] = eigenfunctions[:, i]

    return pv_data