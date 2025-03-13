from typing import Iterable

import h5py
import numpy as np
import pyvista as pv

from embryoharmonics import MeshData
from embryoharmonics.harmonics import Harmonics


def save_mesh(
        file_name: str,
        mesh: pv.UnstructuredGrid
):
    """Store the given mesh in a vtk file.
    
    :param file_name: The name of the file to store the mesh in (must end with .vtu).
    :param mesh: The mesh to store.
    :raises ValueError: If the file name does not end with .vtu.
    """
    if not file_name.endswith('.vtu'):
        raise ValueError('File name must end with ".vtu"')

    mesh.save(file_name)


def load_mesh(
        file_name: str
) -> pv.UnstructuredGrid:
    """Load the mesh from the given file.

    :param file_name: The name of the file to load the mesh from.
    :raises ValueError: If the file does not contain a triangular or tetrahedral mesh.
    """
    mesh = pv.read(file_name).cast_to_unstructured_grid()

    first_cell_type = mesh.celltypes[0]
    if first_cell_type not in [5, 10]:  # TRIANGLE or TETRA
        raise ValueError('Mesh must be triangular or tetrahedral')

    if any(cell_type != first_cell_type for cell_type in mesh.celltypes):
        raise ValueError('Mesh must contain uniform cell types')

    return mesh


def save_mesh_data(
        file_name: str,
        data: MeshData | Iterable[MeshData]
):
    """Store the given data in a HDF5 file.

    :param file_name: The name of the file to store the data in (must end with .h5).
    :param data: The data to store (must be of same length).
    :raises ValueError: If the file name does not end with .h5 or if the data is
        not of the same length.
    """
    if isinstance(data, MeshData):
        data = [data]

    if not file_name.endswith('.h5'):
        raise ValueError('File name must end with ".h5"')

    first_data_length = len(data[0].data)
    if any(len(d.data) != first_data_length for d in data):
        raise ValueError('All data must have the same length')

    # Concatenate the data and names in the same order
    concatenated_data = np.vstack([d.data for d in data])
    concatenated_names = [d.name for d in data]

    with h5py.File(file_name, 'w') as h5file:
        h5file.create_dataset('/data', data=concatenated_data)
        h5file.create_dataset('/names', data=concatenated_names)


def load_mesh_data(
        file_name: str,
        mesh: pv.UnstructuredGrid
) -> list[MeshData]:
    """Load mesh data from the given HDF5 file.

    :param file_name: The name of the file to store the data in (must end with .h5).
    :param mesh: The mesh these :class:`MeshData` objects belong to.
    """
    with h5py.File(file_name, 'r') as h5file:
        data = np.array(h5file['/data'][:])
        names = np.array(h5file['/names'][:]).astype(str)

    mesh_data_list = []
    for name, d in zip(names, data):
        mesh_data_list.append(MeshData(mesh=mesh, name=name, data=d))

    return mesh_data_list



def save_harmonics(
        file_name: str,
        harmonics: Harmonics
):
    """Store the given harmonics data in an HDF5 file.

    :param file_name: The name of the file to store the harmonics data in (must end with .h5).
    :param harmonics: The Harmonics object containing the data to store.
    :raises ValueError: If the file name does not end with .h5.
    """
    save_mesh_data(file_name, list(harmonics))
    with h5py.File(file_name, 'a') as h5file:
        h5file.create_dataset('/eigenvalues', data=harmonics.eigenvalues)


def load_harmonics(
        file_name: str,
        mesh: pv.UnstructuredGrid
) -> Harmonics:
    """Load harmonics from the given HDF5 file.

    :param file_name: The name of the file to store the data in (must end with .h5).
    :param mesh: The mesh these :class:`MeshData` objects belong to.
    """
    # Load the harmonics as raw mesh data and coax them into the correct object
    mesh_data = load_mesh_data(file_name, mesh)

    with h5py.File(file_name, 'r') as h5file:
        eigenvalues = np.array(h5file['/eigenvalues'][:])
    harmonics = np.vstack([d.data for d in mesh_data])

    return Harmonics(mesh, harmonics, eigenvalues)
