import os
import xml.etree.ElementTree as ET
from typing import Iterable

import h5py
import numpy as np
import pyvista as pv

from embryoharmonics.harmonics import Harmonics


# VTK cell types with the corresponding XDMF topology names
_CELL_TYPES = {
    3: (5, 'Triangle'),     # nodes per cell -> (VTK_TRIANGLE, XDMF name)
    4: (10, 'Tetrahedron'),  # nodes per cell -> (VTK_TETRA, XDMF name)
}


def _time_key(time: int) -> str:
    return f"{time:03d}"


def save_time_point(
        file_name: str,
        time: int,
        mesh: pv.UnstructuredGrid,
        harmonics: Harmonics
):
    """Store mesh and harmonics for one time point in an HDF5 file.

    The data is appended to the groups /meshes/<time> and /harmonics/<time>;
    existing data for the same time point is overwritten. A sibling XDMF file
    (same name, .xdmf extension) referencing the HDF5 data is regenerated after
    every call so the time series can be opened in ParaView.

    :param file_name: The name of the HDF5 file (must end with .h5).
    :param time: The time point to store the data under.
    :param mesh: The triangular/tetrahedral mesh at this time point.
    :param harmonics: The harmonics computed on the mesh.
    :raises ValueError: If the file name does not end with .h5 or the mesh is
        not a uniform triangular/tetrahedral mesh.
    """
    if not file_name.endswith('.h5'):
        raise ValueError('File name must end with ".h5"')

    cells_dict = mesh.cells_dict
    if len(cells_dict) != 1 or next(iter(cells_dict)) not in (5, 10):
        raise ValueError('Mesh must be uniformly triangular or tetrahedral')
    cells = next(iter(cells_dict.values()))

    key = _time_key(time)
    with h5py.File(file_name, 'a') as h5file:
        for group in (f'/meshes/{key}', f'/harmonics/{key}'):
            if group in h5file:
                del h5file[group]

        h5file.create_dataset(f'/meshes/{key}/points', data=np.asarray(mesh.points, dtype=np.float64))
        h5file.create_dataset(f'/meshes/{key}/cells', data=cells.astype(np.int64))

        h5file.create_dataset(f'/harmonics/{key}/eigenvalues', data=harmonics.eigenvalues)
        for harmonic in harmonics:
            h5file.create_dataset(f'/harmonics/{key}/{harmonic.name}', data=harmonic.data)

    _write_xdmf(file_name)


def load_mesh(
        file_name: str,
        time: int
) -> pv.UnstructuredGrid:
    """Load only the mesh for one time point from an HDF5 file.

    :param file_name: The name of the HDF5 file written by :func:`save_time_point`.
    :param time: The time point to load.
    :return: The mesh at the given time point.
    """
    key = _time_key(time)
    with h5py.File(file_name, 'r') as h5file:
        points = h5file[f'/meshes/{key}/points'][:]
        cells = h5file[f'/meshes/{key}/cells'][:]

    cell_type, _ = _CELL_TYPES[cells.shape[1]]
    return pv.UnstructuredGrid({cell_type: cells}, points)


def load_harmonics(
        file_name: str,
        time: int,
        mesh: pv.UnstructuredGrid | None = None
) -> Harmonics:
    """Load the harmonics for one time point from an HDF5 file.

    :param file_name: The name of the HDF5 file written by :func:`save_time_point`.
    :param time: The time point to load.
    :param mesh: The mesh the harmonics belong to. If None, it is loaded from
        the same file (see :func:`load_mesh`).
    :return: The harmonics at the given time point.
    """
    if mesh is None:
        mesh = load_mesh(file_name, time)

    key = _time_key(time)
    with h5py.File(file_name, 'r') as h5file:
        harmonics_group = h5file[f'/harmonics/{key}']
        eigenvalues = harmonics_group['eigenvalues'][:]
        harmonic_names = sorted(name for name in harmonics_group if name.startswith('harmonic_'))
        harmonics = np.vstack([harmonics_group[name][:] for name in harmonic_names])

    return Harmonics(mesh, harmonics, eigenvalues)


def time_points(file_name: str) -> list[int]:
    """List all time points stored in the given HDF5 file.

    :param file_name: The name of the HDF5 file written by :func:`save_time_point`.
    :return: The sorted time points.
    """
    with h5py.File(file_name, 'r') as h5file:
        return sorted(int(key) for key in h5file['/meshes'])


def _write_xdmf(file_name: str):
    """Regenerate the sibling XDMF file describing all time points in the
    given HDF5 file. Only dataset shapes are read, no data."""
    h5_name = os.path.basename(file_name)

    xdmf = ET.Element('Xdmf', Version='3.0')
    domain = ET.SubElement(xdmf, 'Domain')
    collection = ET.SubElement(
        domain, 'Grid',
        Name='TimeSeries', GridType='Collection', CollectionType='Temporal'
    )

    def data_item(parent, dataset, number_type):
        dimensions = ' '.join(str(d) for d in dataset.shape)
        item = ET.SubElement(
            parent, 'DataItem',
            Dimensions=dimensions, NumberType=number_type, Precision='8', Format='HDF'
        )
        item.text = f'{h5_name}:{dataset.name}'

    with h5py.File(file_name, 'r') as h5file:
        for key in sorted(h5file['/meshes']):
            points = h5file[f'/meshes/{key}/points']
            cells = h5file[f'/meshes/{key}/cells']
            _, topology_type = _CELL_TYPES[cells.shape[1]]

            grid = ET.SubElement(collection, 'Grid', Name=key, GridType='Uniform')
            ET.SubElement(grid, 'Time', Value=str(int(key)))

            topology = ET.SubElement(
                grid, 'Topology',
                TopologyType=topology_type, NumberOfElements=str(cells.shape[0])
            )
            data_item(topology, cells, 'Int')

            geometry = ET.SubElement(grid, 'Geometry', GeometryType='XYZ')
            data_item(geometry, points, 'Float')

            harmonics_group = h5file[f'/harmonics/{key}']
            for name in sorted(harmonics_group):
                if not name.startswith('harmonic_'):
                    continue
                attribute = ET.SubElement(
                    grid, 'Attribute',
                    Name=name, AttributeType='Scalar', Center='Node'
                )
                data_item(attribute, harmonics_group[name], 'Float')

    tree = ET.ElementTree(xdmf)
    ET.indent(tree)
    tree.write(os.path.splitext(file_name)[0] + '.xdmf', xml_declaration=True)


def encode_matlab_strings(strings: Iterable[str]) -> np.ndarray:
    """Encode a list of strings as an array of space padded ascii-chars, as
    they are stored in mat files.

    :param strings: The strings to encode.
    :return: The encoded strings as a numpy array.
    """
    strings = list(strings)
    max_length = max((len(s) for s in strings), default=0)
    encoded = [s.ljust(max_length).encode('ascii') for s in strings]
    result = np.array([np.frombuffer(s, dtype=np.uint8) for s in encoded], dtype=np.uint16)
    return result.reshape(len(strings), max_length).T
