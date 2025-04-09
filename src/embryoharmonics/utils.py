from typing import Iterable

import numpy as np

from embryoharmonics.fem import FemMatrices
from embryoharmonics.mesh_data import MeshData


def correlate(
        data1: MeshData | Iterable[MeshData],
        data2: MeshData | Iterable[MeshData] = None,
        *,
        normalize: bool = True,
) -> np.ndarray:
    """Compute the correlation between (sets of) mesh data objects. The mesh
    data is optionally normalized by subtracting the mean and dividing by the
    standard deviation, both with respect to the mass matrix of the underlying
    mesh. All involved meshes must be the same.

    :param data1: The first set of mesh data objects.
    :param data2: The second set of mesh data objects; if None, auto-correlation
        of data1 is computed.
    :param normalize: If True, the data is normalized by subtracting the mean
        and dividing by the standard deviation, both with respect to the mass
        matrix of the underlying mesh.
    :return: The correlation coefficient between the two mesh data objects,
        where the rows correspond to data1 and the columns to data2.
    :raise ValueError: If any of the mesh data objects are not on the same mesh.
    """
    # Check if mesh data is on the same mesh
    if data2 is None:
        data2 = data1
    data1 = [data1] if isinstance(data1, MeshData) else data1
    data2 = [data2] if isinstance(data2, MeshData) else data2

    mesh = data1[0].mesh
    for data in data1 + data2:
        if data.mesh != mesh:
            raise ValueError("All mesh data must be on the same mesh.")

    # Extract data into numpy arrays for more convenient and efficient processing
    data1 = np.column_stack([data.data for data in data1])
    data2 = np.column_stack([data.data for data in data2])

    # Get mass matrix for the given mesh
    mass = FemMatrices.compute_for(mesh, stiffness=False).mass

    # Normalize data
    if normalize:
        data1 -= np.sum(mass @ data1, axis=0)
        data1 /= np.sqrt(np.vecdot(data1, mass @ data1, axis=0))
        data2 -= np.sum(mass @ data2, axis=0)
        data2 /= np.sqrt(np.vecdot(data2, mass @ data2, axis=0))

    return data1.T @ (mass @ data2)
