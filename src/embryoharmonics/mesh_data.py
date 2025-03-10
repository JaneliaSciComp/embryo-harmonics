import numpy as np
import pyvista as pv


class MeshData:
    """Node-wise data for a triangular/tetrahedral mesh.
    """

    def __init__(
            self,
            mesh: pv.UnstructuredGrid,
            name: str, data: np.ndarray
    ):
        """
        Initialize the mesh data.

        :param mesh: The triangular/tetrahedral mesh this data is associated with
        :param name: The name of the data
        :param data: The node-wise data
        :raise ValueError: If the data length does not match the number of nodes
            in the mesh
        """
        # Check that the data fits the mesh
        if len(data) != mesh.n_points:
            raise ValueError(f"Data length {len(data)} does not match mesh size {mesh.n_points}")

        self._name = name
        self._data = data


    @property
    def name(self) -> str:
        """The name of the data.
        """
        return self._name


    @property
    def data(self) -> np.ndarray:
        """The node-wise data.
        """
        return self._data
