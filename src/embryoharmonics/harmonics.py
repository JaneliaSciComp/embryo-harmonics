import logging
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike
import pyvista as pv
import scipy.sparse.linalg as spla

from embryoharmonics.fem import FemMatrices
from embryoharmonics.mesh_data import MeshData


_logger = logging.getLogger(__name__)


class Harmonics:
    """
    Harmonics and eigenvalues for a given mesh.
    """
    _mesh: pv.UnstructuredGrid
    _harmonics: ArrayLike
    eigenvalues: ArrayLike

    def __init__(
            self,
            mesh: pv.UnstructuredGrid,
            harmonics: ArrayLike,
            eigenvalues: ArrayLike
    ):
        """
        Initialize the harmonics and eigenvalues for a given mesh.

        :param mesh: The mesh to compute the harmonics on
        :param harmonics: The harmonics to compute
        :param eigenvalues: The eigenvalues of the harmonics
        """
        self._mesh = mesh
        self._harmonics = harmonics
        self.eigenvalues = eigenvalues


    def __getitem__(self, item) -> MeshData:
        if not isinstance(item, int):
            raise TypeError(f"Invalid index type {type(item)}; must be int")

        num_zeros = len(str(len(self._harmonics) - 1))
        name = f"harmonic_{item:0{num_zeros}d}"
        return MeshData(self._mesh, name, self._harmonics[item])


    def __len__(self):
        return len(self._harmonics)


    def __repr__(self):
        return f"Harmonics(num_mesh_nodes={self._mesh.n_points}, num_harmonics={len(self)})"


    def __str__(self):
        return f"Harmonics with {len(self)} harmonics and {self._mesh.n_points} mesh nodes"


    @classmethod
    def compute(
            cls,
            mesh: pv.UnstructuredGrid,
            *,
            n: int = 10,
    ) -> "Harmonics":
        """
        Compute the first n harmonics and some key metrics of the Laplace operator
        with Neumann boundary conditions on a given mesh.

        :param mesh: The triangular/tetrahedral mesh to compute the harmonics on
        :param n: The number of harmonics to compute
        :return: The harmonics (as pyvista data structure), and a dictionary
            containing eigenvalues as numpy arrays
        """
        # Set up lowest-order finite element problem for the Laplace operator
        _logger.info("Computing the first %d harmonics on the given mesh", n)
        fem = FemMatrices.compute_for(mesh)
        eigvals, eigvecs = spla.eigsh(A=fem.stiffness, M=fem.mass, k=n, which='LM', sigma=0.0)

        return Harmonics(mesh, eigvecs.T, eigvals)


    def subset(
            self,
            indices: slice | Iterable[int],
    ) -> 'Harmonics':
        """
        Limit the harmonics to the given indices.

        :param indices: The harmonics to limit to
        :return: The subset of harmonics
        """
        return Harmonics(self._mesh, self._harmonics[indices], self.eigenvalues[indices])


    def decompose(
            self,
            mesh_data: MeshData | Iterable[MeshData],
    ) -> dict[str, ArrayLike]:
        """
        Decompose the given data into the harmonics.

        :param mesh_data: The data to decompose
        :return: The coefficients of the data with respect to the harmonics
        """
        if isinstance(mesh_data, MeshData):
            mesh_data = [mesh_data]

        # Pre-compute the mass matrix of the mesh
        mass = FemMatrices.compute_for(self._mesh, stiffness=False).mass

        harmonic_coefficients = {}
        for data in mesh_data:
            _logger.debug('Decomposing data "%s" into %d harmonics', data.name, len(self))
            harmonic_coefficients[data.name] = self._harmonics @ (mass @ data.data)

        return harmonic_coefficients


    def compose(
            self,
            coefficients: ArrayLike | dict[str, ArrayLike],
    ) -> MeshData | list[MeshData]:
        """
        Compose harmonics weighted by the given eigen coefficients (NaNs are ignored).

        :param eigen_coefficients: The weights to use for composing the
            harmonics. If a single coefficient array is given, the returned data
            is given the name "harmonic_composition". Otherwise, the returned
            data is a list of MeshData objects with the same names as the keys
            in the dictionary.
        """
        if not isinstance(coefficients, dict):
            coefficients = {'harmonic_composition': coefficients}

        composed_data = []
        for name, coeffs in coefficients.items():
            _logger.debug('Composing data "%s" from %d harmonics', name, len(self))
            data = self._harmonics.T @ coeffs
            composed_data.append(MeshData(self._mesh, name, data))

        return composed_data if len(composed_data) > 1 else composed_data[0]
