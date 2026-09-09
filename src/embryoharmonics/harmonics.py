import logging
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike
import pyvista as pv
import scipy.sparse.linalg as spla

from embryoharmonics.fem import FemMatrices
from embryoharmonics.interpolation import interpolation_matrix
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


    @property
    def mesh(self) -> pv.UnstructuredGrid:
        """The mesh the harmonics are defined on."""
        return self._mesh


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
        # The Neumann stiffness is singular (constant mode), which can make the
        # shift-invert factorization at sigma = 0 fail; use a small negative
        # shift (relative to the mean eigenvalue scale) so stiffness - sigma *
        # mass is definite. Any sigma below the lowest eigenvalue leaves the
        # computed modes unchanged.
        sigma = -1e-6 * fem.stiffness.diagonal().sum() / fem.mass.diagonal().sum()
        eigvals, eigvecs = spla.eigsh(A=fem.stiffness, M=fem.mass, k=n, which='LM', sigma=sigma)

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


    def sample(self, locations: ArrayLike) -> np.ndarray:
        """Evaluate the harmonics at the given points.

        :param locations: The points to evaluate at (n_points, 3)
        :return: The values (n_harmonics, n_points); zero outside the mesh
        """
        return self._harmonics @ interpolation_matrix(self._mesh, np.asarray(locations))


    def decompose_point_data(
            self,
            locations: ArrayLike,
            values: ArrayLike,
    ) -> np.ndarray:
        """Decompose point data into the harmonics without interpolating it
        onto the mesh first.

        This equals the decomposition of the L2 projection of each row of the
        values onto the mesh, but the mass matrix of the projection cancels
        against the one of the decomposition: the coefficients are just the
        harmonics sampled at the points, weighted by the values. NaN values
        and points outside the mesh are ignored.

        :param locations: The points the data is given at (n_points, 3)
        :param values: The point data to decompose (n_data, n_points)
        :return: The coefficients (n_data, n_harmonics)
        """
        _logger.debug('Decomposing point data of shape %s into %d harmonics',
                      np.shape(values), len(self))
        return np.nan_to_num(values) @ self.sample(locations).T


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
