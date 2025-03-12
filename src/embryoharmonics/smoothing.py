import abc

import scipy.sparse.linalg as spla

from embryoharmonics.fem import compute_fem_matrices
from embryoharmonics.harmonics import Harmonics
from embryoharmonics.mesh_data import MeshData


class Smoother(abc.ABC):
    """Base class for smoothers on mesh data.
    """

    @abc.abstractmethod
    def smooth(self, data: MeshData) -> MeshData:
        """Smooth the given mesh data.

        :param data: The mesh data to smooth
        :return: The smoothed mesh data
        """


class HarmonicSmoother(Smoother):
    """Smoothing of mesh data using a truncated harmonic series.
    """
    def __init__(self, harmonics: Harmonics):
        """
        Initialize the harmonic smoother.

        :param harmonics: The harmonics to use for smoothing.
        """
        self._harmonics = harmonics


    def smooth(self, data: MeshData) -> MeshData:
        """
        Smooth the given mesh data.

        :param data: The mesh data to smooth
        :return: The smoothed mesh data
        """
        coefficients = self._harmonics.decompose(data)
        name = data.name
        coefficients[f"{name}_smoothed"] = coefficients[name]
        del coefficients[name]
        return self._harmonics.compose(coefficients)


class DiffusionSmoother(Smoother):
    """Smoothing of mesh data using a diffusion process.
    """
    def __init__(
            self,
            mesh,
            smoothness: float = 1,
            n_steps: int = 100
    ):
        """
        Initialize the diffusion smoother. Diffusion smooths the data by simulating
        heat flow on the mesh, where the given data is the initial heat distribution.
        This smoother is mass-preserving.

        :param mesh: The mesh to smooth data on.
        :param smoothness: A measure between 0 and infinity of how smooth the
            interpolated data should be (roughly the radius of the smoothing kernel).
        :param n_steps: The number of time steps to simulate the diffusion process.
        """
        self._mesh = mesh
        self.smoothness = smoothness
        self.n_steps = n_steps

        # Set up system matrices for implicit midpoint rule:
        # (M + tau/2 A) d_n = tau A x_n, x_{n+1} = x_n - d_n
        coeff = smoothness ** 2 / n_steps
        fem_matrices = compute_fem_matrices(mesh)
        self.a = coeff * fem_matrices.stiffness
        m_star = fem_matrices.mass + (coeff / 2) * fem_matrices.stiffness
        self.m_lu = spla.splu(m_star.tocsc())


    def smooth(self, data: MeshData) -> MeshData:
        """
        Smooth the given mesh data.

        :param data: The mesh data to smooth
        :return: The smoothed mesh data
        """
        # Simulate the diffusion process for the given number of time steps
        solution = data.data
        for _ in range(self.n_steps):
            delta = self.m_lu.solve(self.a @ solution)
            solution -= delta

        name = f"{data.name}_smoothed"
        return MeshData(self._mesh, name, solution)
