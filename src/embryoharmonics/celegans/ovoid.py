"""Ovoid geometry of a pre-twitch C. elegans embryo.

The outline is given as a Fourier series of the radius around its centroid,
r(theta) = sum_n a_n cos(n * theta), with theta = 0 along the major axis. Only
the cosine terms are kept, which makes the outline mirror symmetric about the
major axis, so revolving it about that axis gives a body of revolution. The
major axis is the z-axis (anterior-posterior in the gene data), theta is the
polar angle measured from the positive z-axis, and the geometry is in voxels
like the post-twitch embryo models.
"""
import csv
import logging

import h5py
import numpy as np
import pyvista as pv
from netgen import occ
from scipy.optimize import minimize

from embryoharmonics.celegans.embryo_model import _to_vtk_2d
from embryoharmonics.celegans.gene_data import VOXEL_SIZE_UM


# The outlines were traced on 1719 x 1146 pixel images of 67.25 x 44.83 microns
DEFAULT_PIXEL_SIZE_UM = 0.0391

_logger = logging.getLogger(__name__)


class OvoidModel:
    """Body of revolution with a Fourier-series outline r(theta), centered at
    the origin.
    """
    def __init__(self, coefficients):
        """Initialize the ovoid model.

        :param coefficients: The cosine coefficients a_0, ..., a_N of the
            outline r(theta) in voxels
        """
        self.coefficients = np.asarray(coefficients, dtype=float)


    @classmethod
    def from_csv(
            cls,
            path: str,
            *,
            pixel_size: float = DEFAULT_PIXEL_SIZE_UM,
            radial_extension: float = 0.0,
            flip: bool = False,
            column: str = "average_value",
    ) -> "OvoidModel":
        """Load the outline from a csv file of Fourier coefficients (in pixels)
        with a ``coeff`` column naming them (a0, a1, b1, ...). The sine
        coefficients b_n are dropped.

        :param path: The path to the csv file
        :param pixel_size: The pixel size of the coefficients in microns
        :param radial_extension: Scale the outline by 1 + this fraction
        :param flip: Whether to point theta = 0 to the negative z-axis instead
        :param column: The column containing the coefficient values
        :return: The ovoid model in voxels
        """
        with open(path) as file:
            values = {row["coeff"]: float(row[column]) for row in csv.DictReader(file)}
        n_max = max(int(name[1:]) for name in values if name.startswith("a"))
        coefficients = np.array([values[f"a{n}"] for n in range(n_max + 1)])

        coefficients *= pixel_size / VOXEL_SIZE_UM
        coefficients *= 1 + radial_extension
        if flip:
            # r(pi - theta): odd terms change sign
            coefficients *= (-1.0) ** np.arange(n_max + 1)
        return cls(coefficients)


    def radius(self, theta, derivative: bool = False) -> np.ndarray:
        """Evaluate the outline r(theta) (or its derivative r'(theta)).

        :param theta: The polar angles to evaluate at
        :param derivative: Whether to evaluate the derivative instead
        :return: The radii (or derivatives) in voxels
        """
        theta = np.asarray(theta, dtype=float)
        n = np.arange(len(self.coefficients))
        angles = np.multiply.outer(theta, n)
        if derivative:
            return -(np.sin(angles) * n) @ self.coefficients
        return np.cos(angles) @ self.coefficients


    def outline(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample the meridian outline uniformly in theta, from the positive
        tip (theta = 0) to the negative tip (theta = pi). Uniform sampling in
        theta clusters the points near the tips, where the outline is steep as
        a function of z.

        :param n_samples: The number of points to sample
        :return: The radial and axial coordinates (R, z) of the outline
        """
        theta = np.linspace(0, np.pi, n_samples)
        r = self.radius(theta)
        R = r * np.sin(theta)
        R[[0, -1]] = 0.0  # exactly on the axis
        return R, r * np.cos(theta)


    def generate_meridian_mesh(
            self,
            mesh_size: float = 5.0,
            n_interpolation: int = 64,
    ) -> pv.UnstructuredGrid:
        """Generate a 2D triangle mesh of the meridian half-plane domain
        {(r, z): r >= 0, sqrt(r^2 + z^2) <= r(theta)} with points (r, 0, z).

        :param mesh_size: The maximum mesh size
        :param n_interpolation: The number of outline points to interpolate
        :return: A triangle mesh of the meridian domain in the y = 0 plane
        """
        R, z = self.outline(n_interpolation)
        profile = occ.SplineApproximation([occ.Pnt(r, 0, z) for r, z in zip(R, z)])
        axis = occ.Segment(occ.Pnt(0, 0, z[-1]), occ.Pnt(0, 0, z[0]))
        face = occ.Face(occ.Wire([profile, axis]))
        surface_mesh = occ.OCCGeometry(face).GenerateMesh(maxh=mesh_size)
        return _to_vtk_2d(surface_mesh)


    def containment(self, points, center=(0.0, 0.0, 0.0)) -> np.ndarray:
        """Compute how far out each point lies relative to the outline: the
        distance to the center divided by the outline radius in the point's
        direction. Points inside the ovoid have values below 1.

        :param points: The 3D points (n_points, 3) in voxels
        :param center: The position of the ovoid's center
        :return: The relative radius of each point
        """
        d = np.asarray(points) - np.asarray(center)
        rho = np.hypot(d[:, 0], d[:, 1])
        theta = np.arctan2(rho, d[:, 2])
        return np.hypot(rho, d[:, 2]) / self.radius(theta)


    def fit_center(self, points) -> np.ndarray:
        """Find the translation that fits the points into the ovoid best,
        i.e. that minimizes the largest :meth:`containment` value. The
        orientation is fixed (major axis along z).

        :param points: The 3D points (n_points, 3) in voxels
        :return: The best position of the ovoid's center
        """
        points = np.asarray(points)
        result = minimize(
            lambda center: self.containment(points, center).max(),
            points.mean(axis=0),
            method="Nelder-Mead",
            options={"xatol": 1e-3, "fatol": 1e-8, "maxiter": 5000},
        )
        _logger.info(
            "Fitted ovoid center %s with maximal relative radius %.4f",
            result.x, result.fun
        )
        return result.x


def save_ovoid(file_name: str, ovoid: OvoidModel):
    """Store the outline coefficients of an ovoid model in an HDF5 file.

    :param file_name: The name of the HDF5 file
    :param ovoid: The ovoid model to store
    """
    with h5py.File(file_name, "a") as h5file:
        if "ovoid" in h5file:
            del h5file["ovoid"]
        h5file.create_dataset("ovoid/coefficients", data=ovoid.coefficients)


def load_ovoid(file_name: str) -> OvoidModel:
    """Load an ovoid model stored with :func:`save_ovoid`.

    :param file_name: The name of the HDF5 file
    :return: The ovoid model
    """
    with h5py.File(file_name, "r") as h5file:
        return OvoidModel(h5file["ovoid/coefficients"][:])
