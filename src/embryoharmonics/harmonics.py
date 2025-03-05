import logging
from typing import Iterable

import numpy as np
import pyvista as pv
import scipy.sparse.linalg as spla

from embryoharmonics._utils import harmonic_name, all_harmonic_names, retain_harmonics
from embryoharmonics.fem import compute_fem_matrices
from embryoharmonics.gene_data import GeneData


_logger = logging.getLogger(__name__)


def compute_harmonics(
        mesh: pv.UnstructuredGrid,
        *,
        k: int = 10,
) -> tuple[pv.UnstructuredGrid, dict[str, np.ndarray]]:
    """
    Compute the first k harmonics and some key metrics of the Laplace operator
    with Neumann boundary conditions on a given mesh.
    :param mesh: The triangular/tetrahedral mesh to compute the harmonics on
    :param k: The number of harmonics to compute
    :return: The harmonics (as pyvista data structure), and a dictionary
        containing eigenvalues as numpy arrays
    """
    # Set up lowest-order finite element problem for the Laplace operator
    _logger.info("Computing the first %d harmonics on the given mesh", k)
    fem = compute_fem_matrices(mesh)
    eigvals, eigvecs = spla.eigsh(A=fem.stiffness, M=fem.mass, k=k, which='LM', sigma=0.0)

    # Make sure that eigenvalues have the correct sign
    integrals = np.sum(fem.mass @ eigvecs, axis=0)
    eigvecs[:, integrals < 0] *= -1

    # Store the harmonics in the mesh data structure
    for i in range(k):
        name = harmonic_name(i)
        mesh[name] = eigvecs[:, i]

    return mesh, eigvals


def compute_eigen_coefficients(
        pv_data: pv.DataSet,
        gene_data: GeneData | Iterable[GeneData],
) -> dict[str, np.ndarray]:
    """
    Compute the coefficients of the given fields with respect to the harmonics.
    :param pv_data: The mesh data to compute the coefficients for
    :param gene_data: The gene data to compute the coefficients for
    :return: The coefficients of the fields with respect to the harmonics
    """
    if not isinstance(gene_data, Iterable):
        gene_data = [gene_data]
    _logger.info("Computing the coefficients of %s with respect to the harmonics", [data.name for data in gene_data])

    harmonic_names = all_harmonic_names(pv_data)
    coefficients = {}
    _logger.debug("Harmonics to compute eigen coefficients against: %s", all_harmonic_names(pv_data))

    # Make a copy of the mesh data that stores only the harmonics (to avoid interpolating other fields)
    only_harmonics = pv_data.copy(deep=True)
    retain_harmonics(only_harmonics)

    for data in gene_data:
        # Filter Nan values and interpolate grid data onto the gene data locations
        _logger.debug("Interpolating gene %s", data.name)
        filtered_data = data.filter_nan_values()
        points = pv.PolyData(filtered_data.locations)
        interpolated_data = points.sample(only_harmonics)

        # Filter data that could not be interpolated (i.e., outside the mesh)
        is_in_mesh = interpolated_data["vtkValidPointMask"].astype(bool)
        activities = filtered_data.activities[is_in_mesh]
        point_evaluations = [interpolated_data[name][is_in_mesh] for name in harmonic_names]

        if not np.all(is_in_mesh):
            _logger.warning("%d out of %d locations are outside the mesh and could not be interpolated for gene %s",
                            np.sum(~is_in_mesh), len(is_in_mesh), filtered_data.name)

        coefficients[filtered_data.name] = np.array([np.dot(p, activities) for p in point_evaluations])

    return coefficients


def compose_eigen_coefficients(
        pv_data: pv.UnstructuredGrid,
        eigen_coefficients: np.ndarray,
        name: str
) -> None:
    """
    Compose harmonics weighted by the given eigen coefficients (NaNs are ignored).
    :param pv_data: The mesh data to compose the harmonics on
    :param eigen_coefficients: The weights to use for composing the harmonics
    :param name: The name of the composed field (will be added to pv_data)
    """
    _logger.info("Composing harmonics with given coefficients to store in field %s", name)
    indices = [i for i in range(len(eigen_coefficients)) if not np.isnan(eigen_coefficients[i])]
    result = sum(eigen_coefficients[i] * pv_data[harmonic_name(i)] for i in indices)
    pv_data[name] = result
