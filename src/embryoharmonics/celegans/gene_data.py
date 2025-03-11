import logging
from dataclasses import dataclass
from typing import Iterable, List, Dict

import h5py
import numpy as np
import pyvista as pv
import ngsolve as ngs

from embryoharmonics._utils import all_harmonic_names


_logger = logging.getLogger(__name__)


@dataclass
class GeneData:
    """Expression data for a gene at a single point in time.
    """
    name: str
    locations: np.ndarray
    activities: np.ndarray

    def __len__(self) -> int:
        return len(self.activities)

    def filter_nan_values(self) -> "GeneData":
        """Remove NaN values from the gene expression data.
        
        :return: A new :class:`GeneData` object with NaN values removed.
        """
        not_nan = np.where(np.logical_not(np.isnan(self.activities)))[0]
        return GeneData(self.name, self.locations[not_nan], self.activities[not_nan])


class GeneDataLoader:
    """Loader for gene expression data from an HDF5 file.
    """
    def __init__(self, h5file: h5py.File):
        """Initialize the gene data loader.

        :param h5file: The HDF5 file containing the gene expression data
        """
        self.h5file = h5file

        # Load all gene names
        data = h5file["geneact"]
        self._gene_to_index = _convert_raw_names(data["genes"][:])
        self._tissue_to_index = _convert_raw_names(data["tissue_name"][:])

        # Load all time steps
        time = data["timepoints"][0]
        self._time_to_index = {time[i]: i for i in range(len(time))}

        # Store handles to the gene expression data
        self._gene_activities = data["data"]
        self._tissues = data["tissue_id"]
        self._locations = data["XYZ"]

    @property
    def gene_names(self) -> List[str]:
        """Get the names of all genes in the HDF5 file.

        :return: The names of all genes
        """
        return list(self._gene_to_index.keys())

    @property
    def tissue_names(self) -> List[str]:
        """Get the names of all tissues in the HDF5 file.

        :return: The names of all tissues
        """
        return list(self._tissue_to_index.keys())

    @property
    def time_steps(self) -> List[int]:
        """Get the time steps in the HDF5 file.

        :return: The time steps in the HDF5 file
        """
        return list(self._time_to_index.keys())

    def load(
            self,
            gene_name: str,
            time_step: int
    ) -> GeneData:
        """Load the gene expression data for the given gene.

        :param gene_name: The name of the gene to load
        :param time_step: Which time step to load
        :return: A :class:`GeneData` object containing the gene expression data
        """
        _logger.info("Loading gene %s at time step %d from '%s'",
                     gene_name, time_step, self.h5file.filename)
        try:
            t = self._time_to_index[time_step]
        except KeyError as e:
            raise ValueError(f"Time step {time_step} not found in the HDF5 file") from e

        try:
            g = self._gene_to_index[gene_name]
        except KeyError as e:
            raise ValueError(f"Gene {gene_name} not found in the HDF5 file") from e

        return GeneData(gene_name, self._locations[t, :, :], self._gene_activities[t, g, :])

    def load_tissue(
            self,
            tissue_name: str,
            time_step: int
    ) -> GeneData:
        """Load the data for the given tissue (where the activity is just 1 for
        cells in the tissue and 0 otherwise).

        :param tissue_name: The name of the tissue to load
        :param time_step: Which time step to load
        :return: A :class:`GeneData` object containing the tissue data
        """
        try:
            i = self._time_to_index[time_step]
        except KeyError as e:
            raise ValueError(f"Time step {time_step} not found in the HDF5 file") from e

        try:
            tissue_index = self._tissue_to_index[tissue_name]
        except KeyError as e:
            raise ValueError(f"Tissue {tissue_name} not found in the HDF5 file") from e

        _logger.info("Loading tissue %s at time step %d from '%s'",
                     tissue_name, time_step, self.h5file.filename)
        return GeneData(tissue_name, self._locations[i, :, :], self._tissues[tissue_index, :])


def _convert_raw_names(raw_names):
    return {raw_names[:, i].astype(np.uint8).tobytes().decode('ascii').strip(): i
            for i in range(raw_names.shape[1])}


def interpolate_gene_data(
        mesh: ngs.Mesh,
        gene_data: GeneData | Iterable[GeneData],
        pv_data: pv.UnstructuredGrid,
        *,
        smoothness: float = 1.0,
        compute_eigen_coefficients: bool = False
) -> None | Dict[str, np.ndarray]:
    """Interpolate gene expression data onto the mesh by solving a Poisson equation
    with the gene expression as the sources and homogeneous Neumann boundary
    conditions. Nan values are ignored.

    :param mesh: The mesh to interpolate the gene data onto.
    :param gene_data: A :class:`GeneData` object containing the gene expression
        data.
    :param pv_data: A :class:`pyvista.UnstructuredGrid` object where the
        interpolated data is stored as a scalar field
    :param smoothness: A measure between 0 and infinity of how smooth the
        interpolated data should be (roughly the radius of the smoothing kernel)
    :param compute_eigen_coefficients: Whether to compute the coefficients of
        the interpolated data with respect to the harmonics; if True, the
        coefficients are returned as a dictionary
    """
    if not isinstance(gene_data, Iterable):
        gene_data = [gene_data]
    _logger.info("Interpolating gene data for %s onto the mesh",
                 ", ".join(data.name for data in gene_data))

    # Set up lowest-order finite element problem for the Poisson equation
    n_time_steps = 100
    dt = 1 / n_time_steps
    fes = ngs.H1(mesh, order=1)
    u, v = fes.TnT()

    diffusivity = smoothness ** 2
    a = ngs.BilinearForm(fes)
    a += diffusivity * ngs.grad(u) * ngs.grad(v) * ngs.dx
    m = ngs.BilinearForm(fes)
    m += u * v * ngs.dx

    harmonic_names = all_harmonic_names(pv_data)
    eigen_coefficients = {}

    # TODO: use scipy.sparse.linalg.expm_multiply instead?
    # TODO: use numpy factorizations (e.g., scipy.sparse.linalg.splu)?
    with ngs.TaskManager():
        a.Assemble()
        m.Assemble()
        m_inverse = m.mat.Inverse(fes.FreeDofs())
        mstar = m.mat.CreateMatrix()
        mstar.AsVector().data = m.mat.AsVector() + dt * a.mat.AsVector()
        mstar_inverse = mstar.Inverse(fes.FreeDofs())

        for data in gene_data:
            _logger.info("Interpolating gene %s", data.name)

            # Filter Nan values
            filtered_data = data.filter_nan_values()
            is_in_mesh = np.array([mesh.Contains(*p) for p in filtered_data.locations])
            if not np.all(is_in_mesh):
                _logger.warning("%d out of %d locations are outside the mesh"
                                "and could not be interpolated for gene %s",
                                np.sum(~is_in_mesh), len(is_in_mesh), filtered_data.name)

            # Use gene expression data as point sources
            f = ngs.LinearForm(fes)
            for i in range(len(filtered_data)):
                if is_in_mesh[i]:
                    f += (filtered_data.activities[i] * v)(*filtered_data.locations[i])

            # First, L2-interpolate the gene expression data onto the mesh, then
            # smooth it via the heat equation
            f.Assemble()
            solution = ngs.GridFunction(fes)
            solution.vec.data = m_inverse * f.vec
            for _ in range(n_time_steps):
                res = -dt * (a.mat * solution.vec)
                solution.vec.data += mstar_inverse * res

            # Add the interpolated data to the pyvista data object
            pv_data[filtered_data.name] = solution.vec.FV().NumPy().copy()

            if compute_eigen_coefficients:
                m_times_solution = (m.mat * solution.vec).Evaluate().FV().NumPy()
                coeff = np.array([np.dot(m_times_solution, pv_data[h]) for h in harmonic_names])
                eigen_coefficients[filtered_data.name] = coeff

    if compute_eigen_coefficients:
        return eigen_coefficients
