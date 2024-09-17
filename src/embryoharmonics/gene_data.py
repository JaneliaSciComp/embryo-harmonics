from dataclasses import dataclass
from typing import Iterable, List

import h5py
import numpy as np
import pyvista as pv
from ngsolve import H1, dx, BilinearForm, LinearForm, GridFunction, TaskManager, grad
from ngsolve import Mesh as NgsMesh


@dataclass
class GeneData:
    """
    Data class representing the expression data for a gene at a single point in time.
    """
    name: str
    locations: np.ndarray
    activities: np.ndarray

    def __len__(self) -> int:
        return len(self.activities)

    def filter_nan_values(self) -> "GeneData":
        non_nan_indices = np.where(np.logical_not(np.isnan(self.activities)))[0]
        return GeneData(self.name, self.locations[non_nan_indices], self.activities[non_nan_indices])


class GeneDataLoader:
    """
    Class for loading gene expression data from an HDF5 file.
    """
    def __init__(self, h5file: h5py.File):
        """
        Initialize the gene data loader.
        :param h5file: The HDF5 file containing the gene expression data
        """
        self.h5file = h5file

        # Load all gene names
        data = h5file["geneact"]
        names = data["genes"][:].T
        self._name_to_index = {names[i].astype(np.uint8).tobytes().decode('ascii').strip(): i for i in range(len(names))}

        # Load all time steps
        time = data["timepoints"][0]
        self._time_to_index = {time[i]: i for i in range(len(time))}

        # Store handles to the gene expression data
        self._gene_activities = data["data"]
        self._locations = data["XYZ"]

    @property
    def gene_names(self) -> List[str]:
        """
        Get the names of all genes in the HDF5 file.
        :return: The names of all genes
        """
        return list(self._name_to_index.keys())

    @property
    def time_steps(self) -> List[int]:
        """
        Get the time steps in the HDF5 file.
        :return: The time steps in the HDF5 file
        """
        return list(self._time_to_index.keys())

    def load(
            self,
            gene_name: str,
            time_step: int
    ) -> GeneData:
        """
        Load the gene expression data for the given gene.
        :param gene_name: The name of the gene to load
        :param time_step: Which time step to load
        :return: A :class:`GeneData` object containing the gene expression data
        """
        try:
            i = self._time_to_index[time_step]
        except KeyError as e:
            raise ValueError(f"Time step {time_step} not found in the HDF5 file") from e

        try:
            gene_index = self._name_to_index[gene_name]
        except KeyError as e:
            raise ValueError(f"Gene {gene_name} not found in the HDF5 file") from e

        return GeneData(gene_name, self._locations[i, :, :], self._gene_activities[i, gene_index, :])


def interpolate_gene_data(
        mesh: NgsMesh,
        gene_data: GeneData | Iterable[GeneData],
        pv_data: pv.UnstructuredGrid,
        *,
        smoothness: float = 1.0
) -> None:
    """
    Interpolate gene expression data onto the mesh by solving a Poisson equation with the gene expression as the sources
    and homogeneous Neumann boundary conditions. Nan values are ignored.
    :param mesh: The mesh to interpolate the gene data onto
    :param gene_data: A :class:`GeneData` object containing the gene expression data
    :param pv_data: A :class:`pyvista.UnstructuredGrid` object where the interpolated data is stored as a scalar field
    :param smoothness: A measure between 0 and infinity of how smooth the interpolated data should be (roughly the
        radius of the smoothing kernel)
    """
    if not isinstance(gene_data, Iterable):
        gene_data = [gene_data]

    # Set up lowest-order finite element problem for the Poisson equation
    n_time_steps = 100
    dt = 1 / n_time_steps
    fes = H1(mesh, order=1)
    u, v = fes.TnT()

    D = smoothness ** 2
    a = BilinearForm(fes)
    a += D * grad(u) * grad(v) * dx
    m = BilinearForm(fes)
    m += u * v * dx

    with TaskManager():
        a.Assemble()
        m.Assemble()
        m.mat.AsVector().data += dt * a.mat.AsVector()
        mstar_inverse = m.mat.Inverse(fes.FreeDofs())

        for data in gene_data:
            # Filter Nan values
            filtered_data = data.filter_nan_values()
            is_in_mesh = np.array([mesh.Contains(*p) for p in filtered_data.locations])
            if not np.all(is_in_mesh):
                print(f"WARNING: {np.sum(~is_in_mesh)} out of {len(is_in_mesh)} locations are outside the mesh and could not be interpolated for gene {filtered_data.name}")

            # Use gene expression data as point sources
            f = LinearForm(fes)
            for i in range(len(filtered_data)):
                if is_in_mesh[i]:
                    f += (filtered_data.activities[i] * v)(*filtered_data.locations[i])

            f.Assemble()
            solution = GridFunction(fes)
            res = dt * f.vec
            for _ in range(n_time_steps):
                solution.vec.data += mstar_inverse * res
                res = -dt * (a.mat * solution.vec)

            # Add the interpolated data to the pyvista data object
            pv_data[filtered_data.name] = solution.vec.FV().NumPy().copy()


