from dataclasses import dataclass
from typing import Iterable, Tuple, Dict

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


def load_gene_data(
        h5file: h5py.File,
        gene_name: str,
        *,
        time_steps: Iterable[int] = None
) -> Tuple[Dict[int, GeneData], np.ndarray]:
    """
    Load the gene expression data from the given file.
    :param h5file: The name of the HDF5 file containing the gene expression data
    :param gene_name: The name of the gene to load
    :param time_steps: Which time steps to load (if None, all time steps are loaded; 1-based)
    :return: A dictionary of time_step to :class:`GeneData` objects containing the gene expression data and the time
    """
    # TODO: find out what the data format is and make more general
    print("WARNING: Due to unspecified data format, this function most likely cannot deal with general data.")

    time = h5file[f"{gene_name}_time"][0]
    if time_steps is None:
        # there is one extra group (measurements)
        time_steps = range(len(time))

    gene_act = h5file[f"{gene_name}_gene_act"]
    pos = h5file[f"{gene_name}_xyz"]

    gene_data = {}
    time_slice = []
    for time_step in time_steps:
        i = time_step - 1
        gene_data[time_step] = GeneData(gene_name, pos[i], gene_act[i])
        time_slice.append(time[i])

    return gene_data, np.array(time_slice)


def interpolate_gene_data(
        mesh: NgsMesh,
        gene_data: GeneData | Iterable[GeneData],
        pv_data: pv.UnstructuredGrid,
        *,
        smoothing_factor: float = 1.0
) -> None:
    """
    Interpolate gene expression data onto the mesh by solving a Poisson equation with the gene expression as the sources
    and homogeneous Neumann boundary conditions. Nan values are ignored.
    :param mesh: The mesh to interpolate the gene data onto
    :param gene_data: A :class:`GeneData` object containing the gene expression data
    :param pv_data: A :class:`pyvista.UnstructuredGrid` object where the interpolated data is stored as a scalar field
    :param smoothing_factor: A measure between 0 and infinity of how much smoothing to apply to the interpolated data
        (the diffusion coefficient)
    """
    if not isinstance(gene_data, Iterable):
        gene_data = [gene_data]

    # Set up lowest-order finite element problem for the Poisson equation
    n_time_steps = 100
    dt = 1 / n_time_steps
    fes = H1(mesh, order=1)
    u, v = fes.TnT()

    a = BilinearForm(fes)
    a += smoothing_factor * grad(u) * grad(v) * dx
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
            for _ in range(n_time_steps):
                res = dt * (f.vec - a.mat * solution.vec)
                solution.vec.data += mstar_inverse * res

            # Add the interpolated data to the pyvista data object
            pv_data[filtered_data.name] = solution.vec.FV().NumPy().copy()


