from typing import Literal, Tuple, Iterable, Dict

import numpy as np
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from ngsolve import H1, grad, dx, BilinearForm, GridFunction, x, y, Integrate
from ngsolve import Mesh as NgsMesh

from embryoharmonics._utils import harmonic_name, all_harmonic_names, retain_harmonics
from embryoharmonics.gene_data import GeneData


def compute_harmonics(
        mesh,
        *,
        k: int = 10,
        boundary_condition: Literal['dirichlet', 'neumann'] = "neumann",
        store_dirichlet_densities: bool = False,
) -> Tuple[pv.UnstructuredGrid, Dict[str, np.ndarray]]:
    """
    Compute the first k harmonics and some key metrics of the Laplace operator on a given mesh.
    :param mesh: The mesh to compute the harmonics on
    :param k: The number of harmonics to compute
    :param boundary_condition: The boundary condition to apply (either 'dirichlet' or 'neumann')
    :param store_dirichlet_densities: Whether to store the densities of the Dirichlet energy in the radial, angular,
        and z direction as functions as harmonic_XXX_dirichlet_[rpz] in the output data
    :return: The harmonics (as pyvista data structure), and a dictionary containing eigenvalues and the dirichlet
    energy in radial, angular, and z direction as numpy arrays
    """

    # Set up lowest-order finite element problem for the Laplace operator
    fes = H1(mesh, order=1, dirichlet="default")
    u, v = fes.TnT()

    a = BilinearForm(fes, symmetric=True)
    a += grad(u) * grad(v) * dx
    a.Assemble()

    m = BilinearForm(fes, symmetric=True)
    m += u * v * dx
    m.Assemble()

    match boundary_condition:
        case 'dirichlet':
            mask = np.array([free for free in fes.FreeDofs()])
        case 'neumann':
            mask = np.ones(fes.ndof, dtype=bool)
        case _:
            raise ValueError(f"Invalid boundary condition '{boundary_condition}'")

    stiffness = _to_scipy_csr(a, mask)
    mass = _to_scipy_csr(m, mask)
    eigvals, eigvecs = spla.eigsh(A=stiffness, M=mass, k=k, which='LM', sigma=0.0)

    dirichlet_energy = CylindricalDirichletEnergy(fes)
    rpz = np.zeros((k, 3))
    for i in range(k):
        rpz[i, 0], rpz[i, 1], rpz[i, 2] = dirichlet_energy.compute_integral(eigvecs[:, i])
    metrics = dict(eigenvalues=eigvals, dirichlet_r=rpz[:, 0], dirichlet_p=rpz[:, 1], dirichlet_z=rpz[:, 2])

    full_eigvecs = np.zeros((fes.ndof, k))
    full_eigvecs[mask, :] = eigvecs

    pv_data = _to_vtk(mesh)
    for i in range(k):
        name = harmonic_name(i)
        pv_data[name] = full_eigvecs[:, i]
        if store_dirichlet_densities:
            r, p, z = dirichlet_energy.compute_density(eigvecs[:, i])
            pv_data[f"{name}_dirichlet_r"] = r
            pv_data[f"{name}_dirichlet_p"] = p
            pv_data[f"{name}_dirichlet_z"] = z

    return pv_data, metrics


class CylindricalDirichletEnergy:
    def __init__(self, fes):
        self.fes = fes
        self.u = GridFunction(self.fes)
        self.v = GridFunction(self.fes)

    def compute_integral(self, eigvec):
        dr, dp, dz = self._get_components(eigvec)
        return Integrate(dr, self.fes.mesh), Integrate(dp, self.fes.mesh), Integrate(dz, self.fes.mesh)


    def compute_density(self, eigvec):
        dr, dp, dz = self._get_components(eigvec)
        self.v.Set(dr)
        r = self.v.vec.FV().NumPy().copy()
        self.v.Set(dp)
        p = self.v.vec.FV().NumPy().copy()
        self.v.Set(dz)
        z = self.v.vec.FV().NumPy().copy()

        return r, p, z

    def _get_components(self, eigvec):
        r_squared = x ** 2 + y ** 2

        self.u.vec.FV().NumPy()[:] = eigvec[:]
        du = grad(self.u)

        dr = (x * du[0] + y * du[1]) ** 2 / r_squared
        dp = (-y * du[0] + x * du[1]) ** 2 / r_squared
        dz = du[2] ** 2

        return dr, dp, dz


def compute_eigen_coefficients(
        pv_data: pv.UnstructuredGrid,
        gene_data: GeneData | Iterable[GeneData],
) -> Dict[str, np.ndarray]:
    """
    Compute the coefficients of the given fields with respect to the harmonics.
    :param pv_data: The mesh data to compute the coefficients for
    :param gene_data: The gene data to compute the coefficients for
    :return: The coefficients of the fields with respect to the harmonics
    """
    if not isinstance(gene_data, Iterable):
        gene_data = [gene_data]

    harmonic_names = all_harmonic_names(pv_data)
    coefficients = {}

    # Make a copy of the mesh data that stores only the harmonics (to avoid interpolating other fields)
    only_harmonics = pv_data.copy(deep=True)
    retain_harmonics(only_harmonics)

    for data in gene_data:
        # Filter Nan values and interpolate grid data onto the gene data locations
        filtered_data = data.filter_nan_values()
        points = pv.PolyData(filtered_data.locations)
        interpolated_data = points.sample(only_harmonics)

        # Filter data that could not be interpolated (i.e., outside the mesh)
        is_in_mesh = interpolated_data["vtkValidPointMask"].astype(bool)
        activities = filtered_data.activities[is_in_mesh]
        point_evaluations = [interpolated_data[name][is_in_mesh] for name in harmonic_names]

        if not np.all(is_in_mesh):
            print(f"WARNING: {np.sum(~is_in_mesh)} out of {len(is_in_mesh)} locations are outside the mesh and could not be interpolated for gene {filtered_data.name}")

        coefficients[filtered_data.name] = np.array([np.dot(p, activities) for p in point_evaluations])

    return coefficients


def _to_scipy_csr(blf, mask):
    row, col, val = blf.mat.COO()
    sparse = sp.csr_matrix((val, (row, col)))
    return sparse[mask][:, mask]


def _to_vtk(mesh: NgsMesh,) -> pv.UnstructuredGrid:
    """
    Convert an NGSolve mesh to a pyvista.UnstructuredGrid object.
    :param mesh: The NGSolve mesh to convert
    :return: A :class:`pyvista.UnstructuredGrid` object containing the mesh
    """
    points = mesh.ngmesh.Coordinates()
    cells = []
    cell_types = []
    for el in mesh.ngmesh.Elements3D():
        # NGSolve uses 1-based indexing for vertices
        cells.append([4] + [el.vertices[i].nr - 1 for i in range(4)])
        cell_types.append(pv.CellType.TETRA)

    return pv.UnstructuredGrid(cells, cell_types, points)


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
    indices = [i for i in range(len(eigen_coefficients)) if not np.isnan(eigen_coefficients[i])]
    result = sum(eigen_coefficients[i] * pv_data[harmonic_name(i)] for i in indices)
    pv_data[name] = result
