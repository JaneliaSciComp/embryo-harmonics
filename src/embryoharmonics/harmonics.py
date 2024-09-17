from typing import Literal, Tuple, Iterable, Dict, List

import numpy as np
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from ngsolve import H1, grad, dx, BilinearForm, GridFunction, x, y, Integrate
from ngsolve import Mesh as NgsMesh

from embryoharmonics.gene_data import GeneData


def compute_harmonics(
        mesh,
        *,
        k: int = 10,
        boundary_condition: Literal['dirichlet', 'neumann'] = "neumann"
) -> Tuple[pv.UnstructuredGrid, Dict[str, np.ndarray]]:
    """
    Compute the first k harmonics and some key metrics of the Laplace operator on a given mesh.
    :param mesh: The mesh to compute the eigenfunctions on
    :param k: The number of eigenfunctions to compute
    :param boundary_condition: The boundary condition to apply (either 'dirichlet' or 'neumann')
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

    r, phi, z = _compute_cylindrical_dirichlet_energy(fes, eigvecs)
    metrics = dict(eigenvalues=eigvals, dirichlet_r=r, dirichlet_phi=phi, dirichlet_z=z)

    full_eigvecs = np.zeros((fes.ndof, k))
    full_eigvecs[mask, :] = eigvecs

    pv_data = _to_vtk(mesh)
    for i in range(k):
        pv_data[_harmonic_name(i)] = full_eigvecs[:, i]

    return pv_data, metrics


def _compute_cylindrical_dirichlet_energy(fes, eigvecs):
    u = GridFunction(fes)
    n = eigvecs.shape[1]
    dr = np.zeros((n,), dtype=np.float64)
    dphi = np.zeros((n,), dtype=np.float64)
    dz = np.zeros((n,), dtype=np.float64)
    r_squared = x ** 2 + y ** 2

    for i in range(n):
        u.vec.FV().NumPy()[:] = eigvecs[:, i]
        du = grad(u)

        dr[i] = Integrate((x * du[0] + y * du[1]) ** 2 / r_squared, fes.mesh)
        dphi[i] = Integrate((-y * du[0] + x * du[1]) ** 2 / r_squared, fes.mesh)
        dz[i] = Integrate(du[2] ** 2, fes.mesh)

    return dr, dphi, dz


def compute_eigen_coefficients(
        pv_data: pv.UnstructuredGrid,
        gene_data: GeneData | Iterable[GeneData],
) -> Dict[str, np.ndarray]:
    """
    Compute the coefficients of the given fields with respect to the eigenfunctions.
    :param pv_data: The mesh data to compute the coefficients for
    :param gene_data: The gene data to compute the coefficients for
    :return: The coefficients of the fields with respect to the eigenfunctions
    """
    if not isinstance(gene_data, Iterable):
        gene_data = [gene_data]

    harmonic_names = _all_harmonic_names(pv_data)
    coefficients = {}

    # Make a copy of the mesh data that stores only the harmonics (to avoid interpolating other fields)
    only_harmonics = pv_data.copy(deep=True)
    _retain_harmonics(only_harmonics)

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


def _harmonic_name(i: int) -> str:
    return f"harmonic_{i:03d}"


def _all_harmonic_names(data: pv.UnstructuredGrid) -> List[str]:
    return [name for name in data.array_names if name.startswith("harmonic_")]


def _retain_harmonics(data: pv.DataSet) -> pv.DataSet:
    for name in data.array_names:
        if not name.startswith("harmonic_"):
            data.point_data.remove(name)
    return data
