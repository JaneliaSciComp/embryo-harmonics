from ctypes.wintypes import VARIANT_BOOL
from typing import Tuple, Dict, Literal, Iterable

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree
import scipy.sparse as sp
from netgen import occ
from netgen.meshing import FaceDescriptor, Element2D
from netgen.meshing import Mesh as NetgenMesh
from ngsolve import Mesh as NgsMesh
from ngsolve import H1, grad, dx, BilinearForm, LinearForm, FESpace, GridFunction

from embryoharmonics.geometry import GeneData


def _convert_to_volume_mesh(
        surface_mesh: NetgenMesh,
        mesh_size: float,
        max_node_distance: float = None
) -> Tuple[NetgenMesh, Dict[int, int]]:
    """
    Convert a surface mesh to a volume mesh by adding a single domain inside the surface and no domain outside. If the
    surface mesh is not closed (i.e., meshing fails), close nodes can be identified by a maximum distance threshold and
    merged to close it.
    :param surface_mesh: The surface mesh to convert
    :param mesh_size: The maximum mesh size
    :param max_node_distance: Maximum distance between nodes to identify close nodes; if None, no nodes are merged
    :return: The volume mesh and a mapping from old node indices to new ones
    """
    n_nodes = len(surface_mesh.Points())
    node_is_unique = np.ones(n_nodes, dtype=bool)
    new_mesh = NetgenMesh()
    old_to_new = {}

    if max_node_distance is not None:
        # Find all nodes that are too close to each other ("doppelgängers")
        kdtree = KDTree(surface_mesh.Coordinates())
        dist = kdtree.sparse_distance_matrix(kdtree, max_distance=max_node_distance, output_type='ndarray')

        # Remove self-distances and compress i->j / j->i pairs
        dist = dist[dist['i'] != dist['j']]
        pairs = np.vstack([dist['i'], dist['j']])
        pairs = np.sort(pairs, axis=0)
        pairs = np.unique(pairs, axis=1)

        # Mark all nodes that have a doppelgänger
        node_is_unique[pairs[0]] = False
        node_is_unique[pairs[1]] = False

        # Copy non-unique nodes
        for i, j in zip(pairs[0], pairs[1]):
            # Identify doppelgängers in new mesh (node indices are 1-based)
            new_node = new_mesh.Add(surface_mesh[i + 1])
            old_to_new[i + 1] = new_node
            old_to_new[j + 1] = new_node

    # Copy unique nodes
    for i in range(n_nodes):
        if node_is_unique[i]:
            # Node indices are 1-based
            old_to_new[i + 1] = new_mesh.Add(surface_mesh[i + 1])

    # Create a face descriptor that is used for all elements
    # (one single surface with one domain inside and no domain outside)
    face_descriptor = new_mesh.Add(FaceDescriptor(surfnr=1, domin=1, domout=0, bc=1))

    # Copy elements
    for e in surface_mesh.Elements2D():
        new_mesh.Add(Element2D(face_descriptor, [old_to_new[v] for v in e.vertices]))

    # Generate volume mesh from surface
    new_mesh.GenerateVolumeMesh(maxh=mesh_size)
    return new_mesh, old_to_new


def mesh_embryo_geometry(
        worm_geometry: occ.Compound,
        mesh_size: float
) -> NetgenMesh:
    """
    Mesh the geometry of an embryo.
    :param worm_geometry: The geometry of the embryo to mesh
    :param mesh_size: The maximum mesh size
    :return: The volume mesh of the embryo geometry
    """
    geo = occ.OCCGeometry(worm_geometry)
    surface_mesh = geo.GenerateMesh(maxh=mesh_size)
    vol_mesh, _ = _convert_to_volume_mesh(surface_mesh, mesh_size, mesh_size / 10)
    return NgsMesh(vol_mesh)


def compute_harmonics(
        mesh,
        *,
        k: int = 10,
        boundary_condition: Literal['dirichlet', 'neumann'] = "neumann"
) -> Tuple[pv.UnstructuredGrid, np.ndarray, sp.csr_matrix]:
    """
    Compute the first n eigenvectors and eigenvalues of the Laplace operator on a given mesh.
    :param mesh: The mesh to compute the eigenfunctions on
    :param k: The number of eigenfunctions to compute
    :param boundary_condition: The boundary condition to apply (either 'dirichlet' or 'neumann')
    :return: The eigenvectors (as pyvista data structure), eigenvalues, and mass matrix
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

    f = LinearForm(fes)
    f += v * dx
    f.Assemble()

    match boundary_condition:
        case 'dirichlet':
            mask = np.array([free for free in fes.FreeDofs()])
        case 'neumann':
            mask = np.ones(fes.ndof, dtype=bool)
        case _:
            raise ValueError(f"Invalid boundary condition '{boundary_condition}'")

    stiffness = _to_scipy_csr(a, mask)
    mass = _to_scipy_csr(m, mask)
    eigvals, eigvecs = sp.linalg.eigsh(A=stiffness, M=mass, k=k, which='LM', sigma=0.0)

    full_eigvecs = np.zeros((fes.ndof, k))
    full_eigvecs[mask, :] = eigvecs

    pv_data = _to_vtk(mesh)
    for i in range(k):
        pv_data[f"eigenfunction {i}"] = full_eigvecs[:, i]
    pv_data["integration weights"] = f.vec.FV().NumPy()

    return pv_data, eigvals, mass


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
        (the inverse of the diffusion coefficient)
    """
    if not isinstance(gene_data, Iterable):
        gene_data = [gene_data]

    # Set up lowest-order finite element problem for the Poisson equation
    V = H1(mesh, order=1)
    Q = FESpace("number", mesh)
    fes = V * Q

    (u, p), (v, q) = fes.TnT()
    solution = GridFunction(fes)

    a = BilinearForm(fes)
    a += (1 / smoothing_factor) * grad(u) * grad(v) * dx
    a += p * v * dx
    a += q * u * dx
    a.Assemble()
    a_inverse = a.mat.Inverse(fes.FreeDofs())

    for data in gene_data:
        # Filter Nan values
        non_nan_indices = np.where(np.logical_not(np.isnan(data.activity)))[0]

        # Use gene expression data as point sources
        f = LinearForm(fes)
        for i in non_nan_indices:
            f += (data.activity[i] * v)(*data.location[i])

        f.Assemble()
        solution.vec.data = a_inverse * f.vec

        # Add the interpolated data to the pyvista data object
        pv_data[data.name] = solution.components[0].vec.FV().NumPy().copy()


def compute_eigen_coefficients(
        pv_data: pv.UnstructuredGrid,
        mass_matrix: sp.csr_matrix,
        names: str | Iterable[str],
) -> Dict[str, np.ndarray]:
    """
    Compute the coefficients of the given fields with respect to the eigenfunctions.
    :param pv_data: The mesh data to compute the coefficients for
    :param mass_matrix: The mass matrix of the underlying mesh
    :param names: The names of the fields to compute the coefficients for
    :return: The coefficients of the fields with respect to the eigenfunctions
    """
    if isinstance(names, str):
        names = [names]

    array_names = pv_data.array_names
    n_eigenfunctions = sum("eigenfunction" in name for name in array_names)
    integral_weights = pv_data["integration weights"]
    eigenfunctions = np.array([pv_data[f"eigenfunction {i}"] for i in range(n_eigenfunctions)])
    coefficients = {}

    for name in names:
        gene = pv_data[name]
        for j in range(n_eigenfunctions):
            coefficients[name] = np.dot(eigenfunctions, mass_matrix @ gene)

    return coefficients
