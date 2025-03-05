from typing import Literal
import numpy as np
import pytest
import pyvista as pv

from embryoharmonics.fem import compute_fem_matrices


def create_box_mesh(dim: Literal[2, 3]) -> pv.UnstructuredGrid:
    """
    Create surface or volume mesh of a box.
    :return: A pyvista.UnstructuredGrid object representing the mesh and its
        d-dimensional volume
    """
    box_surface_mesh = pv.Box(bounds=(0, 1, 0, 1, 0, 1), level=0, quads=False)

    return (box_surface_mesh.cast_to_unstructured_grid(), 6) if dim == 2 \
        else (box_surface_mesh.delaunay_3d().cast_to_unstructured_grid(), 1)


def constant_function(x):
    """Function that returns a constant value of 1 for each input location."""
    return np.ones(x.shape[0])


def linear_function(x):
    """Function that returns the sum of the coordinates for each input location."""
    return x.sum(axis=1)


@pytest.mark.parametrize("dim", [2, 3])
def test_mass_matrix_integrates_constant(dim):
    """The mass matrix integrates constant functions correctly."""
    mesh, vol = create_box_mesh(dim)
    constant = 3.14

    fem = compute_fem_matrices(mesh, stiffness=False)
    fun_value = constant * constant_function(mesh.points)

    assert (fem.mass @ fun_value).sum() == pytest.approx(constant * vol, rel=1e-2)


@pytest.mark.parametrize("dim", [2, 3])
def test_mass_matrix_integrates_linear(dim):
    """The mass matrix integrates linear functions correctly."""
    mesh, vol = create_box_mesh(dim)

    fem = compute_fem_matrices(mesh, stiffness=False)
    fun_value = linear_function(mesh.points)

    expected = vol * np.ptp(fun_value) / 2
    assert (fem.mass @ fun_value).sum() == pytest.approx(expected, rel=1e-2)


@pytest.mark.parametrize("dim", [2, 3])
def test_stiffness_matrix_integrates_constant(dim):
    """The stiffness matrix integrates constant functions correctly."""
    mesh, _ = create_box_mesh(dim)
    constant = 3.14

    fem = compute_fem_matrices(mesh, mass=False)
    fun_value = constant * constant_function(mesh.points)

    assert (fem.stiffness @ fun_value).sum() == pytest.approx(0, abs=1e-2)


@pytest.mark.parametrize("dim", [2, 3])
def test_stiffness_matrix_integrates_linear(dim):
    """The stiffness matrix integrates linear functions correctly."""
    mesh, vol = create_box_mesh(dim)

    fem = compute_fem_matrices(mesh, mass=False)
    fun_value = linear_function(mesh.points)

    expected = dim * vol
    assert fun_value.dot(fem.stiffness @ fun_value) == pytest.approx(expected, abs=1e-2)
