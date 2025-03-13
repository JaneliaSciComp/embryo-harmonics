import numpy as np
import pytest

import pyvista as pv

from embryoharmonics import FemMatrices, Harmonics


def create_mesh(dim) -> pv.UnstructuredGrid:
    """Create a simple small mesh for testing purposes."""
    surface_mesh = pv.Box(level=1).triangulate().cast_to_unstructured_grid()

    if dim == 2:
        return surface_mesh
    else:
        return surface_mesh.delaunay_3d().cast_to_unstructured_grid()


@pytest.mark.parametrize("dim", [2, 3])
def test_computing_harmonics_works(dim):
    """Test that computing harmonics works."""
    n = 3
    harmonics = Harmonics.compute(create_mesh(dim), n=n)

    assert len(harmonics) == n
    assert len(harmonics.eigenvalues) == n
    assert all(harmonics.eigenvalues > -1e-4)


@pytest.mark.parametrize("dim", [2, 3])
def test_indexing_with_slice_fails(dim):
    """Test that indexing with a slice fails."""
    harmonics = Harmonics.compute(create_mesh(dim), n=3)

    with pytest.raises(TypeError):
        _ = harmonics[:]


@pytest.mark.parametrize("dim", [2, 3])
def test_first_harmonic_is_not_representable_by_rest(dim):
    """Test that the first harmonic is not representable by the rest."""
    mesh = create_mesh(dim)
    n = mesh.n_points - 1
    harmonics = Harmonics.compute(mesh, n=n)

    first = harmonics[0]
    rest = harmonics.subset(slice(1, n, 2))
    coefficients = rest.decompose(first)

    assert np.allclose(coefficients["harmonic_00"], 0, rtol=1e-4)


@pytest.mark.parametrize("dim", [2, 3])
def test_harmonics_are_orthonormal(dim):
    """Test that the harmonics are orthonormal."""
    mesh = create_mesh(dim)
    n = mesh.n_points - 1
    harmonics = Harmonics.compute(mesh, n=n)
    mass = FemMatrices.compute_for(mesh, stiffness=False).mass

    for i in range(n):
        norm = harmonics[i].data @ mass @ harmonics[i].data
        assert abs(norm - 1) < 1e-4

        for j in range(i + 1, n):
            inner_product = harmonics[i].data @ mass @ harmonics [j].data
            assert abs(inner_product) < 1e-4


@pytest.mark.parametrize("dim", [2, 3])
def test_decomposing_harmonic_yields_single_coefficient(dim):
    """Test that decomposing a single harmonic yields a single non-zero coefficient."""
    mesh = create_mesh(dim)
    n = mesh.n_points - 1
    harmonics = Harmonics.compute(mesh, n=n)

    for mesh_data in harmonics:
        coefficients = harmonics.decompose(mesh_data)
        composed = harmonics.compose(coefficients)

        assert np.allclose(mesh_data.data, composed.data, rtol=1e-4)


@pytest.mark.parametrize("dim", [2, 3])
def test_decomposing_and_composing_yields_same(dim):
    """Test that decomposing and composing an array yields the array."""
    mesh = create_mesh(dim)
    n = mesh.n_points // 2
    harmonics = Harmonics.compute(mesh, n=n)

    expected = {'coefficients': np.arange(n)}
    composed = harmonics.compose(expected)
    actual = harmonics.decompose(composed)

    assert np.allclose(expected['coefficients'], actual['coefficients'], rtol=1e-4)
