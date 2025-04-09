import numpy as np
import pytest

import pyvista as pv
import numpy as np

from embryoharmonics import MeshData, correlate


def create_box_mesh(level: int):
    """Create a box mesh with the given level of refinement."""
    return pv.Box(bounds=(0, 1, 0, 1, 0, 1), level=level, quads=False) \
        .triangulate().delaunay_3d().cast_to_unstructured_grid()


@pytest.mark.parametrize("level", [0, 1, 2])
def test_data_have_correct_fields(level: int):
    """Test that correctly created data have correct accessors."""
    mesh = create_box_mesh(level)
    data = MeshData(mesh, "test", mesh.points[:, 0])

    assert data.name == "test"
    assert data.data.shape == (mesh.n_points,)


@pytest.mark.parametrize("level", [0, 1, 2])
def test_wrong_data_length_raises_error(level: int):
    """Test that wrong data length raises an error."""
    mesh = create_box_mesh(level)

    with pytest.raises(ValueError):
        MeshData(mesh, "test", mesh.points[:, 0][:-1])


@pytest.mark.parametrize("level", [0, 1])
def test_resample_data(level):
    """Test that resampling constant data onto a finer mesh is constant."""
    mesh = create_box_mesh(level)
    data = MeshData(mesh, "test", np.full(mesh.n_points, 1.23))

    finer_mesh = create_box_mesh(level + 1)
    finer_data = data.resample(finer_mesh)

    assert np.all(finer_data.data == 1.23)

@pytest.mark.parametrize("level", [0, 1])
def test_correlation_works(level: int):
    """Test that correlation function yields a correlation matrix."""
    mesh = create_box_mesh(level)
    data = [MeshData(mesh, f"test{i}", np.random.randn(mesh.n_points)) for i in range(3)]

    corr = correlate(data)

    assert corr.shape == (3, 3)
    assert np.all(np.logical_and(corr >= -1 - 1e-6, corr <= 1 + 1e-6))
    assert np.all(np.isclose(corr, corr.T))
    assert np.all(np.isclose(np.diag(corr), 1))
