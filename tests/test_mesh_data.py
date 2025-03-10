import pytest

import pyvista as pv

from embryoharmonics import MeshData


@pytest.mark.parametrize("level", [0, 1, 2])
def test_data_have_correct_fields(level: int):
    """Test that correctly created data have correct accessors."""
    mesh = pv.Box(bounds=(0, 1, 0, 1, 0, 1), level=level, quads=False).delaunay_3d()
    data = MeshData(mesh, "test", mesh.points[:, 0])

    assert data.name == "test"
    assert data.data.shape == (mesh.n_points,)


@pytest.mark.parametrize("level", [0, 1, 2])
def test_wrong_data_length_raises_error(level: int):
    """Test that wrong data length raises an error."""
    mesh = pv.Box(bounds=(0, 1, 0, 1, 0, 1), level=level, quads=False).delaunay_3d()

    with pytest.raises(ValueError):
        MeshData(mesh, "test", mesh.points[:, 0][:-1])
