import os
import tempfile
import pytest

import pyvista as pv
import numpy as np

from embryoharmonics import Harmonics, MeshData
import embryoharmonics.io as io


def create_mesh(dim) -> pv.UnstructuredGrid:
    """Create a simple small mesh for testing purposes."""
    surface_mesh = pv.Box(level=1).triangulate().cast_to_unstructured_grid()

    if dim == 2:
        return surface_mesh
    else:
        return surface_mesh.delaunay_3d().cast_to_unstructured_grid()


@pytest.mark.parametrize("dim", [2, 3])
def test_storing_mesh_works(dim):
    """Test that storing and loading a mesh works."""
    mesh = create_mesh(dim)
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_name = os.path.join(tmpdirname, "test.vtu")

        io.save_mesh(file_name, mesh)
        loaded_mesh = io.load_mesh(file_name)

    assert loaded_mesh.n_points == mesh.n_points
    assert loaded_mesh.n_cells == mesh.n_cells


def test_storing_mesh_fails_on_wrong_extension():
    """Test that storing a mesh fails on wrong extension."""
    mesh = create_mesh(2)
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_name = os.path.join(tmpdirname, "test.stl")

        with pytest.raises(ValueError):
            io.save_mesh(file_name, mesh)


@pytest.mark.parametrize("dim", [2, 3])
def test_storing_mesh_data_works(dim):
    """Test that storing and loading data works."""
    mesh = create_mesh(dim)
    data = [
        MeshData(mesh, "zero", np.zeros(mesh.n_points)),
        MeshData(mesh, "non-zero", np.arange(mesh.n_points) + 1)
    ]

    with tempfile.TemporaryDirectory() as tmpdirname:
        file_name = os.path.join(tmpdirname, "test.h5")

        io.save_mesh_data(file_name, data)
        loaded_data = io.load_mesh_data(file_name, mesh)

    assert len(loaded_data) == len(data)
    assert loaded_data[0].name == data[0].name
    assert all(loaded_data[0].data == data[0].data)
    assert loaded_data[1].name == data[1].name
    assert all(loaded_data[1].data == data[1].data)


@pytest.mark.parametrize("dim", [2, 3])
def test_storing_harmonics_works(dim):
    """Test that storing and loading data works."""
    mesh = create_mesh(dim)
    harmonics = Harmonics.compute(mesh, n=mesh.n_points - 1)

    with tempfile.TemporaryDirectory() as tmpdirname:
        file_name = os.path.join(tmpdirname, "test.h5")

        io.save_harmonics(file_name, harmonics)
        loaded_data = io.load_harmonics(file_name, mesh)

    assert len(loaded_data) == len(harmonics)
    assert all(loaded_data.eigenvalues == harmonics.eigenvalues)
    assert all(loaded_data[0].data == harmonics[0].data)
