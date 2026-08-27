import os
import tempfile
import pytest

import pyvista as pv
import numpy as np
from vtkmodules.vtkIOXdmf2 import vtkXdmfReader

from embryoharmonics import Harmonics
from embryoharmonics import io


def create_mesh(dim, level=1) -> pv.UnstructuredGrid:
    """Create a simple small mesh for testing purposes."""
    surface_mesh = pv.Box(level=level).triangulate().cast_to_unstructured_grid()

    if dim == 2:
        return surface_mesh
    else:
        return surface_mesh.delaunay_3d().cast_to_unstructured_grid()


@pytest.mark.parametrize("dim", [2, 3])
def test_storing_time_points_works(dim):
    """Test that storing and loading meshes and harmonics per time point works."""
    times = [0, 1]
    meshes = [create_mesh(dim, level=level + 1) for level in times]
    harmonics = [Harmonics.compute(mesh, n=5) for mesh in meshes]

    with tempfile.TemporaryDirectory() as tmpdirname:
        file_name = os.path.join(tmpdirname, "test.h5")

        for t, mesh, h in zip(times, meshes, harmonics):
            io.save_time_point(file_name, t, mesh, h)

        assert io.time_points(file_name) == times

        for t, mesh, h in zip(times, meshes, harmonics):
            loaded_mesh, loaded_harmonics = io.load_time_point(file_name, t)
            assert loaded_mesh.n_points == mesh.n_points
            assert loaded_mesh.n_cells == mesh.n_cells
            assert np.allclose(loaded_mesh.points, mesh.points)
            assert len(loaded_harmonics) == len(h)
            assert np.allclose(loaded_harmonics.eigenvalues, h.eigenvalues)
            assert np.allclose(loaded_harmonics[0].data, h[0].data)

        # Re-saving a time point overwrites it instead of failing
        io.save_time_point(file_name, times[0], meshes[0], harmonics[0])
        assert io.time_points(file_name) == times


def test_storing_time_point_fails_on_wrong_extension():
    """Test that storing a time point fails on wrong extension."""
    mesh = create_mesh(2)
    harmonics = Harmonics.compute(mesh, n=3)
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_name = os.path.join(tmpdirname, "test.xdmf")

        with pytest.raises(ValueError):
            io.save_time_point(file_name, 0, mesh, harmonics)


@pytest.mark.parametrize("dim", [2, 3])
def test_xdmf_file_is_readable(dim):
    """Test that the generated XDMF file can be read back with VTK.

    The vtk wheel only ships the Xdmf2 reader (ParaView has the Xdmf3 one as
    well); both accept the XML data model written by save_time_point.
    """
    mesh = create_mesh(dim)
    harmonics = Harmonics.compute(mesh, n=3)

    with tempfile.TemporaryDirectory() as tmpdirname:
        file_name = os.path.join(tmpdirname, "test.h5")
        io.save_time_point(file_name, 42, mesh, harmonics)

        reader = vtkXdmfReader()
        reader.SetFileName(os.path.join(tmpdirname, "test.xdmf"))
        reader.Update()
        loaded = pv.wrap(reader.GetOutputDataObject(0))

        # The temporal collection is returned as a multi-block dataset
        block = loaded[0] if isinstance(loaded, pv.MultiBlock) else loaded
        assert block.n_points == mesh.n_points
        assert block.n_cells == mesh.n_cells
        assert "harmonic_0" in block.point_data


@pytest.mark.parametrize("strings", [["foo"], ["foobar", "bar", "baz"]])
def test_encoding_matlab_strings(strings):
    """Test that encoding strings to match the matlab format works."""
    encoded = io.encode_matlab_strings(strings)

    decoded = [s.tobytes().decode('ascii').strip() for s in encoded.astype(np.uint8).T]

    assert len(strings) == len(encoded.T)
    assert all(s == d for s, d in zip(strings, decoded))
