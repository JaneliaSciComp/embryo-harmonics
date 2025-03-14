import pytest

import numpy as np
import pyvista as pv

from embryoharmonics import MeshData, Harmonics, HarmonicSmoother, DiffusionSmoother, FemMatrices


@pytest.mark.parametrize("n", [1, 5, 10])
def test_harmonic_smoother_doesnt_amplify_data(n):
    """Test that data after smoothing is in the same range as before."""
    mesh = pv.Box(level=2).triangulate().delaunay_3d().cast_to_unstructured_grid()
    data = np.zeros(mesh.n_points)
    data[0] = 1.0
    mesh_data = MeshData(mesh, "test", data)
    harmonics = Harmonics.compute(mesh, n=n)
    smoother = HarmonicSmoother(harmonics)

    smoothed = smoother.smooth(mesh_data)

    assert smoothed.name == "test_smoothed"
    assert smoothed.data[0] <= 1.0
    assert smoothed.data[0] >= 0.0


@pytest.mark.parametrize("n_steps", [1, 10, 100, 1000])
def test_diffusion_smoother_conserves_mass(n_steps):
    """Test that the harmonic smoother conserves the total mass of the data."""
    mesh = pv.Box(level=2).triangulate().delaunay_3d().cast_to_unstructured_grid()
    data = np.zeros(mesh.n_points)
    data[0] = 1.0
    mesh_data = MeshData(mesh, "test", data)
    smoother = DiffusionSmoother(mesh, smoothness=2, n_steps=n_steps)
    fem_matrices = FemMatrices.compute_for(mesh, stiffness=False)

    smoothed = smoother.smooth(mesh_data)

    expected_mass = (fem_matrices.mass @ mesh_data.data).sum()
    actual_mass = (fem_matrices.mass @ smoothed.data).sum()
    assert expected_mass == pytest.approx(actual_mass, rel=1e-4)
