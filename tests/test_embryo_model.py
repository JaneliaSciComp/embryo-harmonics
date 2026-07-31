import numpy as np
import pytest

from embryoharmonics.celegans import EmbryoModelLoader


TEST_FILE = "tests/resources/celegans_models.h5"


@pytest.fixture(scope="module")
def embryo_model_loader():
    """Load the embryo model for a single time step."""
    return EmbryoModelLoader(TEST_FILE)


def test_loader_has_correct_time_steps(embryo_model_loader):
    """Load the embryo model and check that it has the expected time steps."""
    assert 420 in embryo_model_loader.time_steps
    assert 421 in embryo_model_loader.time_steps


def test_loading_wrong_time_step_raises_error(embryo_model_loader):
    """Load the embryo model and check that loading a wrong time step raises an error."""
    with pytest.raises(ValueError):
        embryo_model_loader.load(419)


@pytest.mark.parametrize("time_step", [420, 421])
def test_model_has_correct_number_of_properties(time_step, embryo_model_loader):
    """Load the embryo model and check that it has the expected number of properties."""
    embryo_model = embryo_model_loader.load(time_step)

    assert embryo_model.n_transverse_splines == 32
    assert embryo_model.n_seam_cells == 11


def test_model_can_generate_mesh(embryo_mesh):
    """Load the embryo model and check that it can generate a mesh (which is
    generated as part of the fixture in conftest).
    """
    assert embryo_mesh.n_points > 0
    assert embryo_mesh.n_cells > 0


def test_symmetric_model_has_identical_radius_for_all_splines(embryo_model_loader):
    """Load a rotationally symmetric embryo model and check that all
    transverse splines have the same radius profile (i.e. they are just
    rotated copies of the 0-th spline).
    """
    embryo_model = embryo_model_loader.load(420, symmetric=True)
    t = embryo_model.spline_domain
    central = embryo_model.central_spline(t)

    radii = [
        np.linalg.norm(spline(t) - central, axis=1)
        for spline in embryo_model.transverse_splines
    ]
    for radius in radii[1:]:
        assert np.allclose(radius, radii[0])
