import pytest
from embryoharmonics.celegans import EmbryoModelLoader


TEST_FILE = "tests/resources/celegans_models.h5"


def test_loader_has_correct_time_steps():
    """Load the embryo model and check that it has the expected time steps."""
    with EmbryoModelLoader(TEST_FILE) as embryo_model_loader:
        assert 420 in embryo_model_loader.time_steps
        assert 421 in embryo_model_loader.time_steps


def test_loading_wrong_time_step_raises_error():
    """Load the embryo model and check that loading a wrong time step raises an error."""
    with EmbryoModelLoader(TEST_FILE) as embryo_model_loader:
        with pytest.raises(ValueError):
            embryo_model_loader.load(419)


@pytest.mark.parametrize("time_step", [420, 421])
def test_model_has_correct_number_of_properties(time_step):
    """Load the embryo model and check that it has the expected number of properties."""
    with EmbryoModelLoader(TEST_FILE) as embryo_model_loader:
        embryo_model = embryo_model_loader.load(time_step)

        assert embryo_model.n_transverse_splines == 32
        assert embryo_model.n_seam_cells == 11


@pytest.mark.slow
def test_model_can_generate_mesh():
    """Load the embryo model and check that it can generate a mesh."""
    with EmbryoModelLoader(TEST_FILE) as embryo_model_loader:
        embryo_model = embryo_model_loader.load(420)

        mesh = embryo_model.generate_mesh()
        assert mesh.n_cells > 0
