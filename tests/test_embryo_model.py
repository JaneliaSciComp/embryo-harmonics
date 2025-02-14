import pytest
from embryoharmonics.celegans import EmbryoModelLoader


TEST_FILE = "tests/resources/celegans_models.h5"


def test_loader_has_correct_time_steps():
    """Load the embryo model and check that it has the expected time steps."""
    with EmbryoModelLoader(TEST_FILE) as embryo_model_loader:
        assert 420 in embryo_model_loader.time_steps
        assert 421 in embryo_model_loader.time_steps


@pytest.mark.parametrize("time_step", [420, 421])
def test_model_has_correct_number_of_properties(time_step):
    """Load the embryo model and check that it has the expected number of properties."""
    with EmbryoModelLoader(TEST_FILE) as embryo_model_loader:
        embryo_model = embryo_model_loader.load(time_step)

        assert len(embryo_model.transverse_splines) == 32
        assert len(embryo_model.seam_cells) == 11


@pytest.mark.slow
def test_model_can_generate_mesh():
    """Load the embryo model and check that it can generate a mesh."""
    with EmbryoModelLoader(TEST_FILE) as embryo_model_loader:
        embryo_model = embryo_model_loader.load(420)

        mesh = embryo_model.generate_mesh()
        assert mesh.ne > 0
