from embryoharmonics.celegans import EmbryoModelLoader

TEST_FILE = "test/resources/celegans_models.h5"

def test_model_has_correct_time_steps():
    """Load the embryo model and check that it has the expected time steps."""
    embryo_model_loader = EmbryoModelLoader(TEST_FILE)

    assert 420 in embryo_model_loader.time_steps
    assert 421 in embryo_model_loader.time_steps
