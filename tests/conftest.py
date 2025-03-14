import pytest

from embryoharmonics.celegans import GeneDataLoader, EmbryoModelLoader
from embryoharmonics import FemMatrices

# Set up skipping of integration tests per default.
# Run them with `pytest --runslow`.
def pytest_addoption(parser):
    """Add cli option to run slow tests."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    """Add marker for slow tests."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """Skip slow tests if --runslow not given."""
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# Session-wide fixtures
GEOMETRY_PATH = "tests/resources/celegans_models.h5"
GENE_PATH = "tests/resources/celegans_genedata.h5"

@pytest.fixture(scope="session")
def embryo_mesh():
    """Generate an embryo mesh for a single time step."""
    with EmbryoModelLoader(GEOMETRY_PATH) as embryo_model_loader:
        embryo_model = embryo_model_loader.load(420)
        mesh = embryo_model.generate_mesh(mesh_size=20)
        assert mesh.n_cells > 0
        return mesh


@pytest.fixture(scope="session")
def gene_data_loader():
    """Load simple test data for a few genes."""
    return GeneDataLoader(GENE_PATH)


@pytest.fixture(scope="session")
def fem_matrices(embryo_mesh):
    """Generate the finite element matrices for the embryo mesh."""
    return FemMatrices.compute_for(embryo_mesh, stiffness=False)
