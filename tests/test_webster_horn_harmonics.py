import numpy as np
import pytest

from embryoharmonics import Harmonics, correlate
from embryoharmonics.celegans import EmbryoModelLoader
from embryoharmonics.celegans.webster_horn_harmonics import compute_webster_horn_harmonics


TEST_FILE = "tests/resources/celegans_models.h5"


@pytest.fixture(scope="module")
def symmetric_mesh():
    """A rotationally symmetric embryo model and its mesh."""
    model = EmbryoModelLoader(TEST_FILE).load(420, symmetric=True)
    return model, model.generate_mesh(mesh_size=10)


def test_webster_horn_harmonics_match_fem_harmonics(symmetric_mesh):
    """The Webster-Horn approximation should closely match the true FEM
    harmonics (up to sign) for the lowest, non-degenerate axisymmetric modes
    of a symmetric embryo geometry.
    """
    model, mesh = symmetric_mesh
    n = 5
    webster_harmonics = compute_webster_horn_harmonics(mesh, model, n=n)
    fem_harmonics = Harmonics.compute(mesh, n=n)

    corr = correlate(
        [webster_harmonics[i] for i in range(n)],
        [fem_harmonics[i] for i in range(n)],
        normalize=False,
    )

    for i in range(n):
        assert np.max(np.abs(corr[i])) > 0.99
