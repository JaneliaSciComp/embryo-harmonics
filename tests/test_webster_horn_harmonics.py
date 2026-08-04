import numpy as np
import pytest

from embryoharmonics import Harmonics, correlate
from embryoharmonics.celegans import EmbryoModelLoader
from embryoharmonics.celegans.webster_horn_harmonics import (
    compute_webster_horn_harmonics,
    degenerate_clusters,
    match_modes,
)


TEST_FILE = "tests/resources/celegans_models.h5"


@pytest.fixture(scope="module")
def symmetric_mesh():
    """A rotationally symmetric embryo model and its mesh."""
    model = EmbryoModelLoader(TEST_FILE).load(420, symmetric=True)
    return model, model.generate_mesh(mesh_size=10)


def test_degenerate_clusters_groups_modes_by_label():
    """Modes sharing an (m, l, n) label form one cluster; m > 0 labels appear
    twice (cos/sin), m = 0 labels once.
    """
    labels = [(0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 0), (0, 1, 2)]
    assert degenerate_clusters(labels) == [[0], [1], [2, 3], [4]]


def test_match_modes_sees_through_rotation_of_a_degenerate_pair():
    """A rotated degenerate pair spans the same eigenspace, so each target mode
    still has a Webster-Horn eigenfunction matching it exactly, even though no
    single column of the coefficient matrix is close to 1.
    """
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    coefficients = np.eye(3)
    coefficients[1:, 1:] = rotation

    matched, products = match_modes(coefficients, [[0], [1, 2]])

    assert np.allclose(products, 1.0)
    assert list(matched) == [0, 1, 1]


def test_match_modes_reports_partial_products_for_mixed_modes():
    """A target mode split evenly between two non-degenerate Webster-Horn modes
    cannot be reproduced by either, and the scalar product must say so.
    """
    mixed = np.array([[np.sqrt(0.5)], [np.sqrt(0.5)]])

    _, products = match_modes(mixed, [[0], [1]])

    assert products[0] == pytest.approx(np.sqrt(0.5))


def test_webster_horn_harmonics_match_fem_harmonics(symmetric_mesh):
    """The Webster-Horn approximation should closely reproduce the true FEM
    harmonics of a symmetric embryo geometry. Compared cluster-wise, since
    individual modes of a degenerate pair are only defined up to a rotation.
    """
    model, mesh = symmetric_mesh
    webster, labels = compute_webster_horn_harmonics(mesh, model, n=5)
    fem = Harmonics.compute(mesh, n=len(webster))

    coefficients = correlate(
        [webster[i] for i in range(len(webster))],
        [fem[i] for i in range(len(fem))],
        normalize=False,
    )
    _, products = match_modes(coefficients, degenerate_clusters(labels))

    assert np.min(products) > 0.99


def test_webster_horn_degenerate_clusters_are_mass_orthonormal(symmetric_mesh):
    """Projecting onto a cluster as a sum of squared coefficients is only valid
    if its modes are mass-orthonormal, which the raw cos/sin pair is not on an
    unstructured mesh.
    """
    model, mesh = symmetric_mesh
    webster, labels = compute_webster_horn_harmonics(mesh, model, n=12)

    for indices in degenerate_clusters(labels):
        if len(indices) < 2:
            continue
        gram = correlate([webster[i] for i in indices], normalize=False)
        assert np.allclose(gram, np.eye(len(indices)), atol=1e-10)


def test_webster_horn_truncation_keeps_degenerate_clusters_intact(symmetric_mesh):
    """Truncating the mode pool must not split a cos/sin pair in half, so the
    requested count may be exceeded by one.
    """
    model, mesh = symmetric_mesh
    _, labels = compute_webster_horn_harmonics(mesh, model, n=4)

    for indices in degenerate_clusters(labels):
        expected = 1 if labels[indices[0]][0] == 0 else 2
        assert len(indices) == expected
