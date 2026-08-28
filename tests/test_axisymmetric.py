import numpy as np
import pytest

from embryoharmonics import Harmonics, correlate, io
from embryoharmonics.celegans import (
    AxisymmetricHarmonics,
    EmbryoModel,
    EmbryoModelLoader,
    load_axisymmetric_harmonics,
    revolve_meridian,
    save_axisymmetric_harmonics,
)
from embryoharmonics.celegans.webster_horn_harmonics import match_modes
from embryoharmonics.fem import FemMatrices
from embryoharmonics.webster_horn import jp_zero


TEST_FILE = "tests/resources/celegans_models.h5"

CYLINDER_RADIUS = 25.0
CYLINDER_LENGTH = 100.0


@pytest.fixture(scope="module")
def cylinder_meridian():
    """The meridian mesh of a straight cylinder along the z-axis."""
    t = np.linspace(0, 1, 8)
    central = np.column_stack([np.zeros_like(t), np.zeros_like(t), CYLINDER_LENGTH * t])
    transverse = central + np.array([CYLINDER_RADIUS, 0.0, 0.0])
    model = EmbryoModel(["a", "b"], t, central, [transverse], symmetric=True)
    return model.generate_meridian_mesh(mesh_size=3)


@pytest.fixture(scope="module")
def cylinder_harmonics(cylinder_meridian):
    return AxisymmetricHarmonics.compute(cylinder_meridian, n=10)


@pytest.fixture(scope="module")
def symmetric_worm():
    """A rotationally symmetric embryo model with its 3D and meridian meshes."""
    model = EmbryoModelLoader(TEST_FILE).load(420, symmetric=True)
    return model, model.generate_mesh(mesh_size=10), model.generate_meridian_mesh(mesh_size=5)


def _analytic_cylinder_spectrum(n):
    """The lowest Neumann-Laplace eigenvalues of a cylinder with their
    quantum numbers (m, n_r, n_z), expanded into cos/sin pairs for m > 0."""
    modes = []
    for m in range(4):
        for l in range(1, 4):
            for p in range(6):
                eigenvalue = ((jp_zero(m, l) / CYLINDER_RADIUS) ** 2
                              + (p * np.pi / CYLINDER_LENGTH) ** 2)
                modes.extend([(eigenvalue, m, l - 1, p)] * (1 if m == 0 else 2))
    modes.sort()
    return modes[:n]


def test_cylinder_eigenvalues_match_analytic_values(cylinder_harmonics):
    """The axisymmetric eigenvalues of a cylinder must match the analytic
    Neumann spectrum, including the angular labels and cos/sin pairing.
    """
    analytic = _analytic_cylinder_spectrum(len(cylinder_harmonics))

    assert np.allclose(
        cylinder_harmonics.eigenvalues, [m[0] for m in analytic], rtol=2e-2, atol=1e-8
    )
    assert list(cylinder_harmonics.angular_orders) == [m[1] for m in analytic]

    # sin modes always directly follow their cos partner
    for i in np.flatnonzero(cylinder_harmonics.trig_kinds == "sin"):
        assert cylinder_harmonics.trig_kinds[i - 1] == "cos"
        assert cylinder_harmonics.angular_orders[i - 1] == cylinder_harmonics.angular_orders[i]
        assert cylinder_harmonics.eigenvalues[i - 1] == cylinder_harmonics.eigenvalues[i]


def test_nodal_counts_match_analytic_quantum_numbers(cylinder_meridian):
    """Nodal counting recovers the analytic cylinder quantum numbers
    (n_r, n_z). Compared as multisets: near-degenerate eigenvalues may swap
    order at discretization level, but the label set is unambiguous.
    """
    # n = 24 reaches the first n_r = 1 mode and ends at a clean spectral gap
    harmonics = AxisymmetricHarmonics.compute(cylinder_meridian, n=24)
    radial, axial = harmonics.nodal_counts()

    analytic = _analytic_cylinder_spectrum(len(harmonics))
    computed = sorted(zip(harmonics.angular_orders, radial, axial))
    expected = sorted((m[1], m[2], m[3]) for m in analytic)
    assert computed == expected
    assert max(m[2] for m in analytic) == 1  # the radial direction is exercised


def test_truncation_never_splits_a_degenerate_pair(cylinder_meridian):
    """If the requested count would cut a cos/sin pair in half, the sin
    partner is included as well.
    """
    for n in range(3, 8):
        harmonics = AxisymmetricHarmonics.compute(cylinder_meridian, n=n)
        assert len(harmonics) >= n
        assert harmonics.trig_kinds[-1] == "sin" or harmonics.angular_orders[-1] == 0


@pytest.mark.slow
def test_axisymmetric_harmonics_match_fem_harmonics(symmetric_worm):
    """The axisymmetric reduction is exact, so eigenvalues and eigenspaces
    must match a full 3D FEM solve on the symmetric worm up to
    discretization error.
    """
    _, mesh_3d, meridian = symmetric_worm
    n = 8

    axi = AxisymmetricHarmonics.compute(meridian, n=n)
    fem = Harmonics.compute(mesh_3d, n=n)

    assert np.allclose(axi.eigenvalues[:n], fem.eigenvalues, rtol=5e-2, atol=1e-8)

    full = axi.to_full_3d(mesh_3d)
    coefficients = correlate(
        [full[i] for i in range(len(full))],
        [fem[i] for i in range(n)],
        normalize=False,
    )
    _, products = match_modes(coefficients, axi.degenerate_clusters())
    assert np.min(products) > 0.98


@pytest.mark.slow
def test_to_full_3d_clusters_are_mass_orthonormal(symmetric_worm):
    """Reconstructed degenerate pairs must be mass-orthonormal on the 3D mesh."""
    _, mesh_3d, meridian = symmetric_worm

    axi = AxisymmetricHarmonics.compute(meridian, n=8)
    full = axi.to_full_3d(mesh_3d)
    mass = FemMatrices.compute_for(mesh_3d, stiffness=False).mass

    for cluster in axi.degenerate_clusters():
        fields = np.array([full[i].data for i in cluster])
        gram = fields @ (mass @ fields.T)
        assert np.allclose(gram, np.eye(len(cluster)), atol=1e-8)


def test_io_round_trip(cylinder_harmonics, tmp_path):
    """Axisymmetric harmonics survive a save/load cycle, and the plain io
    loader still works on the same file (labels ignored).
    """
    file_name = str(tmp_path / "axisymmetric.h5")
    save_axisymmetric_harmonics(file_name, 42, cylinder_harmonics)
    loaded = load_axisymmetric_harmonics(file_name, 42)

    assert np.allclose(loaded.eigenvalues, cylinder_harmonics.eigenvalues)
    assert np.array_equal(loaded.angular_orders, cylinder_harmonics.angular_orders)
    assert np.array_equal(loaded.trig_kinds, cylinder_harmonics.trig_kinds)
    for i in range(len(cylinder_harmonics)):
        assert np.allclose(loaded[i].data, cylinder_harmonics[i].data)

    plain = io.load_harmonics(file_name, 42)
    assert len(plain) == len(cylinder_harmonics)


def test_revolve_meridian_recovers_cylinder_volume(cylinder_meridian):
    """Revolving the cylinder meridian yields a tet mesh with the right volume."""
    mesh = revolve_meridian(cylinder_meridian, mesh_size=5)

    assert mesh.n_cells > 0
    expected = np.pi * CYLINDER_RADIUS**2 * CYLINDER_LENGTH
    # netgen tet ordering yields negative VTK-signed volumes (as in the
    # existing 3D pipeline); the FEM assembly fixes orientation per element
    assert abs(mesh.volume) == pytest.approx(expected, rel=2e-2)
