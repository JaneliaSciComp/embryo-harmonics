import numpy as np
import pytest

from embryoharmonics.celegans import AxisymmetricHarmonics
from embryoharmonics.celegans.gene_data import VOXEL_SIZE_UM
from embryoharmonics.celegans.ovoid import OvoidModel, load_ovoid, save_ovoid
from embryoharmonics.celegans.ovoid_horn import (
    CAP_TREATMENTS,
    compute_horn_phi_harmonics,
    compute_horn_z_harmonics,
    sph_jp_zero,
)


COEFFICIENT_FILE = "tests/resources/ovoid_coefficients.csv"

# Semi-axes (in pixels) of the ellipse whose Fourier series is the
# 'ellipse_value' column, from data/ovoid/ovoid_fit.csv
ELLIPSE_AXES = (706.0090908148745, 370.13160709765776)

# The lowest nonzero zeros of j_n' for n = 1, 2: the lowest Neumann
# eigenvalues of the unit ball are their squares
BALL_ZEROS = (2.0815759778, 3.3420936578)


def test_ellipse_coefficients_reproduce_the_ellipse():
    """The truncated Fourier series of an ellipse should reproduce it closely,
    and the pixel size should scale it to voxels.
    """
    ovoid = OvoidModel.from_csv(
        COEFFICIENT_FILE, pixel_size=2 * VOXEL_SIZE_UM, column="ellipse_value"
    )
    a, b = ELLIPSE_AXES
    theta = np.linspace(0, np.pi, 500)
    exact = a * b / np.hypot(b * np.cos(theta), a * np.sin(theta))

    assert np.allclose(ovoid.radius(theta), 2 * exact, rtol=1e-3)


def test_flip_and_radial_extension():
    """Flipping mirrors the outline at theta = pi / 2, and the radial extension
    adds a constant radius (in microns).
    """
    ovoid = OvoidModel.from_csv(COEFFICIENT_FILE)
    flipped = OvoidModel.from_csv(COEFFICIENT_FILE, flip=True, radial_extension=1.0)
    theta = np.linspace(0, np.pi, 50)

    assert np.allclose(flipped.radius(np.pi - theta), ovoid.radius(theta) + 1 / VOXEL_SIZE_UM)


def test_radius_derivative():
    """The analytic derivative of the outline should match finite differences."""
    ovoid = OvoidModel.from_csv(COEFFICIENT_FILE)
    theta = np.linspace(0.1, 3.0, 20)
    h = 1e-6
    finite_difference = (ovoid.radius(theta + h) - ovoid.radius(theta - h)) / (2 * h)

    assert np.allclose(ovoid.radius(theta, derivative=True), finite_difference, rtol=1e-5)


def test_meridian_mesh_follows_the_outline():
    """The meridian mesh should span the ovoid from tip to tip."""
    ovoid = OvoidModel.from_csv(COEFFICIENT_FILE)
    mesh = ovoid.generate_meridian_mesh(mesh_size=10)

    r, z = mesh.points[:, 0], mesh.points[:, 2]
    assert r.min() == pytest.approx(0)
    assert z.max() == pytest.approx(ovoid.radius(0.0))
    assert z.min() == pytest.approx(-ovoid.radius(np.pi))
    assert np.all(np.hypot(r, z) <= ovoid.radius(np.arctan2(r, z)) * (1 + 1e-3))


def test_exact_harmonics_of_a_ball():
    """On a ball, the axisymmetric harmonics should reproduce the analytic
    Neumann spectrum (j'_{n,1} / R)^2 with multiplicity 2n + 1.
    """
    radius = 20.0
    mesh = OvoidModel([radius]).generate_meridian_mesh(mesh_size=1.5)
    harmonics = AxisymmetricHarmonics.compute(mesh, n=9)

    expected = np.repeat(np.array((0.0,) + BALL_ZEROS) ** 2, [1, 3, 5]) / radius**2
    assert np.allclose(harmonics.eigenvalues, expected, rtol=1e-2, atol=1e-10)


def test_fitted_center_contains_all_points():
    """Points inside a shifted ovoid should be fitted with a center that
    contains them all, and flipping the ovoid should mirror the center.
    """
    ovoid = OvoidModel.from_csv(COEFFICIENT_FILE)
    flipped = OvoidModel.from_csv(COEFFICIENT_FILE, flip=True)
    shift = np.array([3.0, -5.0, 10.0])

    rng = np.random.default_rng(0)
    direction = rng.normal(size=(2000, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    theta = np.arccos(direction[:, 2])
    points = direction * (0.9 * ovoid.radius(theta) * rng.uniform(size=2000) ** (1 / 3))[:, None]
    points += shift

    center = ovoid.fit_center(points)
    assert ovoid.containment(points, center).max() <= 1

    mirror = np.array([1, 1, -1])
    flipped_center = flipped.fit_center(points * mirror)
    assert np.allclose(flipped_center, center * mirror, atol=0.5)


def test_save_and_load_round_trip(tmp_path):
    """The outline coefficients should survive a round trip through HDF5."""
    ovoid = OvoidModel.from_csv(COEFFICIENT_FILE)
    file_name = str(tmp_path / "ovoid.h5")
    save_ovoid(file_name, ovoid)

    assert np.array_equal(load_ovoid(file_name).coefficients, ovoid.coefficients)


@pytest.fixture(scope="module")
def spheroid():
    """An (almost exact) prolate spheroid and its meridian mesh and exact
    harmonics.
    """
    ovoid = OvoidModel.from_csv(COEFFICIENT_FILE, column="ellipse_value")
    mesh = ovoid.generate_meridian_mesh(mesh_size=6)
    return ovoid, mesh, AxisymmetricHarmonics.compute(mesh, n=20)


@pytest.mark.parametrize("caps", CAP_TREATMENTS)
def test_horn_z_approximates_the_spheroid_spectrum(spheroid, caps):
    """Every cap treatment should approximate the exact spectrum of a spheroid
    (aspect ratio 1.9) to within a few percent.
    """
    ovoid, mesh, exact = spheroid
    horn, labels = compute_horn_z_harmonics(mesh, ovoid, n=20, caps=caps)

    assert len(horn) == len(labels) >= 20
    assert horn.eigenvalues[0] == pytest.approx(0, abs=1e-10)
    errors = horn.eigenvalues[1:20] / exact.eigenvalues[1:20] - 1
    assert np.mean(np.abs(errors)) < 0.05


def test_corrected_caps_leave_the_pure_area_branch_unchanged(spheroid):
    """The wall-slope terms vanish for the m=0, l=1 branch."""
    ovoid, mesh, _ = spheroid
    natural, natural_labels = compute_horn_z_harmonics(mesh, ovoid, n=8, angular_order=0)
    corrected, corrected_labels = compute_horn_z_harmonics(
        mesh, ovoid, n=8, angular_order=0, caps="corrected"
    )

    def area_branch(harmonics, labels):
        return harmonics.eigenvalues[[i for i, (_, l, _) in enumerate(labels) if l == 1]]

    plain = area_branch(natural, natural_labels)
    assert len(plain) >= 3
    assert np.allclose(area_branch(corrected, corrected_labels)[:len(plain)], plain)


def test_unknown_cap_treatment_raises(spheroid):
    ovoid, mesh, _ = spheroid
    with pytest.raises(ValueError):
        compute_horn_z_harmonics(mesh, ovoid, caps="pointy")


def test_spherical_bessel_derivative_zeros():
    """The zeros of j_k' should match the tabulated ball eigenvalues."""
    assert sph_jp_zero(0, 1) == 0.0
    assert sph_jp_zero(1, 1) == pytest.approx(BALL_ZEROS[0])
    assert sph_jp_zero(2, 1) == pytest.approx(BALL_ZEROS[1])
    assert sph_jp_zero(0, 2) == pytest.approx(4.4934094579)


def test_horn_phi_is_exact_for_a_ball():
    """On a ball, the phi-horn modes are the exact ones: eigenvalues
    (kappa_{k,p} / R)^2 with multiplicity 2k + 1.
    """
    radius = 20.0
    ball = OvoidModel([radius])
    mesh = ball.generate_meridian_mesh(mesh_size=4)
    horn, labels = compute_horn_phi_harmonics(mesh, ball, n=16)

    expected = np.repeat(
        np.array([0.0, BALL_ZEROS[0], BALL_ZEROS[1], 4.4934094579]) ** 2, [1, 3, 5, 1]
    ) / radius**2
    assert len(horn) == len(labels) >= 16
    assert np.allclose(horn.eigenvalues[:10], expected, rtol=1e-3, atol=1e-10)


def test_horn_phi_approximates_the_spheroid_spectrum(spheroid):
    """The phi-horn should roughly approximate the exact spectrum of a
    spheroid. It is much worse than the z-horn here: r(phi) is not slowly
    varying at aspect ratio 1.9, and axial (m = 0) modes are off by up to 50%.
    """
    ovoid, mesh, exact = spheroid
    horn, labels = compute_horn_phi_harmonics(mesh, ovoid, n=20)

    assert len(horn) == len(labels) >= 20
    errors = horn.eigenvalues[1:20] / exact.eigenvalues[1:20] - 1
    assert np.mean(np.abs(errors)) < 0.2
