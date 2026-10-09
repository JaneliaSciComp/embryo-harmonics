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


def test_expand_and_extend_tail_is_smooth(embryo_model_loader):
    """The extension must keep the body, join smoothly, and taper linearly."""
    from embryoharmonics.celegans import expand_and_extend_tail

    model = embryo_model_loader.load(420, symmetric=True)
    extended = expand_and_extend_tail(model, radial_expansion=0.05, linear_fraction=0.8)
    t_end, t_new = model.spline_domain[-1], extended.spline_domain[-1]
    assert t_new > t_end
    assert all(spline.x[-1] == t_new for spline in extended.transverse_splines)

    def radius(m, spline, t):
        return np.linalg.norm((spline(t) - m.central_spline(t))[..., :2], axis=-1)

    for original, spline in zip(model.transverse_splines, extended.transverse_splines):
        scaled = lambda t: model.central_spline(t) + 1.05 * (original(t) - model.central_spline(t))
        t = np.linspace(0, t_end, 50)
        np.testing.assert_allclose(spline(t), scaled(t), atol=1e-9)

        # C1 at the new knot
        for k in range(2):
            np.testing.assert_allclose(spline(t_end - 1e-9, k), spline(t_end + 1e-9, k), rtol=1e-5, atol=1e-5)

        # Linear taper: radius drops by 80% of its end value, central stays straight
        np.testing.assert_allclose(radius(extended, spline, t_new), 0.2 * radius(extended, spline, t_end))
        t_tail = np.linspace(t_end, t_new, 200)
        assert np.all(np.diff(radius(extended, spline, t_tail)) < 0)
    np.testing.assert_allclose(extended.central_spline(t_tail)[:, :2], [model.central_spline(t_end)[:2]] * 200, atol=1e-9)


def test_expand_and_extend_tail_absolute_offset(embryo_model_loader):
    """An absolute offset pushes the whole body out by a constant distance."""
    from embryoharmonics.celegans import expand_and_extend_tail

    model = embryo_model_loader.load(420, symmetric=True)
    extended = expand_and_extend_tail(model, radial_offset=12.0, linear_fraction=0.8)
    t_end, t_new = model.spline_domain[-1], extended.spline_domain[-1]
    t = np.linspace(0, t_end, 200)
    for original, spline in zip(model.transverse_splines, extended.transverse_splines):
        r_old = np.linalg.norm((original(t) - model.central_spline(t))[:, :2], axis=1)
        r_new = np.linalg.norm((spline(t) - model.central_spline(t))[:, :2], axis=1)
        np.testing.assert_allclose(r_new - r_old, 12.0, atol=0.05)
        assert spline.x[-1] == t_new


def test_expand_and_extend_tail_without_taper(embryo_model_loader):
    """A zero linear fraction offsets the body but leaves the domain unchanged."""
    from embryoharmonics.celegans import axial_profile, expand_and_extend_tail
    model = embryo_model_loader.load(420, symmetric=True)
    offset = expand_and_extend_tail(model, radial_offset=3.0, linear_fraction=0.0)
    np.testing.assert_array_equal(offset.spline_domain, model.spline_domain)
    _, r_old, _, _ = axial_profile(model, 50)
    _, r_new, _, _ = axial_profile(offset, 50)
    np.testing.assert_allclose(r_new - r_old, 3.0, atol=0.05)
    offset.generate_meridian_mesh(mesh_size=10)
