import numpy as np
import pytest
from scipy.special import jnp_zeros

from embryoharmonics.webster_horn import jp_zero, slope_coupling, sturm_liouville, webster_envelopes


def test_jp_zero_m0_l1_is_trivial():
    """The m=0, l=1 branch is the pure area-profile horn (j'_{0,1} = 0)."""
    assert jp_zero(0, 1) == 0.0


def test_jp_zero_matches_scipy_for_higher_branches():
    """Higher branches should match scipy's Bessel-derivative zeros directly."""
    assert jp_zero(0, 2) == pytest.approx(jnp_zeros(0, 1)[-1])
    assert jp_zero(1, 1) == pytest.approx(jnp_zeros(1, 1)[-1])


def test_constant_radius_recovers_neumann_rod_spectrum():
    """For constant R, the m=0, l=1 branch reduces to the classic Neumann rod
    equation -A'' = lambda A, with eigenvalues (n*pi/L)^2.
    """
    L = 5.0
    s = np.linspace(0, L, 400)
    R = np.ones_like(s)

    eigenvalues, _ = webster_envelopes(s, R, m=0, l=1)
    expected = np.array([(n * np.pi / L) ** 2 for n in range(5)])

    assert np.allclose(eigenvalues[:5], expected, atol=1e-3)


def test_sturm_liouville_fixed_ends_recover_dirichlet_rod_spectrum():
    """With fixed ends, -A'' = lambda A has eigenvalues (n*pi/L)^2, n >= 1,
    and a constant coupling c (A phi)' integrates to zero, so it must not
    change the spectrum.
    """
    L = 3.0
    s = np.linspace(0, L, 400)
    ones = np.ones(len(s) - 1)
    expected = np.array([(n * np.pi / L) ** 2 for n in range(1, 6)])

    eigenvalues, envelopes = sturm_liouville(s, ones, 0 * ones, ones, fixed_ends=True)
    coupled, _ = sturm_liouville(s, ones, 0 * ones, ones, 0.7 * ones, fixed_ends=True)

    assert np.allclose(eigenvalues[:5], expected, rtol=1e-3)
    assert np.allclose(coupled[:5], expected, rtol=1e-3)
    assert np.all(envelopes[[0, -1]] == 0)


def test_slope_coupling():
    """The m=0, l=1 branch has a constant transverse shape and no slope terms;
    otherwise G1 follows from integrating by parts.
    """
    from scipy.integrate import quad
    from scipy.special import jv, jvp

    assert slope_coupling(0, 1) == (0.0, 0.0)

    jp = jp_zero(1, 1)
    G0 = quad(lambda xi: jv(1, jp * xi) ** 2 * xi, 0, 1)[0]
    G1 = quad(lambda xi: jv(1, jp * xi) * jp * jvp(1, jp * xi) * xi ** 2, 0, 1)[0]
    assert slope_coupling(1, 1)[0] == pytest.approx(G1 / G0)


def test_corrected_webster_equals_plain_for_constant_radius():
    """Without wall slope, the corrected equation is the plain one."""
    s = np.linspace(0, 5.0, 200)
    R = np.ones_like(s)

    plain, _ = webster_envelopes(s, R, m=1, l=2)
    corrected, _ = webster_envelopes(s, R, m=1, l=2, corrected=True)

    assert np.allclose(plain, corrected)
