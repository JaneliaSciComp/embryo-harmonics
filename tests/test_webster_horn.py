import numpy as np
import pytest
from scipy.special import jnp_zeros

from embryoharmonics.webster_horn import jp_zero, webster_envelopes


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
