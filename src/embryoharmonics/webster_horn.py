"""Webster-Horn approximation for the Neumann Laplacian on a body of revolution.

Ported from ``ShroffHarmonics.jl`` (``plot_harmonics.jl``, ``webster_envelopes``).
"""
import numpy as np
import scipy.linalg as sla
import scipy.special as sp


def jp_zero(m: int, l: int) -> float:
    """Return the l-th (1-indexed) zero of J_m', the derivative of the m-th
    order Bessel function of the first kind.

    :param m: The azimuthal (Bessel) order
    :param l: The 1-indexed zero to return
    :return: j'_{m,l}
    """
    if m == 0 and l == 1:
        # The trivial zero at 0: the m=0, l=1 branch is the pure
        # area-profile horn (no transverse Bessel structure).
        return 0.0

    n = l - 1 if m == 0 else l
    return float(sp.jnp_zeros(m, n)[-1])


def webster_envelopes(
        s: np.ndarray,
        R: np.ndarray,
        m: int,
        l: int
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the 1D Webster-Horn eigenproblem for the (m, l) branch of a body
    with radius profile R(s):

        -(R^2 A')' + j'_{m,l}^2 A = lambda R^2 A,     R^2 A' = 0 (natural
        Neumann) at the ends.

    This is the adiabatic reduction of the 3D Neumann-Laplacian problem: at
    each axial station the transverse shape is the local disk mode
    J_m(j'_{m,l} * r / R(z)) and A(z) is its slowly-varying amplitude, with
    (j'_{m,l} / R(z))^2 acting as an axial cutoff potential (large where the
    body is thin). j'_{0,1} = 0, so the m=0, l=1 branch is the pure
    area-profile horn. Assembled with linear finite elements (consistent
    mass), so a nonuniform s grid is handled directly.

    :param s: The (possibly nonuniform) axial coordinate grid
    :param R: The radius profile, sampled at the points in ``s``
    :param m: The azimuthal (Bessel) order of the branch
    :param l: The 1-indexed radial zero of the branch
    :return: Eigenvalues (ascending) and the matching envelopes A_n(s)
        (n=0 is the fundamental), as columns of the second return value
    """
    jp2 = jp_zero(m, l) ** 2
    n_s = len(s)
    K = np.zeros((n_s, n_s))  # int R^2 A'phi' + int j'^2 A phi
    M = np.zeros((n_s, n_s))  # int R^2 A phi  (RHS weight)

    for e in range(n_s - 1):
        h = s[e + 1] - s[e]
        w_bar = (R[e] ** 2 + R[e + 1] ** 2) / 2  # element-averaged R^2
        kd = w_bar / h  # stiffness int phi'phi'
        qd = jp2 * h / 6  # potential, consistent mass
        md = w_bar * h / 6  # RHS, consistent mass

        nodes = (e, e + 1)
        for a, ia in enumerate(nodes):
            for b, ib in enumerate(nodes):
                same = a == b
                K[ia, ib] += (kd if same else -kd) + qd * (2 if same else 1)
                M[ia, ib] += md * (2 if same else 1)

    eigenvalues, envelopes = sla.eigh(K, M)
    return eigenvalues, envelopes
