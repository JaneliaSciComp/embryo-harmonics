"""Webster-Horn approximation for the Neumann Laplacian on a body of revolution.

Ported from ``ShroffHarmonics.jl`` (``plot_harmonics.jl``, ``webster_envelopes``).
"""
import numpy as np
import scipy.linalg as sla
import scipy.special as sp
from scipy.integrate import quad


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


def slope_coupling(m: int, l: int) -> tuple[float, float]:
    """Return the coefficients (G1 / G0, G2 / G0) of the wall-slope terms of
    the (m, l) branch, with the transverse shape g(xi) = J_m(j'_{m,l} xi) and

        G0 = int g^2 xi,   G1 = int g g' xi^2 = g(1)^2 / 2 - G0,
        G2 = int g'^2 xi^3,

    all integrated over 0 <= xi <= 1 (see :func:`webster_envelopes`).

    :param m: The azimuthal (Bessel) order
    :param l: The 1-indexed radial zero
    :return: The coefficients of the cross term and the slope potential
    """
    jp = jp_zero(m, l)
    if jp == 0:
        return 0.0, 0.0  # constant transverse shape

    G0 = quad(lambda xi: sp.jv(m, jp * xi) ** 2 * xi, 0, 1)[0]
    G1 = sp.jv(m, jp) ** 2 / 2 - G0
    G2 = quad(lambda xi: (jp * sp.jvp(m, jp * xi)) ** 2 * xi ** 3, 0, 1)[0]
    return G1 / G0, G2 / G0


def webster_envelopes(
        s: np.ndarray,
        R: np.ndarray,
        m: int,
        l: int,
        *,
        corrected: bool = False,
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
    area-profile horn. Solved with :func:`sturm_liouville`, so a nonuniform s
    grid is handled directly.

    The transverse shape J_m(j' r / R(s)) also varies along s where the wall
    is sloped, which the plain equation neglects. With ``corrected``, the
    Rayleigh quotient of the ansatz is reduced exactly instead (Kantorovich),
    which adds the wall-slope terms

        -2 (G1/G0) R R' A A' + (G2/G0) R'^2 A^2

    to the energy density (see :func:`slope_coupling`). They matter where R'
    is large, i.e. near round caps, and vanish for the m=0, l=1 branch.

    :param s: The (possibly nonuniform) axial coordinate grid
    :param R: The radius profile, sampled at the points in ``s``
    :param m: The azimuthal (Bessel) order of the branch
    :param l: The 1-indexed radial zero of the branch
    :param corrected: Whether to include the wall-slope terms
    :return: Eigenvalues (ascending) and the matching envelopes A_n(s)
        (n=0 is the fundamental), as columns of the second return value
    """
    w_bar = (R[:-1] ** 2 + R[1:] ** 2) / 2  # element-averaged R^2
    q = np.full_like(w_bar, jp_zero(m, l) ** 2)
    if not corrected:
        return sturm_liouville(s, w_bar, q, w_bar)

    # ponytail: element-wise constant slope; the R'^2 potential is log-divergent
    # at a round tip, so its weight there depends (weakly) on the grid
    c1, c2 = slope_coupling(m, l)
    slope = np.diff(R) / np.diff(s)
    c = -c1 * (R[:-1] + R[1:]) / 2 * slope
    return sturm_liouville(s, w_bar, q + c2 * slope ** 2, w_bar, c)


def assemble_sturm_liouville(
        s: np.ndarray,
        p: np.ndarray,
        q: np.ndarray,
        w: np.ndarray,
        c: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the stiffness and mass matrices of the weak form solved by
    :func:`sturm_liouville` (same arguments), without boundary conditions.

    :return: The dense stiffness and mass matrices (n_s, n_s)
    """
    n_s = len(s)
    h = np.diff(s)
    K = np.zeros((n_s, n_s))
    M = np.zeros((n_s, n_s))

    e = np.arange(n_s - 1)
    for a, b, sign in ((0, 0, 1), (1, 1, 1), (0, 1, -1), (1, 0, -1)):
        mass_factor = 2 if a == b else 1
        np.add.at(K, (e + a, e + b), sign * p / h + q * h * mass_factor / 6)
        np.add.at(M, (e + a, e + b), w * h * mass_factor / 6)
    if c is not None:
        # int c (A phi' + A' phi) over an element only touches the diagonal
        np.add.at(K, (e, e), -c)
        np.add.at(K, (e + 1, e + 1), c)

    return K, M


def sturm_liouville(
        s: np.ndarray,
        p: np.ndarray,
        q: np.ndarray,
        w: np.ndarray,
        c: np.ndarray | None = None,
        *,
        fixed_ends: bool = False,
        subset: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the 1D Sturm-Liouville eigenproblem given by the weak form

        int p A' phi' + c (A phi' + A' phi) + q A phi = lambda int w A phi

    with natural (or, if ``fixed_ends``, homogeneous Dirichlet) conditions at
    the ends. The coefficients are constant per element, e.g. evaluated at the
    element midpoints, so they may be singular at the end nodes. Assembled with
    linear finite elements (consistent mass) on a possibly nonuniform grid.

    :param s: The grid nodes (n_s,)
    :param p: The stiffness coefficient per element (n_s - 1,)
    :param q: The potential per element (n_s - 1,)
    :param w: The weight of the right-hand side per element (n_s - 1,)
    :param c: The (symmetrized) first-order coupling per element (n_s - 1,);
        ``None`` means no coupling
    :param fixed_ends: Whether to impose A = 0 at both ends
    :param subset: If given, only compute the eigenpairs with these (inclusive,
        0-based) indices
    :return: Eigenvalues (ascending) and the matching eigenfunctions at the
        nodes, as columns of the second return value
    """
    K, M = assemble_sturm_liouville(s, p, q, w, c)
    free = slice(1, -1) if fixed_ends else slice(None)
    eigenvalues, free_envelopes = sla.eigh(
        K[free, free], M[free, free], subset_by_index=subset
    )
    envelopes = np.zeros((len(s), free_envelopes.shape[1]))
    envelopes[free] = free_envelopes
    return eigenvalues, envelopes
