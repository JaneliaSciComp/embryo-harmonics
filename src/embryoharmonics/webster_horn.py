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
    area-profile horn. Solved with :func:`sturm_liouville`, so a nonuniform s
    grid is handled directly.

    :param s: The (possibly nonuniform) axial coordinate grid
    :param R: The radius profile, sampled at the points in ``s``
    :param m: The azimuthal (Bessel) order of the branch
    :param l: The 1-indexed radial zero of the branch
    :return: Eigenvalues (ascending) and the matching envelopes A_n(s)
        (n=0 is the fundamental), as columns of the second return value
    """
    w_bar = (R[:-1] ** 2 + R[1:] ** 2) / 2  # element-averaged R^2
    q = np.full_like(w_bar, jp_zero(m, l) ** 2)
    return sturm_liouville(s, w_bar, q, w_bar)


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

    free = slice(1, -1) if fixed_ends else slice(None)
    eigenvalues, free_envelopes = sla.eigh(
        K[free, free], M[free, free], subset_by_index=subset
    )
    envelopes = np.zeros((n_s, free_envelopes.shape[1]))
    envelopes[free] = free_envelopes
    return eigenvalues, envelopes
