"""Reconstruct 3D harmonic fields on an embryo mesh from the Webster-Horn
approximation, so they can be compared with :class:`Harmonics` computed
directly on the mesh (same mesh, same mass-normalization convention).
"""
import logging

import numpy as np
import pyvista as pv
from scipy.integrate import cumulative_trapezoid
from scipy.special import jv

from embryoharmonics.celegans.embryo_model import EmbryoModel
from embryoharmonics.fem import FemMatrices
from embryoharmonics.harmonics import Harmonics
from embryoharmonics.webster_horn import jp_zero, webster_envelopes


_logger = logging.getLogger(__name__)


def compute_webster_horn_harmonics(
        mesh: pv.UnstructuredGrid,
        embryo_model: EmbryoModel,
        *,
        n: int = 10,
        m_max: int = 4,
        l_max: int = 6,
        n_samples: int = 200,
) -> tuple[Harmonics, list[tuple[int, int, int]]]:
    """Approximate the first ``n`` harmonics of a (rotationally symmetric)
    embryo mesh using the Webster-Horn approximation, reconstructed as fields
    on the given mesh.

    The Webster-Horn equation only separates modes by their (m, l) branch on
    the assumption of an exact body of revolution, so ``embryo_model`` should
    be one loaded with ``symmetric=True``.

    :param mesh: The embryo mesh to reconstruct the fields on (usually the
        mesh generated from ``embryo_model``, but any mesh occupying roughly
        the same region works, since the modes are evaluated analytically)
    :param embryo_model: The (rotationally symmetric) embryo model providing
        the central axis and radius profile
    :param n: The number of harmonics to approximate
    :param m_max: The largest azimuthal (Bessel) order branch to consider
    :param l_max: The largest 1-indexed radial branch to consider, per order
    :param n_samples: The number of points to sample the axial profile at
    :return: The approximated harmonics defined on ``mesh``, and their
        ``(m, l, n)`` mode labels (azimuthal order, radial branch, axial
        index). Modes sharing a label form a degenerate cluster.
    """
    s_samples, R_samples, t_samples, z_samples = _axial_profile(embryo_model, n_samples)
    candidates = _select_modes(s_samples, R_samples, n, m_max, l_max)

    s_node, r_node, theta_node, R_node = _node_axial_coordinates(
        mesh, embryo_model, t_samples, s_samples, R_samples, z_samples
    )

    mass = FemMatrices.compute_for(mesh, stiffness=False).mass

    fields = np.empty((len(candidates), mesh.n_points))
    eigenvalues = np.empty(len(candidates))
    for i, candidate in enumerate(candidates):
        field = _reconstruct_field(candidate, s_samples, s_node, r_node, theta_node, R_node)
        # sparse-dense matmul spuriously raises div/overflow FP warnings on some
        # BLAS backends even though the (verified finite) result is unaffected.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            field /= np.sqrt(field @ (mass @ field))
        fields[i] = field
        eigenvalues[i] = candidate["eigenvalue"]

    labels = [(c["m"], c["l"], c["n"]) for c in candidates]

    # The cos/sin pair of a degenerate cluster is mass-orthogonal on an exactly
    # rotationally symmetric domain, but a tetrahedral mesh never is, so
    # orthonormalize explicitly. Any rotation within a degenerate eigenspace is
    # still an eigenbasis, so this costs nothing and lets projections onto a
    # cluster be computed as a plain sum of squared coefficients.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for indices in degenerate_clusters(labels):
            for position, i in enumerate(indices):
                for j in indices[:position]:
                    fields[i] -= (fields[i] @ (mass @ fields[j])) * fields[j]
                fields[i] /= np.sqrt(fields[i] @ (mass @ fields[i]))

    return Harmonics(mesh, fields, eigenvalues), labels


def degenerate_clusters(
        labels: list[tuple[int, int, int]]
) -> list[list[int]]:
    """Group mode indices by their ``(m, l, n)`` label.

    Modes with the same label are degenerate: for ``m > 0`` the ``cos(m*theta)``
    and ``sin(m*theta)`` modes share an eigenvalue, so any rotation of the pair
    is an equally valid eigenbasis. Only quantities computed per cluster are
    well defined; individual modes within a cluster are not.

    :param labels: The mode labels, as returned by
        :func:`compute_webster_horn_harmonics`
    :return: One list of mode indices per cluster, in ascending eigenvalue order
    """
    clusters = {}
    for index, label in enumerate(labels):
        clusters.setdefault(label, []).append(index)
    return sorted(clusters.values(), key=lambda indices: indices[0])


def match_modes(
        coefficients: np.ndarray,
        clusters: list[list[int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Pair every target mode one-to-one with the approximate eigenfunction it
    matches best, and return that scalar product.

    ``coefficients[i, j]`` is the mass-weighted scalar product of approximate
    mode ``i`` with target mode ``j``, i.e. the output of
    :meth:`Harmonics.decompose` stacked column-wise, with both bases
    mass-normalized.

    For a degenerate cluster ``C`` there is no single approximate mode to pair
    with: for m > 0 the pair ``cos(m*theta)``, ``sin(m*theta)`` spans a 2D
    eigenspace, and every rotation ``cos(m*(theta - phi))`` in it is equally an
    eigenfunction -- the phase is gauge, not part of the approximation. The
    best-matching eigenfunction of that cluster is therefore the normalized
    projection ``P_C u / ||P_C u||``, whose scalar product with ``u`` is
    ``||P_C u|| = sqrt(sum_i in C  coefficients[i, j]^2)``. For a
    non-degenerate cluster this reduces to the plain scalar product
    ``|coefficients[i, j]|``.

    :param coefficients: The scalar products of the target basis with respect to
        the approximate basis
    :param clusters: The degenerate clusters of the approximate basis, from
        :func:`degenerate_clusters`
    :return: For each target mode, the index of the matched cluster and the
        scalar product with the best eigenfunction in it (1 means the target
        mode is reproduced exactly)
    """
    # Scalar product of every target mode with the best eigenfunction of every
    # cluster, i.e. the norm of its projection onto that cluster's eigenspace.
    scores = np.array([
        np.linalg.norm(coefficients[indices, :], axis=0) for indices in clusters
    ])
    matched = scores.argmax(axis=0)
    return matched, scores[matched, np.arange(coefficients.shape[1])]


def _axial_profile(
        embryo_model: EmbryoModel,
        n_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample the axial coordinate (arclength of the central spline) and
    radius profile (distance of the 0-th transverse spline from the central
    axis) of an embryo model.

    :return: Arclength ``s``, radius profile ``R``, spline domain samples
        ``t``, and the corresponding z-coordinates of the central spline
    """
    domain = embryo_model.spline_domain
    t_samples = np.linspace(domain[0], domain[-1], n_samples)

    central = embryo_model.central_spline(t_samples)
    reference = embryo_model.transverse_splines[0](t_samples)

    tangent = embryo_model.central_spline.derivative()(t_samples)
    speed = np.linalg.norm(tangent, axis=1)
    s_samples = cumulative_trapezoid(speed, t_samples, initial=0.0)

    R_samples = np.linalg.norm(reference - central, axis=1)
    z_samples = central[:, 2]

    return s_samples, R_samples, t_samples, z_samples


def _select_modes(
        s_samples: np.ndarray,
        R_samples: np.ndarray,
        n: int,
        m_max: int,
        l_max: int
) -> list[dict]:
    """Solve the Webster-Horn equation for a pool of (m, l) branches and pick
    the lowest ``n`` modes overall, expanding m > 0 branches into their
    degenerate cos/sin pair.
    """
    candidates = []
    for m in range(m_max + 1):
        for l in range(1, l_max + 1):
            eigenvalues, envelopes = webster_envelopes(s_samples, R_samples, m, l)
            jp = jp_zero(m, l)

            for idx in range(min(n, len(eigenvalues))):
                trig_kinds = ("none",) if m == 0 else ("cos", "sin")
                for trig in trig_kinds:
                    candidates.append({
                        "eigenvalue": eigenvalues[idx],
                        "m": m,
                        "l": l,
                        "n": idx,
                        "jp": jp,
                        "envelope": envelopes[:, idx],
                        "trig": trig,
                    })

    candidates.sort(key=lambda c: c["eigenvalue"])
    n_available = len(candidates)
    if n_available < n:
        _logger.warning(
            "Only found %d Webster-Horn candidate modes (requested %d); "
            "consider increasing m_max/l_max", n_available, n
        )

    # Never split a degenerate cos/sin pair across the truncation boundary:
    # a half-cluster would make cluster-wise comparisons meaningless.
    selected = candidates[:n]
    label = lambda c: (c["m"], c["l"], c["n"])
    while len(selected) < n_available and label(candidates[len(selected)]) == label(selected[-1]):
        selected.append(candidates[len(selected)])
    if len(selected) > n:
        _logger.info(
            "Extended to %d modes to keep the degenerate cluster %s intact",
            len(selected), label(selected[-1])
        )
    return selected


def _node_axial_coordinates(
        mesh: pv.UnstructuredGrid,
        embryo_model: EmbryoModel,
        t_samples: np.ndarray,
        s_samples: np.ndarray,
        R_samples: np.ndarray,
        z_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the (arclength, radius, angle, local worm radius) cylindrical
    coordinates of every mesh node relative to the embryo model's central
    axis.
    """
    t_node = np.interp(mesh.points[:, 2], z_samples, t_samples)
    central_node = embryo_model.central_spline(t_node)

    radial = mesh.points[:, :2] - central_node[:, :2]
    r_node = np.linalg.norm(radial, axis=1)
    theta_node = np.arctan2(radial[:, 1], radial[:, 0])

    s_node = np.interp(t_node, t_samples, s_samples)
    R_node = np.interp(t_node, t_samples, R_samples)

    return s_node, r_node, theta_node, R_node


def _reconstruct_field(
        candidate: dict,
        s_samples: np.ndarray,
        s_node: np.ndarray,
        r_node: np.ndarray,
        theta_node: np.ndarray,
        R_node: np.ndarray,
) -> np.ndarray:
    """Reconstruct a single Webster-Horn mode as a field on the mesh nodes:
    ``A(s) * J_m(j'_{m,l} * r / R(s)) * cos_or_sin(m * theta)``.
    """
    amplitude = np.interp(s_node, s_samples, candidate["envelope"])

    bessel_argument = np.divide(
        candidate["jp"] * r_node, R_node,
        out=np.zeros_like(r_node), where=R_node > 1e-12
    )
    profile = jv(candidate["m"], bessel_argument)

    match candidate["trig"]:
        case "cos":
            angular = np.cos(candidate["m"] * theta_node)
        case "sin":
            angular = np.sin(candidate["m"] * theta_node)
        case _:
            angular = 1.0

    return amplitude * profile * angular
