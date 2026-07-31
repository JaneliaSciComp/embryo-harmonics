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
) -> Harmonics:
    """Approximate the first ``n`` harmonics of a (rotationally symmetric)
    embryo mesh using the Webster-Horn approximation, reconstructed as fields
    on the given mesh.

    The Webster-Horn equation only separates modes by their (m, l) branch on
    the assumption of an exact body of revolution, so ``embryo_model`` should
    be one loaded with ``symmetric=True``.

    :param mesh: The embryo mesh to reconstruct the fields on (should be the
        mesh generated from ``embryo_model``)
    :param embryo_model: The (rotationally symmetric) embryo model providing
        the central axis and radius profile
    :param n: The number of harmonics to approximate
    :param m_max: The largest azimuthal (Bessel) order branch to consider
    :param l_max: The largest 1-indexed radial branch to consider, per order
    :param n_samples: The number of points to sample the axial profile at
    :return: The approximated harmonics, defined on ``mesh``
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

    return Harmonics(mesh, fields, eigenvalues)


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
    return candidates[:n]


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
