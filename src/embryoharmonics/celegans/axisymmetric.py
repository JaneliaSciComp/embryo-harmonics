"""Exact axisymmetric (2D meridian) reduction of the Laplace-Neumann
eigenproblem on a body-of-revolution embryo.

For a rotationally symmetric domain, separation of variables u(r, theta, z) =
v(r, z) * cos/sin(k * theta) reduces the 3D eigenproblem to a family of 2D
problems on the meridian half-plane, one per angular order k (see
:func:`embryoharmonics.fem.compute_mass_and_stiffness_axisymmetric`). The 2D
eigenvalues are exactly the 3D ones; each k > 0 mode is doubly degenerate
(cos/sin pair). Unlike the Webster-Horn approximation, no assumption on the
axial variation of the cross-section is made.
"""
import logging

import h5py
import numpy as np
import pyvista as pv
import scipy.sparse.linalg as spla
from netgen import occ
import ngsolve as ngs
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from embryoharmonics import io
from embryoharmonics.celegans.embryo_model import _to_vtk
from embryoharmonics.celegans.webster_horn_harmonics import orthonormalize_clusters
from embryoharmonics.fem import FemMatrices, compute_mass_and_stiffness_axisymmetric
from embryoharmonics.harmonics import Harmonics


_AXIS_TOL = 1e-8  # nodes with r < _AXIS_TOL * max(r) are on the axis

_logger = logging.getLogger(__name__)


class AxisymmetricHarmonics(Harmonics):
    """Harmonics of a body of revolution, represented on its 2D meridian mesh.

    Each entry corresponds to one 3D mode v(r, z) * cos/sin(k * theta); the
    stored field is the meridian part v, and ``angular_orders`` /
    ``trig_kinds`` record the angular factor. A 'sin' mode always directly
    follows its degenerate 'cos' partner (sharing field and eigenvalue).
    Use :meth:`to_full_3d` to reconstruct the modes as a standard
    :class:`Harmonics` basis on a 3D mesh.
    """
    def __init__(
            self,
            mesh: pv.UnstructuredGrid,
            harmonics,
            eigenvalues,
            angular_orders,
            trig_kinds
    ):
        """Initialize the axisymmetric harmonics.

        :param mesh: The 2D meridian mesh with points (r, 0, z)
        :param harmonics: The meridian fields, one 3D mode per row
        :param eigenvalues: The (3D) eigenvalues of the modes
        :param angular_orders: The angular order k of each mode
        :param trig_kinds: The angular factor of each mode, 'cos' or 'sin'
            ('cos' for k = 0)
        """
        super().__init__(mesh, harmonics, eigenvalues)
        self.angular_orders = np.asarray(angular_orders)
        self.trig_kinds = np.asarray(trig_kinds)


    @classmethod
    def compute(
            cls,
            mesh: pv.UnstructuredGrid,
            *,
            n: int = 10,
    ) -> "AxisymmetricHarmonics":
        """Compute the first n harmonics of the body of revolution described
        by the given meridian mesh.

        The angular orders k = 0, 1, ... are solved one by one until no
        further branch can contribute to the lowest n eigenvalues (branch
        minima increase with k, since the k^2/r^2 potential does). The last
        cos/sin pair is never split: if the n-th mode is the cos half of a
        pair, the sin partner is included as well (yielding n + 1 modes).

        :param mesh: The meridian triangle mesh with points (r, 0, z)
        :param n: The number of (3D) harmonics to compute
        :return: The axisymmetric harmonics, sorted by eigenvalue
        """
        _logger.info("Computing the first %d axisymmetric harmonics", n)
        on_axis = mesh.points[:, 0] < _AXIS_TOL * mesh.points[:, 0].max()
        free = np.flatnonzero(~on_axis)

        candidates = []
        expanded_count = 0
        k = 0
        while True:
            fem = compute_mass_and_stiffness_axisymmetric(mesh, angular_order=k)
            mass, stiffness = fem.mass, fem.stiffness
            if k > 0:
                # Modes with angular dependence vanish on the axis
                mass = mass[free][:, free]
                stiffness = stiffness[free][:, free]

            n_modes = min(n, mass.shape[0] - 2)
            if n_modes < n:
                _logger.warning(
                    "Mesh too coarse to compute %d modes per branch; capped at %d",
                    n, n_modes
                )
            eigenvalues, eigenvectors = spla.eigsh(
                A=stiffness, M=mass, k=n_modes, which='LM', sigma=0.0
            )

            fields = eigenvectors.T
            if k > 0:
                full_fields = np.zeros((n_modes, mesh.n_points))
                full_fields[:, free] = fields
                fields = full_fields

            for eigenvalue, field in zip(eigenvalues, fields):
                candidates.append({"eigenvalue": eigenvalue, "k": k, "field": field})
            expanded_count += n_modes if k == 0 else 2 * n_modes
            _logger.debug("Branch k=%d: lowest eigenvalue %g", k, eigenvalues[0])

            if expanded_count >= n:
                cutoff = sorted(
                    c["eigenvalue"] for c in candidates
                    for _ in range(1 if c["k"] == 0 else 2)
                )[n - 1]
                if eigenvalues[0] > cutoff:
                    break
            k += 1

        candidates.sort(key=lambda c: c["eigenvalue"])

        # Expand k > 0 candidates into adjacent cos/sin pairs and truncate,
        # never splitting a pair across the truncation boundary
        fields, eigenvalues, angular_orders, trig_kinds = [], [], [], []
        for candidate in candidates:
            trigs = ("cos",) if candidate["k"] == 0 else ("cos", "sin")
            for trig in trigs:
                fields.append(candidate["field"])
                eigenvalues.append(candidate["eigenvalue"])
                angular_orders.append(candidate["k"])
                trig_kinds.append(trig)
            if len(fields) >= n:
                break
        if len(fields) > n:
            _logger.info(
                "Extended to %d modes to keep the degenerate k=%d pair intact",
                len(fields), angular_orders[-1]
            )

        return cls(
            mesh,
            np.array(fields),
            np.array(eigenvalues),
            np.array(angular_orders),
            np.array(trig_kinds),
        )


    def to_full_3d(
            self,
            mesh: pv.UnstructuredGrid | None = None,
            *,
            mesh_size: float = 5.0,
    ) -> Harmonics:
        """Reconstruct the harmonics as fields on a 3D mesh of the body of
        revolution.

        Each mode is evaluated as v(r, z) * cos/sin(k * theta) at the 3D mesh
        nodes, mass-normalized, and degenerate cos/sin pairs are
        mass-orthonormalized.

        :param mesh: The 3D tetrahedral mesh to reconstruct the fields on. If
            None, the meridian mesh is revolved (see :func:`revolve_meridian`).
        :param mesh_size: The maximum mesh size of the revolved mesh (ignored
            if ``mesh`` is given)
        :return: The harmonics defined on the 3D mesh
        """
        if mesh is None:
            mesh = revolve_meridian(self.mesh, mesh_size=mesh_size)

        r = np.hypot(mesh.points[:, 0], mesh.points[:, 1])
        theta = np.arctan2(mesh.points[:, 1], mesh.points[:, 0])
        z = mesh.points[:, 2]

        # One vector-valued interpolation for all meridian fields at once;
        # nodes just outside the meridian polygon (surface faceting) fall back
        # to nearest-neighbor
        meridian_rz = self.mesh.points[:, [0, 2]]
        values = LinearNDInterpolator(meridian_rz, self._harmonics.T)(r, z)
        outside = np.isnan(values[:, 0])
        if np.any(outside):
            values[outside] = NearestNDInterpolator(meridian_rz, self._harmonics.T)(
                r[outside], z[outside]
            )

        angular = np.where(
            (self.trig_kinds == "sin")[:, None],
            np.sin(self.angular_orders[:, None] * theta),
            np.cos(self.angular_orders[:, None] * theta),
        )
        fields = values.T * angular

        mass = FemMatrices.compute_for(mesh, stiffness=False).mass
        # sparse-dense matmul spuriously raises div/overflow FP warnings on some
        # BLAS backends even though the (verified finite) result is unaffected.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            norms = np.sqrt(np.einsum("ij,ij->i", fields, (mass @ fields.T).T))
        fields /= norms[:, None]
        orthonormalize_clusters(fields, self.degenerate_clusters(), mass)

        return Harmonics(mesh, fields, self.eigenvalues)


    def nodal_counts(self, *, n_samples: int = 256, threshold: float = 0.05):
        """Count nodal crossings of the meridian fields along the radial and
        axial directions, yielding approximate "quantum numbers" (n_r, n_z)
        that complement the exact angular order k.

        Sign changes are counted along coordinate sample lines (axial lines at
        a few fractions of the local radius, radial lines at a few axial
        stations), taking the median over lines and ignoring values below
        ``threshold`` times the mode amplitude. The labels are exact for a
        cylinder, where the modes separate as f(r) * g(z); for a general
        profile the nodal lines need not align with coordinate lines, so they
        are a heuristic.

        :param n_samples: The number of samples per line
        :param threshold: The relative amplitude below which values are
            ignored when counting sign changes
        :return: Two integer arrays (radial counts, axial counts), one entry
            per mode; degenerate cos/sin partners share their counts
        """
        r, z = self.mesh.points[:, 0], self.mesh.points[:, 2]
        interpolator = LinearNDInterpolator(
            self.mesh.points[:, [0, 2]], self._harmonics.T
        )
        amplitudes = np.abs(self._harmonics).max(axis=1)

        # Local radius R(z) from binned maxima of the mesh node radii
        # ponytail: 50-bin max profile; exact boundary polyline if the counts
        # ever misbehave near strongly tapered ends
        bins = np.linspace(z.min(), z.max(), 51)
        bin_index = np.clip(np.digitize(z, bins) - 1, 0, 49)
        radius = np.zeros(50)
        np.maximum.at(radius, bin_index, r)
        centers = (bins[:-1] + bins[1:]) / 2

        def crossings(points_rz):
            values = interpolator(points_rz)
            counts = np.zeros(len(self), dtype=int)
            for j in range(len(self)):
                v = values[:, j]
                v = v[~np.isnan(v)]
                v = v[np.abs(v) > threshold * amplitudes[j]]
                counts[j] = np.count_nonzero(np.diff(np.sign(v)))
            return counts

        z_line = np.linspace(z.min(), z.max(), n_samples + 2)[1:-1]
        r_line = np.interp(z_line, centers, radius)
        axial = np.median([
            crossings(np.column_stack([fraction * r_line, z_line]))
            for fraction in (0.35, 0.55, 0.75)
        ], axis=0).astype(int)

        radial = np.median([
            crossings(np.column_stack([
                np.linspace(0, 0.95 * np.interp(z_station, centers, radius),
                            n_samples)[1:],
                np.full(n_samples - 1, z_station),
            ]))
            for z_station in z.min() + np.array([0.25, 0.4, 0.5, 0.6, 0.75])
            * (z.max() - z.min())
        ], axis=0).astype(int)

        return radial, axial


    def degenerate_clusters(self) -> list[list[int]]:
        """Group mode indices into degenerate clusters: each 'sin' mode joins
        the cluster of the immediately preceding 'cos' partner.

        :return: One list of mode indices per cluster, in eigenvalue order
        """
        clusters = []
        for index, trig in enumerate(self.trig_kinds):
            if trig == "sin":
                clusters[-1].append(index)
            else:
                clusters.append([index])
        return clusters


    def subset(self, indices) -> "AxisymmetricHarmonics":
        """Limit the harmonics to the given indices.

        :param indices: The harmonics to limit to
        :return: The subset of harmonics
        """
        return AxisymmetricHarmonics(
            self.mesh,
            self._harmonics[indices],
            self.eigenvalues[indices],
            self.angular_orders[indices],
            self.trig_kinds[indices],
        )


    def decompose(self, mesh_data):
        raise NotImplementedError(
            "Data lives in 3D; reconstruct the 3D basis with to_full_3d() first"
        )


    def compose(self, coefficients):
        raise NotImplementedError(
            "Data lives in 3D; reconstruct the 3D basis with to_full_3d() first"
        )


def revolve_meridian(
        mesh: pv.UnstructuredGrid,
        *,
        mesh_size: float = 5.0,
) -> pv.UnstructuredGrid:
    """Generate a 3D tetrahedral mesh by revolving a meridian mesh about the
    z-axis.

    The meridian boundary polyline is extracted from the mesh, so no embryo
    model is needed; the revolved domain matches the 2D FEM domain
    polygon-for-polygon.

    :param mesh: The meridian triangle mesh with points (r, 0, z)
    :param mesh_size: The maximum mesh size of the 3D mesh
    :return: The tetrahedral mesh of the body of revolution
    """
    edges = mesh.extract_feature_edges(
        boundary_edges=True, feature_edges=False,
        manifold_edges=False, non_manifold_edges=False,
    )
    loop = edges.strip().get_cell(0).points

    segments = [
        occ.Segment(occ.Pnt(*loop[i]), occ.Pnt(*loop[i + 1]))
        for i in range(len(loop) - 1)
    ]
    face = occ.Face(occ.Wire(segments))
    solid = face.Revolve(occ.Axis(occ.Pnt(0, 0, 0), occ.Z), 360)
    volume_mesh = occ.OCCGeometry(solid).GenerateMesh(maxh=mesh_size)
    return _to_vtk(ngs.Mesh(volume_mesh))


def save_axisymmetric_harmonics(
        file_name: str,
        time: int,
        harmonics: AxisymmetricHarmonics,
):
    """Store meridian mesh and axisymmetric harmonics for one time point in
    an HDF5 file, in the layout of :func:`embryoharmonics.io.save_time_point`
    plus per-mode angular labels.

    :param file_name: The name of the HDF5 file (must end with .h5)
    :param time: The time point to store the data under
    :param harmonics: The axisymmetric harmonics to store
    """
    io.save_time_point(file_name, time, harmonics.mesh, harmonics)
    key = f"{time:03d}"
    with h5py.File(file_name, "a") as h5file:
        h5file.create_dataset(
            f"/harmonics/{key}/angular_orders",
            data=harmonics.angular_orders.astype(np.int64)
        )
        h5file.create_dataset(
            f"/harmonics/{key}/trig_kinds",
            data=harmonics.trig_kinds.astype("S3")
        )


def load_axisymmetric_harmonics(
        file_name: str,
        time: int,
) -> AxisymmetricHarmonics:
    """Load the axisymmetric harmonics for one time point from an HDF5 file.

    :param file_name: The name of the HDF5 file written by
        :func:`save_axisymmetric_harmonics`
    :param time: The time point to load
    :return: The axisymmetric harmonics at the given time point
    """
    mesh = io.load_mesh(file_name, time)
    key = f"{time:03d}"
    with h5py.File(file_name, "r") as h5file:
        group = h5file[f"/harmonics/{key}"]
        eigenvalues = group["eigenvalues"][:]
        angular_orders = group["angular_orders"][:]
        trig_kinds = np.array([t.decode("ascii") for t in group["trig_kinds"][:]])
        names = sorted(name for name in group if name.startswith("harmonic_"))
        harmonics = np.vstack([group[name][:] for name in names])

    return AxisymmetricHarmonics(mesh, harmonics, eigenvalues, angular_orders, trig_kinds)
