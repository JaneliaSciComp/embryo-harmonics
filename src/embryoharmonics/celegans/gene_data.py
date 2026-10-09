import functools
import logging
import os
from dataclasses import dataclass

import h5py
import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import scipy.sparse.linalg as spla
import pyvista as pv
import zarr

from embryoharmonics.fem import FemMatrices
from embryoharmonics.interpolation import HashableWrapper, interpolation_matrix
from embryoharmonics.mesh_data import MeshData


_logger = logging.getLogger(__name__)

# Gene data positions are in microns, the geometry in voxels of 0.1625 microns.
VOXEL_SIZE_UM = 0.1625
LOCATION_SCALE = 1 / VOXEL_SIZE_UM


def buffered_location_scale(buffer: list[float]) -> np.ndarray:
    """Per-axis (x, y, z) scale from microns to voxels, expanded by a buffer in
    percent: either one value for all directions or two values for the radial
    (x, y) and axial (z) direction.
    """
    if len(buffer) not in (1, 2):
        raise ValueError(f"Buffer must be one or two values, got {buffer}")
    radial, axial = buffer if len(buffer) == 2 else buffer * 2
    return (1 + np.array([radial, radial, axial]) / 100) / VOXEL_SIZE_UM


@dataclass
class GeneData:
    """Expression data for a gene at a single point in time.
    """
    name: str
    locations: np.ndarray
    activities: np.ndarray

    def __len__(self) -> int:
        return len(self.activities)


    def __post_init__(self):
        # locations.shape = (n, 3), activities.shape = (n,)
        if len(self.locations) != len(self.activities):
            raise ValueError("The number of locations and activities must be the same")


    def interpolate(self, mesh: pv.UnstructuredGrid) -> MeshData:
        """Interpolate the gene expression data onto the mesh.

        :param mesh: The mesh to interpolate the gene expression data onto
        :return: A :class:`MeshData` object containing the interpolated data
        """
        interpolated_data = _interpolate(mesh, self.locations, self.activities)
        return MeshData(mesh, self.name, interpolated_data)


    def as_point_cloud(self) -> pv.PointSet:
        """Convert the gene expression data to a point cloud.

        :return: A :class:`pyvista.PointSet` object representing the gene expression data.
        """
        point_cloud = pv.PointSet(self.locations)
        point_cloud[self.name] = self.activities
        return point_cloud


class GeneDataLoader:
    """Loader for gene expression data from an HDF5 file.
    """
    def __init__(self, path: str, location_scale: float | np.ndarray = LOCATION_SCALE):
        """Initialize the gene data loader.

        :param path: The path to the HDF5 file containing the gene expression data.
        :param location_scale: Factor (scalar or per axis) from the stored positions (microns) to the geometry units.
        """
        self.location_scale = location_scale
        self.h5file = h5py.File(path, "r")

        # Load all gene names
        data = self.h5file["geneact"]
        self._gene_to_index = _convert_raw_names(data["genes"][:])
        self._tissue_to_index = _convert_raw_names(data["tissue_name"][:])

        # Load all time steps
        time = data["timepoints"][0]
        self._time_to_index = {int(t): i for i, t in enumerate(time)}

        # Store handles to the gene expression data
        self._gene_activities = data["data"]
        self._tissues = data["tissue_id"]
        self._locations = data["XYZ"]


    @property
    def gene_names(self) -> list[str]:
        """Get the names of all genes in the HDF5 file.

        :return: The names of all genes
        """
        return list(self._gene_to_index.keys())


    @property
    def tissue_names(self) -> list[str]:
        """Get the names of all tissues in the HDF5 file.

        :return: The names of all tissues
        """
        return list(self._tissue_to_index.keys())


    @property
    def time_steps(self) -> list[int]:
        """Get the time steps in the HDF5 file.

        :return: The time steps in the HDF5 file
        """
        return list(self._time_to_index.keys())


    @property
    def n_genes(self) -> int:
        """Get the number of genes in the HDF5 file.

        :return: The number of genes in the HDF5 file
        """
        return len(self._gene_to_index)


    @property
    def n_tissues(self) -> int:
        """Get the number of tissues in the HDF5 file.

        :return: The number of tissues in the HDF5 file
        """
        return len(self._tissue_to_index)


    @property
    def n_time_steps(self) -> int:
        """Get the number of time steps in the HDF5 file.

        :return: The number of time steps in the HDF5 file
        """
        return len(self._time_to_index)


    @property
    def n_cells(self) -> int:
        """Get the number of cells in the HDF5 file.

        :return: The number of cells in the HDF5 file
        """
        return self._locations.shape[1]


    def load(
            self,
            gene_name: str,
            time_step: int,
            remove_nans: bool = True
    ) -> GeneData:
        """Load the gene expression data for the given gene.

        :param gene_name: The name of the gene to load.
        :param time_step: Which time step to load.
        :param remove_nans: Whether to remove NaN values from the gene data.
        :return: A :class:`GeneData` object containing the gene expression data.
        """
        _logger.info("Loading gene %s at time step %d from '%s'",
                     gene_name, time_step, self.h5file.filename)
        try:
            t = self._time_to_index[time_step]
        except KeyError as e:
            raise ValueError(f"Time step {time_step} not found in the HDF5 file") from e

        try:
            g = self._gene_to_index[gene_name]
        except KeyError as e:
            raise ValueError(f"Gene {gene_name} not found in the HDF5 file") from e

        locations = self._locations[t, :, :] * self.location_scale
        activities = self._gene_activities[t, g, :]

        if remove_nans:
            _logger.debug("Filtering NaN values from gene data %s at time step %d",
                          gene_name, time_step)
            locations, activities = _filter_nan_values(activities, locations)

        return GeneData(gene_name, locations, activities)


    def load_all(self, time_step: int) -> tuple[np.ndarray, np.ndarray]:
        """Load the cell locations (n_cells, 3) and the activities of all genes
        and tissues at once as a (n_genes + n_tissues, n_cells) matrix, rows
        ordered like ``gene_names + tissue_names``.
        """
        try:
            t = self._time_to_index[time_step]
        except KeyError as e:
            raise ValueError(f"Time step {time_step} not found in the HDF5 file") from e

        _logger.info("Loading all genes and tissues at time step %d from '%s'",
                     time_step, self.h5file.filename)
        activities = np.concatenate((self._gene_activities[t], self._tissues[:]))
        return self._locations[t] * self.location_scale, activities


    def load_tissue(
            self,
            tissue_name: str,
            time_step: int
    ) -> GeneData:
        """Load the data for the given tissue (where the activity is just 1 for
        cells in the tissue and 0 otherwise).

        :param tissue_name: The name of the tissue to load.
        :param time_step: Which time step to load.
        :return: A :class:`GeneData` object containing the tissue data.
        """
        try:
            t = self._time_to_index[time_step]
        except KeyError as e:
            raise ValueError(f"Time step {time_step} not found in the HDF5 file") from e

        try:
            tissue_index = self._tissue_to_index[tissue_name]
        except KeyError as e:
            raise ValueError(f"Tissue {tissue_name} not found in the HDF5 file") from e

        _logger.info("Loading tissue %s at time step %d from '%s'",
                     tissue_name, time_step, self.h5file.filename)
        return GeneData(tissue_name, self._locations[t, :, :] * self.location_scale,
                        self._tissues[tissue_index, :])


class ParquetGeneDataLoader:
    """Loader for gene expression data from a directory of parquet files
    (``genes.parquet``, ``lineages.parquet``, ``cpm.parquet``, ``xyz.parquet``,
    ``xyz-lineages.parquet``). Expression values (keyed by lineage and time
    point) are joined onto the cell positions by lineage name. The format has
    no tissue data, so ``tissue_names`` is empty.
    """
    def __init__(self, directory: str, location_scale: float | np.ndarray = LOCATION_SCALE):
        """Initialize the gene data loader.

        :param directory: The directory containing the parquet files.
        :param location_scale: Factor (scalar or per axis) from the stored positions (microns) to the geometry units.
        """
        self.location_scale = location_scale
        self.directory = directory
        path = functools.partial(os.path.join, directory)

        genes = pq.read_table(path("genes.parquet")).to_pydict()
        self._gene_to_index = dict(zip(genes["GE"], genes["i"]))

        # Expression and position tables use different lineage ID lists; build
        # a lookup from expression lineage ID to position lineage ID by name
        cpm_lineages = pq.read_table(path("lineages.parquet")).to_pydict()
        xyz_lineages = pq.read_table(path("xyz-lineages.parquet")).to_pydict()
        xyz_name_to_index = dict(zip(xyz_lineages["LI"], xyz_lineages["i"]))
        self._cpm_to_xyz_lineage = np.full(max(cpm_lineages["i"]) + 1, -1)
        for i, name in zip(cpm_lineages["i"], cpm_lineages["LI"]):
            self._cpm_to_xyz_lineage[i] = xyz_name_to_index.get(name, -1)

        self._xyz = pq.read_table(path("xyz.parquet"))
        self._cpm = ds.dataset(path("cpm.parquet"))

        # Time steps need both expression and position data
        cpm_times = set()
        for batch in self._cpm.scanner(columns=["TI"]).to_batches():
            cpm_times.update(np.unique(batch["TI"].to_numpy()).tolist())
        xyz_times = set(pc.unique(self._xyz["TI"]).to_pylist())
        self._time_steps = sorted(int(t) for t in cpm_times & xyz_times)

        # Expression matrix and locations of the most recently loaded time step
        self._cached_time_step = None
        self._cached_locations = None
        self._cached_activities = None


    @property
    def gene_names(self) -> list[str]:
        return list(self._gene_to_index.keys())


    @property
    def tissue_names(self) -> list[str]:
        return []


    @property
    def time_steps(self) -> list[int]:
        return list(self._time_steps)


    @property
    def n_genes(self) -> int:
        return len(self._gene_to_index)


    @property
    def n_tissues(self) -> int:
        return 0


    @property
    def n_time_steps(self) -> int:
        return len(self._time_steps)


    def _load_time_step(self, time_step: int) -> None:
        """Read all expression values and positions of a time step and arrange
        them like the HDF5 layout: a (gene, cell) matrix plus (cell, 3) locations.
        """
        if self._cached_time_step == time_step:
            return

        if time_step not in self._time_steps:
            raise ValueError(f"Time step {time_step} not found in '{self.directory}'")

        _logger.info("Reading time step %d from '%s'", time_step, self.directory)
        xyz = self._xyz.filter(pc.field("TI") == time_step)
        xyz_ids = xyz["iLI"].to_numpy()
        locations = np.stack([xyz[c].to_numpy() for c in ("LR", "DV", "AP")], axis=1) * self.location_scale

        cpm = self._cpm.to_table(filter=pc.field("TI") == time_step)
        xyz_id = self._cpm_to_xyz_lineage[cpm["iLI"].to_numpy()]
        # Lineages with expression may have no position at this time step
        row_of_xyz_id = np.full(max(xyz_ids.max(), xyz_id.max()) + 1, -1)
        row_of_xyz_id[xyz_ids] = np.arange(len(xyz_ids))
        rows = np.where(xyz_id >= 0, row_of_xyz_id[np.clip(xyz_id, 0, None)], -1)
        valid = rows >= 0
        if not valid.all():
            _logger.warning("Dropping %d expression values without cell position at time step %d",
                            (~valid).sum(), time_step)

        activities = np.full((max(self._gene_to_index.values()) + 1, len(xyz_ids)), np.nan,
                             dtype=np.float32)
        activities[cpm["iGE"].to_numpy()[valid], rows[valid]] = cpm["CPM_mean"].to_numpy()[valid]

        self._cached_time_step = time_step
        self._cached_locations = locations
        self._cached_activities = activities


    def load(
            self,
            gene_name: str,
            time_step: int,
            remove_nans: bool = True
    ) -> GeneData:
        """Load the gene expression data for the given gene; see
        :meth:`GeneDataLoader.load`. Cells without expression data get NaN.
        """
        try:
            g = self._gene_to_index[gene_name]
        except KeyError as e:
            raise ValueError(f"Gene {gene_name} not found in '{self.directory}'") from e

        self._load_time_step(time_step)
        locations = self._cached_locations
        activities = self._cached_activities[g]

        if remove_nans:
            locations, activities = _filter_nan_values(activities, locations)

        return GeneData(gene_name, locations, activities)


    def load_all(self, time_step: int) -> tuple[np.ndarray, np.ndarray]:
        """Load the cell locations (n_cells, 3) and the activities of all genes
        at once as a (n_genes, n_cells) matrix, rows ordered like ``gene_names``;
        see :meth:`GeneDataLoader.load_all`.
        """
        self._load_time_step(time_step)
        return self._cached_locations, self._cached_activities[list(self._gene_to_index.values())]


    def load_tissue(self, tissue_name: str, time_step: int) -> GeneData:
        raise ValueError(f"No tissue data in gene data '{self.directory}'")


class ZarrGeneDataLoader(ParquetGeneDataLoader):
    """Loader for gene expression data from a zarr store with a ``cpm``
    subgroup holding a ``cpm`` array (TI, LI, GE) and an ``xyz`` subgroup
    holding the cell positions in an ``xyz`` array (TI, CO, LI). Both subgroups
    have their own time point and lineage lists; expression values are joined
    onto the positions by time point and lineage name. The format has no tissue data, so ``tissue_names`` is empty.
    """
    def __init__(self, path: str, location_scale: float | np.ndarray = LOCATION_SCALE):
        """Initialize the gene data loader.

        :param path: The zarr store containing the expression data and cell positions.
        :param location_scale: Factor (scalar or per axis) from the stored positions (microns) to the geometry units.
        """
        self.location_scale = location_scale
        self.directory = path
        root = zarr.open_group(path, mode="r")
        cpm, xyz = root["cpm"], root["xyz"]

        self._gene_to_index = {name: i for i, name in enumerate(cpm["GE"][:].tolist())}

        # Build a lookup from position lineage ID to expression lineage ID by name
        cpm_lineage_to_index = {name: i for i, name in enumerate(cpm["LI"][:].tolist())}
        self._xyz_to_cpm_lineage = np.array(
            [cpm_lineage_to_index.get(name, -1) for name in xyz["LI"][:].tolist()])

        # Time steps need both expression and position data
        cpm_times, xyz_times = cpm["TI"][:].tolist(), xyz["TI"][:].tolist()
        self._cpm_time_to_index = {t: i for i, t in enumerate(cpm_times)}
        self._xyz_time_to_index = {t: i for i, t in enumerate(xyz_times)}
        self._time_steps = sorted(set(cpm_times) & set(xyz_times))

        # Positions are small; reorder the axes to (x, y, z) = (LR, DV, AP)
        axes = [xyz["CO"][:].tolist().index(c) for c in ("LR", "DV", "AP")]
        self._xyz = xyz["xyz"][:][:, axes, :]
        self._cpm_array = cpm["cpm"]
        self._cpm = None

        # Expression matrix and locations of the most recently loaded time step
        self._cached_time_step = None
        self._cached_locations = None
        self._cached_activities = None


    def _load_time_step(self, time_step: int) -> None:
        """Arrange the expression values and positions of a time step like the
        HDF5 layout: a (gene, cell) matrix plus (cell, 3) locations.
        """
        if self._cached_time_step == time_step:
            return

        if time_step not in self._time_steps:
            raise ValueError(f"Time step {time_step} not found in '{self.directory}'")

        if self._cpm is None:
            # ponytail: chunks span all time points, so reading one time point decompresses
            # the whole array anyway; keep it in memory (~15 GB), rechunk along TI if that's too much
            _logger.info("Reading all expression data from '%s'", self.directory)
            self._cpm = self._cpm_array[:]

        positions = self._xyz[self._xyz_time_to_index[time_step]].T
        has_position = ~np.isnan(positions).any(axis=1)
        locations = positions[has_position] * self.location_scale

        # Lineages with a position may have no expression data
        cpm_ids = self._xyz_to_cpm_lineage[has_position]
        has_cpm = cpm_ids >= 0
        activities = np.full((self.n_genes, len(cpm_ids)), np.nan, dtype=np.float32)
        activities[:, has_cpm] = self._cpm[self._cpm_time_to_index[time_step]][cpm_ids[has_cpm]].T

        self._cached_time_step = time_step
        self._cached_locations = locations
        self._cached_activities = activities


def open_gene_data_loader(
    path: str, location_scale: float | np.ndarray = LOCATION_SCALE
) -> GeneDataLoader | ParquetGeneDataLoader:
    """Open gene expression data in any format: a zarr store, a directory of
    parquet files or a single HDF5 file.
    """
    if os.path.normpath(path).endswith(".zarr"):
        loader = ZarrGeneDataLoader
    else:
        loader = ParquetGeneDataLoader if os.path.isdir(path) else GeneDataLoader
    return loader(path, location_scale)


def _convert_raw_names(raw_names):
    return {raw_names[:, i].astype(np.uint8).tobytes().decode('ascii').strip(): i
            for i in range(raw_names.shape[1])}


def _filter_nan_values(
        activities: np.ndarray,
        locations: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
    """Remove NaN values from the gene expression data.
    
    :param activities: The gene expression activities.
    :param locations: The 3D locations of the gene expression activities.
    :return: A tuple containing the filtered activities and locations.
    """
    not_nan = np.where(np.logical_not(np.isnan(activities)))[0]
    return locations[not_nan], activities[not_nan]


def _interpolate(
        mesh: pv.UnstructuredGrid,
        locations: np.ndarray,
        values: np.ndarray
) -> np.ndarray:
    """Interpolate the given point cloud data onto the mesh using the L2-orthogonal
    projection onto the lowest-order finite element space of the mesh. NaN values
    are ignored.

    :param mesh: The mesh to interpolate the data onto.
    :param locations: The locations of the data points (n, 3).
    :param values: The values of the data points (n,).
    :return: The interpolated data.
    """
    # Compute L2-orthogonal projection of the pointwise data onto the mesh; the
    # LU decomposition of the mass matrix is cached across calls
    mass_lu = _compute_lu(HashableWrapper(mesh, [mesh.points, mesh.cells]))
    rhs = interpolation_matrix(mesh, locations) @ np.nan_to_num(values)
    return mass_lu.solve(rhs)


@functools.lru_cache(maxsize=5)
def _compute_lu(
        mesh_wrapper: HashableWrapper,
) -> spla.splu:
    """Compute the LU decomposition of the mass matrix of the given mesh.
    """
    _logger.debug("Computing LU decomposition for mesh with %d points and %d cells",
                    mesh_wrapper.data.n_points, mesh_wrapper.data.n_cells)
    fem_matrices = FemMatrices.compute_for(mesh_wrapper.data, stiffness=False)
    _logger.debug("LU decomposition done")
    return spla.splu(fem_matrices.mass.tocsc())
