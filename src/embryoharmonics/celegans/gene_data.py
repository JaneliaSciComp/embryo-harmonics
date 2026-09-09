import functools
import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import scipy.sparse.linalg as spla
import pyvista as pv

from embryoharmonics.fem import FemMatrices
from embryoharmonics.mesh_data import MeshData


_logger = logging.getLogger(__name__)

# Gene data positions are in microns, the geometry in voxels of 0.1625 microns.
# The exact factor 1/0.1625 leaves many cells outside the geometry, since the
# gene data are smoothed in time and the geometries are not; 5 keeps almost all
# cells inside until that is resolved.
LOCATION_SCALE = 5.0


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
    def __init__(self, path: str):
        """Initialize the gene data loader.

        :param path: The path to the HDF5 file containing the gene expression data.
        """
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

        locations = self._locations[t, :, :] * LOCATION_SCALE
        activities = self._gene_activities[t, g, :]

        if remove_nans:
            _logger.debug("Filtering NaN values from gene data %s at time step %d",
                          gene_name, time_step)
            locations, activities = _filter_nan_values(activities, locations)

        return GeneData(gene_name, locations, activities)


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
        return GeneData(tissue_name, self._locations[t, :, :] * LOCATION_SCALE,
                        self._tissues[tissue_index, :])


class ParquetGeneDataLoader:
    """Loader for gene expression data from a directory of parquet files
    (``genes.parquet``, ``lineages.parquet``, ``cpm.parquet``, ``xyz.parquet``,
    ``xyz-lineages.parquet``). Expression values (keyed by lineage and time
    point) are joined onto the cell positions by lineage name. The format has
    no tissue data, so ``tissue_names`` is empty.
    """
    def __init__(self, directory: str):
        """Initialize the gene data loader.

        :param directory: The directory containing the parquet files.
        """
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
        locations = np.stack([xyz[c].to_numpy() for c in ("LR", "DV", "AP")], axis=1) * LOCATION_SCALE

        cpm = self._cpm.to_table(filter=pc.field("TI") == time_step)
        row_of_xyz_id = np.full(xyz_ids.max() + 1, -1)
        row_of_xyz_id[xyz_ids] = np.arange(len(xyz_ids))
        xyz_id = self._cpm_to_xyz_lineage[cpm["iLI"].to_numpy()]
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


    def load_tissue(self, tissue_name: str, time_step: int) -> GeneData:
        raise ValueError(f"No tissue data in parquet gene data '{self.directory}'")


def open_gene_data_loader(path: str) -> GeneDataLoader | ParquetGeneDataLoader:
    """Open gene expression data in either format: a directory of parquet
    files or a single HDF5 file.
    """
    return ParquetGeneDataLoader(path) if os.path.isdir(path) else GeneDataLoader(path)


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
    # Make locations, points, and cells hashable
    # This is some effort, but still a lot less than the cost of recomputing the
    # LU decomposition and the mesh query for each interpolation
    mesh_wrapper = HashableWrapper(mesh, [mesh.points, mesh.cells])
    locations_wrapper = HashableWrapper(locations, [locations])

    # Compute the LU decomposition of the mass matrix for orthogonal projection
    # Find containing cells, skip points outside the mesh
    mass_lu = _compute_lu(mesh_wrapper)
    barycentric, cell_indices = _compute_barycentric_coordinates(mesh_wrapper, locations_wrapper)

    is_outside = np.any(np.isnan(barycentric), axis=1)
    is_nan = np.isnan(values)
    _logger.debug("Interpolating %d points, skipping %d points outside the mesh and %d NaN values",
                    len(locations), np.sum(is_outside), np.sum(is_nan))

    skip = is_outside | is_nan
    values = values[~skip]
    barycentric = barycentric[~skip]
    cells = mesh.cell_connectivity.reshape(-1, 4)[cell_indices[~skip]]

    rhs = np.zeros(mesh.n_points)
    np.add.at(rhs, cells.ravel(), (values[:, None] * barycentric).ravel())

    # Compute L2-orthogonal projection of the pointwise data onto the mesh
    return mass_lu.solve(rhs)


class HashableWrapper:
    """A wrapper class to make various numpy arrays hashable for caching
    purposes.
    """
    def __init__(self, data: Any, arrays_to_hash: list[np.ndarray]):
        """Set up the wrapper to hash the given data.

        :param data: The data to wrap.
        :param arrays_to_hash: The arrays to compute the hash from.
        """
        self.data = data

        hash_accumulator = hashlib.md5(np.ascontiguousarray(arrays_to_hash[0]).data)
        for array in arrays_to_hash[1:]:
            hash_accumulator.update(np.ascontiguousarray(array).data)
        self.checksum = hash_accumulator.hexdigest()

    def __hash__(self):
        return hash(self.checksum)

    def __eq__(self, other):
        if isinstance(other, HashableWrapper):
            return np.array_equal(self.checksum, other.checksum)
        return False


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


@functools.lru_cache(maxsize=50)
def _compute_barycentric_coordinates(
        mesh_wrapper: HashableWrapper,
        locations_wrapper: HashableWrapper
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the barycentric coordinates of the given locations with respect
    to the given mesh and the indices of their containing cells. If the
    locations are outside the mesh, the barycentric coordinates are set to NaN.
    """
    mesh = mesh_wrapper.data
    locations = locations_wrapper.data
    n_locs = locations.shape[0]

    # Find containing cells, skip points outside the mesh
    _logger.debug("Finding containing cells for %d points in mesh with %d points and %d cells",
                    len(locations), mesh.n_points, mesh.n_cells)
    containing_cells = mesh.find_containing_cell(locations)

    is_outside: np.ndarray = containing_cells == -1
    _logger.debug("Interpolating %d points, skipping %d points outside the mesh",
                    len(locations), np.sum(is_outside))

    locations = locations[~is_outside]
    cell_indices = containing_cells[~is_outside]

    # Get points of the cells containing the points
    m = locations.shape[0]
    cells = mesh.cell_connectivity.reshape(-1, 4)[cell_indices]
    cell_points = mesh.points[cells.flatten()].reshape(-1, 4, 3)

    # Compute barycentric coordinates of the points in the cells by solving
    # a linear system of equations for each point
    element_matrices = np.concatenate((
        np.transpose(cell_points, (0, 2, 1)),
        np.ones((m, 1, 4))
    ), axis=1)
    b = np.concatenate((locations, np.ones((m, 1))), axis=1)
    b = b[:, :, None]

    barycentric = np.full((n_locs, 4), np.nan)
    barycentric[~is_outside] = np.linalg.solve(element_matrices, b).squeeze()
    return barycentric, containing_cells
