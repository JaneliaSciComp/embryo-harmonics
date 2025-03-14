import logging
from dataclasses import dataclass

import h5py
import numpy as np
import scipy.sparse.linalg as spla
import pyvista as pv

from embryoharmonics.fem import FemMatrices
from embryoharmonics.mesh_data import MeshData


_logger = logging.getLogger(__name__)


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
        interpolator = PointInterpolator(mesh)
        interpolated_data = interpolator.interpolate(self.locations, self.activities)
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

        locations = self._locations[t, :, :]
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
        return GeneData(tissue_name, self._locations[t, :, :], self._tissues[tissue_index, :])


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


class PointInterpolator():
    """Interpolates pointwise data onto a mesh.
    """
    class LuDecompositionCache:
        """Cache for LU decompositions of the mass matrix based on the mesh size.
        """
        def __init__(self):
            self._cache = {}

        def get(self, mesh: pv.UnstructuredGrid) -> 'PointInterpolator':
            """Cache the LU decomposition of the mass matrix based on mesh
            properties that are unlikely to be the same for different meshes.
            """
            key = (mesh.n_points, mesh.n_cells, mesh.bounds)
            if key not in self._cache:
                fem_matrices = FemMatrices.compute_for(mesh, stiffness=False)
                self._cache[key] = spla.splu(fem_matrices.mass.tocsc())
            return self._cache[key]


    # Create a global cache instance
    lu_cache = LuDecompositionCache()


    def __init__(self, mesh: pv.UnstructuredGrid):
        """Initialize the point interpolator.

        :param mesh: The mesh to interpolate the data onto.
        """
        self.mesh = mesh
        self.mass_lu = PointInterpolator.lu_cache.get(mesh)


    def interpolate(self, locations: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Interpolate the data from the point cloud onto the mesh.

        :param locations: The locations of the data points (n, 3).
        :param values: The values of the data points (n,).
        :return: The interpolated data.
        """
        # Find containing cells, skip points outside the mesh
        containing_cells = self.mesh.find_containing_cell(locations)
        rhs = np.zeros(self.mesh.n_points)

        is_outside: np.ndarray = containing_cells == -1
        _logger.debug("Interpolating %d points, skipping %d points outside the mesh",
                      len(locations), np.sum(is_outside))

        locations = locations[~is_outside]
        values = values[~is_outside]
        containing_cells = containing_cells[~is_outside]

        # Compute L2-orthogonal projection of the pointwise data onto the mesh
        for loc, val, cell_id in zip(locations, values, containing_cells):
            cell = self.mesh.get_cell(cell_id).point_ids
            points = self.mesh.points[cell]
            element_matrix = np.vstack((points.T, np.ones(4)))
            b = np.append(loc, 1)
            rhs[cell] += val * np.linalg.solve(element_matrix, b)

        return self.mass_lu.solve(rhs)
