from dataclasses import dataclass

from numpy.typing import ArrayLike
import pyvista as pv


@dataclass(frozen=True)
class MeshData:
    """Node-wise data for a triangular/tetrahedral mesh.

    :param mesh: The triangular/tetrahedral mesh this data is associated with.
    :param name: The name of the data.
    :param data: The node-wise data.
    :raise ValueError: If the data length does not match the number of nodes
        in the mesh.
    """
    mesh: pv.UnstructuredGrid
    name: str
    data: ArrayLike

    def __post_init__(self):
        """Check that the data fits the mesh"""
        if len(self.data) != self.mesh.n_points:
            raise ValueError(f"Data length {len(self.data)} " \
                             f"does not match mesh size {self.mesh.n_points}")

    def resample(
            self,
            target: pv.UnstructuredGrid,
            *,
            project_outside_data: bool = False
    ):
        """Resample the data onto a target mesh.

        :param target: The target mesh to resample the data onto.
        :param project_outside_data: If True, data outside the target mesh will
            be projected back to the mesh.
        :return: A new MeshData object on the target mesh with the resampled data.
        """
        # Use pyvista's sampling to interpolate the data onto the target mesh
        self.mesh.point_data[self.name] = self.data
        result = target.sample(self.mesh, snap_to_closest_point=project_outside_data)
        interpolated_data = result.point_data[self.name]
        del result.point_data[self.name]

        # Return a new MeshData object with the resampled data
        return MeshData(target, self.name, interpolated_data)
