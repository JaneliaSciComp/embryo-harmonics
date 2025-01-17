# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from typing import Tuple
import pyvista as pv
import numpy as np

# %%
mesh_earlier = pv.read( "/Users/innerbergerm/Data/worm-geometry/example-geometries/data_420.vtu")
mesh_later = pv.read( "/Users/innerbergerm/Data/worm-geometry/example-geometries/data_421.vtu")
pv.set_jupyter_backend('client')

# %%
# Visualize the meshes if desired
# mesh_earlier.plot()
# mesh_later.plot()

# %%
def smallest_containing_cylinder(mesh: pv.PolyData) -> Tuple[float, float, float]:
    """
    Find the smallest cylinder that contains the mesh.
    :param mesh: a general 3D mesh
    :return: a cylinder represented as a tuple (radius, z_min, z_max)
    """
    x, y, z = mesh.points[:, 0], mesh.points[:, 1], mesh.points[:, 2]
    r = np.sqrt(x**2 + y**2)
    return np.max(r), np.min(z), np.max(z)

cylinder_earlier = smallest_containing_cylinder(mesh_earlier)
cylinder_later = smallest_containing_cylinder(mesh_later)
containing_cylinder = (
    max(cylinder_earlier[0], cylinder_later[0]),
    min(cylinder_earlier[1], cylinder_later[1]),
    max(cylinder_earlier[2], cylinder_later[2])
)
print(f"Containing cylinder: z={containing_cylinder[1]:.2f} to {containing_cylinder[2]:.2f}, r={containing_cylinder[0]:.2f}")

# %%
def transform_mesh_to_cylinder(mesh: pv.PolyData, cylinder: Tuple[float, float, float]) -> None:
    """
    Transform the embryo mesh to the given cylinder.
    :param mesh: a mesh that already has kind of a cylindrical shape
    :param cylinder: a cylinder represented as a tuple (radius, z_min, z_max)
    """
    x, y, z = mesh.points[:, 0], mesh.points[:, 1], mesh.points[:, 2]
    _, _, _, _, z_min, z_max = mesh.bounds
    r_cyl_max, z_cyl_min, z_cyl_max = cylinder
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Stretch points in z direction
    z_factor = (z_cyl_max - z_cyl_min) / (z_max - z_min)
    mesh.points[:, 2] = (z - z_min) * z_factor + z_cyl_min

    # Stretch points in radial direction
    surface = mesh.extract_surface()
    surface_indices = mesh.surface_indices()
    r_factor = np.zeros_like(r)
    ray_start = mesh.points
    ray_end = np.column_stack([np.cos(theta) * r_cyl_max * 1.1, np.sin(theta) * r_cyl_max * 1.1, z])
    for i, _ in enumerate(r):
        if i in surface_indices:
            # Easy: just project to the cylinder surface
            r_factor[i] = r_cyl_max / r[i]
        else:
            # Hard: find the distance to the mesh surface and stretch accordingly
            point, _ = surface.ray_trace(ray_start[i], ray_end[i])
            r_surf = np.linalg.norm(point[:2])
            r_factor[i] = r[i] * r_cyl_max / r_surf

    mesh.points[:, :2] *= r_factor[:, None]

transform_mesh_to_cylinder(mesh_earlier, containing_cylinder)
transform_mesh_to_cylinder(mesh_later, containing_cylinder)



    
    


# %%
