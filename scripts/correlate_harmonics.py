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
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

# %%
mesh_earlier = pv.read( "/Users/innerbergerm/Data/worm-geometry/example-geometries/data_420.vtu")
mesh_later = pv.read( "/Users/innerbergerm/Data/worm-geometry/example-geometries/data_421.vtu")
pv.set_jupyter_backend('client')
print(f"Earlier timestep bounds: {mesh_earlier.bounds}")
print(f"Later timestep bounds: {mesh_later.bounds}")

# %%
# Visualize the meshes if desired
# mesh_earlier.plot()
# mesh_later.plot()

# %%
def transform_mesh(
        source: pv.UnstructuredGrid,
        target: pv.UnstructuredGrid,
        *,
        fudge_factor: float = 1e-4,
) -> None:
    """
    Transform the source mesh to a target mesh (both are supposed to be embryo geometries).
    :param source: a mesh that is transformed in place to match the target geometry
    :param target: a second mesh with a kind of similar shape
    :param fudge_factor: a small number to make sure the transformed source stays inside the target
    """
    x, y, z = source.points[:, 0], source.points[:, 1], source.points[:, 2]

    # Stretch points in z direction (shifting by the center to respect the fudge factor)
    _, x_src_max, _, y_src_max, z_src_min, z_src_max = source.bounds
    _, x_trg_max, _, y_trg_max, z_trg_min, z_trg_max = target.bounds

    z_src_center = (z_src_min + z_src_max) / 2
    z_trg_center = (z_trg_min + z_trg_max) / 2
    z_factor = (z_trg_max - z_trg_min) * (1 - fudge_factor) / (z_src_max - z_src_min)
    source.points[:, 2] = (z - z_src_center) * z_factor + z_trg_center

    # Stretch points in radial direction
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    r_factor = np.ones_like(r)

    surface_src = source.extract_surface()
    surface_trg = target.extract_surface()
    surface_indices = source.surface_indices()

    r_max = np.max([np.sqrt(x_trg_max**2 + y_trg_max**2), np.sqrt(x_src_max**2 + y_src_max**2)])
    ray_start = np.column_stack([np.zeros_like(x), np.zeros_like(y), z])
    ray_end = np.column_stack([np.cos(theta) * r_max * 1.1, np.sin(theta) * r_max * 1.1, z])

    for i, r_current in enumerate(r):
        if np.isclose(r_current, 0):
            r_factor[i] = 1
            continue

        # Find the distance to the target surface
        point, _ = surface_trg.ray_trace(ray_start[i], ray_end[i])
        r_surf_trg = np.linalg.norm(point[:, :2])

        if i in surface_indices:
            # Easy: just project to the target surface
            r_factor[i] = r_surf_trg / r_current
        else:
            # Harder: find the distance to the source mesh surface and stretch accordingly
            point, _ = surface_src.ray_trace(ray_start[i], ray_end[i])
            r_surf_src = np.linalg.norm(point[:, :2])
            r_factor[i] = r_surf_trg / r_surf_src

    source.points[:, :2] *= r_factor[:, None] * (1 - fudge_factor)
    print(r_factor)

transform_mesh(mesh_earlier, mesh_later)
print(f"Earlier timestep bounds after transformation: {mesh_earlier.bounds}")
print(f"Later timestep bounds after transformation: {mesh_later.bounds}")

# %%
# Weed out some of the harmonics for testing purposes
names_to_keep = [f"harmonic_{i:03}" for i in range(100)]
for name in mesh_earlier.array_names:
    if name not in names_to_keep:
        mesh_earlier.point_data.remove(name)
        mesh_later.point_data.remove(name)
resampled = mesh_earlier.sample(mesh_later)

# %%
# Compare harmonics visually
p = pv.Plotter(shape=(1, 3))
name = "harmonic_013"
p.add_mesh(mesh_later, scalars=name)
p.subplot(0, 1)
p.add_mesh(resampled, scalars=name)
p.subplot(0, 2)
p.add_mesh(mesh_earlier, scalars=name)
p.link_views()
p.show()

# %%
# Compare harmonics numerically
# TODO: the dot product should feature the mass matrix; refactor so that mass is always available
dot_products = np.zeros((len(names_to_keep), len(names_to_keep)))
for i, name_i in enumerate(names_to_keep):
    for j, name_j in enumerate(names_to_keep):
        dot_products[i, j] = np.dot(resampled[name_i], mesh_earlier[name_j])
plt.imshow(dot_products)

# %%
# Find rearrangement of harmonics between time steps, e.g., 13 and 14 are swapped
# TODO: some harmonics have a different sign in different time steps -> invert coefficients!
row_max_idx = np.argmax(np.abs(dot_products), axis=1)
for i, idx in enumerate(row_max_idx):
    if i != idx:
        print(f"{i:03} -> {idx:03}")

# %%
