import pyvista as pv

from embryoharmonics.common import get_harmonic_name


def plot_eigenfunction(
        pv_data: pv.UnstructuredGrid,
        k: int,
        cmap: str = 'turbo',
) -> pv.Plotter:
    """
    Plot the k-th eigenfunction of a given mesh.
    :param pv_data: The mesh data to plot
    :param k: The index of the eigenfunction to plot
    :param cmap: The colormap to use (default: 'turbo')
    :return: The plotter object
    """
    scalar_name = get_harmonic_name(k)

    camera = pv.Camera()
    camera.position = (-400.0, 400.0, -500.0)
    camera.focal_point = (100.0, 50.0, 5.0)

    axes = pv.Axes(show_actor=True, actor_scale=2.0, line_width=5)
    axes.origin = (3.0, 3.0, 3.0)

    plotter = pv.Plotter()
    plotter.camera = camera
    slice1 = pv_data.slice(normal=[1, 0, 0]).translate((400, 0, -200))
    plotter.add_mesh(slice1, scalars=scalar_name, cmap=cmap)
    slice2 = pv_data.slice(normal=[0, 1, 0]).translate((200, 0, -100))
    plotter.add_mesh(slice2, scalars=scalar_name, cmap=cmap)
    slice3 = pv_data.slice_along_axis(n=10, axis="z")
    plotter.add_mesh(slice3, scalars=scalar_name, cmap=cmap)
    return plotter
