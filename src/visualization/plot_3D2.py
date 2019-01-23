# ==================================================================================================
# IMPORTS
import numpy as np
import scipy as sc
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d
from numpy import sin, cos, tan, pi

from .. import mathematics as m

# ==================================================================================================
# FUNCTIONS


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Source:
    https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# --------------------------------------------------------------------------------------------------


def generate_blank_3D_plot(title=None, show_origin=True):

    # Create figure and apply defaults
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", proj_type="persp")

    # Add labels
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title(title)

    if show_origin:

        # Plot coordinate system
        ax.quiver(
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            color="black",
        )
        ax.scatter([0], [0], [0], color="red")

    return ax, fig


# --------------------------------------------------------------------------------------------------


def plot_macro_surface(macrosurface_aero_grid, ax):

    for surface_aero_grid in macrosurface_aero_grid:
        xx = surface_aero_grid["xx"]
        yy = surface_aero_grid["yy"]
        zz = surface_aero_grid["zz"]
        ax.plot_surface(xx, yy, zz)

    return ax


# --------------------------------------------------------------------------------------------------


def plot_macrosurface_aero_grid(
    macrosurface_aero_grid, ax, line_color="darkblue", alpha=1
):

    for surface_aero_grid in macrosurface_aero_grid:
        xx = surface_aero_grid["xx"]
        yy = surface_aero_grid["yy"]
        zz = surface_aero_grid["zz"]
        ax.plot_wireframe(xx, yy, zz, color=line_color, alpha=alpha)

    return ax


# --------------------------------------------------------------------------------------------------


def plot_structure(struct_elements, ax, show_nodes=True, line_color="k", alpha=1):

    for component_elements in struct_elements:
        for beam_element in component_elements:

            point_A = [
                beam_element.node_A.x,
                beam_element.node_A.y,
                beam_element.node_A.z,
            ]
            point_B = [
                beam_element.node_B.x,
                beam_element.node_B.y,
                beam_element.node_B.z,
            ]
            x = [point_A[0], point_B[0]]
            y = [point_A[1], point_B[1]]
            z = [point_A[2], point_B[2]]

            if show_nodes:
                ax.plot(x, y, z, color=line_color, marker="o", markersize=2, alpha=alpha)
            else:
                ax.plot(x, y, z, color=line_color, alpha=alpha)

    return ax


# --------------------------------------------------------------------------------------------------


def plot_deformed_structure(
    struct_elements,
    struct_deformations,
    ax,
    scale_factor=1,
    show_nodes=True,
    line_color="k",
    alpha=1,
):

    for component_elements in struct_elements:
        for beam_element in component_elements:

            node_A = beam_element.node_A
            node_B = beam_element.node_B

            point_A = [
                node_A.x + scale_factor * struct_deformations[node_A.number][0],
                node_A.y + scale_factor * struct_deformations[node_A.number][1],
                node_A.z + scale_factor * struct_deformations[node_A.number][2],
            ]

            point_B = [
                node_B.x + scale_factor * struct_deformations[node_B.number][0],
                node_B.y + scale_factor * struct_deformations[node_B.number][1],
                node_B.z + scale_factor * struct_deformations[node_B.number][2],
            ]

            x = [point_A[0], point_B[0]]
            y = [point_A[1], point_B[1]]
            z = [point_A[2], point_B[2]]

            if show_nodes:
                ax.plot(x, y, z, color=line_color, marker="o", markersize=2, alpha=alpha)
            else:
                ax.plot(x, y, z, color=line_color, alpha=alpha)

    return ax


# --------------------------------------------------------------------------------------------------


def generate_aircraft_grids_plot(
    aircraft_macrosurfaces_aero_grids,
    aircraft_struct_fem_elements,
    title=None,
    ax=None,
    fig=None,
    show_origin=True,
    show_nodes=False,
    line_color="k",
    alpha=1,
):

    if ax is None:
        ax, fig = generate_blank_3D_plot(title, show_origin)

    for macrosurface_aero_grid in aircraft_macrosurfaces_aero_grids:
        ax = plot_macrosurface_aero_grid(
            macrosurface_aero_grid,
            ax,
            line_color=line_color,
            alpha=alpha,
        )

    for macrosurface_struct_fem_elements in aircraft_struct_fem_elements[
        "macrosurfaces_fem_elements"
    ]:
        ax = plot_structure(macrosurface_struct_fem_elements, ax, show_nodes=show_nodes, line_color=line_color, alpha=alpha)

    ax = plot_structure(aircraft_struct_fem_elements["beams_fem_elements"], ax, show_nodes=show_nodes, line_color=line_color, alpha=alpha)

    set_axes_equal(ax)

    return ax, fig


# --------------------------------------------------------------------------------------------------


def generate_deformed_aircraft_grids_plot(
    aircraft_deformed_macrosurfaces_aero_grids,
    aircraft_struct_fem_elements,
    aircraft_struct_deformations,
    title=None,
    ax=None,
    fig=None,
    show_origin=True,
    show_nodes=False,
    line_color="k",
    alpha=1,
):

    if ax is None:
        ax, fig = generate_blank_3D_plot(title, show_origin)

    for macrosurface_aero_grid in aircraft_deformed_macrosurfaces_aero_grids:
        ax = plot_macrosurface_aero_grid(macrosurface_aero_grid, ax, line_color=line_color, alpha=alpha)

    for macrosurface_struct_fem_elements in aircraft_struct_fem_elements[
        "macrosurfaces_fem_elements"
    ]:

        ax = plot_deformed_structure(
            macrosurface_struct_fem_elements,
            aircraft_struct_deformations,
            ax,
            scale_factor=1,
            show_nodes=show_nodes, line_color=line_color, alpha=alpha
        )

    ax = plot_deformed_structure(
        aircraft_struct_fem_elements["beams_fem_elements"],
        aircraft_struct_deformations,
        ax,
        scale_factor=1,
        show_nodes=show_nodes, line_color=line_color, alpha=alpha
    )

    set_axes_equal(ax)

    return ax, fig

# --------------------------------------------------------------------------------------------------


def plot_results(
    aircraft_deformed_macrosurfaces_aero_grids,
    aircraft_struct_fem_elements,
    aircraft_struct_deformations,
    aircraft_panel_loads,
    results_string="delta_p_grid",
    title=None,
    colorbar_label="Delta Pressure [Pa]",
    ax=None,
    fig=None,
    show_origin=True,
    colormap="coolwarm",
):

    if ax is None:
        ax, fig = generate_blank_3D_plot(title, show_origin)

    # Configure Color Map
    min_result_value = 0
    max_result_value = 0

    for component_panel_loads in aircraft_panel_loads:

        grid_min = component_panel_loads[results_string].min()
        grid_max = component_panel_loads[results_string].max()

        if grid_min < min_result_value:
            min_result_value = grid_min

        if grid_max > max_result_value:
            max_result_value = grid_max

    norm = matplotlib.colors.Normalize(min_result_value, max_result_value)
    m = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    m.set_array([])

    fig.colorbar(m, shrink=0.5, aspect=5, label=colorbar_label)

    # Plot Aircraft Surfaces
    for i, deformed_macrosurfaces_aero_grid in enumerate(aircraft_deformed_macrosurfaces_aero_grids):

        results = aircraft_panel_loads[i][results_string]

        index = 0

        for deformed_surface_aero_grid in deformed_macrosurfaces_aero_grid:
            n_chord_panels = np.shape(deformed_surface_aero_grid["xx"])[0] - 1
            n_span_panels = np.shape(deformed_surface_aero_grid["xx"])[1] - 1

            results_slice = results[:, index : (index + n_span_panels)]
            index += n_span_panels

            xx = deformed_surface_aero_grid["xx"]
            yy = deformed_surface_aero_grid["yy"]
            zz = deformed_surface_aero_grid["zz"]
            fcolors = m.to_rgba(results_slice)
            surf = ax.plot_surface(
                xx,
                yy,
                zz,
                facecolors=fcolors,
                vmin=min_result_value,
                vmax=max_result_value,
                shade=False,
                linewidth=0.5,
                antialiased=False,
            )

    # Plot Aircraft Deformed Structure
    for macrosurface_struct_fem_elements in aircraft_struct_fem_elements[
        "macrosurfaces_fem_elements"
    ]:

        ax = plot_deformed_structure(
            macrosurface_struct_fem_elements,
            aircraft_struct_deformations,
            ax,
            scale_factor=1,
            show_nodes=False, line_color="k", alpha=1
        )

    ax = plot_deformed_structure(
        aircraft_struct_fem_elements["beams_fem_elements"],
        aircraft_struct_deformations,
        ax,
        scale_factor=1,
        show_nodes=False, line_color="k", alpha=1
    )

    set_axes_equal(ax)

    return ax, fig
