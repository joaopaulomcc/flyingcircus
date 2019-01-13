"""
visualization.py

Routines for results visualization

Author: João Paulo Monteiro Cruvinel da Costa
email: joaopaulomcc@gmail.com / joao.cruvinel@embraer.com.br
github: joaopaulomcc
"""
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
# Functions


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


def plot_surface(meshs):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", proj_type="persp")

    for mesh in meshs:
        xx = mesh["xx"]
        yy = mesh["yy"]
        zz = mesh["zz"]
        ax.plot_surface(xx, yy, zz)

    # Plot coordinate system
    ax.quiver(
        [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], color="black"
    )
    ax.scatter([0], [0], [0], color="red")

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    set_axes_equal(ax)
    plt.show(block=False)

    return ax, fig


# --------------------------------------------------------------------------------------------------


def plot_mesh(meshs):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", proj_type="ortho")

    for mesh in meshs:
        xx = mesh["xx"]
        yy = mesh["yy"]
        zz = mesh["zz"]
        ax.plot_wireframe(xx, yy, zz)

    # Plot coordinate system
    ax.quiver(
        [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], color="black"
    )
    ax.scatter([0], [0], [0], color="red")

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    set_axes_equal(ax)
    plt.show(block=False)

    return ax, fig
# --------------------------------------------------------------------------------------------------

def plot_surface_grid(surface_grid, matplotlib_axis, grid_color="blue"):

    ax = matplotlib_axis
    xx = surface_grid["xx"]
    yy = surface_grid["yy"]
    zz = surface_grid["zz"]

    ax.plot_wireframe(xx, yy, zz, alpha=0.85)

    return ax

# --------------------------------------------------------------------------------------------------


def plot_results(
    aircraft_aero_mesh, results_grids, title=None, label=None, colormap="coolwarm"
):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", proj_type="ortho")

    min_result_value = 0
    max_result_value = 0

    for grid in results_grids:

        grid_min = grid.min()
        grid_max = grid.max()

        if grid_min < min_result_value:
            min_result_value = grid_min

        if grid_max > max_result_value:
            max_result_value = grid_max

    norm = matplotlib.colors.Normalize(min_result_value, max_result_value)
    m = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    m.set_array([])

    fig.colorbar(m, shrink=0.5, aspect=5, label=label)

    for i, component_mesh in enumerate(aircraft_aero_mesh):
        results = results_grids[i]

        index = 0

        for surface in component_mesh:
            n_chord_panels = np.shape(surface["xx"])[0] - 1
            n_span_panels = np.shape(surface["xx"])[1] - 1

            results_slice = results[:, index : (index + n_span_panels)]
            index += n_span_panels

            xx = surface["xx"]
            yy = surface["yy"]
            zz = surface["zz"]
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

    # Plot coordinate system
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    maximum = np.array([x_limits, y_limits, z_limits]).max()
    quiver_length = maximum / 10
    ax.quiver(
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [quiver_length, 0, 0],
        [0, quiver_length, 0],
        [0, 0, quiver_length],
        color="black",
    )
    ax.scatter([0], [0], [0], color="red")

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title(title)
    set_axes_equal(ax)
    plt.show(block=False)

    return ax, fig


# --------------------------------------------------------------------------------------------------


def plot_node(Node, plot_axis):

    x = Node.x
    y = Node.y
    z = Node.z
    x_axis = Node.x_axis
    y_axis = Node.y_axis
    z_axis = Node.z_axis

    plot_axis.quiver([x], [y], [z], [x_axis[0]], [x_axis[1]], [x_axis[2]], color="red")
    plot_axis.quiver(
        [x], [y], [z], [y_axis[0]], [y_axis[1]], [y_axis[2]], color="green"
    )
    plot_axis.quiver([x], [y], [z], [z_axis[0]], [z_axis[1]], [z_axis[2]], color="blue")

    plot_axis.scatter([x], [y], [z], color="black")


# --------------------------------------------------------------------------------------------------


def plot_structure(struct_elements):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", proj_type="persp")

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
            ax.plot(x, y, z, "-ko", markersize=2)

    # Plot coordinate system
    ax.quiver(
        [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], color="black"
    )
    ax.scatter([0], [0], [0], color="red")

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    set_axes_equal(ax)
    plt.show(block=False)

    return ax, fig

# --------------------------------------------------------------------------------------------------

def plot_beam_elements(struct_elements, matplotlib_axis):

    ax = matplotlib_axis

    for beam_element in struct_elements:

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
        ax.plot(x, y, z, "-ko", linewidth=5, markersize=2)

    return ax

# --------------------------------------------------------------------------------------------------


def plot_deformed_structure(struct_elements, node_vector, deformations, scale_factor=1):

    ax, fig = plot_structure(struct_elements)

    deformed_grid = []
    for i, node in enumerate(node_vector):
        deformed_grid.append(
            [
                node.x + scale_factor * deformations[i][0],
                node.y + scale_factor * deformations[i][1],
                node.z + scale_factor * deformations[i][2],
                deformations[i][3],
                deformations[i][4],
                deformations[i][5],
            ]
        )

    for component_elements in struct_elements:
        for beam_element in component_elements:

            point_A = [
                deformed_grid[beam_element.node_A.number][0],
                deformed_grid[beam_element.node_A.number][1],
                deformed_grid[beam_element.node_A.number][2],
            ]

            point_B = [
                deformed_grid[beam_element.node_B.number][0],
                deformed_grid[beam_element.node_B.number][1],
                deformed_grid[beam_element.node_B.number][2],
            ]

            x = [point_A[0], point_B[0]]
            y = [point_A[1], point_B[1]]
            z = [point_A[2], point_B[2]]
            ax.plot(x, y, z, "-ro", markersize=2)

    set_axes_equal(ax)
    plt.show(block=False)

    return ax, fig


# --------------------------------------------------------------------------------------------------


def plot_aircraft(aircraft, title=None):

    # Create figure and apply defaults
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", proj_type="persp")

    # Add labels
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title(aircraft.name)

    # Define color map to be used. With this is possible to plot each component in a different color
    n_colors = len(aircraft.macrosurfaces)
    color_map = plt.get_cmap("tab20c")
    color_norm = matplotlib.colors.Normalize(vmin=0, vmax=n_colors - 1)
    scalar_map = matplotlib.cm.ScalarMappable(norm=color_norm, cmap=color_map)

    """
    # Plot coordinate system
    ax.quiver(
        [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], color="black"
    )
    ax.scatter([0], [0], [0], color="red")
    """

    # Plot CG
    if aircraft.inertial_properties:
        cg_x = aircraft.inertial_properties.position[0]
        cg_y = aircraft.inertial_properties.position[1]
        cg_z = aircraft.inertial_properties.position[2]

        ax.scatter([cg_x], [cg_y], [cg_z], marker="D", color="black", s=50)

    # Plot Engine and Thrust Vector
    if aircraft.engines:
        for engine in aircraft.engines:
            eng_x = engine.position[0]
            eng_y = engine.position[1]
            eng_z = engine.position[2]

            eng_t_x = engine.thrust_vector[0]
            eng_t_y = engine.thrust_vector[1]
            eng_t_z = engine.thrust_vector[2]

            ax.scatter([eng_x], [eng_y], [eng_z], marker="P", color="red", s=50)
            ax.quiver(
                [eng_x], [eng_y], [eng_z], [eng_t_x], [eng_t_y], [eng_t_z], color="red"
            )

    # Plot Aircraf Components
    for i, macrosurface in enumerate(aircraft.macrosurfaces):

        # Generate component mesh
        n_chord_panels = 5
        n_span_panels_list = [3 for i in range(len(macrosurface.surface_list))]
        n_beam_elements_list = [3 for i in range(len(macrosurface.surface_list))]
        chord_discretization = "linear"
        span_discretization_list = [
            "linear" for i in range(len(macrosurface.surface_list))
        ]
        torsion_function_list = [
            "linear" for i in range(len(macrosurface.surface_list))
        ]

        macrosurface_aero_grid, macrosurface_nodes_list = macrosurface.create_grids(
            n_chord_panels,
            n_span_panels_list,
            n_beam_elements_list,
            chord_discretization,
            span_discretization_list,
            torsion_function_list,
        )

        for mesh in macrosurface_aero_grid:
            xx = mesh["xx"]
            yy = mesh["yy"]
            zz = mesh["zz"]
            ax.plot_surface(
                xx, yy, zz, color=scalar_map.to_rgba(i), shade=False, alpha=0.85
            )

        for surface_node_list in macrosurface_nodes_list:
            x = []
            y = []
            z = []

            for node in surface_node_list:
                x.append(node.xyz[0])
                y.append(node.xyz[1])
                z.append(node.xyz[2])

            ax.plot(x, y, z, c="black")

        # Plot aircraft beams
        if aircraft.beams:
            for beam in aircraft.beams:
                x = np.array([beam.root_point[0], beam.tip_point[0]])
                y = np.array([beam.root_point[1], beam.tip_point[1]])
                z = np.array([beam.root_point[2], beam.tip_point[2]])

                if beam.ElementProperty.material == "rigid_connection":
                    ax.plot(x, y, z, c="blue", ls="--")
                else:
                    ax.plot(x, y, z, c="black")

    # Fix axes scale
    set_axes_equal(ax)

    # Generate Plot
    plt.show()

    # Write plot title
    if title:
        ax.set_title(title)

    return ax, fig

# --------------------------------------------------------------------------------------------------


def generate_blank_3D_plot(title=None, plot_origin=True):

    # Create figure and apply defaults
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", proj_type="persp")

    # Add labels
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title(title)

    if plot_origin:

        # Plot coordinate system
        ax.quiver(
            [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], color="black"
        )
        ax.scatter([0], [0], [0], color="red")

    return ax, fig

# --------------------------------------------------------------------------------------------------

def plot_aircraft_grids(aircraft_grids, aircraft_fem_elements, title=None):

    ax, fig = generate_blank_3D_plot(title)

    # Plot aerodynamic grids
    for macrosurface_aero_grid in aircraft_grids["macrosurfaces_aero_grids"]:

        for surface_aero_grid in macrosurface_aero_grid:

            ax = plot_surface_grid(surface_aero_grid, ax, grid_color="blue")

    # Plot macrosurfaces structures
    for macrosurface_fem_elements in aircraft_fem_elements["macrosurfaces_fem_elements"]:

        for surface_fem_elements in macrosurface_fem_elements:

            ax = plot_beam_elements(surface_fem_elements, ax)


    # Plot aircraft beams structure
    if aircraft_fem_elements["beams_fem_elements"]:

        for beam_fem_elements in aircraft_fem_elements["beams_fem_elements"]:

            ax = plot_beam_elements(beam_fem_elements, ax)

    set_axes_equal(ax)

    return ax, fig

# --------------------------------------------------------------------------------------------------


def plot_deformation(elements, nodes, deformations, scale=1):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", proj_type="persp")

    # Original Structure
    for element in elements:
        point_A = nodes[element.point_A_index]
        point_B = nodes[element.point_B_index]
        x = [point_A[0], point_B[0]]
        y = [point_A[1], point_B[1]]
        z = [point_A[2], point_B[2]]
        ax.plot(x, y, z, "--ko", markersize=2)

    # Deformed Structure
    deformations = np.delete(deformations, [3, 4, 5], axis=1)
    deformed_nodes = nodes + deformations * scale

    for element in elements:
        point_A = deformed_nodes[element.point_A_index]
        point_B = deformed_nodes[element.point_B_index]
        x = [point_A[0], point_B[0]]
        y = [point_A[1], point_B[1]]
        z = [point_A[2], point_B[2]]
        ax.plot(x, y, z, "-ro", markersize=2)

    # Plot coordinate system
    ax.quiver(
        [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], color="black"
    )
    ax.scatter([0], [0], [0], color="red")

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    set_axes_equal(ax)
    plt.show(block=False)

    return ax, fig
