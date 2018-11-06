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


# --------------------------------------------------------------------------------------------------


def plot_mesh(meshs):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", proj_type="persp")

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


# --------------------------------------------------------------------------------------------------


def plot_results(meshs, color_variable_vector):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", proj_type="persp")

    minn, maxx = (
        np.array(color_variable_vector).min(),
        np.array(color_variable_vector).max(),
    )
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    m.set_array([])

    fig.colorbar(m, shrink=0.5, aspect=5)

    for i, mesh in enumerate(meshs):
        xx = mesh["xx"]
        yy = mesh["yy"]
        zz = mesh["zz"]
        fcolors = m.to_rgba(color_variable_vector[i])
        surf = ax.plot_surface(
            xx,
            yy,
            zz,
            facecolors=fcolors,
            vmin=minn,
            vmax=maxx,
            shade=False,
            linewidth=0.5,
            antialiased=False,
        )

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


# --------------------------------------------------------------------------------------------------


def plot_structure(structure):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", proj_type="persp")

    for beam in structure.beams:
        point_A = structure.points[beam.point_A_index]
        point_B = structure.points[beam.point_B_index]
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


# --------------------------------------------------------------------------------------------------


def plot_aircraft(aircraft):

    # Create figure and apply defaults
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", proj_type="persp")

    # Add labels
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title(aircraft.name)

    # Define color map to be used. With this is possible to plot each component in a different color
    n_colors = len(aircraft.components)
    color_map = plt.get_cmap("tab20c")
    color_norm = matplotlib.colors.Normalize(vmin=0, vmax=n_colors-1)
    scalar_map = matplotlib.cm.ScalarMappable(norm=color_norm, cmap=color_map)

    # Plot coordinate system
    ax.quiver(
        [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], color="black"
    )
    ax.scatter([0], [0], [0], color="red")

    # Plot CG
    cg_x = aircraft.inertial_properties.cg_position[0]
    cg_y = aircraft.inertial_properties.cg_position[1]
    cg_z = aircraft.inertial_properties.cg_position[2]

    ax.scatter([cg_x], [cg_y], [cg_z], marker="D", color="black", s=50)

    # Plot Engine and Thrust Vector
    for engine in aircraft.engines:
        eng_x = engine.position[0]
        eng_y = engine.position[1]
        eng_z = engine.position[2]

        eng_t_x = engine.thrust_vector[0]
        eng_t_y = engine.thrust_vector[1]
        eng_t_z = engine.thrust_vector[2]

        ax.scatter([eng_x], [eng_y], [eng_z], marker="P", color="red", s=50)
        ax.quiver([eng_x], [eng_y], [eng_z], [eng_t_x], [eng_t_y], [eng_t_z], color="red")

    # Plot Aircraf Components
    for i, component in enumerate(aircraft.components):

        # Generate component mesh
        n_chord_panels = 5
        n_span_panels_list = [3 for i in range(len(component.surface_list))]
        chord_discretization = "linear"
        span_discretization_list = [
            "linear" for i in range(len(component.surface_list))
        ]
        torsion_function_list = ["linear" for i in range(len(component.surface_list))]

        component_mesh = component.create_mesh(
            n_chord_panels,
            n_span_panels_list,
            chord_discretization,
            span_discretization_list,
            torsion_function_list,
        )

        for mesh in component_mesh:
            xx = mesh["xx"]
            yy = mesh["yy"]
            zz = mesh["zz"]
            ax.plot_surface(xx, yy, zz, color=scalar_map.to_rgba(i), shade=False)

    # Fix axes scale
    set_axes_equal(ax)

    # Generate Plot
    plt.show(block=False)


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

