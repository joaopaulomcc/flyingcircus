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
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Source:
    https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

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
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# --------------------------------------------------------------------------------------------------


def plot_surface(meshs):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type="persp")

    for mesh in meshs:
        xx = mesh[0]
        yy = mesh[1]
        zz = mesh[2]
        ax.plot_surface(xx, yy, zz)

    # Plot coordinate system
    ax.quiver([0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], color="black")
    ax.scatter([0], [0], [0], color="red")

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    set_axes_equal(ax)
    plt.show(block=False)

# --------------------------------------------------------------------------------------------------


def plot_mesh(meshs): 

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type="persp")

    for mesh in meshs:
        xx = mesh[0]
        yy = mesh[1]
        zz = mesh[2]
        ax.plot_wireframe(xx, yy, zz)

    # Plot coordinate system
    ax.quiver([0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], color="black")
    ax.scatter([0], [0], [0], color="red")

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    set_axes_equal(ax)
    plt.show(block=False)

# --------------------------------------------------------------------------------------------------


def plot_results(xx, yy, zz, color_variable):

    minn, maxx = color_variable.min(), color_variable.max()
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    m.set_array([])
    fcolors = m.to_rgba(color_variable)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type="persp")
    surf = ax.plot_surface(xx, yy, zz, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False,
                           linewidth=0.5, antialiased=False)
    fig.colorbar(m,  shrink=0.5, aspect=5)

    # Plot coordinate system
    ax.quiver([0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], color="black")
    ax.scatter([0], [0], [0], color="red")

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    set_axes_equal(ax)
    plt.show(block=False)

# --------------------------------------------------------------------------------------------------


def plot_structure(structure):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type="persp")

    for beam in structure.beams:
        point_A = structure.points[beam.point_A_index]
        point_B = structure.points[beam.point_B_index]
        x = [point_A[0], point_B[0]]
        y = [point_A[1], point_B[1]]
        z = [point_A[2], point_B[2]]
        ax.plot(x, y, z, "-ko", markersize=2)

    # Plot coordinate system
    ax.quiver([0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], color="black")
    ax.scatter([0], [0], [0], color="red")
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    set_axes_equal(ax)
    plt.show(block=False)

# --------------------------------------------------------------------------------------------------


def plot_aircraft(xx, yy, zz, structure):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type="persp")

    for beam in structure.beams:
        point_A = structure.points[beam.point_A_index]
        point_B = structure.points[beam.point_B_index]
        x = [point_A[0], point_B[0]]
        y = [point_A[1], point_B[1]]
        z = [point_A[2], point_B[2]]
        ax.plot(x, y, z, "-ko", markersize=2)

    ax.plot_wireframe(xx, yy, zz)

    # Plot coordinate system
    ax.quiver([0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], color="black")
    ax.scatter([0], [0], [0], color="red")
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    set_axes_equal(ax)
    plt.show(block=False)

# --------------------------------------------------------------------------------------------------


def plot_deformation(elements, nodes, deformations, scale=1):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type="persp")

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
    ax.quiver([0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], color="black")
    ax.scatter([0], [0], [0], color="red")
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    set_axes_equal(ax)
    plt.show(block=False)

