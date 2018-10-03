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

from .fast_operations import dot, cross, norm, normalize



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


def plot_mesh(xx, yy, zz):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type="persp")
    ax.autoscale(False)
    ax.plot_wireframe(xx, yy, zz)
    set_axes_equal(ax)
    plt.show()

# --------------------------------------------------------------------------------------------------


def plot_results(xx, yy, zz, color_variable):

    minn, maxx = color_variable.min(), color_variable.max()
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    m.set_array([])
    fcolors = m.to_rgba(color_variable)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type="persp")
    ax.autoscale(False)
    surf = ax.plot_surface(xx, yy, zz, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False,
                           linewidth=0.5, antialiased=False)
    fig.colorbar(m,  shrink=0.5, aspect=5)
    set_axes_equal(ax)
    plt.show()