"""
geometry.py

Collection of function for geometric manipulation

Author: João Paulo Monteiro Cruvinel da Costa
email: joaopaulomcc@gmail.com / joao.cruvinel@embraer.com.br
github: joaopaulomcc
"""
# ==================================================================================================
# IMPORTS
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from numpy import sin, cos, tan, pi, dot
from numpy.linalg import norm

from numba import jit, double
import math
import numpy as np

@jit
def cross(vec1, vec2):
    """ Calculate the cross product of two 3d vectors. """
    result = np.zeros(3)
    return cross_(vec1, vec2, result)


@jit(nopython=True)
def cross_(vec1, vec2, result):
    """ Calculate the cross product of two 3d vectors. """
    a1, a2, a3 = double(vec1[0]), double(vec1[1]), double(vec1[2])
    b1, b2, b3 = double(vec2[0]), double(vec2[1]), double(vec2[2])
    result[0] = a2 * b3 - a3 * b2
    result[1] = a3 * b1 - a1 * b3
    result[2] = a1 * b2 - a2 * b1
    return result


def rotate_point(point_coord, rot_axis, rot_center, rot_angle, degrees=False):
    """Rotates a point around an axis

    Args:
        point_coord [[float, float, float]]: x, y and z coordinates of the points, every column is a point
        rot_axis [float, float, float]: vector that will be used as rotation axis
        rot_center [float, float, float]: point that will be used as rotation center
        rot_angle [float]: angle of rotation in radians (default) or degrees if degrees = True
        degrees [bool]: True if the user wants to use angles in degrees

    Returns:
        point [float, float, float]: coordinates of the rotated point
    """

    # Converts inputs to numpy arrays, normalizes axis vector
    point_coord = np.array(point_coord)
    rot_center = np.array(rot_center)
    rot_center = (rot_center[np.newaxis]).transpose()
    U = np.array(rot_axis) / norm(rot_axis)

    if degrees:
        theta = np.radians(rot_angle)
    else:
        theta = rot_angle

    u0 = U[0]
    u1 = U[1]
    u2 = U[2]

    # Calculating rotation matrix
    # reference: https://en.wikipedia.org/wiki/Rotation_matrix - "Rotation matrix from axis and angle"

    # Identity matrix
    I = np.identity(3)

    # Cross product matrix
    CPM = np.array([[  0, -u2,  u1],
                    [ u2,   0, -u0],
                    [-u1,  u0,   0]])

    # Tensor product U X U, this is NOT a cross product
    TP = np.tensordot(U, U, axes=0)

    # Rotation Matrix
    R = cos(theta) * I + sin(theta) * CPM + (1 - cos(theta)) * TP

    # Calculating rotated point

    # Translates points so rotation center is the origin of the coordinate system
    point_coord = point_coord - rot_center

    # Rotates all points
    rotated_points = R @ point_coord

    # Undo translation
    rotated_points = rotated_points + rot_center

    return rotated_points


def velocity_vector(true_airspeed, alpha, beta, gamma):
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    origin = np.array([0, 0, 0])

    vector = np.array([true_airspeed, 0, 0])[np.newaxis].transpose()

    # Rotate around y for alfa
    vector = rotate_point(vector, y_axis, origin, -alpha, degrees=True)

    # Rotate around z for beta
    vector = rotate_point(vector, z_axis, origin, -beta, degrees=True)

    # Rotate around x for gamma
    vector = rotate_point(vector, x_axis, origin, -gamma, degrees=True)

    return vector







