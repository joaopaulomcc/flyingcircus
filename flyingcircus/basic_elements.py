"""
basic_elements.py

Collection of basic elements both aerodynamic and structural
Except where explicitly stated all units are S.I.

Author: João Paulo Monteiro Cruvinel da Costa
email: joaopaulomcc@gmail.com / joao.cruvinel@embraer.com.br
github: joaopaulomcc
"""
# ==================================================================================================
# IMPORTS
import numpy as np
import scipy as sc

from numpy import sin, cos, tan, pi
from numba import jit

from . import geometry
from .fast_operations import dot, cross, norm
# ==================================================================================================
# AERODYNAMICS


@jit(nopython=True)
def vortex_segment(first_point, second_point, target_point, circulation):
    """
    Calculates the induced velocity by a vortex line segment at a point.

    Notation and algorithm can be found on the book "Low-Speed Aerodynamics" by Joseph Katz and
    Allen Plotkin, 2nd edition, pages 251-255

    Args:
    first_point (float, float, float): tupple or array with x, y and z coordinates of the first
                                       point of the vortex segment

    second_point (float, float, float): tupple or array with x, y and z coordinates of the second
                                        point of the vortex segment

    target_point (float, float, float): tupple or array with x, y and z coordinates of the point
                                        where the induced velocity is to be calculated

    circulation (float): intensity of the vortex segment circulation

    Returns:
    np.array([float, float, float]): numpy array of the induced velocity vector
    """

    # Converting inputs to numpy arrays and changing names to reference notation
    p1 = first_point
    p2 = second_point
    P = target_point
    gamma = circulation

#    p1 = np.array(first_point)
#    p2 = np.array(second_point)
#    P = np.array(target_point)
#    gamma = circulation

    # Calculating position vectors
    R1 = P - p1
    R2 = P - p2
    R0 = R1 - R2

    # Verification to deal with cases where the point of interest is situated in the vicinity of the
    # vortex line which for numerical purposes is assumed to have a very small radius
    # epsilon = 0.0001 (one milimiter, if meters are being used as the reference unit)
    epsilon = 0.0001

    if (norm(R1) < epsilon or norm(R2) < epsilon or norm(cross(R1, R2)) ** 2 < epsilon):
        induced_velocity = np.array([0.0, 0.0, 0.0])
        return induced_velocity

    # Calculating induced velocity vector
    K = (gamma / (4 * pi * norm(cross(R1, R2)) ** 2)) * \
        (dot(R0, R1) / norm(R1) - dot(R0, R2) / norm(R2))

    induced_velocity = K * cross(R1, R2)

    return induced_velocity

# --------------------------------------------------------------------------------------------------


# @jit(nopython=True)
def vortex_ring(vertex_coordinates, target_point, circulation):
    """
    Calculates the induced velocity by a vortex ring at a point.

    Notation and algorithm can be found on the book "Low-Speed Aerodynamics" by Joseph Katz and
    Allen Plotkin, 2nd edition, pages 255-256

    Args:
    vertex_coordinates 4 x (float, float, float): tupple or array with x, y and z coordinates of the
                                                  ring vertex. Each line contains one point and the
                                                  order of the vertex defines the direction of the
                                                  circulation

    target_point (float, float, float): tupple or array with x, y and z coordinates of the point
                                        where the induced velocity is to be calculated

    circulation (float): intensity of the vortex segment circulation

    Returns:
    np.array([float, float, float]): numpy array of the induced velocity vector
    """

    # Converting inputs to numpy arrays and changing names to reference notation
    vertex_array = np.array(vertex_coordinates)
    P = np.array(target_point)
    gamma = circulation

    induced_velocity = np.array([0.0, 0.0, 0.0])
    n_points = np.shape(vertex_array)[1]

    for i in range(n_points):

        # If the point is not the last vertex, connects it to the next vertex
        if i != n_points - 1:
            Q = vortex_segment(vertex_array[:, i], vertex_array[:, i + 1], P, gamma)
            induced_velocity = induced_velocity + Q

        # If the point is the last vertex, connects it to the first vertex
        else:
            Q = vortex_segment(vertex_array[:, i], vertex_array[:, 0], P, gamma)
            induced_velocity = induced_velocity + Q

    return induced_velocity

# --------------------------------------------------------------------------------------------------


@jit(nopython=True)
def vortex_horseshoe(vertex_coordinates, target_point, circulation):
    """
    Calculates the induced velocity by a horseshoe vortex at a point.

    Notation and algorithm can be found on the book "Low-Speed Aerodynamics" by Joseph Katz and
    Allen Plotkin, 2nd edition, pages 256-258;331-334

    Args:
    vertex_coordinates 4 x (float, float, float): tupple or array with x, y and z coordinates of the
                                                  ring vertex. Each column contains one point and the
                                                  order of the vertex defines the direction of the
                                                  circulation

    target_point (float, float, float): tupple or array with x, y and z coordinates of the point
                                        where the induced velocity is to be calculated

    circulation (float): intensity of the vortex segment circulation

    Returns:
    np.array([float, float, float]): numpy array of the induced velocity vector
    """

    # Converting inputs to numpy arrays and changing names to reference notation
    P = target_point
    gamma = circulation

    induced_velocity = np.array([0.0, 0.0, 0.0])
    wake_induced_velocity = np.array([0.0, 0.0, 0.0])

    n_points = (vertex_coordinates).shape[1]
    for i in range(n_points - 1):

        # If the vortex segment doesn't belong to the wake
        if (i != 0) and (i != n_points - 2):
            Q = vortex_segment(vertex_coordinates[:, i], vertex_coordinates[:, i + 1], P, gamma)
            induced_velocity = induced_velocity + Q

        # If the vortex segment belongs to the wake
        else:
            Q = vortex_segment(vertex_coordinates[:, i], vertex_coordinates[:, i + 1], P, gamma)
            induced_velocity = induced_velocity + Q
            wake_induced_velocity = wake_induced_velocity + Q

    return induced_velocity, wake_induced_velocity

# --------------------------------------------------------------------------------------------------

# ==================================================================================================
# STRUCTURES


def beam_3D_stiff(E, A, L, G, J, Iy, Iz):

    K = np.zeros((12, 12))

    K[0][0] = E * A / L
    K[0][6] = -E * A / L
    K[6][0] = K[0][6]

    K[1][1] = 12 * E * Iz / L ** 3
    K[1][5] = 6 * E * Iz / L ** 2
    K[5][1] = K[1][5]
    K[1][7] = -12 * E * Iz / L ** 3
    K[7][1] = K[1][7]
    K[1][11] = 6 * E * Iz / L ** 2
    K[11][1] = K[1][11]

    K[2][2] = 12 * E * Iy / L ** 3
    K[2][4] = -6 * E * Iy / L ** 2
    K[4][2] = K[2][4]
    K[2][8] = -12 * E * Iy / L ** 3
    K[8][2] = K[2][8]
    K[2][10] = -6 * E * Iy / L ** 2
    K[10][2] = K[2][10]

    K[3][3] = G * J / L
    K[3][9] = -G * J / L
    K[9][3] = K[3][9]

    K[4][4] = 4 * E * Iy / L
    K[4][8] = 6 * E * Iy / L ** 2
    K[8][4] = K[4][8]
    K[4][10] = 2 * E * Iy / L
    K[10][4] = K[4][10]

    K[5][5] = 4 * E * Iz / L
    K[5][7] = -6 * E * Iz / L ** 2
    K[7][5] = K[5][7]
    K[5][11] = 2 * E * Iz / L
    K[11][5] = K[5][11]

    K[6][6] = E * A / L

    K[7][7] = 12 * E * Iz / L ** 3
    K[7][11] = -6 * E * Iz / L ** 2
    K[11][7] = K[7][11]

    K[8][8] = 12 * E * Iy / L ** 3
    K[8][10] = 6 * E * Iy / L ** 2
    K[10][8] = K[8][10]

    K[9][9] = G * J / L

    K[10][10] = 4 * E * Iy / L

    K[11][11] = 4 * E * Iz / L

    return K

# --------------------------------------------------------------------------------------------------


def beam_3D_rot(global_coord, local_coord):

    zero = np.zeros((3, 3))
    r = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            r[i][j] = geometry.cos_between(local_coord[:, i], global_coord[:, j])

    R = np.block([[r, zero, zero, zero],
                  [zero, r, zero, zero],
                  [zero, zero, r, zero],
                  [zero, zero, zero, r]])

    return R
