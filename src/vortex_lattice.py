"""
Collection of vortex lattice related functions

Except where explicitly stated all units are S.I.
"""
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from numpy import pi, dot, cross
from numpy.linalg import norm

from .functions import rotate_point, velocity_vector
# from .objects import Panel
from .mesh_generation import generate_panel_matrix, generate_mesh

from scipy.sparse.linalg import gmres


def vortex_line(first_point, second_point, target_point, circulation):
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
    p1 = np.array(first_point)
    p2 = np.array(second_point)
    P = np.array(target_point)
    gamma = circulation

    # Calculating position vectors
    R1 = P - p1
    R2 = P - p2
    R0 = R1 - R2

    # Verification to deal with cases where the point of interest is situated in the vicinity of the
    # vortex line which for numerical purposes is assumed to have a very small radius
    # epsilon = 0.0001 (one milimiter, if meters are being used as the reference unit)
    epsilon = 0.001

    if (norm(R1) < epsilon or norm(R2) < epsilon or norm(cross(R1, R2)) ** 2 < epsilon):
        induced_velocity = np.array([0.0, 0.0, 0.0])
        return induced_velocity

    # Calculating induced velocity vector
    K = (gamma / (4 * pi * norm(cross(R1, R2)) ** 2)) * \
        (dot(R0, R1) / norm(R1) - dot(R0, R2) / norm(R2))

    induced_velocity = K * cross(R1, R2)

    return induced_velocity


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
            Q = vortex_line(vertex_array[:, i], vertex_array[:, i + 1], P, gamma)
            induced_velocity = induced_velocity + Q

        # If the point is the last vertex, connects it to the first vertex
        else:
            Q = vortex_line(vertex_array[:, i], vertex_array[:, 0], P, gamma)
            induced_velocity = induced_velocity + Q

    return induced_velocity


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
    vertex_array = np.array(vertex_coordinates)
    P = np.array(target_point)
    gamma = circulation

    induced_velocity = np.array([0.0, 0.0, 0.0])
    wake_induced_velocity = np.array([0.0, 0.0, 0.0])

    n_points = np.shape(vertex_array)[1]
    for i in range(n_points - 1):

        # If the vortex segment doesn't belong to the wake
        if (i != 0) and (i != n_points - 2):
            Q = vortex_line(vertex_array[:, i], vertex_array[:, i + 1], P, gamma)
            induced_velocity = induced_velocity + Q

        # If the vortex segment belongs to the wake
        else:
            Q = vortex_line(vertex_array[:, i], vertex_array[:, i + 1], P, gamma)
            induced_velocity = induced_velocity + Q
            wake_induced_velocity = wake_induced_velocity + Q

    return induced_velocity, wake_induced_velocity


def panel_matrix_to_panel_vector(panel_matrix):

    panel_vector = [item for sublist in panel_matrix for item in sublist]

    return panel_vector


def vortex_solver(panel_vector, flow_velocity_vector, infinity):

    n_panels = len(panel_vector)

    influence_coef_matrix = np.zeros((n_panels, n_panels))
    right_hand_side_vector = np.zeros((n_panels, 1))

    # for each colocation point i calculate the influence of panel j with a cirulation of 1

    for i in range(n_panels):
        right_hand_side_vector[i][0] = dot(-flow_velocity_vector, panel_vector[i].n)

        for j in range(n_panels):
            ind_vel, _ = panel_vector[j].hs_induced_velocity(panel_vector[i].col_point, 1, infinity)
            influence_coef_matrix[i][j] = dot(ind_vel, panel_vector[i].n)

    # Solve linear system using scipy library
    gamma, exitCode = gmres(influence_coef_matrix, right_hand_side_vector)

    return gamma, exitCode


# def lifting_line_horse_shoe(wing_object, attitude_vector, true_airspeed, altitude,
#                             n_semi_wingspam_panels=5, n_chord_panels=1,
#                             wingspam_discretization_type="linear",
#                             chord_discretization_type="linear",
#                             infinity_mult=25):
#
#     alpha, beta, gamma = attitude_vector[0], attitude_vector[1], attitude_vector[2]
#
#     flow_velocity_vector = velocity_vector(true_airspeed, alpha, beta, gamma)[:, 0]
#
#     infinity = infinity_mult * wing_object.wing_span
#
#     xx, yy, zz = generate_mesh(wing_object, n_semi_wingspam_panels, n_chord_panels,
#                                wingspam_discretization_type, chord_discretization_type)
#
#     panel_matrix = generate_panel_matrix(xx, yy, zz)
#     panel_vector = panel_matrix_to_panel_vector(panel_matrix)
#     circulation, exitCode = vortex_solver(panel_vector, flow_velocity_vector, infinity)
#
#     return circulation, exitCode