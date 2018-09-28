"""
vortex_lattice_method.py

Implementation of the vortex lattice method.

Reference: "Low-Speed Aerodynamics", Second Edition, Joseph Katz and Allen Plotkin

Author: João Paulo Monteiro Cruvinel da Costa
email: joaopaulomcc@gmail.com / joao.cruvinel@embraer.com.br
github: joaopaulomcc
"""
# ==================================================================================================
# IMPORTS
import numpy as np
import scipy as sc
import time
import matplotlib.pyplot as plt

from numpy import sin, cos, tan, pi, dot, cross
from numpy.linalg import norm
from scipy.sparse.linalg import gmres

# from numba import jit
# ==================================================================================================
# VORTEX LATTICE METHOD


def flatten(panel_matrix):
    """Flatten a matrix into a vector, basicaly the lines are concatenated one into another
    into a long vector
    """

    panel_vector = [item for sublist in panel_matrix for item in sublist]

    return panel_vector

# --------------------------------------------------------------------------------------------------


def vortex_solver(panel_vector, flow_velocity_vector, infinity):
    """Receives a vector of panel objects and the airflow velocity. Using this information
    calculates the influence matrix, the right hand side velocity vector and solves the resulting
    linear system. Returns a vector with the circulation for each one of the panels.

    Args:

    Returns:

    """

    n_panels = len(panel_vector)

    influence_coef_matrix = np.zeros((n_panels, n_panels))
    right_hand_side_vector = np.zeros((n_panels, 1))

    # For each colocation point i calculate the influence of panel j with a cirulation of 1
    print("Calculating influence matrix ...")
    start1 = time.time()

    for i in range(n_panels):
        print("Calculating right_hand_side_vector ...")
        start = time.time()
        right_hand_side_vector[i][0] = dot(-flow_velocity_vector, panel_vector[i].n)
        end = time.time()
        print(F"Calculating right_hand_side_vector executed in {end - start} seconds")

        for j in range(n_panels):
            print("Calculating ind_vel ...")
            start = time.time()
            ind_vel, _ = panel_vector[j].hs_induced_velocity(panel_vector[i].col_point, 1, infinity)
            end = time.time()
            print(F"Calculating ind_vel executed in {end - start} seconds")

            print("Calculating influence_coef_matrix[i][j] ...")
            start = time.time()
            influence_coef_matrix[i][j] = dot(ind_vel, panel_vector[i].n)
            end = time.time()
            print(F"Calculating influence_coef_matrix[i][j] executed in {end - start} seconds")

    end = time.time()
    print(F"Calculating influence matrix executed in {end - start1} seconds")
    # Solve linear system using scipy library
    # gamma = np.linalg.solve(influence_coef_matrix, right_hand_side_vector)
    print("Solving System...")
    start = time.time()
    gamma, _ = sc.sparse.linalg.lgmres(influence_coef_matrix, right_hand_side_vector)
    end = time.time()
    print(F"Solving System executed in {end - start} seconds")
    return gamma

# --------------------------------------------------------------------------------------------------

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
#     panel_vector = flatten(panel_matrix)
#     circulation, exitCode = vortex_solver(panel_vector, flow_velocity_vector, infinity)
#
#     return circulation, exitCode