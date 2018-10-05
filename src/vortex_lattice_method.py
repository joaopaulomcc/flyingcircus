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

from numpy import sin, cos, tan, pi
import scipy.sparse.linalg as spla

from .fast_operations import dot, cross, norm, normalize
from numba import jit
# ==================================================================================================
# VORTEX LATTICE METHOD


def flatten(panel_matrix):
    """Flatten a matrix into a vector, basicaly the lines are concatenated one into another
    into a long vector
    """

    panel_vector = [item for sublist in panel_matrix for item in sublist]

    return panel_vector

# --------------------------------------------------------------------------------------------------


#@jit
def gamma_solver(panel_vector, flow_velocity_vector, infinity):
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

    for i in range(n_panels):

        right_hand_side_vector[i][0] = dot(-flow_velocity_vector, panel_vector[i].n)

        right_hand_side_vector[i][0] = dot(-flow_velocity_vector, panel_vector[i].n)

        for j in range(n_panels):

            ind_vel, wake_ind_velocity = panel_vector[j].hs_induced_velocity(panel_vector[i].col_point, 1)

            influence_coef_matrix[i][j] = dot(ind_vel, panel_vector[i].n)

    # Solve linear system using scipy library

    gamma, _ = spla.gmres(influence_coef_matrix, right_hand_side_vector)

    #np.savetxt("influence.csv", influence_coef_matrix, delimiter=",")
    #np.savetxt("RHS.csv", right_hand_side_vector, delimiter=",")
    #np.savetxt("gamma.csv", gamma, delimiter=",")

    return gamma

# --------------------------------------------------------------------------------------------------


@jit
def downwash_solver(panel_vector, gamma):
    """Receives a vector of panel objects and the cirulation.
    Returns a vector with the downwash for each one of the panels.

    Args:

    Returns:

    """

    n_panels = len(panel_vector)

    downwash_influence_coef_matrix = np.zeros((n_panels, n_panels))

    # For each colocation point i calculate the influence of panel j with a cirulation of 1

    for i in range(n_panels):

        for j in range(n_panels):

            ind_vel, wake_ind_velocity = panel_vector[j].hs_induced_velocity(panel_vector[i].col_point, 1)

            downwash_influence_coef_matrix[i][j] = dot(wake_ind_velocity, panel_vector[i].n)

    # Solve linear system using scipy library

    downwash = downwash_influence_coef_matrix @ gamma

    return downwash

# --------------------------------------------------------------------------------------------------


@jit
def lift_drag(panel_vector, gamma, downwash, free_stream_vel, air_density):

    n_panels = len(panel_vector)
    lift = np.zeros(n_panels)
    drag = np.zeros(n_panels)

    for i, panel in enumerate(panel_vector):

        lift[i] = air_density * free_stream_vel * gamma[i] * panel.span
        drag[i] = -air_density * downwash[i] * gamma[i] * panel.span

    return lift, drag


# --------------------------------------------------------------------------------------------------
