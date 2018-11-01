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

from .. import mathematics
from . import objects
from numba import jit
# ==================================================================================================
# VORTEX LATTICE METHOD


def create_panel_grid(macro_surface_mesh):

    n_span_panels = 0
    n_chord_panels = 0

    # Count number of chord and spam panels
    for surface_mesh in macro_surface_mesh:
        i, j = np.shape(surface_mesh["xx"])
        n_chord_panels = (i - 1)
        n_span_panels += (j - 1)

    # Initialize panel grid
    panel_grid = np.empty((n_chord_panels, n_span_panels), dtype="object")

    # Populate Panel Grid
    span_index = 0
    for surface_mesh in macro_surface_mesh:
        n_x, n_y = np.shape(surface_mesh["xx"])
        n_x -= 1
        n_y -= 1

        for i in range(n_x):
            for j in range(n_y):
                xx_slice = surface_mesh["xx"][i:i + 2, j:j + 2]
                yy_slice = surface_mesh["yy"][i:i + 2, j:j + 2]
                zz_slice = surface_mesh["zz"][i:i + 2, j:j + 2]
                panel_grid[i][j + span_index] = objects.PanelHorseShoe(xx_slice, yy_slice, zz_slice)

        span_index += n_y

    return panel_grid

# --------------------------------------------------------------------------------------------------


def flatten(panel_matrix):
    """Flatten a matrix into a vector, basicaly the lines are concatenated one into another
    into a long vector
    """

    panel_vector = [item for sublist in panel_matrix for item in sublist]

    return panel_vector

# --------------------------------------------------------------------------------------------------


#@jit
def gamma_solver(panel_vector, flow_velocity_vector):
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
