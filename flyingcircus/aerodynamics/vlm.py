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

from .. import mathematics as m
from .. import geometry as geo
from . import objects
from . import functions
from numba import jit

# ==================================================================================================
# VORTEX LATTICE METHOD

#@jit
def create_panel_grid(macro_surface_mesh):

    n_span_panels = 0
    n_chord_panels = 0

    # Count number of chord and spam panels
    for surface_mesh in macro_surface_mesh:
        i, j = np.shape(surface_mesh["xx"])
        n_chord_panels = i - 1
        n_span_panels += j - 1

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
                xx_slice = surface_mesh["xx"][i : i + 2, j : j + 2]
                yy_slice = surface_mesh["yy"][i : i + 2, j : j + 2]
                zz_slice = surface_mesh["zz"][i : i + 2, j : j + 2]
                panel_grid[i][j + span_index] = objects.PanelHorseShoe(
                    xx_slice, yy_slice, zz_slice
                )

        span_index += n_y

    return panel_grid


# --------------------------------------------------------------------------------------------------


def flatten(panel_matrix):
    """Flatten a matrix into a vector, basicaly the lines are concatenated one into another
    into a long vector
    """

    # shape = np.shape(panel_matrix)
    panel_vector = np.copy(np.reshape(panel_matrix, np.size(panel_matrix)))

    # panel_vector = [item for sublist in panel_matrix for item in sublist]

    return panel_vector


# --------------------------------------------------------------------------------------------------


@jit
def gamma_solver(influence_coef_matrix, right_hand_side_vector):
    """Receives a vector of panel objects and the airflow velocity. Using this information
    calculates the influence matrix, the right hand side velocity vector and solves the resulting
    linear system. Returns a vector with the circulation for each one of the panels.

    Args:

    Returns:

    """
    # Solve linear system using scipy library

    gamma, info = spla.gmres(influence_coef_matrix, right_hand_side_vector)

    # Warn user if the solver can not find a solution for the system
    if info != 0:

        print("aerodynamics.vlm.gamma_solver : ERROR: Solver did not converge!")
        return None

    else:
        return gamma


# --------------------------------------------------------------------------------------------------


#@jit
def aero_loads(
    aircraft_aero_mesh,
    velocity_vector,
    rotation_vector,
    attitude_vector,
    altitude,
    center,
    influence_coef_matrix=None,
):

    velocity_field_function = geo.functions.velocity_field_function_generator(
        velocity_vector, rotation_vector, attitude_vector, center
    )

    aircraft_panel_vector = np.array([], dtype="object")
    shapes = []
    components_panel_grid = []
    components_panel_vector = []

    #print("Generating Panel Grid")
    for component_mesh in aircraft_aero_mesh:

        # Generates components panel vector
        component_panel_grid = create_panel_grid(component_mesh)
        component_panel_vector = flatten(component_panel_grid)

        # Adds component panel vector to aircraft
        aircraft_panel_vector = np.copy(
            np.append(aircraft_panel_vector, component_panel_vector)
        )
        # aircraft_panel_vector += component_panel_vector

        # Save shape for reconstruction
        shapes.append(np.shape(component_panel_grid))

        # Save panel vector and grid
        components_panel_grid.append(component_panel_grid)
        components_panel_vector.append(component_panel_vector)

    # Calculate Influence Coefficient Matrix
    #print("Calculating Influence Coefficient Matrix")
    if influence_coef_matrix is None:
        influence_coef_matrix = calc_influence_matrix(aircraft_panel_vector)

    # Calculate right hand side vector
    #print("Calculating Right Hand Side Vector")
    right_hand_side_vector = calc_rhs_vector(
        aircraft_panel_vector, velocity_field_function
    )

    # Calculate vortex circulation intensity
    #print("Solving system to find gamma")
    gamma_vector = gamma_solver(influence_coef_matrix, right_hand_side_vector)

    if gamma_vector is None:

        print("FATAL ERROR")
        return None

    # Calculate Local Flow Vector

    flow_vector = calc_local_flow_vector(
        aircraft_panel_vector,
        gamma_vector,
        velocity_vector,
        rotation_vector,
        attitude_vector,
        center,
    )

    # Calculate Aerodynamic Forces
    air_density, air_pressure, air_temperature = functions.ISA(altitude)

    force_vector = np.empty(len(gamma_vector), dtype="object")

    #print("Calculating Aerodynamic Forces")
    for i, panel_info in enumerate(zip(gamma_vector, aircraft_panel_vector)):

        panel_gamma = panel_info[0]
        panel = panel_info[1]

        # Calculate force acting on panel in the geometrical reference system
        force_vector[i] = panel.aero_force(panel_gamma, flow_vector[i], air_density)

    # Separate results by component
    components_force_vector = []
    components_force_grid = []

    components_gamma_vector = []
    components_gamma_grid = []

    first_index = 0
    for shape in shapes:
        n_itens = shape[0] * shape[1]

        force_vector_slice = force_vector[(first_index) : (first_index + n_itens)]
        components_force_vector.append(force_vector_slice)
        components_force_grid.append(np.reshape(force_vector_slice, shape))

        gamma_vector_slice = gamma_vector[(first_index) : (first_index + n_itens)]
        components_gamma_vector.append(gamma_vector_slice)
        components_gamma_grid.append(np.reshape(gamma_vector_slice, shape))

        first_index += n_itens

    return (
        components_force_vector,
        components_panel_vector,
        components_gamma_vector,
        components_force_grid,
        components_panel_grid,
        components_gamma_grid,
        influence_coef_matrix,
    )


# --------------------------------------------------------------------------------------------------

@jit
def calc_influence_matrix(panel_vector):

    n_panels = len(panel_vector)
    influence_coef_matrix = np.zeros((n_panels, n_panels))

    # For each colocation point i calculate the influence of panel j with a cirulation of 1

    for i in range(n_panels):

        for j in range(n_panels):

            ind_vel = panel_vector[j].induced_velocity(panel_vector[i].col_point, 1)

            influence_coef_matrix[i][j] = m.dot(ind_vel, panel_vector[i].n)

    return influence_coef_matrix


# --------------------------------------------------------------------------------------------------

@jit
def calc_rhs_vector(panel_vector, velocity_field_function):

    n_panels = len(panel_vector)
    right_hand_side_vector = np.zeros((n_panels, 1))

    # For each colocation point i calculate the influence of panel j with a cirulation of 1

    for i, panel in enumerate(panel_vector):
        flow_velocity = velocity_field_function(panel.col_point)
        right_hand_side_vector[i][0] = -m.dot(flow_velocity, panel.n)

    return right_hand_side_vector


# --------------------------------------------------------------------------------------------------

@jit
def calc_panels_ind_velocity(panel_vector, gamma_vector, point):

    total_ind_velocity = np.zeros(3)

    for panel, gamma in zip(panel_vector, gamma_vector):

        ind_velocity = panel.induced_velocity(point, gamma)

        total_ind_velocity += ind_velocity

    return total_ind_velocity


# --------------------------------------------------------------------------------------------------

@jit
def calc_local_flow_vector(
    panel_vector,
    gamma_vector,
    velocity_vector,
    rotation_vector,
    attitude_vector,
    attitude_center,
):

    flow_vector = np.empty(len(gamma_vector), dtype="object")

    velocity_field_function = geo.functions.velocity_field_function_generator(
        velocity_vector, rotation_vector, attitude_vector, attitude_center
    )

    for i, panel in enumerate(panel_vector):

        # Calculate Flow vector at panel aerodynamic center

        flow_vector[i] = calc_panels_ind_velocity(
            panel_vector, gamma_vector, panel.aero_center
        ) + velocity_field_function(panel.aero_center)

    return flow_vector


# --------------------------------------------------------------------------------------------------

@jit
def calc_panels_delta_pressure(panel_grid, force_grid):

    n_chord_panels = np.shape(panel_grid)[0]
    n_spam_panels = np.shape(panel_grid)[1]

    delta_p_grid = np.zeros(np.shape(panel_grid))
    force_magnitude_grid = np.zeros(np.shape(panel_grid))

    for i in range(n_chord_panels):
        for j in range(n_spam_panels):
            force_magnitude_grid[i][j] = m.norm(force_grid[i][j])
            delta_p_grid[i][j] = force_magnitude_grid[i][j] / panel_grid[i][j].area

    return delta_p_grid, force_magnitude_grid
