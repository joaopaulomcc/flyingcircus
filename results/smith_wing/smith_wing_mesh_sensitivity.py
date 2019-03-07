"""
====================================================================================================
CFD-Based Analysis of Nonlinear Aeroelastic Behavior of High-Aspect Ratio Wings

M. J. Smith, M. J. Patil, D. H. Hodges

Georgia Institute fo Technology, Atlanta

====================================================================================================

Comparisson of the results obtained by in the paper above with those generated by the tool developed
in this work

Author: João Paulo Monteiro Cruvinel da Costa
"""

# ==================================================================================================
# IMPORTS

# Import python scientific libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import sys
import pickle

# Import code sub packages
from context import src
from src import aerodynamics as aero
from src import aeroelasticity as aelast
from src import control
from src import flight_mechanics as flmec
from src import geometry as geo
from src import loads
from src import structures as struct
from src import visualization as vis

from smith_wing_data import smith_wing

# ==================================================================================================
# GRID CREATION

results_list = []
iteration_results_list = []

for i in range(1, 11):

    # Number of panels and finite elements
    N_CHORD_PANELS = i
    N_SPAN_PANELS = int(16 / (2 * (1 / i)))
    N_BEAM_ELEMENTS = 2 * N_SPAN_PANELS
    CHORD_DISCRETIZATION = "linear"
    SPAN_DISCRETIZATION = "linear"
    TORSION_FUNCTION = "linear"
    CONTROL_SURFACE_DEFLECTION_DICT = dict()

    wing_grid_data = {
        "n_chord_panels": N_CHORD_PANELS,
        "n_span_panels_list": [N_SPAN_PANELS, N_SPAN_PANELS],
        "n_beam_elements_list": [N_BEAM_ELEMENTS, N_BEAM_ELEMENTS],
        "chord_discretization": CHORD_DISCRETIZATION,
        "span_discretization_list": [SPAN_DISCRETIZATION, SPAN_DISCRETIZATION],
        "torsion_function_list": [TORSION_FUNCTION, TORSION_FUNCTION],
        "control_surface_deflection_dict": CONTROL_SURFACE_DEFLECTION_DICT,
    }

    smith_wing_grid_data = {
        "macrosurfaces_grid_data": [wing_grid_data],
        "beams_grid_data": None,
    }
    # ==================================================================================================
    # STRUCTURE DEFINITION

    # Generate Constraints
    wing_fixation = {
        "component_identifier": "left_wing",
        "fixation_point": "ROOT",
        "dof_constraints": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    }

    smith_wing_constrains_data = [wing_fixation]

    # ==================================================================================================
    # AERODYNAMIC LOADS CALCULATION - CASE 2 - ALPHA 2º - Flexible Wing

    # Flight Conditions definition

    # Translation velocities
    V_X = 25
    V_Y = 0
    V_Z = 0

    # Rotation velocities
    R_X = 0
    R_Y = 0
    R_Z = 0

    # Aircraft Attitude in relation to the wind axis, in degrees
    ALPHA = 2  # Pitch angle
    BETA = 0  # Yaw angle
    GAMMA = 0  # Roll angle

    # Center of rotation, usually the aircraft CG position
    CENTER_OF_ROTATION = np.array([0, 0, 0])

    # Flight altitude, used to calculate atmosppheric conditions, in meters
    ALTITUDE = 20000

    # Atmospheric turbulence, function that calculates the air speeds in relation to the ground, given
    # a point coordinates


    def ATM_TURBULENCE_FUNCTION(point_coordinates):
       return np.zeros(3)

    FLIGHT_CONDITIONS_DATA = {
        "translation_velocity": np.array([V_X, V_Y, V_Z]),
        "rotation_velocity": np.array([R_X, R_Y, R_Z]),
        "attitude_angles_deg": np.array([ALPHA, BETA, GAMMA]),
        "center_of_rotation": CENTER_OF_ROTATION,
        "altitude": ALTITUDE,
        "atm_turbulenc_function": ATM_TURBULENCE_FUNCTION,
        "center_of_rotation": CENTER_OF_ROTATION,
    }

    SIMULATION_OPTIONS = {
        "flexible_aircraft": True,
        "status_messages": True,
        "control_node_string": "left_wing-TIP",
        "max_iterations": 100,
        "bending_convergence_criteria": 0.01,
        "torsion_convergence_criteria": 0.01,
        "fem_prop_choice": "ROOT",
        "interaction_algorithm": "closest",
        "output_iteration_results": True,
    }

    results, iteration_results = aelast.functions.calculate_aircraft_loads(
        aircraft_object=smith_wing,
        aircraft_grid_data=smith_wing_grid_data,
        aircraft_constraints_data=smith_wing_constrains_data,
        flight_condition_data=FLIGHT_CONDITIONS_DATA,
        simulation_options=SIMULATION_OPTIONS,
        influence_coef_matrix=None,
    )

    results_list.append(results)
    iteration_results_list.append(iteration_results)

    print("==============================================================")
    print(f"Chord Panels: {N_CHORD_PANELS}, Span panels: {N_SPAN_PANELS}")
    print("Total force:")
    print(results["aircraft_force_grid"][0].sum())

    f = open("results\\smith_wing\\mesh_sensitivity.pckl", "wb")
    pickle.dump([results_list, iteration_results_list], f)
    f.close()

