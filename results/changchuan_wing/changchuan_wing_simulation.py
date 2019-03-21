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

# Import code sub packages
from context import flyingcircus
from flyingcircus import aerodynamics as aero
from flyingcircus import aeroelasticity as aelast
from flyingcircus import control
from flyingcircus import flight_mechanics as flmec
from flyingcircus import geometry as geo
from flyingcircus import loads
from flyingcircus import structures as struct
from flyingcircus import visualization as vis

# ==================================================================================================
# print()
# print("============================================================")
# print("= VALIDATION OF AEROELASTIC CALCULATION                    =")
# print("= VALIDATION CASE: CFD-Based Analysis of Nonlinear         =")
# print("= Aeroelastic Behavior of High-Aspect Ratio Wings          =")
# print("= AUTHORS: M. J. Smith, M. J. Patil, D. H. Hodges          =")
# print("============================================================")
# ==================================================================================================
# GEOMETRY DEFINITION

print("# Importing geometric data...")
from changchuan_wing_data import changchuan_wing

# vis.plot_3D.plot_aircraft(changchuan_wing, title="Smith Wing")


# ==================================================================================================
# GRID CREATION


# Number of panels and finite elements
N_CHORD_PANELS = 8
N_SPAN_PANELS = 64
N_BEAM_ELEMENTS = 2 * N_SPAN_PANELS
CHORD_DISCRETIZATION = "linear"
SPAN_DISCRETIZATION = "linear"
TORSION_FUNCTION = "linear"
CONTROL_SURFACE_DEFLECTION_DICT = dict()

wing_grid_data = {
    "n_chord_panels": N_CHORD_PANELS,
    "n_span_panels_list": [N_SPAN_PANELS],
    "n_beam_elements_list": [N_BEAM_ELEMENTS],
    "chord_discretization": CHORD_DISCRETIZATION,
    "span_discretization_list": [SPAN_DISCRETIZATION],
    "torsion_function_list": [TORSION_FUNCTION],
    "control_surface_deflection_dict": CONTROL_SURFACE_DEFLECTION_DICT,
}

changchuan_wing_grid_data = {
    "macrosurfaces_grid_data": [wing_grid_data],
    "beams_grid_data": None,
}

# Creation of the smith wing grids

changchuan_wing_grids = aelast.functions.generate_aircraft_grids(
    aircraft_object=changchuan_wing, aircraft_grid_data=changchuan_wing_grid_data
)

# ==================================================================================================
# STRUCTURE DEFINITION

# Create wing finite elements

changchuan_wing_fem_elements = struct.fem.generate_aircraft_fem_elements(
    aircraft=changchuan_wing, aircraft_grids=changchuan_wing_grids, prop_choice="ROOT"
)

# Generate Constraints
wing_fixation = {
    "component_identifier": "main_wing",
    "fixation_point": "ROOT",
    "dof_constraints": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
}

changchuan_wing_constrains_data = [wing_fixation]

changchuan_wing_constraints = aelast.functions.generate_aircraft_constraints(
    aircraft=changchuan_wing,
    aircraft_grids=changchuan_wing_grids,
    constraints_data_list=changchuan_wing_constrains_data,
)

# ax, fig = vis.plot_3D.plot_aircraft_grids(changchuan_wing_grids, changchuan_wing_fem_elements, title="Smith Wing Grids")

# fig.show()

# ==================================================================================================
# AERODYNAMIC LOADS CALCULATION - CASE 1 - ALPHA 2º
print()
print("# CASE 001:")
print(f"    - Altitude: 0 m")
print(f"    - True Airspeed: 34m/s")
print(f"    - Alpha: 0.4º")
print(f"    - Flexible Wing")
print()

# Flight Conditions definition

# Translation velocities
V_X = 34
V_Y = 0
V_Z = 0

# Rotation velocities
R_X = 0
R_Y = 0
R_Z = 0

# Aircraft Attitude in relation to the wind axis, in degrees
ALPHA = 0.4  # Pitch angle
BETA = 0  # Yaw angle
GAMMA = 0  # Roll angle

# Center of rotation, usually the aircraft CG position
CENTER_OF_ROTATION = np.array([0, 0, 0])

# Flight altitude, used to calculate atmosppheric conditions, in meters
ALTITUDE = 0

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
    "control_node_string": "main_wing-TIP",
    "max_iterations": 100,
    "bending_convergence_criteria": 0.01,
    "torsion_convergence_criteria": 0.01,
    "fem_prop_choice": "ROOT",
    "interaction_algorithm": "closest",
    "output_iteration_results": True,
}

results, iteration_results = aelast.functions.calculate_aircraft_loads(
    aircraft_object=changchuan_wing,
    aircraft_grid_data=changchuan_wing_grid_data,
    aircraft_constraints_data=changchuan_wing_constrains_data,
    flight_condition_data=FLIGHT_CONDITIONS_DATA,
    simulation_options=SIMULATION_OPTIONS,
    influence_coef_matrix=None,
)

