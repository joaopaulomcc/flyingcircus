"""
====================================================================================================
Comparisson between results found in the literature and those obtained by using Flying Circus.
Definition of the simulation data

Author: João Paulo Monteiro Cruvinel da Costa

Literature results:

NACA Technical Note No.1270 - EXPERIMENTAL AND CALCULATED CHARACTERISTICS OF SEVERALNACA 44-SERIES
WINGS WITH ASPECT RATIOS OF 8, 10, AND 12 AND TAPER RATIOS OF 2.5 AND 3.5

Authors: Robert H. Neely, Thomas V. Bollech, Gertrude C. Westrick, and Robert R. Graham

Langley Memorial Aeronautical Laboratory
Langley Field, Va.

Washington, May 1947
====================================================================================================
"""

# ==================================================================================================
# IMPORTS

# Import python scientific libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import sys

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
from hale_aircraft_data import hale_aircraft

vis.plot_3D.plot_aircraft(hale_aircraft, title="Smith Wing")


# WING GRID DATA
N_CHORD_PANELS = 5

WING_N_SPAN_PANELS = 20
WING_N_BEAM_ELEMENTS = 2 * WING_N_SPAN_PANELS

AILERON_N_SPAN_PANELS = 4
AILERON_N_BEAM_ELEMENTS = 2 * AILERON_N_SPAN_PANELS

CHORD_DISCRETIZATION = "linear"
SPAN_DISCRETIZATION = "linear"
TORSION_FUNCTION = "linear"
CONTROL_SURFACE_DEFLECTION_DICT = {"left_aileron": -15, "right_aileron": +15}

wing_grid_data = {
    "n_chord_panels": N_CHORD_PANELS,
    "n_span_panels_list": [
        AILERON_N_SPAN_PANELS,
        WING_N_SPAN_PANELS,
        WING_N_SPAN_PANELS,
        AILERON_N_SPAN_PANELS,
    ],
    "n_beam_elements_list": [
        AILERON_N_BEAM_ELEMENTS,
        WING_N_BEAM_ELEMENTS,
        WING_N_BEAM_ELEMENTS,
        AILERON_N_BEAM_ELEMENTS,
    ],
    "chord_discretization": CHORD_DISCRETIZATION,
    "span_discretization_list": [
        SPAN_DISCRETIZATION,
        SPAN_DISCRETIZATION,
        SPAN_DISCRETIZATION,
        SPAN_DISCRETIZATION,
    ],
    "torsion_function_list": [
        TORSION_FUNCTION,
        TORSION_FUNCTION,
        TORSION_FUNCTION,
        TORSION_FUNCTION,
    ],
    "control_surface_deflection_dict": CONTROL_SURFACE_DEFLECTION_DICT,
}

# --------------------------------------------------------------------------------------------------

# TAIL GRID DATA
TAIL_N_CHORD_PANELS = 5

TAIL_N_SPAN_PANELS = 10
TAIL_N_BEAM_ELEMENTS = 2 * TAIL_N_SPAN_PANELS

TAIL_CHORD_DISCRETIZATION = "linear"
TAIL_SPAN_DISCRETIZATION = "linear"
TAIL_TORSION_FUNCTION = "linear"
TAIL_CONTROL_SURFACE_DEFLECTION_DICT = {"left_elevator": 15, "right_elevator": -15}

tail_grid_data = {
    "n_chord_panels": TAIL_N_CHORD_PANELS,
    "n_span_panels_list": [
        TAIL_N_SPAN_PANELS,
        TAIL_N_SPAN_PANELS,
    ],
    "n_beam_elements_list": [
        TAIL_N_BEAM_ELEMENTS,
        TAIL_N_BEAM_ELEMENTS,
    ],
    "chord_discretization": TAIL_CHORD_DISCRETIZATION,
    "span_discretization_list": [
        TAIL_SPAN_DISCRETIZATION,
        TAIL_SPAN_DISCRETIZATION,
    ],
    "torsion_function_list": [
        TAIL_TORSION_FUNCTION,
        TAIL_TORSION_FUNCTION,
    ],
    "control_surface_deflection_dict": TAIL_CONTROL_SURFACE_DEFLECTION_DICT,
}

# --------------------------------------------------------------------------------------------------
# FUSELAGE AND TAIL BOOM GRID DATA

fuselage_grid_data = {"n_elements": 3}

tail_boom_grid_data = {"n_elements": 10}

# --------------------------------------------------------------------------------------------------
# HALE AIRCRAFT GRID DATA

hale_aircraft_grid_data = {
    "macrosurfaces_grid_data": [wing_grid_data, tail_grid_data],
    "beams_grid_data": [fuselage_grid_data, tail_boom_grid_data],
}

# ==================================================================================================
# GRID CREATION

# Creation of the smith wing grids
hale_aircraft_grids = aelast.functions.generate_aircraft_grids(
    aircraft_object=hale_aircraft, aircraft_grid_data=hale_aircraft_grid_data
)

# ==================================================================================================
# STRUCTURE DEFINITION

# Create wing finite elements

hale_aircraft_fem_elements = struct.fem.generate_aircraft_fem_elements(
    aircraft=hale_aircraft, aircraft_grids=hale_aircraft_grids, prop_choice="ROOT"
)

# Generate Constraints
cg_fixation = {
    "component_identifier": "fuselage",
    "fixation_point": "TIP",
    "dof_constraints": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
}

hale_aircraft_constrains_data = [cg_fixation]

hale_aircraft_constraints = aelast.functions.generate_aircraft_constraints(
    aircraft=hale_aircraft,
    aircraft_grids=hale_aircraft_grids,
    constraints_data_list=hale_aircraft_constrains_data,
)

# ax, fig = vis.plot_3D.plot_aircraft_grids(hale_aircraft_grids, hale_aircraft_fem_elements, title="Hale Aircraft Grids")

# fig.show()

# ==================================================================================================
# AERODYNAMIC LOADS CALCULATION - CASE 1 - ALPHA 2º
print()
print("# CASE 001:")
print(f"    - Altitude: 20000m")
print(f"    - True Airspeed: 25m/s")
print(f"    - Alpha: 2º")
print(f"    - Flexible Wing")
print()

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
CENTER_OF_ROTATION = hale_aircraft.inertial_properties.position

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
    "control_node_string": "left_aileron-TIP",
    "max_iterations": 100,
    "bending_convergence_criteria": 0.01,
    "torsion_convergence_criteria": 0.01,
    "fem_prop_choice": "ROOT",
    "interaction_algorithm": "closest",
    "output_iteration_results": True,
}

results, iteration_results = aelast.functions.calculate_aircraft_loads(
    aircraft_object=hale_aircraft,
    aircraft_grid_data=hale_aircraft_grid_data,
    aircraft_constraints_data=hale_aircraft_constrains_data,
    flight_condition_data=FLIGHT_CONDITIONS_DATA,
    simulation_options=SIMULATION_OPTIONS,
    influence_coef_matrix=None,
)

