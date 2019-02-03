"""
====================================================================================================
Comparisson between results found in the literature and those obtained by using Flying Circus.
Definition of the simulation parameters

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
# IMPORT GEOMETRICAL DATA

print("# Importing geometric data...")
from broeren_wing_data import broeren_wing_i

# from broeren_wing_data import broeren_wing_ii
# from broeren_wing_data import broeren_wing_iii
# from broeren_wing_data import broeren_wing_iv
# from broeren_wing_data import broeren_wing_v
# from broeren_wing_data import broeren_wing_vi
# from broeren_wing_data import broeren_wing_vii

vis.plot_3D.plot_aircraft(broeren_wing_i, title="BROEREN WING I: 2.5-08-4416")


# ==================================================================================================
# GRID CREATION

# The same grid parameters will be used for all the wing models

# Number of panels and finite elements
N_CHORD_PANELS = 10
N_SPAN_PANELS = 20
N_BEAM_ELEMENTS = 40
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

broeren_wing_grid_data = {
    "macrosurfaces_grid_data": [wing_grid_data],
    "beams_grid_data": None,
}

# Creation of the smith wing grids

broeren_wing_i_grids = aelast.functions.generate_aircraft_grids(
    aircraft_object=broeren_wing_i, aircraft_grid_data=broeren_wing_grid_data
)
# broeren_wing_ii_grids = aelast.functions.generate_aircraft_grids(
#    aircraft_object=broeren_wing_ii, aircraft_grid_data=broeren_wing_grid_data
# )
# broeren_wing_iii_grids = aelast.functions.generate_aircraft_grids(
#    aircraft_object=broeren_wing_iii, aircraft_grid_data=broeren_wing_grid_data
# )
# broeren_wing_iv_grids = aelast.functions.generate_aircraft_grids(
#    aircraft_object=broeren_wing_iv, aircraft_grid_data=broeren_wing_grid_data
# )
# broeren_wing_v_grids = aelast.functions.generate_aircraft_grids(
#    aircraft_object=broeren_wing_v, aircraft_grid_data=broeren_wing_grid_data
# )
# broeren_wing_vi_grids = aelast.functions.generate_aircraft_grids(
#    aircraft_object=broeren_wing_vi, aircraft_grid_data=broeren_wing_grid_data
# )
# broeren_wing_vii_grids = aelast.functions.generate_aircraft_grids(
#    aircraft_object=broeren_wing_vii, aircraft_grid_data=broeren_wing_grid_data
# )

# ==================================================================================================
# STRUCTURE DEFINITION

# Create wing finite elements

# smith_wing_fem_elements = struct.fem.generate_aircraft_fem_elements(
#    aircraft=smith_wing, aircraft_grids=smith_wing_grids, prop_choice="ROOT"
# )
#
## Generate Constraints
# wing_fixation = {
#    "component_identifier": "left_wing",
#    "fixation_point": "ROOT",
#    "dof_constraints": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
# }
#
# smith_wing_constrains_data = [wing_fixation]
#
# smith_wing_constraints = aelast.functions.generate_aircraft_constraints(
#    aircraft=smith_wing,
#    aircraft_grids=smith_wing_grids,
#    constraints_data_list=smith_wing_constrains_data,
# )

ax, fig = vis.plot_3D2.generate_aircraft_grids_plot(
    broeren_wing_i_grids["macrosurfaces_aero_grids"],
    title="BROEREN WING I: 2.5-08-4416",
)
fig.show()

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
ALPHA = 5  # Pitch angle
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
    "flexible_aircraft": False,
    "status_messages": True,
    "control_node_string": "left_wing-TIP",
    "max_iterations": 100,
    "bending_convergence_criteria": 0.01,
    "torsion_convergence_criteria": 0.01,
    "fem_prop_choice": "ROOT",
    "interaction_algorithm": "closest",
    "output_iteration_results": True,
}

broeren_wing_i_results = aelast.functions.calculate_aircraft_loads(
    aircraft_object=broeren_wing_i,
    aircraft_grid_data=broeren_wing_grid_data,
    flight_condition_data=FLIGHT_CONDITIONS_DATA,
    simulation_options=SIMULATION_OPTIONS,
    influence_coef_matrix=None,
)

