# ==================================================================================================
# IMPORTS
import numpy as np
import scipy as sc

from numpy import sin, cos, tan, pi
from numba import jit

from .. import mathematics as m

# ==================================================================================================
# Functions


def cg_aero_loads(aircraft, components_force_vector, components_panel_vector):

    component_cg_aero_loads = []

    cg_position = aircraft.inertial_properties.cg_position
    
    for component_force_slice, component_panel_slice in zip(components_force_vector, components_panel_vector):

        cg_aero_forces = np.zeros(3)
        cg_aero_moments = np.zeros(3)

        for force, panel in zip(component_force_slice, component_panel_slice):
            cg_aero_forces += force

            # Moment lever calculation
            force_application_point = panel.aero_center
            lever_arm = cg_position - force_application_point

            cg_aero_moments += m.cross(lever_arm, force)

        component_cg_aero_loads.append([np.copy(cg_aero_forces), np.copy(cg_aero_moments)])

    total_cg_aero_force = np.zeros(3)
    total_cg_aero_moment = np.zeros(3)

    for component in component_cg_aero_loads:

        total_cg_aero_force += component[0]
        total_cg_aero_moment += component[1]

    return total_cg_aero_force, total_cg_aero_moment, component_cg_aero_loads

# --------------------------------------------------------------------------------------------------


def cg_engine_loads(aircraft, throtle_list, parameters_list):

    engine_loads = []
    engine_force = np.zeros(3)
    engine_moment = np.zeros(3)

    cg_position = aircraft.inertial_properties.cg_position

    for i, engine in enumerate(aircraft.engines):

        # Thrust calculation
        thrust = engine.thrust(throtle_list[i], parameters_list[i])

        # Moment calculation
        lever_arm = cg_position - engine.position

        moment = m.cross(lever_arm, thrust)

        engine_loads.append([np.copy(thrust), np.copy(moment)])

        engine_force += thrust
        engine_moment += moment
    
    return engine_force, engine_moment, engine_loads

