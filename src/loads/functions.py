# ==================================================================================================
# IMPORTS
import numpy as np
import scipy as sc

from numpy import sin, cos, tan, pi
from numba import jit
from pyquaternion import Quaternion

from .. import aerodynamics as aero
from .. import mathematics as m
from .. import geometry as geo

# ==================================================================================================
# Functions


def calc_aero_loads_at_point(point, aircraft_force_grid, aircraft_panel_grid):

    macrosurfaces_aero_loads = []

    for macrosurface_force_grid, macrosurface_panel_grid in zip(
        aircraft_force_grid, aircraft_panel_grid
    ):

        # Transform into a vector
        macrosurface_force_vector = np.copy(
            np.reshape(macrosurface_force_grid, np.size(macrosurface_force_grid))
        )
        macrosurface_panel_vector = np.copy(
            np.reshape(macrosurface_panel_grid, np.size(macrosurface_panel_grid))
        )

        # Compute forces
        aero_forces = np.zeros(3)
        aero_moments = np.zeros(3)

        for force, panel in zip(macrosurface_force_vector, macrosurface_panel_vector):
            aero_forces += force

            # Moment lever calculation
            force_application_point = panel.aero_center
            lever_arm = point - force_application_point

            aero_moments += m.cross(lever_arm, force)

        macrosurfaces_aero_loads.append([np.copy(aero_forces), np.copy(aero_moments)])

    total_aero_force = np.zeros(3)
    total_aero_moment = np.zeros(3)

    for component in macrosurfaces_aero_loads:

        total_aero_force += component[0]
        total_aero_moment += component[1]

    return total_aero_force, total_aero_moment, macrosurfaces_aero_loads


# --------------------------------------------------------------------------------------------------


def calc_engine_loads_at_point(aircraft, point, throtle_list, parameters_list):

    engine_loads = []
    engine_force = np.zeros(3)
    engine_moment = np.zeros(3)

    for i, engine in enumerate(aircraft.engines):

        # Thrust calculation
        thrust = engine.thrust(throtle_list[i], parameters_list[i])

        # Moment calculation
        lever_arm = point - engine.position

        moment = m.cross(lever_arm, thrust)

        engine_loads.append([np.copy(thrust), np.copy(moment)])

        engine_force += thrust
        engine_moment += moment

    return engine_force, engine_moment, engine_loads


# --------------------------------------------------------------------------------------------------


def calc_lift_drag(
    aircraft,
    point,
    speed,
    altitude,
    attitude_vector,
    aircraft_force_grid,
    aircraft_panel_grid,
):
    """Calculates the lift and drag of the aircraft
    """

    # Calculate wind coord system in relation to aircraft coodinate system

    alpha = np.radians(attitude_vector[0])
    beta = np.radians(attitude_vector[1])
    gamma = np.radians(attitude_vector[2])

    wind_coord_sys = geo.objects.Node(xyz=np.array([0, 0, 0]), quaternion=Quaternion())

    # Apply YAW rotation
    wind_coord_sys = wind_coord_sys.rotate(
        rotation_quaternion=Quaternion(axis=wind_coord_sys.z_axis, angle=-beta)
    )

    # Apply PITCH rotation
    wind_coord_sys = wind_coord_sys.rotate(
        rotation_quaternion=Quaternion(axis=wind_coord_sys.y_axis, angle=-alpha)
    )

    # Apply ROLL rotation
    wind_coord_sys = wind_coord_sys.rotate(
        rotation_quaternion=Quaternion(axis=wind_coord_sys.x_axis, angle=-gamma)
    )

    # Calculate aerodynamic forces in the aircraft coordinates system
    total_cg_aero_force, total_cg_aero_moment, component_cg_aero_loads = calc_aero_loads_at_point(
        point, aircraft_force_grid, aircraft_panel_grid
    )

    # Transform aerodynamic forces from aircraft coordinate system to wind coordinate system
    aero_forces = geo.functions.change_coord_sys(
        total_cg_aero_force,
        wind_coord_sys.x_axis,
        wind_coord_sys.y_axis,
        wind_coord_sys.z_axis,
    )

    lift = aero_forces[2]
    drag = aero_forces[0]
    sideforce = aero_forces[1]

    density, pressure, temperature = aero.functions.ISA(altitude)
    Cl = lift / (0.5 * density * (speed ** 2) * aircraft.ref_area)
    Cd = drag / (0.5 * density * (speed ** 2) * aircraft.ref_area)

    aero_moments = geo.functions.change_coord_sys(
        total_cg_aero_moment,
        wind_coord_sys.x_axis,
        wind_coord_sys.y_axis,
        wind_coord_sys.z_axis,
    )

    roll_moment = aero_moments[0]
    pitch_moment = aero_moments[1]
    yaw_moment = aero_moments[2]

    Cm = pitch_moment / (
        0.5 * density * (speed ** 2) * aircraft.ref_area * aircraft.mean_aero_chord
    )

    # Prepare Output
    forces = {"lift": lift, "drag": drag, "sideforce": sideforce}
    moments = {
        "roll_moment": roll_moment,
        "pitch_moment": pitch_moment,
        "yaw_moment": yaw_moment,
    }
    coefficients = {"Cl": Cl, "Cd": Cd, "Cm": Cm}

    return forces, moments, coefficients


# --------------------------------------------------------------------------------------------------


def calc_load_distribution(
    aircraft_force_grid, aircraft_panel_grid, attitude_vector, altitude, speed
):

    density, pressure, temperature = aero.functions.ISA(altitude)

    components_loads = []

    for component_force_grid, component_panel_grid in zip(
        aircraft_force_grid, aircraft_panel_grid
    ):

        # Calculate wind coord system in relation to aircraft coodinate system

        alpha = np.radians(attitude_vector[0])
        beta = np.radians(attitude_vector[1])
        gamma = np.radians(attitude_vector[2])

        wind_coord_sys = geo.objects.Node(
            xyz=np.array([0, 0, 0]), quaternion=Quaternion()
        )

        # Apply YAW rotation
        wind_coord_sys = wind_coord_sys.rotate(
            rotation_quaternion=Quaternion(axis=wind_coord_sys.z_axis, angle=-beta)
        )

        # Apply PITCH rotation
        wind_coord_sys = wind_coord_sys.rotate(
            rotation_quaternion=Quaternion(axis=wind_coord_sys.y_axis, angle=-alpha)
        )

        # Apply ROLL rotation
        wind_coord_sys = wind_coord_sys.rotate(
            rotation_quaternion=Quaternion(axis=wind_coord_sys.x_axis, angle=-gamma)
        )

        n_chord_panels, n_span_panels = np.shape(component_force_grid)

        y_values = np.zeros(n_span_panels)

        x_force = np.zeros(n_span_panels)
        y_force = np.zeros(n_span_panels)
        z_force = np.zeros(n_span_panels)

        lift = np.zeros(n_span_panels)
        drag = np.zeros(n_span_panels)
        side = np.zeros(n_span_panels)
        Cl = np.zeros(n_span_panels)
        Cd = np.zeros(n_span_panels)

        for i in range(n_span_panels):
            section_force = 0
            section = component_force_grid[:, i]
            panels_section = np.array(component_panel_grid)[:, i]

            for force in section:
                section_force += force

            # Calculate section area
            section_area = 0
            for panel in panels_section:
                section_area += panel.area

            x_force[i] = section_force[0]
            y_force[i] = section_force[1]
            z_force[i] = section_force[2]

            # Transform aerodynamic forces from aircraft coordinate system to wind coordinate system
            section_aero_forces = geo.functions.change_coord_sys(
                section_force,
                wind_coord_sys.x_axis,
                wind_coord_sys.y_axis,
                wind_coord_sys.z_axis,
            )

            lift[i] = section_aero_forces[2]
            drag[i] = section_aero_forces[0]
            side[i] = section_aero_forces[1]

            y_values[i] = panels_section[0].aero_center[1]

            Cl[i] = lift[i] / (0.5 * density * (speed ** 2) * section_area)
            Cd[i] = drag[i] / (0.5 * density * (speed ** 2) * section_area)

        loads = {
            "x_force": x_force,
            "y_force": y_force,
            "z_force": z_force,
            "lift": lift,
            "drag": drag,
            "side": side,
            "Cl": Cl,
            "Cd": Cd,
            "y_values": y_values,
        }

        components_loads.append(loads)

    return components_loads


# --------------------------------------------------------------------------------------------------


def calculate_surface_panels_loads(surface_panel_grid, surface_force_grid):

    n_chord_panels = np.shape(surface_panel_grid)[0]
    n_spam_panels = np.shape(surface_panel_grid)[1]

    delta_p_grid = np.zeros(np.shape(surface_panel_grid))
    force_magnitude_grid = np.zeros(np.shape(surface_panel_grid))
    force_x_grid = np.zeros(np.shape(surface_panel_grid))
    force_y_grid = np.zeros(np.shape(surface_panel_grid))
    force_z_grid = np.zeros(np.shape(surface_panel_grid))

    for i in range(n_chord_panels):
        for j in range(n_spam_panels):
            force_x_grid[i][j] = surface_force_grid[i][j][0]
            force_y_grid[i][j] = surface_force_grid[i][j][1]
            force_z_grid[i][j] = surface_force_grid[i][j][2]

            force_magnitude_grid[i][j] = m.norm(surface_force_grid[i][j])
            delta_p_grid[i][j] = (
                force_magnitude_grid[i][j] / surface_panel_grid[i][j].area
            )

    return {
        "delta_p_grid": delta_p_grid,
        "force_magnitude_grid": force_magnitude_grid,
        "force_x_grid": force_x_grid,
        "force_y_grid": force_y_grid,
        "force_z_grid": force_z_grid,
    }


# --------------------------------------------------------------------------------------------------


def calculate_aircraft_panel_loads(aircraft_panel_grids, aircraft_force_grids):

    aircraft_panel_loads = []

    for component_panel_grid, component_force_grid in zip(
        aircraft_panel_grids, aircraft_force_grids
    ):

        component_panel_loads = calculate_surface_panels_loads(
            component_panel_grid, component_force_grid
        )

        aircraft_panel_loads.append(component_panel_loads)

    return aircraft_panel_loads

