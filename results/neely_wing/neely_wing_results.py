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

# EXECUTE CALCULATION

from neely_wing_data import neely_wing_i
from neely_wing_simulation import (
    FLIGHT_CONDITIONS_DATA,
    SIMULATION_OPTIONS,
    neely_wing_i_results,
)

# ==================================================================================================
# PROCESSING RESULTS

components_delta_p_grids = []
components_force_mag_grids = []

for panels, forces in zip(
    neely_wing_i_results["aircraft_macrosurfaces_panels"],
    neely_wing_i_results["aircraft_force_grid"],
):

    delta_p, force = aero.vlm.calc_panels_delta_pressure(panels, forces)
    components_delta_p_grids.append(delta_p)
    components_force_mag_grids.append(force)

ax, fig = vis.plot_3D.plot_results(
    neely_wing_i_results["aircraft_original_grids"]["macrosurfaces_aero_grids"],
    components_delta_p_grids,
    title="BROEREN WING I: 2.5-08-4416, V=25 m/s, $\\alpha$=5º",
    label="Delta Pressure [Pa]",
    colormap="coolwarm",
)

interest_point = neely_wing_i.inertial_properties.position

# Aerodynamic forces in the aircraft coordinate system
total_cg_aero_force, total_cg_aero_moment, component_cg_aero_loads = loads.functions.calc_aero_loads_at_point(
    interest_point,
    neely_wing_i_results["aircraft_force_grid"],
    neely_wing_i_results["aircraft_macrosurfaces_panels"],
)

print()
print(f"# Total loads at aircraft CG:")
print(f"    FX: {total_cg_aero_force[0]} N")
print(f"    FY: {total_cg_aero_force[1]} N")
print(f"    FZ: {total_cg_aero_force[2]} N")
print(f"    RX: {total_cg_aero_moment[0]} N")
print(f"    RY: {total_cg_aero_moment[1]} N")
print(f"    RZ: {total_cg_aero_moment[2]} N")

forces, moments, coefficients = loads.functions.calc_lift_drag(
    neely_wing_i,
    interest_point,
    FLIGHT_CONDITIONS_DATA["translation_velocity"][0],
    FLIGHT_CONDITIONS_DATA["altitude"],
    FLIGHT_CONDITIONS_DATA["attitude_angles_deg"],
    neely_wing_i_results["aircraft_force_grid"],
    neely_wing_i_results["aircraft_macrosurfaces_panels"],
)

print()
print("# Aerodynamic Coeffients:")
print(f"    - Lift: {forces['lift']} N")
print(f"    - Cl: {coefficients['Cl']} N")
print(f"    - Drag: {forces['drag']} N")
print(f"    - Cd: {coefficients['Cd']} N")

# Create load distribution plots
components_loads = loads.functions.calc_load_distribution(
    aircraft_force_grid=neely_wing_i_results["aircraft_force_grid"],
    aircraft_panel_grid=neely_wing_i_results["aircraft_macrosurfaces_panels"],
    aircraft_gamma_grid=neely_wing_i_results["aircraft_gamma_grid"],
    attitude_vector=FLIGHT_CONDITIONS_DATA["attitude_angles_deg"],
    altitude=FLIGHT_CONDITIONS_DATA["altitude"],
    speed=FLIGHT_CONDITIONS_DATA["translation_velocity"][0],
)

for component in components_loads:
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_title("Lift Distribution")
    ax1.set_xlabel("Spam Position [m]")
    ax1.set_ylabel("Lift [N]")
    ax1.plot(component["y_values"], component["lift"])
    ax1.grid()

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_title("Cl Distribution")
    ax2.set_xlabel("Spam Position [m]")
    ax2.set_ylabel("Cl")
    ax2.plot(component["y_values"], component["Cl"])
    ax2.grid()

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.set_title("Drag Distribution")
    ax3.set_xlabel("Spam Position [m]")
    ax3.set_ylabel("Drag [N]")
    ax3.plot(component["y_values"], component["drag"])
    ax3.grid()

    plt.tight_layout()

plt.show()