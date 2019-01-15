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
print()
print("============================================================")
print("= VALIDATION OF AEROELASTIC CALCULATION                    =")
print("= VALIDATION CASE: CFD-Based Analysis of Nonlinear         =")
print("= Aeroelastic Behavior of High-Aspect Ratio Wings          =")
print("= AUTHORS: M. J. Smith, M. J. Patil, D. H. Hodges          =")
print("============================================================")
# ==================================================================================================
# EXECUTE CALCULATION

from smith_wing_data import smith_wing
from smith_wing_simulation import results, iteration_results

# ==================================================================================================
# PROCESSING RESULTS

components_delta_p_grids = []
components_force_mag_grids = []

for panels, forces in zip(
    results["aircraft_deformed_macrosurfaces_aero_panels"],
    results["aircraft_force_grid"],
):

    delta_p, force = aero.vlm.calc_panels_delta_pressure(panels, forces)
    components_delta_p_grids.append(delta_p)
    components_force_mag_grids.append(force)

ax, fig = vis.plot_3D.plot_results(
    results["aircraft_deformed_macrosurfaces_aero_grids"],
    components_delta_p_grids,
    title="Smith Wing - alpha: 2º - 10 Iterations",
    label="Delta Pressure [Pa]",
    colormap="coolwarm",
)

plt.show()

print(results["aircraft_struct_deformations"])

# Aerodynamic forces in the aircraft coordinate system
total_cg_aero_force, total_cg_aero_moment, component_cg_aero_loads = loads.functions.cg_aero_loads(
    smith_wing,
    components_force_vector, components_panel_vector
)

# Aerodynamic forces in the wind coordinate system
forces, moments, coefficients = loads.functions.lift_drag(
    smith_wing,
    velocity_vector,
    altitude,
    attitude_vector,
    components_force_vector,
    components_panel_vector,
)

print("- Results:")
print(f"    - Lift: {forces['lift']} N")
print(f"    - Cl: {coefficients['Cl']} N")
print(f"    - Drag: {forces['drag']} N")
print(f"    - Cd: {coefficients['Cd']} N")
# Write Results to file
report_file = open("results/smith_wing_report.txt", "w")
report_file.write("============================================================\n")
report_file.write("= VALIDATION OF AEROELASTIC CALCULATION                    =\n")
report_file.write("= VALIDATION CASE: CFD-Based Analysis of Nonlinear         =\n")
report_file.write("= Aeroelastic Behavior of High-Aspect Ratio Wings          =\n")
report_file.write("= AUTHORS: M. J. Smith, M. J. Patil, D. H. Hodges          =\n")
report_file.write("============================================================\n")
report_file.write("\n")
report_file.write("CASE 1:\n")
report_file.write("# Aerodynamic loads in the aircraft coordinate system:\n")
report_file.write(f"- X force: {total_cg_aero_force[0]} [N]\n")
report_file.write(f"- Y force: {total_cg_aero_force[1]} [N]\n")
report_file.write(f"- Z force: {total_cg_aero_force[2]} [N]\n")
report_file.write(f"- X moment: {total_cg_aero_moment[0]} [N.m]\n")
report_file.write(f"- Y moment: {total_cg_aero_moment[1]} [N.m]\n")
report_file.write(f"- Z moment: {total_cg_aero_moment[2]} [N.m]\n")
report_file.write("\n")
report_file.write("# Aerodynamic loads in the wind coordinate system:\n")
report_file.write(f"- Lift: {forces['lift']} [N]\n")
report_file.write(f"- Drag: {forces['drag']} [N]\n")
report_file.write(f"- Sideforce: {forces['sideforce']} [N]\n")
report_file.write(f"- Roll moment: {moments['roll_moment']} [N.m]\n")
report_file.write(f"- Pitch moment: {moments['pitch_moment']} [N.m]\n")
report_file.write(f"- Yaw moment: {moments['yaw_moment']} [N.m]\n")
report_file.write("\n")
report_file.write("# Aerodynamic Coefficients:\n")
report_file.write(f"- Cl : {coefficients['Cl']}\n")
report_file.write(f"- Cd : {coefficients['Cd']}\n")
report_file.write(f"- Cm : {coefficients['Cm']}\n")
report_file.write("\n")

# Create load distribution plots
components_loads = loads.functions.load_distribution(
    components_force_grid,
    components_panel_grid,
    attitude_vector,
    altitude,
    velocity_vector,
)

for component in components_loads:
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_title("Lift Distribution")
    ax1.set_xlabel("Spam Position [m]")
    ax1.set_ylabel("Lift [N]")
    ax1.plot(component["y_values"], component["lift"])

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_title("Cl Distribution")
    ax2.set_xlabel("Spam Position [m]")
    ax2.set_ylabel("Cl")
    ax2.plot(component["y_values"], component["Cl"])
    plt.tight_layout()

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.set_title("Drag Distribution")
    ax3.set_xlabel("Spam Position [m]")
    ax3.set_ylabel("Drag [N]")
    ax3.plot(component["y_values"], component["drag"])
    plt.tight_layout()

sys.exit()


# ==================================================================================================
# AERODYNAMIC LOADS CALCULATION - CASE 2 - ALPHA 4º

print()
print("# CASE 002:")
print(f"    - Altitude: 20000m")
print(f"    - True Airspeed: 25m/s")
print(f"    - Alpha: 4º")
print(f"    - Rigid Wing")
print("- Setting flight conditions...")

# Attitude angles in degrees
alpha = 4
beta = 0
gamma = 0

attitude_vector = np.array([alpha, beta, gamma])

print("- Running calculation:")
(
    components_force_vector,
    components_panel_vector,
    components_gamma_vector,
    components_force_grid,
    components_panel_grid,
    components_gamma_grid,
) = aero.vlm.aero_loads(
    aero_grid, velocity_vector, rotation_vector, attitude_vector, altitude, center
)

# ==================================================================================================
# PROCESSING RESULTS


# Aerodynamic forces in the aircraft coordinate system
total_cg_aero_force, total_cg_aero_moment, component_cg_aero_loads = loads.functions.cg_aero_loads(
    smith_wing, components_force_vector, components_panel_vector
)

# Aerodynamic forces in the wind coordinate system
forces, moments, coefficients = loads.functions.lift_drag(
    smith_wing,
    velocity_vector,
    altitude,
    attitude_vector,
    components_force_vector,
    components_panel_vector,
)

print("- Results:")
print(f"    - Lift: {forces['lift']} N")
print(f"    - Cl: {coefficients['Cl']} N")
print(f"    - Drag: {forces['drag']} N")
print(f"    - Cd: {coefficients['Cd']} N")

report_file.write("\n")
report_file.write("CASE 2:\n")
report_file.write("# Aerodynamic loads in the aircraft coordinate system:\n")
report_file.write(f"- X force: {total_cg_aero_force[0]} [N]\n")
report_file.write(f"- Y force: {total_cg_aero_force[1]} [N]\n")
report_file.write(f"- Z force: {total_cg_aero_force[2]} [N]\n")
report_file.write(f"- X moment: {total_cg_aero_moment[0]} [N.m]\n")
report_file.write(f"- Y moment: {total_cg_aero_moment[1]} [N.m]\n")
report_file.write(f"- Z moment: {total_cg_aero_moment[2]} [N.m]\n")
report_file.write("\n")
report_file.write("# Aerodynamic loads in the wind coordinate system:\n")
report_file.write(f"- Lift: {forces['lift']} [N]\n")
report_file.write(f"- Drag: {forces['drag']} [N]\n")
report_file.write(f"- Sideforce: {forces['sideforce']} [N]\n")
report_file.write(f"- Roll moment: {moments['roll_moment']} [N.m]\n")
report_file.write(f"- Pitch moment: {moments['pitch_moment']} [N.m]\n")
report_file.write(f"- Yaw moment: {moments['yaw_moment']} [N.m]\n")
report_file.write("\n")
report_file.write("# Aerodynamic Coefficients:\n")
report_file.write(f"- Cl : {coefficients['Cl']}\n")
report_file.write(f"- Cd : {coefficients['Cd']}\n")
report_file.write(f"- Cm : {coefficients['Cm']}\n")
report_file.write("\n")

components_delta_p_grids = []
components_force_mag_grids = []

for panels, forces in zip(components_panel_grid, components_force_grid):

    delta_p, force = aero.vlm.calc_panels_delta_pressure(panels, forces)
    components_delta_p_grids.append(delta_p)
    components_force_mag_grids.append(force)

ax, fig = vis.plot_3D.plot_results(
    aero_grid,
    components_delta_p_grids,
    title="Smith Wing - alpha: 4º",
    label="Delta Pressure [Pa]",
    colormap="coolwarm",
)

report_file.close()
plt.show()
