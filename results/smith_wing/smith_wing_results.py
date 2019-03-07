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

#from smith_wing_simulation import (
#    results_case1,
#    results_case2,
#    iteration_results_case2,
#    results_case3,
#    iteration_results_case3,
#)

#f = open("results\\smith_wing\\results.pckl", "wb")
#pickle.dump(
#    [
#        results_case1,
#        results_case2,
#        iteration_results_case2,
#        results_case3,
#        iteration_results_case3,
#    ],
#    f,
#)
#f.close()

f = open("results\\smith_wing\\results\\results.pckl", "rb")
results_case1, results_case2, iteration_results_case2, results_case3, iteration_results_case3 = pickle.load(
    f
)
f.close()

# Draw Aircraft
aircraft_ax, aircraft_fig = vis.plot_3D2.generate_aircraft_plot(
    smith_wing, title="Smith Wing"
)

# ==================================================================================================
# PROCESSING RESULTS

# CASE 001:
#   - Alpha: 2º
#   - Speed: 25 m/s
#   - Altitude: 20000 m
#   - Rigid

# Generate Original vs Deformed Grid Plot

# Draw original grids
grids_ax, grids_fig = vis.plot_3D2.generate_aircraft_grids_plot(
    aircraft_macrosurfaces_aero_grids=results_case1["aircraft_original_grids"]["macrosurfaces_aero_grids"],
    aircraft_struct_fem_elements=None,
    title="Smith Wing - Case 001 - Aerodynamic Grids",
    ax=None,
    show_origin=True,
    show_nodes=False,
    line_color="k",
    alpha=0.5,
)


# Calculate Loads on each of the aerodynamic panels
aircraft_panel_loads = loads.functions.calculate_aircraft_panel_loads(
    results_case1["aircraft_macrosurfaces_panels"], results_case1["aircraft_force_grid"]
)

results_ax, results_fig = vis.plot_3D2.generate_results_plot(
    aircraft_deformed_macrosurfaces_aero_grids=results_case1["aircraft_original_grids"]["macrosurfaces_aero_grids"],
    aircraft_panel_loads=aircraft_panel_loads,
    aircraft_struct_fem_elements=None,
    aircraft_struct_deformations=None,
    results_string="delta_p_grid",
    title="Smith Wing - Case 001 - Delta Pressure [Pa]",
    colorbar_label="Delta Pressure [Pa]",
    ax=None,
    fig=None,
    show_origin=True,
    colormap="coolwarm",
)

interest_point = smith_wing.inertial_properties.position

# Aerodynamic forces in the aircraft coordinate system
total_cg_aero_force, total_cg_aero_moment, component_cg_aero_loads = loads.functions.calc_aero_loads_at_point(
    interest_point,
    results_case1["aircraft_force_grid"],
    results_case1["aircraft_macrosurfaces_panels"],
)

print()
print("########################################")
print("#           CASE 001 RESULTS           #")
print("########################################")

print()
print(f"# Total loads at aircraft CG:")
print(f"    FX: {total_cg_aero_force[0]} N")
print(f"    FY: {total_cg_aero_force[1]} N")
print(f"    FZ: {total_cg_aero_force[2]} N")
print(f"    RX: {total_cg_aero_moment[0]} N")
print(f"    RY: {total_cg_aero_moment[1]} N")
print(f"    RZ: {total_cg_aero_moment[2]} N")

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
CENTER_OF_ROTATION = smith_wing.inertial_properties.position

# Flight altitude, used to calculate atmosppheric conditions, in meters
ALTITUDE = 20000

forces, moments, coefficients = loads.functions.calc_lift_drag(
    aircraft=smith_wing,
    point=interest_point,
    speed=V_X,
    altitude=ALTITUDE,
    attitude_vector=np.array([ALPHA, BETA, GAMMA]),
    aircraft_force_grid=results_case1["aircraft_force_grid"],
    aircraft_panel_grid=results_case1["aircraft_macrosurfaces_panels"],
)

print()
print("# Aerodynamic Coeffients:")
print(f"    - Lift: {forces['lift']} N")
print(f"    - Cl: {coefficients['Cl']}")
print(f"    - Drag: {forces['drag']} N")
print(f"    - Cd: {coefficients['Cd']}")
print(f"    - Pitch Moment: {moments['pitch_moment']} N.m")
print(f"    - Cm: {coefficients['Cm']}")

# Create load distribution plots
components_loads = loads.functions.calc_load_distribution(
    aircraft_force_grid=results_case1["aircraft_force_grid"],
    aircraft_panel_grid=results_case1["aircraft_macrosurfaces_panels"],
    aircraft_gamma_grid=results_case1["aircraft_gamma_grid"],
    attitude_vector=np.array([ALPHA, BETA, GAMMA]),
    altitude=ALTITUDE,
    speed=V_X,
)

for component in components_loads:
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_title("Smith Wing - Case 001 - Lift Distribution")
    ax1.set_xlabel("Spam Position [m]")
    ax1.set_ylabel("Lift [N]")
    ax1.plot(component["y_values"], component["lift"])
    ax1.grid()

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_title("Smith Wing - Case 001 -Cl Distribution")
    ax2.set_xlabel("Spam Position [m]")
    ax2.set_ylabel("Cl")
    ax2.plot(component["y_values"], component["Cl"])
    ax2.grid()

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.set_title("Smith Wing - Case 001 - Drag Distribution")
    ax3.set_xlabel("Spam Position [m]")
    ax3.set_ylabel("Drag [N]")
    ax3.plot(component["y_values"], component["drag"])
    ax3.grid()
    plt.tight_layout()

#plt.show()

# --------------------------------------------------------------------------------------------------

# CASE 002:
#   - Alpha: 2º
#   - Speed: 25 m/s
#   - Altitude: 20000 m
#   - Flexible

# Generate Original vs Deformed Grid Plot

# Draw original grids
grids_ax, grids_fig = vis.plot_3D2.generate_aircraft_grids_plot(
    results_case2["aircraft_original_grids"]["macrosurfaces_aero_grids"],
    results_case2["aircraft_struct_fem_elements"],
    title="Smith Wing - Case 002 - Original vs Deformed Grids",
    ax=None,
    show_origin=True,
    show_nodes=False,
    line_color="k",
    alpha=0.5,
)


# Draw deformed Grids
grids_ax, grids_fig = vis.plot_3D2.generate_deformed_aircraft_grids_plot(
    results_case2["aircraft_deformed_macrosurfaces_aero_grids"],
    results_case2["aircraft_struct_fem_elements"],
    results_case2["aircraft_struct_deformations"],
    ax=grids_ax,
    fig=grids_fig,
    show_origin=True,
    show_nodes=False,
    line_color="r",
    alpha=1,
)

# Calculate Loads on each of the aerodynamic panels
aircraft_panel_loads = loads.functions.calculate_aircraft_panel_loads(
    results_case2["original_aircraft_panel_grid"], results_case2["aircraft_force_grid"]
)

results_ax, results_fig = vis.plot_3D2.generate_results_plot(
    aircraft_deformed_macrosurfaces_aero_grids=results_case2["aircraft_deformed_macrosurfaces_aero_grids"],
    aircraft_panel_loads=aircraft_panel_loads,
    aircraft_struct_fem_elements=results_case2["aircraft_struct_fem_elements"],
    aircraft_struct_deformations=results_case2["aircraft_struct_deformations"],
    results_string="delta_p_grid",
    title="Smith Wing - Case 002 - Delta Pressure [Pa]",
    colorbar_label="Delta Pressure [Pa]",
    ax=None,
    fig=None,
    show_origin=True,
    colormap="coolwarm",
)

# plt.show()

# Deformation plot

deformation_table = aelast.functions.calculate_deformation_table(
    results_case2["aircraft_original_grids"],
    results_case2["aircraft_struct_deformations"],
)

# sort nodes by desired column, in this case the Y coordinate

nodes = deformation_table["aircraft_macrosurfaces_deformed_nodes"][0]
nodes = nodes[nodes[:, 1].argsort()]

# Plot Bending
fig, ax = plt.subplots()
ax.plot(nodes[:, 1], nodes[:, 2])
ax.grid()
ax.set_title("Smith Wing - Case 002 - Bending")
ax.set_ylabel("Bending [m]")
ax.set_xlabel("Span [m]")

# Plot Torsion
fig, ax = plt.subplots()
ax.plot(nodes[:, 1], np.degrees(nodes[:, 4]))
ax.grid()
ax.set_title("Smith Wing - Case 002 - Torsion")
ax.set_ylabel("Torsion [degrees]")
ax.set_xlabel("Span [m]")

interest_point = smith_wing.inertial_properties.position

# Aerodynamic forces in the aircraft coordinate system
total_cg_aero_force, total_cg_aero_moment, component_cg_aero_loads = loads.functions.calc_aero_loads_at_point(
    interest_point,
    results_case2["aircraft_force_grid"],
    results_case2["aircraft_deformed_macrosurfaces_aero_panels"],
)

print()
print("########################################")
print("#           CASE 002 RESULTS           #")
print("########################################")
print()
print(f"# Total loads at aircraft CG:")
print(f"    FX: {total_cg_aero_force[0]} N")
print(f"    FY: {total_cg_aero_force[1]} N")
print(f"    FZ: {total_cg_aero_force[2]} N")
print(f"    RX: {total_cg_aero_moment[0]} N")
print(f"    RY: {total_cg_aero_moment[1]} N")
print(f"    RZ: {total_cg_aero_moment[2]} N")

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
CENTER_OF_ROTATION = smith_wing.inertial_properties.position

# Flight altitude, used to calculate atmosppheric conditions, in meters
ALTITUDE = 20000

forces, moments, coefficients = loads.functions.calc_lift_drag(
    aircraft=smith_wing,
    point=interest_point,
    speed=V_X,
    altitude=ALTITUDE,
    attitude_vector=np.array([ALPHA, BETA, GAMMA]),
    aircraft_force_grid=results_case2["aircraft_force_grid"],
    aircraft_panel_grid=results_case2["aircraft_deformed_macrosurfaces_aero_panels"],
)

print()
print("# Aerodynamic Coeffients:")
print(f"    - Lift: {forces['lift']} N")
print(f"    - Cl: {coefficients['Cl']}")
print(f"    - Drag: {forces['drag']} N")
print(f"    - Cd: {coefficients['Cd']}")
print(f"    - Pitch Moment: {moments['pitch_moment']} N.m")
print(f"    - Cm: {coefficients['Cm']}")

# Create load distribution plots
components_loads = loads.functions.calc_load_distribution(
    aircraft_force_grid=results_case2["aircraft_force_grid"],
    aircraft_panel_grid=results_case2["original_aircraft_panel_grid"],
    aircraft_gamma_grid=results_case1["aircraft_gamma_grid"],
    attitude_vector=np.array([ALPHA, BETA, GAMMA]),
    altitude=ALTITUDE,
    speed=V_X,
)

for component in components_loads:
    fig = plt.figure()

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_title("Smith Wing - Case 002 - Lift Distribution")
    ax1.set_xlabel("Spam Position [m]")
    ax1.set_ylabel("Lift [N]")
    ax1.plot(component["y_values"], component["lift"])
    ax1.grid()

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_title("Smith Wing - Case 002 - Cl Distribution")
    ax2.set_xlabel("Spam Position [m]")
    ax2.set_ylabel("Cl")
    ax2.plot(component["y_values"], component["Cl"])
    ax2.grid()

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.set_title("Smith Wing - Case 002 - Drag Distribution")
    ax3.set_xlabel("Spam Position [m]")
    ax3.set_ylabel("Drag [N]")
    ax3.plot(component["y_values"], component["drag"])
    ax3.grid()
    plt.tight_layout()

#plt.show()

# --------------------------------------------------------------------------------------------------

# CASE 003:
#   - Alpha: 4º
#   - Speed: 25 m/s
#   - Altitude: 20000 m
#   - Flexible

# Generate Original vs Deformed Grid Plot

# Draw original grids
grids_ax, grids_fig = vis.plot_3D2.generate_aircraft_grids_plot(
    results_case3["aircraft_original_grids"]["macrosurfaces_aero_grids"],
    results_case3["aircraft_struct_fem_elements"],
    title="Smith Wing - Case 003 - Original vs Deformed Grids",
    ax=None,
    show_origin=True,
    show_nodes=False,
    line_color="k",
    alpha=0.5,
)


# Draw deformed Grids
grids_ax, grids_fig = vis.plot_3D2.generate_deformed_aircraft_grids_plot(
    results_case3["aircraft_deformed_macrosurfaces_aero_grids"],
    results_case3["aircraft_struct_fem_elements"],
    results_case3["aircraft_struct_deformations"],
    ax=grids_ax,
    fig=grids_fig,
    show_origin=True,
    show_nodes=False,
    line_color="r",
    alpha=1,
)

# Calculate Loads on each of the aerodynamic panels
aircraft_panel_loads = loads.functions.calculate_aircraft_panel_loads(
    results_case3["original_aircraft_panel_grid"], results_case3["aircraft_force_grid"]
)

results_ax, results_fig = vis.plot_3D2.generate_results_plot(
    aircraft_deformed_macrosurfaces_aero_grids=results_case3["aircraft_deformed_macrosurfaces_aero_grids"],
    aircraft_panel_loads=aircraft_panel_loads,
    aircraft_struct_fem_elements=results_case3["aircraft_struct_fem_elements"],
    aircraft_struct_deformations=results_case3["aircraft_struct_deformations"],
    results_string="delta_p_grid",
    title="Smith Wing - Case 003 - Delta Pressure [Pa]",
    colorbar_label="Delta Pressure [Pa]",
    ax=None,
    fig=None,
    show_origin=True,
    colormap="coolwarm",
)

# plt.show()

# Deformation plot

deformation_table = aelast.functions.calculate_deformation_table(
    results_case3["aircraft_original_grids"],
    results_case3["aircraft_struct_deformations"],
)

# sort nodes by desired column, in this case the Y coordinate

nodes = deformation_table["aircraft_macrosurfaces_deformed_nodes"][0]
nodes = nodes[nodes[:, 1].argsort()]

# Plot Bending
fig, ax = plt.subplots()
ax.plot(nodes[:, 1], nodes[:, 2])
ax.grid()
ax.set_title("Smith Wing - Case 003 - Bending")
ax.set_ylabel("Bending [m]")
ax.set_xlabel("Span [m]")

# Plot Torsion
fig, ax = plt.subplots()
ax.plot(nodes[:, 1], np.degrees(nodes[:, 4]))
ax.grid()
ax.set_title("Smith Wing - Case 003 - Torsion")
ax.set_ylabel("Torsion [degrees]")
ax.set_xlabel("Span [m]")

interest_point = smith_wing.inertial_properties.position

# Aerodynamic forces in the aircraft coordinate system
total_cg_aero_force, total_cg_aero_moment, component_cg_aero_loads = loads.functions.calc_aero_loads_at_point(
    interest_point,
    results_case3["aircraft_force_grid"],
    results_case3["aircraft_deformed_macrosurfaces_aero_panels"],
)

print()
print("########################################")
print("#           CASE 003 RESULTS           #")
print("########################################")
print()
print(f"# Total loads at aircraft CG:")
print(f"    FX: {total_cg_aero_force[0]} N")
print(f"    FY: {total_cg_aero_force[1]} N")
print(f"    FZ: {total_cg_aero_force[2]} N")
print(f"    RX: {total_cg_aero_moment[0]} N")
print(f"    RY: {total_cg_aero_moment[1]} N")
print(f"    RZ: {total_cg_aero_moment[2]} N")

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
CENTER_OF_ROTATION = smith_wing.inertial_properties.position

# Flight altitude, used to calculate atmosppheric conditions, in meters
ALTITUDE = 20000

forces, moments, coefficients = loads.functions.calc_lift_drag(
    aircraft=smith_wing,
    point=interest_point,
    speed=V_X,
    altitude=ALTITUDE,
    attitude_vector=np.array([ALPHA, BETA, GAMMA]),
    aircraft_force_grid=results_case3["aircraft_force_grid"],
    aircraft_panel_grid=results_case3["aircraft_deformed_macrosurfaces_aero_panels"],
)

print()
print("# Aerodynamic Coeffients:")
print(f"    - Lift: {forces['lift']} N")
print(f"    - Cl: {coefficients['Cl']}")
print(f"    - Drag: {forces['drag']} N")
print(f"    - Cd: {coefficients['Cd']}")
print(f"    - Pitch Moment: {moments['pitch_moment']} N.m")
print(f"    - Cm: {coefficients['Cm']}")

# Create load distribution plots
components_loads = loads.functions.calc_load_distribution(
    aircraft_force_grid=results_case3["aircraft_force_grid"],
    aircraft_panel_grid=results_case3["original_aircraft_panel_grid"],
    aircraft_gamma_grid=results_case1["aircraft_gamma_grid"],
    attitude_vector=np.array([ALPHA, BETA, GAMMA]),
    altitude=ALTITUDE,
    speed=V_X,
)

for component in components_loads:
    fig = plt.figure()

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_title("Smith Wing - Case 003 - Lift Distribution")
    ax1.set_xlabel("Spam Position [m]")
    ax1.set_ylabel("Lift [N]")
    ax1.plot(component["y_values"], component["lift"])
    ax1.grid()

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_title("Smith Wing - Case 003 -Cl Distribution")
    ax2.set_xlabel("Spam Position [m]")
    ax2.set_ylabel("Cl")
    ax2.plot(component["y_values"], component["Cl"])
    ax2.grid()

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.set_title("Smith Wing - Case 003 - Drag Distribution")
    ax3.set_xlabel("Spam Position [m]")
    ax3.set_ylabel("Drag [N]")
    ax3.plot(component["y_values"], component["drag"])
    ax3.grid()
    plt.tight_layout()


a = results_case1["aircraft_force_grid"]
b = results_case2["aircraft_force_grid"]
c = results_case3["aircraft_force_grid"]
plt.show()
