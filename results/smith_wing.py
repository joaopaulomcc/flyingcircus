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
# GEOMETRY DEFINITION
print()
print("# Generating geometry...")
# Wing section

# Aifoil name, only informative
naca0012 = "NACA 0012"

# Material properties, all equal to one as Smith et al only provies the stiffness characteristics
# of the wing
material = struct.objects.Material(
    name="material",
    density=1,
    elasticity_modulus=1,
    rigidity_modulus=1,
    poisson_ratio=1,
    yield_tensile_stress=1,
    ultimate_tensile_stress=1,
    yield_shear_stress=1,
    ultimate_shear_stress=1,
)

# Wing section properties
wing_section = geo.objects.Section(
    identifier=naca0012,
    material=material,
    area=0.75,
    Iyy=2e4,
    Izz=2e4,
    J=1e4,
    shear_center=0.5,
)

# --------------------------------------------------------------------------------------------------

# Wing surface

# Definition of the wing planform
left_wing_surface = geo.objects.Surface(
    identifier="left_wing",
    root_chord=1,
    root_section=wing_section,
    tip_chord=1,
    tip_section=wing_section,
    length=16,
    leading_edge_sweep_angle_deg=0,
    dihedral_angle_deg=0,
    tip_torsion_angle_deg=0,
    control_surface_hinge_position=None,
)

right_wing_surface = geo.objects.Surface(
    identifier="right_wing",
    root_chord=1,
    root_section=wing_section,
    tip_chord=1,
    tip_section=wing_section,
    length=16,
    leading_edge_sweep_angle_deg=0,
    dihedral_angle_deg=0,
    tip_torsion_angle_deg=0,
    control_surface_hinge_position=None,
)

# Creation of the wing macrosurface
wing = geo.objects.MacroSurface(
    position=np.array([0, 0, 0]),
    incidence=0,
    surface_list=[left_wing_surface, right_wing_surface],
    symmetry_plane="XZ",
    torsion_center=0.5,
)

# --------------------------------------------------------------------------------------------------

# Aircraft definition

smith_wing = geo.objects.Aircraft(
    name="Smith Wing",
    macro_surfaces=[wing],
    inertial_properties=geo.objects.MaterialPoint(),
)

# vis.plot_3D.plot_aircraft(smith_wing)


# ==================================================================================================
# GRID CREATION
print("# Generating aerodynamic and structural grid...")

# Number of panels and finite elements
n_chord_panels = 3
n_span_panels = 3
n_beam_elements = 6

wing_n_chord_panels = n_chord_panels
wing_n_span_panels_list = [n_span_panels, n_span_panels]
wing_n_beam_elements_list = [n_beam_elements, n_beam_elements]

# Types of discretization to be used
wing_chord_discretization = "linear"
wing_span_discretization_list = ["linear", "linear"]
wing_torsion_function_list = ["linear", "linear"]

# Creation of the wing grids
wing_aero_grid, wing_struct_grid = wing.create_grids(
    wing_n_chord_panels,
    wing_n_span_panels_list,
    wing_n_beam_elements_list,
    wing_chord_discretization,
    wing_span_discretization_list,
    wing_torsion_function_list,
)

aero_grid = [wing_aero_grid]
struct_grid = wing_struct_grid

# ==================================================================================================
# STRUCTURE DEFINITION

# Creation of the wing structural connections
struct_connections = struct.fem.create_macrosurface_connections(wing)

# Number the wing nodes
struct.fem.number_nodes(
    [left_wing_surface, right_wing_surface], wing_struct_grid, struct_connections
)

# Create wing finite elements
wing_fem_elements = struct.fem.generate_macrosurface_fem_elements(
    macrosurface=wing,
    macrosurface_nodes_list=wing_struct_grid,
    prop_choice="ROOT"
)

struct_elements = wing_fem_elements

# Generate Constraints
wing_fixation = struct.objects.Constraint(
    application_node=wing_struct_grid[0][0],
    dof_constraints=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
)

struct_constraints = [wing_fixation]

vis.plot_3D.plot_structure(struct_elements)

# ==================================================================================================
# FLUID STRUCTURE INTERACTION MATRICES CALCULATION

loads_to_nodes_matrix = aelast.functions.loads_to_nodes_weight_matrix(
    wing_aero_grid, wing_struct_grid
)

deformation_to_aero_grid_weight_matrix = aelast.functions.deformation_to_aero_grid_weight_matrix(
    wing_aero_grid, wing_struct_grid
)

# ==================================================================================================
# AERODYNAMIC LOADS CALCULATION - CASE 1 - ALPHA 2º
print()
print("# CASE 001:")
print(f"    - Altitude: 20000m")
print(f"    - True Airspeed: 25m/s")
print(f"    - Alpha: 2º")
print(f"    - Rigid Wing")
print("- Setting flight conditions...")
# Flight Conditions definition

# Translation velocities
V_x = 25
V_y = 0
V_z = 0
velocity_vector = np.array([V_x, V_y, V_z])

# Rotation velocities
rotation_vector = np.array([0, 0, 0])

# Attitude angles in degrees
alpha = 2
beta = 0
gamma = 0

attitude_vector = np.array([alpha, beta, gamma])

# Center of rotation is set to the origin
center = np.array([0.5, 0, 0])

# Altitude is set to sea level
altitude = 20000


print("- Running calculation...")
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

components_delta_p_grids = []
components_force_mag_grids = []

for panels, forces in zip(components_panel_grid, components_force_grid):

    delta_p, force = aero.vlm.calc_panels_delta_pressure(panels, forces)
    components_delta_p_grids.append(delta_p)
    components_force_mag_grids.append(force)

ax, fig = vis.plot_3D.plot_results(
    aero_grid,
    components_delta_p_grids,
    title="Smith Wing - alpha: 2º",
    label="Delta Pressure [Pa]",
    colormap="coolwarm",
)

# ==================================================================================================
# DEFORMATION CALCULATION USING

macro_surface_loads = aelast.functions.generated_aero_loads(wing_aero_grid, components_force_grid[0], wing_struct_grid)

struct_loads = macro_surface_loads

deformed_grid, force_vector, deformations, node_vector = struct.fem.structural_solver(
    struct_grid, struct_elements, struct_loads, struct_constraints
)

vis.plot_3D.plot_deformed_structure(struct_elements, node_vector, deformations, scale_factor=1)

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
