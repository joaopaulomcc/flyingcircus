import numpy as np
import matplotlib.pyplot as plt

from context import src
from src import geometry as geo
from src import visualization as vis
from src import aerodynamics as aero
from src import loads

# ==================================================================================================
# ==================================================================================================
# Simple Aircraft for testing purposes

# WING

# Stub
root_chord = 2
root_section = "stub_root_section"
tip_chord = 1.5
tip_section = "stub_tip_section"
length = 2
leading_edge_sweep_angle_deg = 30
dihedral_angle_deg = 5
tip_torsion_angle_deg = -1
control_surface_hinge_position = None

surface_identifier = "right_stub"
right_stub = geo.objects.Surface(
    surface_identifier,
    root_chord,
    root_section,
    tip_chord,
    tip_section,
    length,
    leading_edge_sweep_angle_deg,
    dihedral_angle_deg,
    tip_torsion_angle_deg,
    control_surface_hinge_position,
)

surface_identifier = "left_stub"
left_stub = geo.objects.Surface(
    surface_identifier,
    root_chord,
    root_section,
    tip_chord,
    tip_section,
    length,
    leading_edge_sweep_angle_deg,
    dihedral_angle_deg,
    tip_torsion_angle_deg,
    control_surface_hinge_position,
)

# --------------------------------------------------------------------------------------------------
# Aileron

root_chord = 1.5
root_section = "aileron_root_section"
tip_chord = 1
tip_section = "aileron_tip_section"
length = 2
leading_edge_sweep_angle_deg = 30
dihedral_angle_deg = 5
tip_torsion_angle_deg = -2
control_surface_hinge_position = 0.75

surface_identifier = "right_aileron"
right_aileron = geo.objects.Surface(
    surface_identifier,
    root_chord,
    root_section,
    tip_chord,
    tip_section,
    length,
    leading_edge_sweep_angle_deg,
    dihedral_angle_deg,
    tip_torsion_angle_deg,
    control_surface_hinge_position,
)

surface_identifier = "left_aileron"
left_aileron = geo.objects.Surface(
    surface_identifier,
    root_chord,
    root_section,
    tip_chord,
    tip_section,
    length,
    leading_edge_sweep_angle_deg,
    dihedral_angle_deg,
    tip_torsion_angle_deg,
    control_surface_hinge_position,
)

# --------------------------------------------------------------------------------------------------
# Winglet

root_chord = 1
root_section = "aileron_root_section"
tip_chord = 0.3
tip_section = "aileron_tip_section"
length = 1
leading_edge_sweep_angle_deg = 45
dihedral_angle_deg = 90
tip_torsion_angle_deg = 0
control_surface_hinge_position = None

surface_identifier = "right_winglet"
right_winglet = geo.objects.Surface(
    surface_identifier,
    root_chord,
    root_section,
    tip_chord,
    tip_section,
    length,
    leading_edge_sweep_angle_deg,
    dihedral_angle_deg,
    tip_torsion_angle_deg,
    control_surface_hinge_position,
)

surface_identifier = "left_winglet"
left_winglet = geo.objects.Surface(
    surface_identifier,
    root_chord,
    root_section,
    tip_chord,
    tip_section,
    length,
    leading_edge_sweep_angle_deg,
    dihedral_angle_deg,
    tip_torsion_angle_deg,
    control_surface_hinge_position,
)

# --------------------------------------------------------------------------------------------------
# Wing macrosurface
wing_surface_list = [
    left_winglet,
    left_aileron,
    left_stub,
    right_stub,
    right_aileron,
    right_winglet,
]
# wing_surface_list = [left_stub, right_stub]
wing_incidence = 2
wing_position = np.array([0.0, 0, 0])
wing_symmetry_plane = "XZ"

wing = geo.objects.MacroSurface(
    wing_position, wing_incidence, wing_surface_list, symmetry_plane=wing_symmetry_plane
)

# ==================================================================================================
# Horizontal Tail
root_chord = 1
root_section = "elevator_root_section"
tip_chord = 0.6
tip_section = "elevator_tip_section"
length = 2
leading_edge_sweep_angle_deg = 25
dihedral_angle_deg = 0
tip_torsion_angle_deg = 0
control_surface_hinge_position = 0.75

surface_identifier = "right_elevator"
right_elevator = geo.objects.Surface(
    surface_identifier,
    root_chord,
    root_section,
    tip_chord,
    tip_section,
    length,
    leading_edge_sweep_angle_deg,
    dihedral_angle_deg,
    tip_torsion_angle_deg,
    control_surface_hinge_position,
)

surface_identifier = "left_elevator"
left_elevator = geo.objects.Surface(
    surface_identifier,
    root_chord,
    root_section,
    tip_chord,
    tip_section,
    length,
    leading_edge_sweep_angle_deg,
    dihedral_angle_deg,
    tip_torsion_angle_deg,
    control_surface_hinge_position,
)

h_tail_surface_list = [left_elevator, right_elevator]
h_tail_incidence = 0
h_tail_position = np.array([7, 0, 0.5])
h_tail_symmetry_plane = "XZ"

h_tail = geo.objects.MacroSurface(
    h_tail_position,
    h_tail_incidence,
    h_tail_surface_list,
    symmetry_plane=h_tail_symmetry_plane,
)

# ==================================================================================================
# Vertical Tail
root_chord = 1
root_section = "right_aileron_root_section"
tip_chord = 0.5
tip_section = "right_aileron_tip_section"
length = 1.5
leading_edge_sweep_angle_deg = 45
dihedral_angle_deg = 90
tip_torsion_angle_deg = 0
control_surface_hinge_position = 0.6

surface_identifier = "rudder"
rudder = geo.objects.Surface(
    surface_identifier,
    root_chord,
    root_section,
    tip_chord,
    tip_section,
    length,
    leading_edge_sweep_angle_deg,
    dihedral_angle_deg,
    tip_torsion_angle_deg,
    control_surface_hinge_position,
)
v_tail_surface_list = [rudder]
v_tail_incidence = 0
v_tail_position = np.array([7, 0, 0.5])
v_tail_symmetry_plane = None

v_tail = geo.objects.MacroSurface(
    v_tail_position,
    v_tail_incidence,
    v_tail_surface_list,
    symmetry_plane=v_tail_symmetry_plane,
)

# ==================================================================================================
# ==================================================================================================
# Engine Definition


def engine_thrust_function(throtle, parameters):

    return 4000 * throtle


engine_position = np.array([0, -2.5, 0])
engine_inertial_properties = "engine inertial properties"
engine_thrust_vector = np.array([-1, 0, 0])

aircraft_engine_1 = geo.objects.Engine(
    engine_position,
    engine_inertial_properties,
    engine_thrust_vector,
    engine_thrust_function,
)

engine_position = np.array([0, 2.5, 0])
aircraft_engine_2 = geo.objects.Engine(
    engine_position,
    engine_inertial_properties,
    engine_thrust_vector,
    engine_thrust_function,
)

# ==================================================================================================
# ==================================================================================================
# Aircraft CG Definition

mass = 4000
cg_position = np.array([3, 0, 0])
Ixx = 1
Iyy = 1
Izz = 1
Ixy = 1
Ixz = 1
Iyz = 1

aircraft_inertia = geo.objects.MaterialPoint(
    mass, cg_position, Ixx, Iyy, Izz, Ixy, Ixz, Iyz
)

# ==================================================================================================
# ==================================================================================================
# Aircraft Definition

name = "Simple Aircraft"

components = [wing, h_tail, v_tail]

engines = [aircraft_engine_1, aircraft_engine_2]

inertial_properties = aircraft_inertia

simple_aircraft = geo.objects.Aircraft(name, components, engines, inertial_properties)


vis.plot_3D.plot_aircraft(simple_aircraft)

# ==================================================================================================
# ==================================================================================================
# Aerodynamic Loads Calculation

# Mesh Generation
n_chord_panels = 5
n_span_panels = 10

# Wing
wing_n_chord_panels = n_chord_panels
wing_n_span_panels_list = [n_span_panels, n_span_panels, n_span_panels, n_span_panels]
wing_chord_discretization = "linear"
wing_span_discretization_list = ["linear", "linear", "linear", "linear"]
wing_torsion_function_list = ["linear", "linear", "linear", "linear"]

wing_control_surface_deflection_dict = {"left_aileron": 10, "right_aileron": 0}

wing_mesh = wing.create_mesh(
    wing_n_chord_panels,
    wing_n_span_panels_list,
    wing_chord_discretization,
    wing_span_discretization_list,
    wing_torsion_function_list,
    wing_control_surface_deflection_dict,
)

# ==================================================================================================
# Horizontal Tail
h_tail_n_chord_panels = n_chord_panels
h_tail_n_span_panels_list = [n_span_panels, n_span_panels]
h_tail_chord_discretization = "linear"
h_tail_span_discretization_list = ["linear", "linear"]
h_tail_torsion_function_list = ["linear", "linear"]

h_tail_control_surface_deflection_dict = {"left_elevator": 9.65, "right_elevator": 9.65}

h_tail_mesh = h_tail.create_mesh(
    h_tail_n_chord_panels,
    h_tail_n_span_panels_list,
    h_tail_chord_discretization,
    h_tail_span_discretization_list,
    h_tail_torsion_function_list,
    h_tail_control_surface_deflection_dict,
)

# ==================================================================================================
# Vertical Tail
v_tail_n_chord_panels = n_chord_panels
v_tail_n_span_panels_list = [n_span_panels]
v_tail_chord_discretization = "linear"
v_tail_span_discretization_list = ["linear"]
v_tail_torsion_function_list = ["linear"]

v_tail_control_surface_deflection_dict = {"rudder": 0}

v_tail_mesh = v_tail.create_mesh(
    v_tail_n_chord_panels,
    v_tail_n_span_panels_list,
    v_tail_chord_discretization,
    v_tail_span_discretization_list,
    v_tail_torsion_function_list,
    v_tail_control_surface_deflection_dict,
)

# ==================================================================================================
# Aircraft Mesh
simple_aircraft_aero_mesh = [wing_mesh, h_tail_mesh, v_tail_mesh]
# simple_aircraft_aero_mesh = [wing_mesh, h_tail_mesh]

# vis.plot_3D.plot_mesh(simple_aircraft_aero_mesh)


aircraft_aero_mesh = simple_aircraft_aero_mesh
velocity_vector = np.array([100, 0, 0])
rotation_vector = np.array([0, 0, 0])
attitude_vector = np.array([5, 0, 0])
center = simple_aircraft.inertial_properties.cg_position
altitude = 0

"""
print()
print("# Running VLM:")
(
    components_force_vector,
    components_panel_vector,
    components_gamma_vector,
    components_force_grid,
    components_panel_grid,
    components_gamma_grid,
) = aero.vlm.aero_loads(
    aircraft_aero_mesh,
    velocity_vector,
    rotation_vector,
    attitude_vector,
    altitude,
    center,
)

components_delta_p_grids = []
components_force_mag_grids = []

for panels, forces in zip(components_panel_grid, components_force_grid):

    delta_p, force = aero.vlm.calc_panels_delta_pressure(panels, forces)
    components_delta_p_grids.append(delta_p)
    components_force_mag_grids.append(force)

vis.plot_3D.plot_results(simple_aircraft_aero_mesh, components_force_mag_grids)

total_cg_aero_force, total_cg_aero_moment, component_cg_aero_loads = loads.functions.cg_aero_loads(
    simple_aircraft, components_force_vector, components_panel_vector
)

throtle_list = np.array([1, 1])
parameters_list = [None, None]
engine_force, engine_moment, engine_loads = loads.functions.cg_engine_loads(
    simple_aircraft, throtle_list, parameters_list
)

print()
print("# CG Aero Loads:")
print(f"- Total force: {total_cg_aero_force}")
print(f"- Total Moment: {total_cg_aero_moment}")
print()
print(f"- Wing Force: {component_cg_aero_loads[0][0]}")
print(f"- Wing Moment: {component_cg_aero_loads[0][1]}")
print(f"- Htail Force: {component_cg_aero_loads[1][0]}")
print(f"- Htail Moment: {component_cg_aero_loads[1][1]}")
print(f"- Vtail Force: {component_cg_aero_loads[2][0]}")
print(f"- Vtail Moment: {component_cg_aero_loads[2][1]}")
print()
print("# Engine CG Loads")
print(f"- Force: {engine_force}")
print(f"- Moment: {engine_moment}")
print()
plt.show()
input("Press any key to quit...")
"""

plt.show()

def generate_structure(component):
    pass