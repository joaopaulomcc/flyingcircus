import numpy as np
import matplotlib.pyplot as plt

from context import src
from pyquaternion import Quaternion
from src import geometry as geo
from src import visualization as vis
from src import aerodynamics as aero
from src import loads
from src import structures as struct

# ==================================================================================================
# ==================================================================================================
# Simple Aircraft for testing purposes

# WING

airfoil = "NACA 0012"

material = struct.objects.Material(
    name="Aluminum 7075-T6",
    density=2810,
    elasticity_modulus=7.170_547e10,
    rigidity_modulus=2.69e10,
    poisson_ratio=0.33,
    yield_tensile_stress=5.033_172e08,
    ultimate_tensile_stress=5.722_648e08,
    yield_shear_stress=3.309_484e08,
    ultimate_shear_stress=0,
)

wing_section = geo.objects.Section(
    airfoil=airfoil,
    material=material,
    area=0.15,
    Iyy=0.000_281_25,
    Izz=0.0125,
    J=0.001_018_692,
    shear_center=0.5,
)

# Stub
root_chord = 2
root_section = wing_section
tip_chord = 1.5
tip_section = wing_section
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
root_section = wing_section
tip_chord = 1
tip_section = wing_section
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
root_section = wing_section
tip_chord = 0.3
tip_section = wing_section
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
root_section = wing_section
tip_chord = 0.6
tip_section = wing_section
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
root_section = wing_section
tip_chord = 0.5
tip_section = wing_section
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
# Fuselage Definition

front_root_point = np.array([1, 0, 0])
front_tip_point = np.array([3, 0, 0])

back_root_point = np.array([3, 0, 0])
back_tip_point = np.array([7.5, 0, 0.5])

fuselage_material = material

fuselage_section = geo.objects.Section(
    airfoil="Circle",
    material=fuselage_material,
    area=0.282_743_338_8,
    Iyy=0.028_981_192_2,
    Izz=0.028_981_192_2,
    J=0.057_962_384_5,
    shear_center=0.5,
)

fuselage_property = struct.objects.ElementProperty(fuselage_section, fuselage_material)

front_fuselage = geo.objects.Beam(
    identifier="front_fuselage",
    root_point=front_root_point,
    tip_point=front_tip_point,
    ElementProperty=fuselage_property,
)

back_fuselage = geo.objects.Beam(
    identifier="back_fuselage",
    root_point=back_root_point,
    tip_point=back_tip_point,
    ElementProperty=fuselage_property,
)


# ==================================================================================================
# ==================================================================================================
# Engine Definition


def engine_thrust_function(throtle, parameters):

    return 4000 * throtle


left_engine_position = np.array([0, -2.5, 0])
engine_inertial_properties = "engine inertial properties"
engine_thrust_vector = np.array([-1, 0, 0])
orientation_quaternion = Quaternion(axis=np.array([0, 0, 1]), angle=np.pi)

left_engine = geo.objects.Engine(
    identifier="left_engine",
    position=left_engine_position,
    orientation_quaternion=orientation_quaternion,
    inertial_properties=engine_inertial_properties,
    thrust_function=engine_thrust_function,
)

right_engine_position = np.array([0, 2.5, 0])
right_engine = geo.objects.Engine(
    identifier="right_engine",
    position=right_engine_position,
    orientation_quaternion=orientation_quaternion,
    inertial_properties=engine_inertial_properties,
    thrust_function=engine_thrust_function,
)

# Engines pylons
left_engine_pylon = geo.objects.Beam(
    identifier="right_engine_pylon",
    root_point=left_engine.position,
    tip_point=wing.tip_nodes[2].xyz,
    ElementProperty=struct.objects.RigidConnection(),
)

right_engine_pylon = geo.objects.Beam(
    identifier="right_engine_pylon",
    root_point=right_engine.position,
    tip_point=wing.tip_nodes[3].xyz,
    ElementProperty=struct.objects.RigidConnection(),
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
    identifier="aircraft_cg",
    orientation_quaternion=Quaternion(),
    mass=mass,
    position=cg_position,
    Ixx=Ixx,
    Iyy=Iyy,
    Izz=Izz,
    Ixy=Ixy,
    Ixz=Ixz,
    Iyz=Iyz,
)

# ==================================================================================================
# ==================================================================================================
# Aircraft Structure Conncetions


front_fuselage_to_wing = struct.objects.Connection(
    right_stub, "ROOT", front_fuselage, "ROOT"
)

front_fuselage_to_cg = struct.objects.Connection(
    front_fuselage, "TIP", aircraft_inertia, "ROOT"
)

front_fuselage_to_back_fuselage = struct.objects.Connection(
    front_fuselage, "TIP", back_fuselage, "ROOT"
)

back_fuselage_to_h_tail = struct.objects.Connection(
    back_fuselage, "TIP", right_elevator, "ROOT"
)

h_tail_to_v_tail = struct.objects.Connection(right_elevator, "ROOT", rudder, "ROOT")

left_engine_to_left_pylon = struct.objects.Connection(
    left_engine, "ROOT", left_engine_pylon, "ROOT"
)

left_pylon_to_wing = struct.objects.Connection(
    left_engine_pylon, "TIP", left_stub, "ROOT"
)

right_engine_to_right_pylon = struct.objects.Connection(
    right_engine, "ROOT", right_engine_pylon, "ROOT"
)

right_pylon_to_wing = struct.objects.Connection(
    right_engine_pylon, "TIP", right_stub, "ROOT"
)

aircraft_struct_connections = [
    front_fuselage_to_wing,
    front_fuselage_to_cg,
    front_fuselage_to_back_fuselage,
    back_fuselage_to_h_tail,
    h_tail_to_v_tail,
    left_engine_to_left_pylon,
    left_pylon_to_wing,
    right_engine_to_right_pylon,
    right_pylon_to_wing,
]

# ==================================================================================================
# ==================================================================================================
# Aircraft Definition

name = "Simple Aircraft"

macro_surfaces = [wing, h_tail, v_tail]

beams = [front_fuselage, back_fuselage, left_engine_pylon, right_engine_pylon]

engines = [left_engine, right_engine]

inertial_properties = aircraft_inertia

connections = [
    front_fuselage_to_wing,
    front_fuselage_to_cg,
    front_fuselage_to_back_fuselage,
    back_fuselage_to_h_tail,
    h_tail_to_v_tail,
    left_engine_to_left_pylon,
    left_pylon_to_wing,
    right_engine_to_right_pylon,
    right_pylon_to_wing,
]

simple_aircraft = geo.objects.Aircraft(
    name="Simple Aircraft",
    macro_surfaces=macro_surfaces,
    beams=beams,
    engines=engines,
    inertial_properties=inertial_properties,
    connections=connections,
)

"""
print("# Plotting Geometry...")

vis.plot_3D.plot_aircraft(simple_aircraft)


print()
user_input = input("# Proceed with calculation? [Y/N]\n")

if user_input == "N" or user_input == "n":
    print("# Stopping execution...")
    raise SystemExit(...)
"""

# ==================================================================================================
# ==================================================================================================
# Aerodynamic Loads Calculation

# Mesh Generation
n_chord_panels = 5
n_span_panels = 10

# Wing
wing_n_chord_panels = n_chord_panels
wing_n_span_panels_list = [
    n_span_panels,
    n_span_panels,
    n_span_panels,
    n_span_panels,
    n_span_panels,
    n_span_panels,
]
wing_n_beam_elements_list = [
    n_span_panels,
    n_span_panels,
    n_span_panels,
    n_span_panels,
    n_span_panels,
    n_span_panels,
]
wing_chord_discretization = "linear"
wing_span_discretization_list = [
    "linear",
    "linear",
    "linear",
    "linear",
    "linear",
    "linear",
]
wing_torsion_function_list = [
    "linear",
    "linear",
    "linear",
    "linear",
    "linear",
    "linear",
]

wing_control_surface_deflection_dict = {"left_aileron": 10, "right_aileron": 0}

wing_aero_grid, wing_struct_grid = wing.create_grids(
    wing_n_chord_panels,
    wing_n_span_panels_list,
    wing_n_beam_elements_list,
    wing_chord_discretization,
    wing_span_discretization_list,
    wing_torsion_function_list,
    wing_control_surface_deflection_dict,
)

# ==================================================================================================
# Horizontal Tail
h_tail_n_chord_panels = n_chord_panels
h_tail_n_span_panels_list = [n_span_panels, n_span_panels]
h_tail_n_beam_elements_list = [n_span_panels, n_span_panels]
h_tail_chord_discretization = "linear"
h_tail_span_discretization_list = ["linear", "linear"]
h_tail_torsion_function_list = ["linear", "linear"]

h_tail_control_surface_deflection_dict = {"left_elevator": 9.65, "right_elevator": 9.65}

h_tail_aero_grid, h_tail_struct_grid = h_tail.create_grids(
    h_tail_n_chord_panels,
    h_tail_n_span_panels_list,
    h_tail_n_beam_elements_list,
    h_tail_chord_discretization,
    h_tail_span_discretization_list,
    h_tail_torsion_function_list,
    h_tail_control_surface_deflection_dict,
)

# ==================================================================================================
# Vertical Tail
v_tail_n_chord_panels = n_chord_panels
v_tail_n_span_panels_list = [n_span_panels]
v_tail_n_beam_elements_list = [n_span_panels]
v_tail_chord_discretization = "linear"
v_tail_span_discretization_list = ["linear"]
v_tail_torsion_function_list = ["linear"]

v_tail_control_surface_deflection_dict = {"rudder": 0}

v_tail_aero_grid, v_tail_struct_grid = v_tail.create_grids(
    v_tail_n_chord_panels,
    v_tail_n_span_panels_list,
    v_tail_n_beam_elements_list,
    v_tail_chord_discretization,
    v_tail_span_discretization_list,
    v_tail_torsion_function_list,
    v_tail_control_surface_deflection_dict,
)

# ==================================================================================================
# Fuselage and pylons

front_fuselage_struct_grid = front_fuselage.create_grid(n_elements=5)
back_fuselage_struct_grid = back_fuselage.create_grid(n_elements=5)
left_engine_pylon_struct_grid = left_engine_pylon.create_grid(n_elements=1)
right_engine_pylon_struct_grid = right_engine_pylon.create_grid(n_elements=1)

# ==================================================================================================
# Create FEM elements

# Create connections

wing_struct_connections = struct.fem.create_macrosurface_connections(wing)
v_tail_struct_connections = struct.fem.create_macrosurface_connections(v_tail)
h_tail_struct_connections = struct.fem.create_macrosurface_connections(h_tail)

connections = [
    wing_struct_connections,
    v_tail_struct_connections,
    h_tail_struct_connections,
]

for connect in connections:
    if connect:
        aircraft_struct_connections += connect

# Number nodes
# Aircraft Components:
# - left winglet
# - left aileron
# - left stub
# - right stub
# - right aileron
# - right winglet
# - left elevator
# - right elevator
# - rudder
# - left engine
# - left pylon
# - right engine
# - right pylon
# - front fuselage
# - back fuselage
# - CG

components_list = (
    wing.surface_list
    + h_tail.surface_list
    + v_tail.surface_list
    + [left_engine]
    + [left_engine_pylon]
    + [right_engine]
    + [right_engine_pylon]
    + [front_fuselage]
    + [back_fuselage]
    + [aircraft_inertia]
)

components_nodes_list = (
    wing_struct_grid
    + h_tail_struct_grid
    + v_tail_struct_grid
    + [[left_engine.node]]
    + [left_engine_pylon_struct_grid]
    + [[right_engine.node]]
    + [right_engine_pylon_struct_grid]
    + [front_fuselage_struct_grid]
    + [back_fuselage_struct_grid]
    + [[aircraft_inertia.node]]
)

struct.fem.number_nodes(
    components_list, components_nodes_list, aircraft_struct_connections
)

"""
# ==================================================================================================
# Aircraft Mesh
simple_aircraft_aero_mesh = [wing_aero_grid, h_tail_aero_grid, v_tail_aero_grid]
# simple_aircraft_aero_mesh = [wing_mesh, h_tail_mesh]

# vis.plot_3D.plot_mesh(simple_aircraft_aero_mesh)


aircraft_aero_mesh = simple_aircraft_aero_mesh
velocity_vector = np.array([100, 0, 0])
rotation_vector = np.array([0, 0, 0])
attitude_vector = np.array([5, 0, 0])
center = simple_aircraft.inertial_properties.position
altitude = 0


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
"""
# ==================================================================================================
# Aircraft FEM mesh
wing_fem_elements = struct.fem.generate_macrosurface_fem_elements(
    macrosurface=wing, macrosurface_nodes_list=wing_struct_grid, prop_choice="MIDDLE"
)

h_tail_fem_elements = struct.fem.generate_macrosurface_fem_elements(
    macrosurface=h_tail, macrosurface_nodes_list=h_tail_struct_grid, prop_choice="MIDDLE"
)

v_tail_fem_elements = struct.fem.generate_macrosurface_fem_elements(
    macrosurface=v_tail, macrosurface_nodes_list=v_tail_struct_grid, prop_choice="MIDDLE"
)



print()