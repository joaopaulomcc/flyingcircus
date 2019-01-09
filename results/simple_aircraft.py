import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion


from context import src
from src import geometry as geo
from src import visualization as vis
from src import aerodynamics as aero
from src import loads
from src import structures as struct

# ==================================================================================================
# ==================================================================================================
# Simple Aircraft for testing purposes

# WING
# Definition of the wing macro surface and it's components

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
    identifier=airfoil,
    material=material,
    area=0.15,
    Iyy=0.000_281_25,
    Izz=0.0125,
    J=0.001_018_692,
    shear_center=0.5,
)

# --------------------------------------------------------------------------------------------------
# Stub
# Definition of the right and left wing stubs

STUB_ROOT_CHORD = 2
STUB_ROOT_SECTION = wing_section
STUB_TIP_CHORD = 1.5
STUB_TIP_SECTION = wing_section
STUB_LENGTH = 2
STUB_LEADING_EDGE_SWEEP_ANGLE_DEG = 30
STUB_DIHEDRAL_ANGLE_DEG = 5
STUB_TIP_TORSION_ANGLE_DEG = -1
STUB_CONTROL_SURFACE_HINGE_POSITION = None

right_stub = geo.objects.Surface(
    identifier="right_stub",
    root_chord=STUB_ROOT_CHORD,
    root_section=STUB_ROOT_SECTION,
    tip_chord=STUB_TIP_CHORD,
    tip_section=STUB_TIP_SECTION,
    length=STUB_LENGTH,
    leading_edge_sweep_angle_deg=STUB_LEADING_EDGE_SWEEP_ANGLE_DEG,
    dihedral_angle_deg=STUB_DIHEDRAL_ANGLE_DEG,
    tip_torsion_angle_deg=STUB_TIP_TORSION_ANGLE_DEG,
    control_surface_hinge_position=STUB_CONTROL_SURFACE_HINGE_POSITION,
)

left_stub = geo.objects.Surface(
    identifier="left_stub",
    root_chord=STUB_ROOT_CHORD,
    root_section=STUB_ROOT_SECTION,
    tip_chord=STUB_TIP_CHORD,
    tip_section=STUB_TIP_SECTION,
    length=STUB_LENGTH,
    leading_edge_sweep_angle_deg=STUB_LEADING_EDGE_SWEEP_ANGLE_DEG,
    dihedral_angle_deg=STUB_DIHEDRAL_ANGLE_DEG,
    tip_torsion_angle_deg=STUB_TIP_TORSION_ANGLE_DEG,
    control_surface_hinge_position=STUB_CONTROL_SURFACE_HINGE_POSITION,
)

# --------------------------------------------------------------------------------------------------
# Aileron
# Definition of the right and left wing ailerons

AILERON_ROOT_CHORD = 1.5
AILERON_ROOT_SECTION = wing_section
AILERON_TIP_CHORD = 1
AILERON_TIP_SECTION = wing_section
AILERON_LENGTH = 2
AILERON_LEADING_EDGE_SWEEP_ANGLE_DEG = 30
AILERON_DIHEDRAL_ANGLE_DEG = 5
AILERON_TIP_TORSION_ANGLE_DEG = -2
AILERON_CONTROL_SURFACE_HINGE_POSITION = 0.75

right_aileron = geo.objects.Surface(
    identifier="right_aileron",
    root_chord=AILERON_ROOT_CHORD,
    root_section=AILERON_ROOT_SECTION,
    tip_chord=AILERON_TIP_CHORD,
    tip_section=AILERON_TIP_SECTION,
    length=AILERON_LENGTH,
    leading_edge_sweep_angle_deg=AILERON_LEADING_EDGE_SWEEP_ANGLE_DEG,
    dihedral_angle_deg=AILERON_DIHEDRAL_ANGLE_DEG,
    tip_torsion_angle_deg=AILERON_TIP_TORSION_ANGLE_DEG,
    control_surface_hinge_position=AILERON_CONTROL_SURFACE_HINGE_POSITION,
)

left_aileron = geo.objects.Surface(
    identifier="left_aileron",
    root_chord=AILERON_ROOT_CHORD,
    root_section=AILERON_ROOT_SECTION,
    tip_chord=AILERON_TIP_CHORD,
    tip_section=AILERON_TIP_SECTION,
    length=AILERON_LENGTH,
    leading_edge_sweep_angle_deg=AILERON_LEADING_EDGE_SWEEP_ANGLE_DEG,
    dihedral_angle_deg=AILERON_DIHEDRAL_ANGLE_DEG,
    tip_torsion_angle_deg=AILERON_TIP_TORSION_ANGLE_DEG,
    control_surface_hinge_position=AILERON_CONTROL_SURFACE_HINGE_POSITION,
)

# --------------------------------------------------------------------------------------------------
# Winglet
# Definition of the right and left wing winglets

WINGLET_ROOT_CHORD = 1
WINGLET_ROOT_SECTION = wing_section
WINGLET_TIP_CHORD = 0.3
WINGLET_TIP_SECTION = wing_section
WINGLET_LENGTH = 1
WINGLET_LEADING_EDGE_SWEEP_ANGLE_DEG = 45
WINGLET_DIHEDRAL_ANGLE_DEG = 90
WINGLET_TIP_TORSION_ANGLE_DEG = 0
WINGLET_CONTROL_SURFACE_HINGE_POSITION = None

right_winglet = geo.objects.Surface(
    identifier="right_winglet",
    root_chord=WINGLET_ROOT_CHORD,
    root_section=WINGLET_ROOT_SECTION,
    tip_chord=WINGLET_TIP_CHORD,
    tip_section=WINGLET_TIP_SECTION,
    length=WINGLET_LENGTH,
    leading_edge_sweep_angle_deg=WINGLET_LEADING_EDGE_SWEEP_ANGLE_DEG,
    dihedral_angle_deg=WINGLET_DIHEDRAL_ANGLE_DEG,
    tip_torsion_angle_deg=WINGLET_TIP_TORSION_ANGLE_DEG,
    control_surface_hinge_position=WINGLET_CONTROL_SURFACE_HINGE_POSITION,
)

left_winglet = geo.objects.Surface(
    identifier="left_winglet",
    root_chord=WINGLET_ROOT_CHORD,
    root_section=WINGLET_ROOT_SECTION,
    tip_chord=WINGLET_TIP_CHORD,
    tip_section=WINGLET_TIP_SECTION,
    length=WINGLET_LENGTH,
    leading_edge_sweep_angle_deg=WINGLET_LEADING_EDGE_SWEEP_ANGLE_DEG,
    dihedral_angle_deg=WINGLET_DIHEDRAL_ANGLE_DEG,
    tip_torsion_angle_deg=WINGLET_TIP_TORSION_ANGLE_DEG,
    control_surface_hinge_position=WINGLET_CONTROL_SURFACE_HINGE_POSITION,
)

# --------------------------------------------------------------------------------------------------
# Wing macrosurface
# Creation of the wing macrosurface object

WING_POSITION = np.array([2.0, 0, 0])
WING_INCIDENCE = 2
WING_SYMMETRY_PLANE = "XZ"

WING_SURFACE_LIST = [
    left_winglet,
    left_aileron,
    left_stub,
    right_stub,
    right_aileron,
    right_winglet,
]

wing = geo.objects.MacroSurface(
    position=WING_POSITION,
    incidence=WING_INCIDENCE,
    surface_list=WING_SURFACE_LIST,
    symmetry_plane=WING_SYMMETRY_PLANE,
)

# ==================================================================================================
# Horizontal Tail
# Definition of the horizontal tail macrosurface and it's components

H_TAIL_ROOT_CHORD = 1
H_TAIL_ROOT_SECTION = wing_section
H_TAIL_TIP_CHORD = 0.6
H_TAIL_TIP_SECTION = wing_section
H_TAIL_LENGTH = 2
H_TAIL_LEADING_EDGE_SWEEP_ANGLE_DEG = 25
H_TAIL_DIHEDRAL_ANGLE_DEG = 0
H_TAIL_TIP_TORSION_ANGLE_DEG = 0
H_TAIL_CONTROL_SURFACE_HINGE_POSITION = 0.75

right_elevator = geo.objects.Surface(
    identifier="right_elevator",
    root_chord=H_TAIL_ROOT_CHORD,
    root_section=H_TAIL_ROOT_SECTION,
    tip_chord=H_TAIL_TIP_CHORD,
    tip_section=H_TAIL_TIP_SECTION,
    length=H_TAIL_LENGTH,
    leading_edge_sweep_angle_deg=H_TAIL_LEADING_EDGE_SWEEP_ANGLE_DEG,
    dihedral_angle_deg=H_TAIL_DIHEDRAL_ANGLE_DEG,
    tip_torsion_angle_deg=H_TAIL_TIP_TORSION_ANGLE_DEG,
    control_surface_hinge_position=H_TAIL_CONTROL_SURFACE_HINGE_POSITION,
)

left_elevator = geo.objects.Surface(
    identifier="left_elevator",
    root_chord=H_TAIL_ROOT_CHORD,
    root_section=H_TAIL_ROOT_SECTION,
    tip_chord=H_TAIL_TIP_CHORD,
    tip_section=H_TAIL_TIP_SECTION,
    length=H_TAIL_LENGTH,
    leading_edge_sweep_angle_deg=H_TAIL_LEADING_EDGE_SWEEP_ANGLE_DEG,
    dihedral_angle_deg=H_TAIL_DIHEDRAL_ANGLE_DEG,
    tip_torsion_angle_deg=H_TAIL_TIP_TORSION_ANGLE_DEG,
    control_surface_hinge_position=H_TAIL_CONTROL_SURFACE_HINGE_POSITION,
)

H_TAIL_POSITION = np.array([7, 0, 0.5])
H_TAIL_INCIDENCE = 0
H_TAIL_SURFACE_LIST = [left_elevator, right_elevator]
H_TAIL_SYMMETRY_PLANE = "XZ"

h_tail = geo.objects.MacroSurface(
    position=H_TAIL_POSITION,
    incidence=H_TAIL_INCIDENCE,
    surface_list=H_TAIL_SURFACE_LIST,
    symmetry_plane=H_TAIL_SYMMETRY_PLANE,
)

# ==================================================================================================
# Vertical Tail
# Definition of the vertical tail macrosurface and it's components

V_TAIL_ROOT_CHORD = 1
V_TAIL_ROOT_SECTION = wing_section
V_TAIL_TIP_CHORD = 0.5
V_TAIL_TIP_SECTION = wing_section
V_TAIL_LENGTH = 1.5
V_TAIL_LEADING_EDGE_SWEEP_ANGLE_DEG = 45
V_TAIL_DIHEDRAL_ANGLE_DEG = 90
V_TAIL_TIP_TORSION_ANGLE_DEG = 0
V_TAIL_CONTROL_SURFACE_HINGE_POSITION = 0.6

rudder = geo.objects.Surface(
    identifier="rudder",
    root_chord=V_TAIL_ROOT_CHORD,
    root_section=V_TAIL_ROOT_SECTION,
    tip_chord=V_TAIL_TIP_CHORD,
    tip_section=V_TAIL_TIP_SECTION,
    length=V_TAIL_LENGTH,
    leading_edge_sweep_angle_deg=V_TAIL_LEADING_EDGE_SWEEP_ANGLE_DEG,
    dihedral_angle_deg=V_TAIL_DIHEDRAL_ANGLE_DEG,
    tip_torsion_angle_deg=V_TAIL_TIP_TORSION_ANGLE_DEG,
    control_surface_hinge_position=V_TAIL_CONTROL_SURFACE_HINGE_POSITION,
)

V_TAIL_POSITION = np.array([7, 0, 0.5])
V_TAIL_INCIDENCE = 0
V_TAIL_SURFACE_LIST = [rudder]
V_TAIL_SYMMETRY_PLANE = None

v_tail = geo.objects.MacroSurface(
    position=V_TAIL_POSITION,
    incidence=V_TAIL_INCIDENCE,
    surface_list=V_TAIL_SURFACE_LIST,
    symmetry_plane=V_TAIL_SYMMETRY_PLANE,
)

# ==================================================================================================
# ==================================================================================================
# Fuselage Definition
# Definition of the fuselage as a Beam object

FRONT_ROOT_POINT = np.array([0, 0, 0])
FRONT_TIP_POINT = np.array([3, 0, 0])

BACK_ROOT_POINT = np.array([3, 0, 0])
BACK_TIP_POINT = np.array([7.5, 0, 0])

fuselage_material = material

fuselage_section = geo.objects.Section(
    airfoil="Circle",
    material=fuselage_material,
    area=1,
    Iyy=1,
    Izz=1,
    J=1,
    shear_center=0.5,
)

fuselage_property = struct.objects.ElementProperty(fuselage_section, fuselage_material)


front_fuselage = geo.objects.Beam(
    identifier="front_fuselage",
    root_point=FRONT_ROOT_POINT,
    tip_point=FRONT_TIP_POINT,
    ElementProperty=fuselage_property,
)

back_fuselage = geo.objects.Beam(
    identifier="back_fuselage",
    root_point=BACK_ROOT_POINT,
    tip_point=BACK_TIP_POINT,
    ElementProperty=fuselage_property,
)


# ==================================================================================================
# ==================================================================================================
# Engine Definition


def engine_thrust_function(throtle, parameters):

    return 4000 * throtle


ENGINE_ORIENTATION_QUATERNION = Quaternion(axis=np.array([0, 0, 1]), angle=np.pi)
ENGINE_INERTIA = geo.objects.MaterialPoint(
    identifier="engine_inertia",
    orientation_quaternion=Quaternion(),
    mass=1,
    position=np.array([0, 0, 0]),
    Ixx=1,
    Iyy=1,
    Izz=1,
    Ixy=1,
    Ixz=1,
    Iyz=1,
)

LEFT_ENGINE_POSITION = np.array([2, -2, -0.5])
RIGHT_ENGINE_POSITION = np.array([2, 2, -0.5])


left_engine = geo.objects.Engine(
    identifier="left_engine",
    position=LEFT_ENGINE_POSITION,
    orientation_quaternion=ENGINE_ORIENTATION_QUATERNION,
    inertial_properties=ENGINE_INERTIA,
    thrust_function=engine_thrust_function,
)

right_engine = geo.objects.Engine(
    identifier="right_engine",
    position=RIGHT_ENGINE_POSITION,
    orientation_quaternion=ENGINE_ORIENTATION_QUATERNION,
    inertial_properties=ENGINE_INERTIA,
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
# Aircraft Structure Connections


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


print("# Plotting Geometry...")

vis.plot_3D.plot_aircraft(simple_aircraft)


print()
"""
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

front_fuselage_elements = struct.fem.generate_beam_fem_elements(
    beam=front_fuselage, beam_nodes_list=front_fuselage_struct_grid, prop_choice="MIDDLE"
)

back_fuselage_elements = struct.fem.generate_beam_fem_elements(
    beam=back_fuselage, beam_nodes_list=back_fuselage_struct_grid, prop_choice="MIDDLE"
)

right_pylon_elements = struct.fem.generate_beam_fem_elements(
    beam=right_engine_pylon, beam_nodes_list=right_engine_pylon_struct_grid, prop_choice="MIDDLE"
)

left_pylon_elements = struct.fem.generate_beam_fem_elements(
    beam=left_engine_pylon, beam_nodes_list=left_engine_pylon_struct_grid, prop_choice="MIDDLE"
)


print()