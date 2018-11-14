import numpy as np
import matplotlib.pyplot as plt
from context import src
from src import geometry
from src import visualization as vis
from src import aerodynamics as aero

# ==================================================================================================
# ==================================================================================================
# Super Tucano Geometrical Data

# WING

# Stub
root_chord = 2.24
root_section = "stub_root_section"
tip_chord = 2.24
tip_section = "stub_tip_section"
length = 0.615
leading_edge_sweep_angle_deg = 0
dihedral_angle_deg = 0
tip_torsion_angle_deg = 0
control_surface_hinge_position = None

surface_identifier = "right_stub"
right_stub = geometry.objects.Surface(
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
left_stub = geometry.objects.Surface(
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
# Flap
root_chord = 2.24
root_section = "flap_root_section"
tip_chord = 1.6
tip_section = "flap_tip_section"
length = 2.582
leading_edge_sweep_angle_deg = 4.04
dihedral_angle_deg = 5.59
tip_torsion_angle_deg = -0.5432
control_surface_hinge_position = 0.75

surface_identifier = "right_flap"
right_flap = geometry.objects.Surface(
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

surface_identifier = "left_flap"
left_flap = geometry.objects.Surface(
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
root_chord = 1.6
root_section = "aileron_root_section"
tip_chord = 1.04
tip_section = "aileron_tip_section"
length = 2.361
leading_edge_sweep_angle_deg = 4.04
dihedral_angle_deg = 5.59
tip_torsion_angle_deg = -0.4968
control_surface_hinge_position = 0.75

surface_identifier = "right_aileron"
right_aileron = geometry.objects.Surface(
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
left_aileron = geometry.objects.Surface(
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
# Wing
wing_surface_list = [
    left_aileron,
    left_flap,
    left_stub,
    right_stub,
    right_flap,
    right_aileron,
]
wing_incidence = 1.04
wing_position = np.array([3.2, 0, -0.347])
wing_symmetry_plane = "XZ"

wing = geometry.objects.MacroSurface(
    wing_position, wing_incidence, wing_surface_list, symmetry_plane=wing_symmetry_plane
)

# ==================================================================================================
# Horizontal Tail
root_chord = 1.29
root_section = "elevator_root_section"
tip_chord = 0.72
tip_section = "elevator_tip_section"
length = 2.24
leading_edge_sweep_angle_deg = 9.88
dihedral_angle_deg = 0
tip_torsion_angle_deg = 0
control_surface_hinge_position = 0.6

surface_identifier = "right_elevator"
right_elevator = geometry.objects.Surface(
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
left_elevator = geometry.objects.Surface(
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
h_tail_position = np.array([9.48, 0, 0.547])
h_tail_symmetry_plane = "XZ"

h_tail = geometry.objects.MacroSurface(
    h_tail_position,
    h_tail_incidence,
    h_tail_surface_list,
    symmetry_plane=h_tail_symmetry_plane,
)

# ==================================================================================================
# Vertical Tail
root_chord = 1.33
root_section = "right_aileron_root_section"
tip_chord = 0.68
tip_section = "right_aileron_tip_section"
length = 2.53
leading_edge_sweep_angle_deg = 24.66
dihedral_angle_deg = 90
tip_torsion_angle_deg = 0
control_surface_hinge_position = 0.43

surface_identifier = "rudder"
rudder = geometry.objects.Surface(
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
v_tail_position = np.array([9.82, 0, 0])
v_tail_symmetry_plane = None

v_tail = geometry.objects.MacroSurface(
    v_tail_position,
    v_tail_incidence,
    v_tail_surface_list,
    symmetry_plane=v_tail_symmetry_plane,
)

# ==================================================================================================
# ==================================================================================================
# Mesh Generation

n_chord_panels = 5
n_span_panels = 5

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

wing_control_surface_deflection_dict = {
    "left_aileron": 35,
    "left_flap": 45,
    "right_flap": 45,
    "right_aileron": -35,
}

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

h_tail_control_surface_deflection_dict = {"left_elevator": 30, "right_elevator": 30}

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

v_tail_control_surface_deflection_dict = {"rudder": 30}

v_tail_mesh = v_tail.create_mesh(
    v_tail_n_chord_panels,
    v_tail_n_span_panels_list,
    v_tail_chord_discretization,
    v_tail_span_discretization_list,
    v_tail_torsion_function_list,
    v_tail_control_surface_deflection_dict,
)

# ==================================================================================================
# Aircraft
aircraft_mesh = wing_mesh + h_tail_mesh + v_tail_mesh

# visualization.plot_3D.plot_surface(aircraft_mesh)
# plt.show()

aircraft_aero_mesh = [wing_mesh, h_tail_mesh, v_tail_mesh]
velocity_vector = np.array([100, 0, 0])
rotation_vector = np.array([0, 0, 0])
attitude_vector = np.array([5, 0, 0])
center = np.array([0, 0, 0])
altitude = 0

(
    components_force_vector,
    components_panel_vectors,
    components_force_grids,
    components_panel_grids,
    components_gamma_vector,
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

for panels, forces in zip(components_panel_grids, components_force_grids):

    delta_p, force = aero.vlm.calc_panels_delta_pressure(panels, forces)
    components_delta_p_grids.append(delta_p)
    components_force_mag_grids.append(force)

vis.plot_3D.plot_results(aircraft_aero_mesh, components_gamma_grid)
# print(components_gamma_vector)
input("Press any key to quit...")
