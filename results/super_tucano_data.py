import numpy as np
import matplotlib.pyplot as plt
from context import src
from src import geometry
from src import visualization

# Super Tucano Geometrical Data

# Stub
root_chord = 2.24
root_section = "stub_root_section"
tip_chord = 2.24
tip_section = "stub_tip_section"
length = 0.615
leading_edge_sweep_angle_deg = 0
dihedral_angle_deg = 0
tip_torsion_angle_deg = 0
control_surface_hinge_position = 0.75

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


wing_control_surface_deflection_dict = {
    "left_aileron": 5,
    "left_flap": 45,
    "right_flap": 45,
    "right_aileron": -5,
}

n_chord_panels = 10
n_span_panels = 10
chord_discretization = "linear"
span_discretization = "linear"
torsion_function = "linear"

wing_mesh = wing.create_mesh(
    n_chord_panels,
    n_span_panels,
    wing_control_surface_deflection_dict,
    chord_discretization=chord_discretization,
    span_discretization=span_discretization,
    torsion_function=torsion_function,
)

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

h_tail_control_surface_deflection_dict = {"left_elevator": -10, "right_elevator": -10}

n_chord_panels = 10
n_span_panels = 10
chord_discretization = "linear"
span_discretization = "linear"
torsion_function = "linear"

h_tail_mesh = h_tail.create_mesh(
    n_chord_panels,
    n_span_panels,
    h_tail_control_surface_deflection_dict,
    chord_discretization=chord_discretization,
    span_discretization=span_discretization,
    torsion_function=torsion_function,
)

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

n_chord_panels = 10
n_span_panels = 10
chord_discretization = "linear"
span_discretization = "linear"
torsion_function = "linear"

v_tail_control_surface_deflection_dict = {"rudder": 10}

v_tail_mesh = v_tail.create_mesh(
    n_chord_panels,
    n_span_panels,
    v_tail_control_surface_deflection_dict,
    chord_discretization=chord_discretization,
    span_discretization=span_discretization,
    torsion_function=torsion_function,
)

aircraft_mesh = wing_mesh + h_tail_mesh + v_tail_mesh

visualization.plot_3D.plot_surface(aircraft_mesh)
plt.show()
