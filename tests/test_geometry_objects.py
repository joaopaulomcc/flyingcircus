import numpy as np
import matplotlib.pyplot as plt
from context import src
from src import geometry
from src import visualization

airfoil = "NACA0012"
material = "Aluminium"
area = 1
Iyy = 1
Izz = 1
J = 0.5
shear_center = 0.5

surface_identifier = "right_aileron"
root_chord = 2
root_section = geometry.objects.Section(
    airfoil, material, area, Iyy, Izz, J, shear_center
)
tip_chord = 2
tip_section = geometry.objects.Section(
    airfoil, material, area, Iyy, Izz, J, shear_center
)
torsion_center = 0.2

length = 5
leading_edge_sweep_angle_deg = 30
dihedral_angle_deg = 20
tip_torsion_angle_deg = 30
control_surface_hinge_position = 0.75

right_aileron = geometry.objects.Surface2(
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
root_chord = 2
tip_chord = 2
length = 5
leading_edge_sweep_angle_deg = 30
dihedral_angle_deg = 20
tip_torsion_angle_deg = 30
control_surface_hinge_position = 0.75

left_aileron = geometry.objects.Surface2(
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

surface_identifier = "right_stub"
root_chord = 5
tip_chord = 2
length = 5
leading_edge_sweep_angle_deg = 45
dihedral_angle_deg = 30
tip_torsion_angle_deg = 30
control_surface_hinge_position = None

right_stub = geometry.objects.Surface2(
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
root_chord = 5
tip_chord = 2
length = 5
leading_edge_sweep_angle_deg = 45
dihedral_angle_deg = 30
tip_torsion_angle_deg = 30
control_surface_hinge_position = None

left_stub = geometry.objects.Surface2(
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

surface_list = [
    left_aileron,
    left_aileron,
    left_stub,
    right_stub,
    right_aileron,
    right_aileron,
]
#surface_list = [left_aileron, right_aileron]
control_surface_deflection_dict = {"left_aileron": 0, "right_aileron": 0}

position = np.array([10, 5, 2])
incidence = 0

wing = geometry.objects.MacroSurface2(
    position, incidence, surface_list, symmetry_plane="XZ", torsion_center=0.5
)

n_span_panels_list = [20, 20, 20, 20, 20, 20]
n_chord_panels = 10

chord_discretization = "linear"
# span_discretization_list = ["linear", "linear"]
# torsion_function_list = ["linear", "linear"]

span_discretization_list = ["linear", "linear", "linear", "linear", "linear", "linear"]
torsion_function_list = ["linear", "linear", "linear", "linear", "linear", "linear"]
n_beam_elements = 10
n_beam_elements_list = [
    n_beam_elements,
    n_beam_elements,
    n_beam_elements,
    n_beam_elements,
    n_beam_elements,
    n_beam_elements,
]
wing_aero_grid, wing_nodes_list = mesh = wing.create_aero_grid(
    n_chord_panels,
    n_span_panels_list,
    n_beam_elements_list,
    chord_discretization,
    span_discretization_list,
    torsion_function_list,
    control_surface_deflection_dict,
)


# n_elements_list = [n_span_panels_list[0], n_span_panels_list[0], n_span_panels_list[0], n_span_panels_list[0]]
n_elements_list = [2, 2, 2, 2]
# n_nodes = n_span_panels_list[0]
# node_list_right = right_aileron.generate_structure_nodes(
#    n_nodes, torsion_center=torsion_center
# )
# node_list_left = left_aileron.generate_structure_nodes(
#    n_nodes, torsion_center=torsion_center, mirror=True
# )

ax, fig = visualization.plot_3D.plot_mesh(wing_aero_grid)

for surface in wing_nodes_list:
    for node in surface:
        visualization.plot_3D.plot_node(node, ax)

    # mirror_node_prop = geometry.functions.mirror_node_xz(node)

plt.show()
