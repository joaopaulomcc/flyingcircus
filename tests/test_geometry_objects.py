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
length = 5
leading_edge_sweep_angle_deg = 30
dihedral_angle_deg = 10
tip_torsion_angle_deg = 15
control_surface_hinge_position = 0.75
torsion_center = 0.25

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

surface_list = [left_aileron, left_aileron, right_aileron, right_aileron]
control_surface_deflection_dict = {"left_aileron": 0, "right_aileron": 0}

position = np.array([0, 0, 0])
incidence = 0

wing = geometry.objects.MacroSurface(
    position,
    incidence,
    surface_list,
    symmetry_plane="XZ",
    torsion_center=torsion_center,
)

n_span_panels_list = [5, 5]
n_chord_panels = 5
control_surface_deflection = 45
chord_discretization = "cos_sim"
span_discretization = "cos_sim"

chord_discretization = "linear"
span_discretization_list = ["linear", "linear"]
torsion_function_list = ["linear", "linear"]

wing_mesh = wing.create_mesh(
    n_chord_panels,
    n_span_panels_list,
    chord_discretization,
    span_discretization_list,
    torsion_function_list,
    control_surface_deflection_dict,
)

n_elements_list = [n_span_panels_list[0], n_span_panels_list[0], n_span_panels_list[0], n_span_panels_list[0]]

#n_nodes = n_span_panels_list[0]
#node_list_right = right_aileron.generate_structure_nodes(
#    n_nodes, torsion_center=torsion_center
#)
#node_list_left = left_aileron.generate_structure_nodes(
#    n_nodes, torsion_center=torsion_center, mirror=True
#)

ax, fig = visualization.plot_3D.plot_mesh(wing_mesh)
node_list = wing.create_struct_mesh(n_elements_list)

for surface in node_list:
    for node in surface:
        visualization.plot_3D.plot_node(node, ax)

    # mirror_node_prop = geometry.functions.mirror_node_xz(node)

plt.show()
