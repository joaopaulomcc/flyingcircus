# Python Imports
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

# SRC imports
from context import src
from src import atmosphere
from src import basic_objects
from src import finite_element_method
from src import geometry
from src import mesh
from src import translator
from src import vortex_lattice_method
from src import visualization

# Wing Properties
wing_area = 20
aspect_ratio = 5
taper_ratio = 1
sweep_quarter_chord = 0
dihedral = 0
incidence = 0
torsion = 0
position = [0, 0, 0]

# Creation of the wing object
wing = basic_objects.Wing(wing_area, aspect_ratio, taper_ratio, sweep_quarter_chord, dihedral,
                          incidence, torsion, position)

# Mesh Properties
n_semi_wingspam_panels = 3
n_chord_panels = 3
wingspam_discretization_type = "linear"
chord_discretization_type = "linear"

# Creation of the mesh
xx, yy, zz = mesh.generate_mesh(wing, n_semi_wingspam_panels, n_chord_panels,
                                wingspam_discretization_type, chord_discretization_type)

# visualization.plot_mesh(xx, yy, zz)

# Flight Conditions
alpha = 2
beta = 0
gamma = 0
attitude_vector = [alpha, beta, gamma]
altitude = 0
density, pressure, temperature = atmosphere.ISA(altitude)
true_airspeed = 100
flow_velocity_vector = geometry.velocity_vector(true_airspeed, alpha, beta, gamma)[:,0]
infinity_mult = 25

# Material
name = "Aluminium 7075-T6"
density = 2810
young_modulus = 7.170547E+10
shear_modulus = 2.69E+10
poisson_ratio = 0.33
yield_strength = 5.033172E+08
ultimate_strength = 5.722648E+08

mat_aluminium7075 = basic_objects.Material(name, young_modulus, shear_modulus, poisson_ratio, density, yield_strength, ultimate_strength)

# Section
#section_area = 7.854e-3
#m_inertia_y = 4.909e-6
#m_inertia_z = 4.909e-6
#polar_moment = 9.817e-6
#rotation = 0

# Hollow rectangle
# width: 2
# height: 0.3
# thickness: 0.02 and 0.04

section_area = 0.100
m_inertia_y = 0.005
m_inertia_z = 0.040
polar_moment = 0.035
rotation = 0


constraint = basic_objects.Constraint(1, [0, 0, 0, 0, 0, 0])
#circular_section = basic_objects.Section(section_area, rotation, m_inertia_y, m_inertia_z, polar_moment)
rectangular_section = basic_objects.Section(section_area, rotation, m_inertia_y, m_inertia_z, polar_moment)

# Structure
structure_points = np.array([[1, -5, 0],
                             [1, -4.16667, 0],
                             [1, -3.33333, 0],
                             [1, -2.5, 0],
                             [1, -1.66667, 0],
                             [1, -0.83333, 0],
                             [1, 0, 0],
                             [1, 0.83333, 0],
                             [1, 1.66667, 0],
                             [1, 2.5, 0],
                             [1, 3.33333, 0],
                             [1, 4.16666, 0],
                             [1, 5, 0]])

left_wing_0 = basic_objects.Beam(structure_points, 0, 1, rectangular_section, mat_aluminium7075, 1)
left_wing_1 = basic_objects.Beam(structure_points, 1, 2, rectangular_section, mat_aluminium7075, 1)
left_wing_2 = basic_objects.Beam(structure_points, 2, 3, rectangular_section, mat_aluminium7075, 1)
left_wing_3 = basic_objects.Beam(structure_points, 3, 4, rectangular_section, mat_aluminium7075, 1)
left_wing_4 = basic_objects.Beam(structure_points, 4, 5, rectangular_section, mat_aluminium7075, 1)
left_wing_5 = basic_objects.Beam(structure_points, 5, 6, rectangular_section, mat_aluminium7075, 1)
right_wing_0 = basic_objects.Beam(structure_points, 6, 7, rectangular_section, mat_aluminium7075, 1)
right_wing_1 = basic_objects.Beam(structure_points, 7, 8, rectangular_section, mat_aluminium7075, 1)
right_wing_2 = basic_objects.Beam(structure_points, 8, 9, rectangular_section, mat_aluminium7075, 1)
right_wing_3 = basic_objects.Beam(structure_points, 9, 10, rectangular_section, mat_aluminium7075, 1)
right_wing_4 = basic_objects.Beam(structure_points, 10, 11, rectangular_section, mat_aluminium7075, 1)
right_wing_5 = basic_objects.Beam(structure_points, 11, 12, rectangular_section, mat_aluminium7075, 1)

structure_beams = [left_wing_0, left_wing_1, left_wing_2, left_wing_3, left_wing_4, left_wing_5,
                   right_wing_0, right_wing_1, right_wing_2, right_wing_3, right_wing_4, right_wing_5]

wing_structure = basic_objects.Structure(structure_points, structure_beams)

# visualization.plot_structure(wing_structure)

# visualization.plot_aircraft(xx, yy, zz, wing_structure)

# Constrains

constrain = basic_objects.Constraint(6, [0, 0, 0, 0, 0, 0])
constraints = [constrain]

orig_xx, orig_yy, orig_zz = xx, yy, zz

for i in range(50):
    # Aerodynamic Loads Calculation
    panel_matrix = mesh.generate_panel_matrix(xx, yy, zz, wing.wing_span)
    panel_vector = vortex_lattice_method.flatten(panel_matrix)
    gamma = vortex_lattice_method.gamma_solver(panel_vector, flow_velocity_vector,
                                               infinity_mult * wing.wing_span)
    downwash = vortex_lattice_method.downwash_solver(panel_vector, gamma)
    lift, drag = vortex_lattice_method.lift_drag(panel_vector, gamma, downwash, true_airspeed, density)
    gamma_matrix = np.reshape(gamma, np.shape(panel_matrix))
    lift_matrix = np.reshape(lift, np.shape(panel_matrix))
    drag_matrix = np.reshape(drag, np.shape(panel_matrix))

    if i == 0:
        print()
        print("# Original")
        print(f"Lift: {lift_matrix.sum() / 1000} kN")
        print(f"Drag: {drag_matrix.sum() / 1000} kN")
        print()
        visualization.plot_results(xx, yy, zz, lift_matrix)

    # Loads Translation
    node_forces, node_moments = translator.node_loads(panel_matrix, lift_matrix, drag_matrix, structure_points)
    loads = translator.loads_generator(node_forces, node_moments)

    deformed_grid, force_vector, deformations, nodes, fem_elements = finite_element_method.structural_solver(wing_structure, loads, constraints)

    if i == 0:
        visualization.plot_deformation(fem_elements, nodes, deformations, scale=1)

    print(f"- Iter {i} tip rotation: {deformations[-1][4]} rad")

    original_nodes = nodes
    deformations_trans = np.delete(deformations, [3, 4, 5], axis=1)
    deformed_nodes = original_nodes + deformations_trans

    n_points = np.shape(xx)[1]

    xx_deformed = []
    yy_deformed = []
    zz_deformed = []

    for i in range(n_points):

        x = orig_xx[:, i]
        y = orig_yy[:, i]
        z = orig_zz[:, i]

        xyz = np.block([[x], [y], [z]])

        rot_angle = deformations[2 * i][4]
        rot_axis = np.array([0, 1, 0])
        rot_center = original_nodes[2 * i]

        x_translation = deformations[2 * i][0]
        y_translation = deformations[2 * i][1]
        z_translation = deformations[2 * i][2]

        xyz_rotated = geometry.rotate_point(xyz, rot_axis, rot_center, rot_angle)

        xyz_translated = xyz_rotated + [[x_translation], [y_translation], [z_translation]]

        xx_deformed.append(xyz_translated[0, :])
        yy_deformed.append(xyz_translated[1, :])
        zz_deformed.append(xyz_translated[2, :])

    xx_deformed = np.array(xx_deformed).transpose()
    yy_deformed = np.array(yy_deformed).transpose()
    zz_deformed = np.array(zz_deformed).transpose()

    #visualization.plot_results(xx_deformed, yy_deformed, zz_deformed, lift_matrix)

    xx = xx_deformed
    yy = yy_deformed
    zz = zz_deformed


visualization.plot_deformation(fem_elements, nodes, deformations, scale=1)
visualization.plot_results(xx_deformed, yy_deformed, zz_deformed, lift_matrix)

print()
print("# Deformed")
print(f"Lift: {lift_matrix.sum() / 1000} kN")
print(f"Drag: {drag_matrix.sum() / 1000} kN")
print()
plt.show()