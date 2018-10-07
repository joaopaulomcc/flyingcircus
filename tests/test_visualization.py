"""
test_visualization.py

Testing suite for visualization module

Author: João Paulo Monteiro Cruvinel da Costa
email: joaopaulomcc@gmail.com / joao.cruvinel@embraer.com.br
github: joaopaulomcc
"""
# ==================================================================================================
# IMPORTS

import numpy as np
import scipy as sc

import timeit
import time

from numpy import sin, cos, tan, pi, dot, cross
from numpy.linalg import norm

from context import src
from src import atmosphere
from src import vortex_lattice_method
from src import mesh
from src import basic_objects
from src import geometry
from src import visualization
from samples import wing_simple

from numba import jit
# ==============================================================================================

@jit
def test_plot_mesh():

    xx, yy, zz = mesh.generate_mesh(wing, n_semi_wingspam_panels, n_chord_panels,
                                    wingspam_discretization_type, chord_discretization_type)

    visualization.plot_mesh(xx, yy, zz)

# --------------------------------------------------------------------------------------------------


@jit
def test_plot_results():

    xx, yy, zz = mesh.generate_mesh(wing, n_semi_wingspam_panels, n_chord_panels,
                                    wingspam_discretization_type, chord_discretization_type)

    panel_matrix = mesh.generate_panel_matrix(xx, yy, zz, wing.wing_span)

    panel_vector = vortex_lattice_method.flatten(panel_matrix)

    gamma = vortex_lattice_method.gamma_solver(panel_vector, flow_velocity_vector,
                                               infinity_mult * wing.wing_span)

    downwash = vortex_lattice_method.downwash_solver(panel_vector, gamma)

    lift, drag = vortex_lattice_method.lift_drag(panel_vector, gamma, downwash, true_airspeed, density)

    gamma_grid = np.reshape(gamma, np.shape(panel_matrix))
    lift_grid = np.reshape(lift, np.shape(panel_matrix))
    drag_grid = np.reshape(drag, np.shape(panel_matrix))

    visualization.plot_results(xx, yy, zz, lift_grid)

# --------------------------------------------------------------------------------------------------


def test_plot_structure():

    name = "Steel"
    density = 8000
    young_modulus = 200e9
    shear_modulus = 80e9
    poisson_ratio = 0.25
    yield_strength = 350e6
    ultimate_strength = 420e6

    mat_steel = basic_objects.Material(name, young_modulus, shear_modulus, poisson_ratio, density, yield_strength, ultimate_strength)

    area = 2e-2
    m_inertia_y = 10e-5
    m_inertia_z = 20e-5
    polar_moment = 5e-5
    rotation = pi / 6

    section = basic_objects.Section(area, rotation, m_inertia_y, m_inertia_z, polar_moment)

    structure_points = np.array([[0, 0, 0],
                                 [2, 0, 0],
                                 [5, 0, 0],
                                 [5, 0, 1],
                                 [2, -5, 0.4],
                                 [2, 5, 0.4],
                                 [5, -2, 1],
                                 [5, 2, 1]])

    fuselage_frontal = basic_objects.Beam(structure_points, 0, 1, section, mat_steel)
    fuselage_posterior = basic_objects.Beam(structure_points, 1, 2, section, mat_steel)
    wing_left = basic_objects.Beam(structure_points, 1, 4, section, mat_steel)
    wing_right = basic_objects.Beam(structure_points, 1, 5, section, mat_steel)
    tail_vertical = basic_objects.Beam(structure_points, 2, 3, section, mat_steel)
    tail_horizontal_left = basic_objects.Beam(structure_points, 3, 6, section, mat_steel)
    tail_horizontal_right = basic_objects.Beam(structure_points, 3, 7, section, mat_steel)

    beams = [fuselage_frontal,
             fuselage_posterior,
             wing_left,
             wing_right,
             tail_vertical,
             tail_horizontal_left,
             tail_horizontal_right]

    constraint = basic_objects.Constraint(1, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    constraints = [[constraint]]

    lift_left_wing = basic_objects.Load(4, np.array([0, 10000, 0, 0, 100, 0]))
    lift_right_wing = basic_objects.Load(5, np.array([0, 10000, 0, 0, 100, 0]))

    loads = [lift_left_wing, lift_right_wing]

    aircraft_structure = basic_objects.Structure(structure_points, beams, loads, constraints)

    visualization.plot_structure(aircraft_structure)

# ==================================================================================================
# RUNNING TESTS

if __name__ == "__main__":

    print()
    print("================================")
    print("= Testing visualization module =")
    print("================================")
    print()

    area = 19.4
    aspect_ratio = 6.39
    taper_ratio = 0.42
    sweep_quarter_chord = 0
    dihedral = 7
    incidence = 0
    torsion = 0
    position = [0, 0, 0]

    n_semi_wingspam_panels = 50
    n_chord_panels = 6
    wingspam_discretization_type = "linear"
    chord_discretization_type = "linear"

    alpha = 3.6
    beta = 0
    gamma = 0
    attitude_vector = [alpha, beta, gamma]
    altitude = 0

    density, pressure, temperature = atmosphere.ISA(altitude)

    true_airspeed = 113.
    flow_velocity_vector = geometry.velocity_vector(true_airspeed, alpha, beta, gamma)[:,0]
    infinity_mult = 25.

    wing = basic_objects.Wing(area, aspect_ratio, taper_ratio, sweep_quarter_chord, dihedral,
                              incidence, torsion, position)

    #print("Testing plot_results")
    #test_plot_mesh()
    #test_plot_results()
    test_plot_structure()