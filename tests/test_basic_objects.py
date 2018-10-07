"""
test_basic_elements.py

Testing suite for basic_elements module

Author: João Paulo Monteiro Cruvinel da Costa
email: joaopaulomcc@gmail.com / joao.cruvinel@embraer.com.br
github: joaopaulomcc
"""
# ==================================================================================================
# IMPORTS

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from numpy import sin, cos, tan, pi, dot, cross
from numpy.linalg import norm

from context import src
from src import basic_objects

# ==================================================================================================
# TESTS


def test_wing():

    area = 20
    aspect_ratio = 5
    taper_ratio = 1
    sweep_quarter_chord = 0
    dihedral = 0
    incidence = 0
    torsion = 0
    position = [0, 0, 0]

    simple_rectangular = basic_objects.Wing(area, aspect_ratio, taper_ratio, sweep_quarter_chord,
                                            dihedral, incidence, torsion, position)

    print()
    print("TESTING Wing")
    print(f"area: {simple_rectangular.area}")
    print(f"AR: {simple_rectangular.AR}")
    print(f"taper_ratio: {simple_rectangular.taper_ratio}")
    print(f"sweep: {simple_rectangular.sweep}")
    print(f"sweep_rad: {simple_rectangular.sweep_rad}")
    print(f"dihedral: {simple_rectangular.dihedral}")
    print(f"dihedral_rad: {simple_rectangular.dihedral_rad}")
    print(f"incidence: {simple_rectangular.incidence}")
    print(f"incidence_rad: {simple_rectangular.incidence_rad}")
    print(f"torsion: {simple_rectangular.torsion}")
    print(f"torsion_rad: {simple_rectangular.torsion_rad}")
    print(f"position: {simple_rectangular.position}")
    print(f"wing_span: {simple_rectangular.wing_span}")
    print(f"semi_wing_span: {simple_rectangular.semi_wing_span}")
    print(f"root_chord: {simple_rectangular.root_chord}")
    print(f"tip_chord: {simple_rectangular.tip_chord}")

# --------------------------------------------------------------------------------------------------


def test_panel():
    x = np.array([0, 2])
    y = np.array([0, 2])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    zz = np.zeros((2, 2))

    target_point = np.array([1, 1, 0])
    circulation = 1
    infinity = 25

    P = basic_objects.Panel(xx, yy, zz, infinity)
    induced_velocity, wake_induced_velocity = P.hs_induced_velocity(target_point, circulation)

    print()
    print("TESTING Panel")
    print(f"Vector AC: {P.AC}")
    print(f"Vector BD: {P.BD}")
    print(f"Vector n: {P.n}")
    print(f"Area n: {P.area}")
    print(f"Collocation Point n: {P.col_point}")
    print(f"induced_velocity: {induced_velocity}")
    print(f"wake_induced_velocity: {wake_induced_velocity}")

# --------------------------------------------------------------------------------------------------


def test_beam_element():

    E = 210e9
    G = 84e9
    A = 2e-2
    Iy = 10e-5
    Iz = 20e-5
    J = 5e-5

    point_A = np.array([0, 0, 0])
    point_B = np.array([3, 0, 0])
    rotation = pi / 3
    correlation_vector = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    beam = basic_objects.BeamElement(point_A, point_B, rotation, correlation_vector, E, A, G, J, Iy, Iz)
    print(beam.__dict__)

# --------------------------------------------------------------------------------------------------


def test_material():

    name = "Steel"
    density = 8000
    young_modulus = 200e9
    shear_modulus = 80e9
    poisson_ratio = 0.25
    yield_strength = 350e6
    ultimate_strength = 420e6

    mat_steel = basic_objects.Material(name, young_modulus, shear_modulus, poisson_ratio, density, yield_strength, ultimate_strength)
    print(mat_steel.__dict__)

# --------------------------------------------------------------------------------------------------


def test_section():

    area = 2e-2
    m_inertia_y = 10e-5
    m_inertia_z = 20e-5
    polar_moment = 5e-5
    rotation = pi / 6

    section = basic_objects.Section(area, rotation, m_inertia_y, m_inertia_z, polar_moment)
    print(section.__dict__)

# --------------------------------------------------------------------------------------------------


def test_beam():

    structure_points = np.array([[0, 0, 0],
                                 [3, 3, 3]])

    point_A_index = 0
    point_B_index = 1

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

    n_elements = 5

    beam = basic_objects.Beam(structure_points, point_A_index, point_B_index, section, mat_steel)
    print(beam.__dict__)

    print("# Mesh points: ")
    print(beam.mesh(n_elements))

# --------------------------------------------------------------------------------------------------

def test_structure():

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
    print(aircraft_structure.__dict__)
    


# ==================================================================================================
# RUN TESTS

if __name__ == "__main__":

    print()
    print("============================")
    print("= Testing objects module =")
    print("============================")
    print()
    #test_wing()
    #test_panel()
    #test_beam_element()
    #test_material()
    #test_section()
    test_beam()
    #test_structure()
