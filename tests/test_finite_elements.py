"""
test_finite_elementes.py

Testing suite for finite_elements module

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
from src import finite_element_method

from samples import wing_simple

from numba import jit
# ==================================================================================================
#    name = "Steel"
#    density = 8000
#    young_modulus = 200e9
#    shear_modulus = 80e9
#    poisson_ratio = 0.25
#    yield_strength = 350e6
#    ultimate_strength = 420e6
#
#    mat_steel = basic_objects.Material(name, young_modulus, shear_modulus, poisson_ratio, density, yield_strength, ultimate_strength)
#
#    area = 2e-2
#    m_inertia_y = 10e-5
#    m_inertia_z = 20e-5
#    polar_moment = 5e-5
#    rotation = pi / 6
#
#    section = basic_objects.Section(area, rotation, m_inertia_y, m_inertia_z, polar_moment)
#
#    structure_points = np.array([[0, 0, 0],
#                                 [0, 0, 3],
#                                 [0, 0, 6],
#                                 [3, 3, 3]])
#
#    beam_0 = basic_objects.Beam(structure_points, 0, 1, section, mat_steel, 3)
#    beam_1 = basic_objects.Beam(structure_points, 1, 2, section, mat_steel, 3)
#    beam_2 = basic_objects.Beam(structure_points, 1, 3, section, mat_steel, 3)
#
#    beams = [beam_0,
#             beam_1,
#             beam_2]
#
#    constraint = basic_objects.Constraint(0, [0, 0, 0, 0, 0, 0])
#    constraints = [constraint]
#
#    force = basic_objects.Load(3, np.array([0, 0, -1000, 0, 0, 0]))
#    moment = basic_objects.Load(2, np.array([0, 0, 0, 0, 0, -100]))
#
#    loads = [force, moment]
#
#    structure = basic_objects.Structure(structure_points, beams, loads, constraints)


def test_generate_FEM_mesh():

    # Material definition
    name = "Aluminiun"
    density = 0.0975
    young_modulus = 1e7
    shear_modulus = 3770000
    poisson_ratio = 0.33
    yield_strength = 350e6
    ultimate_strength = 420e6

    mat_aluminiun = basic_objects.Material(name, young_modulus, shear_modulus, poisson_ratio, density, yield_strength, ultimate_strength)

    # Section definition
    area = 1
    m_inertia_y = 0.08333333
    m_inertia_z = 0.08333333
    polar_moment = 0.1408333
    rotation = 0

    section = basic_objects.Section(area, rotation, m_inertia_y, m_inertia_z, polar_moment)

    # Structure points definition
    structure_points = np.array([[0, 0, 0],
                                 [10, 0, 0],
                                 [20, 0, 0]])

    # Structure beams definition

    beam_0 = basic_objects.Beam(structure_points, 0, 1, section, mat_aluminiun, 5)
    beam_1 = basic_objects.Beam(structure_points, 1, 2, section, mat_aluminiun, 5)

    beams = [beam_0,
             beam_1]

    # Structure ddefinition
    structure = basic_objects.Structure(structure_points, beams)

    # Element Length
    element_length = 2

    print("# Testing generate_FEM_mesh")
    start = time.time()
    nodes, fem_elements = finite_element_method.generate_FEM_mesh(structure, element_length)
    end = time.time()
    print(f"- Test completed in {end - start}")

# --------------------------------------------------------------------------------------------------


def test_create_global_FEM_matrices():

    # Material definition
    name = "Aluminiun"
    density = 0.0975
    young_modulus = 1e7
    shear_modulus = 3770000
    poisson_ratio = 0.33
    yield_strength = 350e6
    ultimate_strength = 420e6

    mat_aluminiun = basic_objects.Material(name, young_modulus, shear_modulus, poisson_ratio, density, yield_strength, ultimate_strength)

    # Section definition
    area = 1
    m_inertia_y = 0.08333333
    m_inertia_z = 0.08333333
    polar_moment = 0.1408333
    rotation = 0

    section = basic_objects.Section(area, rotation, m_inertia_y, m_inertia_z, polar_moment)

    # Structure points definition
    structure_points = np.array([[0, 0, 0],
                                 [10, 0, 0],
                                 [20, 0, 0]])

    # Structure beams definition

    beam_0 = basic_objects.Beam(structure_points, 0, 1, section, mat_aluminiun, 5)
    beam_1 = basic_objects.Beam(structure_points, 1, 2, section, mat_aluminiun, 5)

    beams = [beam_0,
             beam_1]

    # Structure definition
    structure = basic_objects.Structure(structure_points, beams)

    # Element Length
    element_length = 2

    nodes, fem_elements = finite_element_method.generate_FEM_mesh(structure, element_length)

    # Loads definition
    force = basic_objects.Load(1, np.array([0, 100, 0, 0, 0, 0]))
    loads = [force]

    print("# Testing create_global_FEM_matrices")
    start = time.time()
    K_global, F_global = finite_element_method.create_global_FEM_matrices(nodes, fem_elements, loads)
    end = time.time()
    print(f"- Test completed in {end - start}")

# --------------------------------------------------------------------------------------------------


def test_FEM_solver():

    # Material definition
    name = "Aluminiun"
    density = 0.0975
    young_modulus = 1e7
    shear_modulus = 3770000
    poisson_ratio = 0.33
    yield_strength = 350e6
    ultimate_strength = 420e6

    mat_aluminiun = basic_objects.Material(name, young_modulus, shear_modulus, poisson_ratio, density, yield_strength, ultimate_strength)

    # Section definition
    area = 1
    m_inertia_y = 0.08333333
    m_inertia_z = 0.08333333
    polar_moment = 0.1408333
    rotation = 0

    section = basic_objects.Section(area, rotation, m_inertia_y, m_inertia_z, polar_moment)

    # Structure points definition
    structure_points = np.array([[0, 0, 0],
                                 [10, 0, 0],
                                 [20, 0, 0]])

    # Structure beams definition
    beam_0 = basic_objects.Beam(structure_points, 0, 1, section, mat_aluminiun, 5)
    beam_1 = basic_objects.Beam(structure_points, 1, 2, section, mat_aluminiun, 5)

    beams = [beam_0,
             beam_1]

    # Structure definition
    structure = basic_objects.Structure(structure_points, beams)

    # Element Length
    element_length = 2

    # Mesh generation
    nodes, fem_elements = finite_element_method.generate_FEM_mesh(structure, element_length)

    # Loads definition
    force = basic_objects.Load(1, np.array([0, 100, 0, 0, 0, 0]))
    loads = [force]

    # Global matrices
    K_global, F_global = finite_element_method.create_global_FEM_matrices(nodes, fem_elements, loads)

    # Constraints definition
    constraint_1 = basic_objects.Constraint(0, [0, 0, 0, 0, None, None])
    constraint_2 = basic_objects.Constraint(2, [0, 0, 0, None, None, None])
    constraints = [constraint_1, constraint_2]

    print("# Testing FEM_solver")
    start = time.time()
    X_global = finite_element_method.FEM_solver(K_global, F_global, constraints)
    end = time.time()
    print(f"- Test completed in {end - start}")

# --------------------------------------------------------------------------------------------------


def test_structural_solver():

    # Material definition
    name = "Aluminiun"
    density = 0.0975
    young_modulus = 1e7
    shear_modulus = 3770000
    poisson_ratio = 0.33
    yield_strength = 350e6
    ultimate_strength = 420e6

    mat_aluminiun = basic_objects.Material(name, young_modulus, shear_modulus, poisson_ratio, density, yield_strength, ultimate_strength)

    # Section definition
    area = 1
    m_inertia_y = 0.08333333
    m_inertia_z = 0.08333333
    polar_moment = 0.1408333
    rotation = 0

    section = basic_objects.Section(area, rotation, m_inertia_y, m_inertia_z, polar_moment)

    # Structure points definition
    structure_points = np.array([[0, 0, 0],
                                 [10, 0, 0],
                                 [20, 0, 0]])

    # Structure beams definition
    beam_0 = basic_objects.Beam(structure_points, 0, 1, section, mat_aluminiun, 5)
    beam_1 = basic_objects.Beam(structure_points, 1, 2, section, mat_aluminiun, 5)

    beams = [beam_0,
             beam_1]

    # Structure definition
    structure = basic_objects.Structure(structure_points, beams)

    # Element Length
    element_length = 2

    # Loads definition
    force = basic_objects.Load(1, np.array([0, 100, 0, 0, 0, 0]))
    loads = [force]

    # Constraints definition
    constraint_1 = basic_objects.Constraint(0, [0, 0, 0, 0, None, None])
    constraint_2 = basic_objects.Constraint(2, [0, 0, 0, None, None, None])
    constraints = [constraint_1, constraint_2]

    print("# Testing structural_solver")
    start = time.time()
    deformed_grid, deformations, force_vector = finite_element_method.structural_solver(structure, loads, constraints, element_length)
    end = time.time()
    print(f"- Test completed in {end - start}")

# ===============================================================================================

if __name__ == "__main__":

    print()
    print("========================================")
    print("= Testing finite_element_method module =")
    print("========================================")
    print()
    test_generate_FEM_mesh()
    print()
    test_create_global_FEM_matrices()
    print()
    test_FEM_solver()
    print()
    test_structural_solver()
