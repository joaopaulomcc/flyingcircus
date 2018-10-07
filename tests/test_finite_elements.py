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
from samples import wing_simple

from numba import jit
# ==================================================================================================

def test_generate_FEM_mesh():

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

    name = "Aluminiun"
    density = 0.0975
    young_modulus = 10e7
    shear_modulus = 3770000
    poisson_ratio = 0.33
    yield_strength = 350e6
    ultimate_strength = 420e6

    mat_aluminiun = basic_objects.Material(name, young_modulus, shear_modulus, poisson_ratio, density, yield_strength, ultimate_strength)

    area = 1
    m_inertia_y = 0.08333333
    m_inertia_z = 0.08333333
    polar_moment = 0.1408333
    rotation = 0

    section = basic_objects.Section(area, rotation, m_inertia_y, m_inertia_z, polar_moment)

    structure_points = np.array([[0, 0, 0],
                                 [10, 0, 0],
                                 [20, 0, 0]])

    beam_0 = basic_objects.Beam(structure_points, 0, 1, section, mat_aluminiun, 5)
    beam_1 = basic_objects.Beam(structure_points, 1, 2, section, mat_aluminiun, 5)

    beams = [beam_0,
             beam_1]

    constraint_1 = basic_objects.Constraint(0, [0, 0, 0, None, None, None])
    constraint_2 = basic_objects.Constraint(2, [0, 0, 0, None, None, None])
    constraints = [constraint_1, constraint_2]

    force = basic_objects.Load(1, np.array([0, 100, 0, 0, 0, 0]))

    loads = [force]

    structure = basic_objects.Structure(structure_points, beams, loads, constraints)

    n_points = len(structure.points)
    n_nodes = n_points

    n_elements = 0

    nodes = []
    fem_elements = []

    for point in structure.points:

        nodes.append(point)
    
    for beam in structure.beams:

        mesh_points = beam.mesh(beam.n_elements)

        for i, point in enumerate(mesh_points):
            if i == 0:
                continue

            elif i == 1:
                nodes.append(point)
                n_nodes += 1
                fem_element = basic_objects.BeamElement(beam.point_A_index, n_nodes - 1, 
                                                        0,
                                                        beam.material.young_modulus,
                                                        beam.section.area,
                                                        beam.material.shear_modulus,
                                                        beam.section.polar_moment,
                                                        beam.section.m_inertia_y,
                                                        beam.section.m_inertia_z)
                fem_elements.append(fem_element)
                n_elements += 1

            elif i == len(mesh_points) - 1:

                fem_element = basic_objects.BeamElement(n_nodes - 1, beam.point_B_index, 
                                                        0,
                                                        beam.material.young_modulus,
                                                        beam.section.area,
                                                        beam.material.shear_modulus,
                                                        beam.section.polar_moment,
                                                        beam.section.m_inertia_y,
                                                        beam.section.m_inertia_z)
                fem_elements.append(fem_element)
                n_elements += 1

            else:
                nodes.append(point)
                n_nodes += 1
                fem_element = basic_objects.BeamElement(n_nodes - 2, n_nodes - 1, 
                                                        0,
                                                        beam.material.young_modulus,
                                                        beam.section.area,
                                                        beam.material.shear_modulus,
                                                        beam.section.polar_moment,
                                                        beam.section.m_inertia_y,
                                                        beam.section.m_inertia_z)
                fem_elements.append(fem_element)
                n_elements += 1

    # Generate global stiffness matrix
    K_global = np.zeros((n_nodes * 6, n_nodes * 6))
    F_global = np.zeros((n_nodes * 6, 1))
    X_global = np.zeros((n_nodes * 6, 1))

    for fem_element in fem_elements:
        K_element = fem_element.calc_K_global(nodes)
        A_index = fem_element.point_A_index * 6
        B_index = fem_element.point_B_index * 6
        correlation_vector = [A_index, A_index + 1, A_index + 2, A_index + 3, A_index + 4, A_index + 5,
                              B_index, B_index + 1, B_index + 2, B_index + 3, B_index + 4, B_index + 5]

        for i in range(len(correlation_vector)):
            for j in range(len(correlation_vector)):
                K_global[correlation_vector[i]][correlation_vector[j]] += K_element[i][j]

    # Generate Force Matrix
    for load in structure.loads:
        node_index = load.application_point_index * 6
        correlation_vector = [node_index, node_index + 1, node_index + 2, node_index + 3, node_index + 4, node_index + 5]

        for i in range(len(correlation_vector)):
            F_global[correlation_vector[i]] += load.components[i]

    # Find constrained degrees of freedom
    constrained_dof = [False for i in range(n_nodes * 6)]

    for constraint in constraints:
        node_index = constraint.application_point_index * 6
        correlation_vector = [node_index, node_index + 1, node_index + 2, node_index + 3, node_index + 4, node_index + 5]

        for i in range(len(correlation_vector)):
            if constraint.dof_constraints[i] != None:
                constrained_dof[correlation_vector[i]] = True
                X_global[correlation_vector[i]] += constraint.dof_constraints[i]

    
    # Find deformations
    red_K_global = np.copy(K_global)
    red_F_global = np.copy(F_global)

    dof_to_delete = []
    for i, dof in enumerate(constrained_dof):
        if dof:
            dof_to_delete.append(i)

    red_F_global = np.delete(red_F_global, dof_to_delete, 0)
    red_K_global = np.delete(red_K_global, dof_to_delete, 0)
    red_K_global = np.delete(red_K_global, dof_to_delete, 1)

    # Solve linear System
    red_X_global = np.linalg.solve(red_K_global, red_F_global)

    # Copy results do deformation vector
    counter = 0
    for i, dof in enumerate(constrained_dof):
        if not dof:
            X_global[i] = red_X_global[counter]
            counter += 1
    
    # Find support reactions

    new_F = K_global @ X_global


    # Deformed grid
    deformations = np.reshape(X_global, (n_nodes, 6))

    nodes_deformed = []
    for i, node in enumerate(nodes):
        nodes_deformed.append([node[0] + deformations[i][0],
                               node[1] + deformations[i][1],
                               node[2] + deformations[i][2]])
    
    

    print()

if __name__ == "__main__":
    test_generate_FEM_mesh()
