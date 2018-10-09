"""
finite_elements_method.py

Implementation of the finite element method for 3D beam

Author: João Paulo Monteiro Cruvinel da Costa
email: joaopaulomcc@gmail.com / joao.cruvinel@embraer.com.br
github: joaopaulomcc
"""
# ==================================================================================================
# IMPORTS
import numpy as np
import scipy as sc

from numpy import sin, cos, tan, pi
from .fast_operations import dot, cross, norm
from . import basic_objects


def generate_FEM_mesh(structure, element_length=None):
    """ Generate two lists: nodes contains the coordinates of all the grid points, fem_elements
    contains all the finite element objects of the model. """

    n_points = len(structure.points)

    n_nodes = n_points
    n_elements = 0

    nodes = []
    fem_elements = []

    # Add all the structure points to the nodes list
    for point in structure.points:

        nodes.append(point)

    # Create mesh for each of the beams
    for beam in structure.beams:

        # If an element length was defined use it to calculate number of elements per beam, otherwise
        # use the beam defined number of elements
        if element_length is not None:
            n_elements = int(beam.L // element_length)
            mesh_points = beam.mesh(n_elements)
        else:
            mesh_points = beam.mesh(beam.n_elements)

        # Generate beam elements and nodes
        for i, point in enumerate(mesh_points):

            # TODO calculation of section rotation
            section_rotation = 0

            if i == 0:
                continue

            # Second node is connected to initial point
            elif i == 1:
                nodes.append(point)
                n_nodes += 1
                fem_element = basic_objects.BeamElement(beam.point_A_index, n_nodes - 1,
                                                        section_rotation,
                                                        beam.material.young_modulus,
                                                        beam.section.area,
                                                        beam.material.shear_modulus,
                                                        beam.section.polar_moment,
                                                        beam.section.m_inertia_y,
                                                        beam.section.m_inertia_z)
                fem_elements.append(fem_element)
                n_elements += 1

            # Penultimum node is connected to the final point
            elif i == len(mesh_points) - 1:

                fem_element = basic_objects.BeamElement(n_nodes - 1, beam.point_B_index,
                                                        section_rotation,
                                                        beam.material.young_modulus,
                                                        beam.section.area,
                                                        beam.material.shear_modulus,
                                                        beam.section.polar_moment,
                                                        beam.section.m_inertia_y,
                                                        beam.section.m_inertia_z)
                fem_elements.append(fem_element)
                n_elements += 1

            # Intermediary nodes are connected to the previous node
            else:
                nodes.append(point)
                n_nodes += 1
                fem_element = basic_objects.BeamElement(n_nodes - 2, n_nodes - 1,
                                                        section_rotation,
                                                        beam.material.young_modulus,
                                                        beam.section.area,
                                                        beam.material.shear_modulus,
                                                        beam.section.polar_moment,
                                                        beam.section.m_inertia_y,
                                                        beam.section.m_inertia_z)
                fem_elements.append(fem_element)
                n_elements += 1

    return nodes, fem_elements

# --------------------------------------------------------------------------------------------------


def create_global_FEM_matrices(nodes, fem_elements, loads):

    n_nodes = len(nodes)

    # Generate global stiffness matrix
    K_global = np.zeros((n_nodes * 6, n_nodes * 6))
    F_global = np.zeros((n_nodes * 6, 1))

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
    for load in loads:
        node_index = load.application_point_index * 6
        correlation_vector = [node_index, node_index + 1, node_index + 2, node_index + 3, node_index + 4, node_index + 5]

        for i in range(len(correlation_vector)):
            F_global[correlation_vector[i]] += load.components[i]

    return K_global, F_global

# --------------------------------------------------------------------------------------------------


def FEM_solver(K_global, F_global, constraints):

    n_dof = len(F_global)
    X_global = np.zeros((n_dof, 1))

    # Find constrained degrees of freedom
    constrained_dof = [False for i in range(n_dof)]

    for constraint in constraints:
        node_index = constraint.application_point_index * 6
        correlation_vector = [node_index, node_index + 1, node_index + 2, node_index + 3, node_index + 4, node_index + 5]

        for i in range(len(correlation_vector)):
            if constraint.dof_constraints[i] is not None:
                constrained_dof[correlation_vector[i]] = True
                X_global[correlation_vector[i]] += constraint.dof_constraints[i]

    # Created reduced stiffess and force matrices
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

    return X_global

# --------------------------------------------------------------------------------------------------


def structural_solver(structure, loads, constraints, element_length=None):

    nodes, fem_elements = generate_FEM_mesh(structure, element_length)

    K_global, F_global = create_global_FEM_matrices(nodes, fem_elements, loads)

    X_global = FEM_solver(K_global, F_global, constraints)

    # Find support reactions
    force_vector = K_global @ X_global

    # Deformed grid
    deformations = np.reshape(X_global, (len(nodes), 6))

    deformed_grid = []
    for i, node in enumerate(nodes):
        deformed_grid.append([node[0] + deformations[i][0],
                              node[1] + deformations[i][1],
                              node[2] + deformations[i][2],
                              deformations[i][3],
                              deformations[i][4],
                              deformations[i][5]])

    return deformed_grid, deformations, force_vector
