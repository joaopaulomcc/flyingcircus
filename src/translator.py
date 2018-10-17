"""
translation.py

Translation of inputs and outputs between geometry, aerodynamics and structures

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

# ==================================================================================================
# FUNCTIONS


def node_loads(panel_matrix, lift_matrix, drag_matrix, nodes):

    n_chord, n_span = np.shape(panel_matrix)
    node_forces = []
    node_moments = []

    for j in range(n_span):

        node_forces.append(np.zeros(3))
        node_moments.append(np.zeros(3))

        node = nodes[2 * j + 1]

        force = np.zeros(3)
        moment = np.zeros(3)

        for i in range(n_chord):

            panel = panel_matrix[i][j]
            r = panel.aero_center - node

            lift = lift_matrix[i][j] * np.array([0, 0, 1])
            drag = drag_matrix[i][j] * np.array([1, 0, 0])
            total_force = lift + drag

            force = force + total_force

            moment = moment + cross(r, total_force)

        node_forces.append(force)
        node_moments.append(moment)

    node_forces.append(np.zeros(3))
    node_moments.append(np.zeros(3))

    return node_forces, node_moments

# --------------------------------------------------------------------------------------------------


def loads_generator(node_forces, node_moments):

    loads = []
    i = 0

    for force, moment in zip(node_forces, node_moments):

        load = basic_objects.Load(i, np.array([force[0], force[1], force[2],
                                               moment[0], moment[1], moment[2]]))

        loads.append(load)

        i += 1

    return loads









