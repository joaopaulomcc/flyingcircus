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
from src import basic_elements

import timeit
# ==================================================================================================
# TESTS


def test_vortex_segment():

    first_point = np.array([1, 5, 0])
    second_point = np.array([3, 1, 0])
    target_point = np.array([8, 5, 0])
    vicinity_point = np.array([3, 1, 0.001])
    circulation = 1

    induced_velocity = basic_elements.vortex_segment(first_point, second_point, target_point, circulation)
    induced_velocity_v = basic_elements.vortex_segment(first_point, second_point, vicinity_point, circulation)

    if not (abs(induced_velocity[0] - 0) < 0.0000001 and abs(induced_velocity[1] - 0) < 0.0000001 and
       abs(induced_velocity[2] - 0.008347229344620727) < 0.0000001):

        print(f"# Vortex Segment Test: FAIL")

    elif not (abs(induced_velocity_v[0] - 0) < 0.0000001 and abs(induced_velocity_v[1] - 0) < 0.0000001 and
              abs(induced_velocity_v[2] - 0) < 0.0000001):

        print(f"# Vortex Segment Test: FAIL")

    else:

        print(f"# Vortex Segment Test: PASS")

# --------------------------------------------------------------------------------------------------


def test_vortex_ring():

    vertex_coordinates = np.array([[2, 0, 0],
                                   [0, 0, 0],
                                   [0, 2, 0],
                                   [2, 2, 0]]).transpose()

    target_point = [1, 1, 0]
    circulation = 1

    induced_velocity = basic_elements.vortex_ring(vertex_coordinates, target_point, circulation)

    if not (abs(induced_velocity[0] - 0) < 0.0000001 and abs(induced_velocity[1] - 0) < 0.0000001 and
       abs(induced_velocity[2] - (-0.45015816)) < 0.0000001):

        print(f"# Vortex Ring Test: FAIL")

    else:

        print(f"# Vortex Ring Test: PASS")

# --------------------------------------------------------------------------------------------------


def test_vortex_horseshoe():

    vertex_coordinates = np.array([[2, 0, 0],
                                   [0, 0, 0],
                                   [0, 2, 0],
                                   [2, 2, 0]]).transpose()

    target_point = np.array([1, 1, 0], dtype=)
    circulation = 1.0

    induced_velocity, wake_induced_velocity = basic_elements.vortex_horseshoe(vertex_coordinates,
                                                                              target_point,
                                                                              circulation)

    print(f"Induced velocity = {induced_velocity}")
    print(f"Wake Induced Velocity = {wake_induced_velocity}")

    if not (abs(induced_velocity[0] - 0) < 0.0000001 and
            abs(induced_velocity[1] - 0) < 0.0000001 and
            abs(induced_velocity[2] - (-0.33761862)) < 0.0000001 and
            abs(wake_induced_velocity[0] - 0) < 0.0000001 and
            abs(wake_induced_velocity[1] - 0) < 0.0000001 and
            abs(wake_induced_velocity[2] - (-0.22507908)) < 0.0000001):

        print(f"# Vortex Horseshoe Test: FAIL")

    else:

        print(f"# Vortex Horseshoe Test: PASS")


# ==================================================================================================
# RUNNING TESTS
if __name__ == "__main__":

    print()
    print("=================================")
    print("= Testing Basic Elements module =")
    print("=================================")
    print()
    print(timeit.timeit(test_vortex_segment, number=1))
    print(timeit.timeit(test_vortex_ring, number=1))
    print(timeit.timeit(test_vortex_horseshoe, number=1))
