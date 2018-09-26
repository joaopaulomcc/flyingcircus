"""
test_geometry.py

Testing suite for geometry module

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
from src import geometry

# ==================================================================================================
# TESTS


def test_rotate_point():

    points_coord = np.array([[5, 5, 0], [0, 10, 0]]).transpose()
    axis = [1, 0, 0]
    rotation_angle = 45
    rotation_center = [0, 5, 0]

    rot_point = geometry.rotate_point(points_coord, axis, rotation_center, rotation_angle, degrees=True)

    print()
    print("TESTING rotate_point")
    print(f"Rotated Point:")
    print(f"{rot_point}")

# --------------------------------------------------------------------------------------------------


def test_velocity_vector():

    true_airspeed = 200
    alpha = 5
    beta = 5
    gamma = 0

    velocity_vector = geometry.velocity_vector(true_airspeed, alpha, beta, gamma)

    print()
    print("TESTING velocity_vector:")
    print(f"Velocity Vector: ")
    print(f"{velocity_vector}")

# ==================================================================================================
# TESTS

if __name__ == "__main__":

    print()
    print("===========================")
    print("= Testing geometry module =")
    print("===========================")
    print()
    test_rotate_point()
    test_velocity_vector()
