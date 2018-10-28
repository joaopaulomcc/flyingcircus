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
    axis = np.array([1, 0, 0])
    rotation_angle = 45
    rotation_center = np.array([0, 5, 0])

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

# --------------------------------------------------------------------------------------------------


def test_discretization():

    print(f"- linear: {geometry.functions.discretization('linear', 11)}")
    print(f"- cos: {geometry.functions.discretization('cos', 11)}")
    print(f"- sin: {geometry.functions.discretization('sin', 11)}")
    print(f"- cos_sim: {geometry.functions.discretization('cos_sim', 11)}")

# --------------------------------------------------------------------------------------------------


def test_replace_closest():

    original = np.linspace(0, 1, 11)
    value = 0.75
    new_array, closest_index = geometry.functions.replace_closest(original, value)
    print(f"- original: {original}")
    print(f"- new_array: {new_array}")
    print(f"- closest_index: {closest_index}")

# --------------------------------------------------------------------------------------------------


def test_grid_to_vector():

    x_grid = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    y_grid = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    z_grid = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]])
    points_vector = geometry.functions.grid_to_vector(x_grid, y_grid, z_grid)
    print(points_vector)

# --------------------------------------------------------------------------------------------------




# ==================================================================================================
# TESTS

if __name__ == "__main__":

    print()
    print("==============================")
    print("= Testing geometry.functions =")
    print("==============================")
    print()
    #test_rotate_point()
    #test_velocity_vector()

    print("# discretization")
    test_discretization()
    print()

    print("# replace_closest")
    test_replace_closest()
    print()

    print("# grid_to_vector")
    test_grid_to_vector()
    print()