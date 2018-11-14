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
from mpl_toolkits.mplot3d import axes3d

from numpy import sin, cos, tan, pi, dot, cross
from numpy.linalg import norm

from context import src
from src import geometry as geo

# ==================================================================================================
# TESTS


def test_rotate_point():

    points_coord = np.array([[5, 5, 0], [0, 10, 0]]).transpose()
    axis = np.array([1, 0, 0])
    rotation_angle = 45
    rotation_center = np.array([0, 5, 0])

    rot_point = geo.functions.rotate_point(
        points_coord, axis, rotation_center, rotation_angle, degrees=True
    )

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

    velocity_vector = geo.functions.velocity_vector(true_airspeed, alpha, beta, gamma)

    print()
    print("TESTING velocity_vector:")
    print(f"Velocity Vector: ")
    print(f"{velocity_vector}")


# --------------------------------------------------------------------------------------------------


def test_discretization():

    print(f"- linear: {geo.functions.discretization('linear', 11)}")
    print(f"- cos: {geo.functions.discretization('cos', 11)}")
    print(f"- sin: {geo.functions.discretization('sin', 11)}")
    print(f"- cos_sim: {geo.functions.discretization('cos_sim', 11)}")


# --------------------------------------------------------------------------------------------------


def test_replace_closest():

    original = np.linspace(0, 1, 11)
    value = 0.75
    new_array, closest_index = geo.functions.replace_closest(original, value)
    print(f"- original: {original}")
    print(f"- new_array: {new_array}")
    print(f"- closest_index: {closest_index}")


# --------------------------------------------------------------------------------------------------


def test_grid_to_vector():

    x_grid = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    y_grid = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    z_grid = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]])
    points_vector = geo.functions.grid_to_vector(x_grid, y_grid, z_grid)
    print(points_vector)


# --------------------------------------------------------------------------------------------------


def test_velocity_field_function_generator():

    velocity_vector = np.array([100, 0, 0])
    rotation_vector = np.array([10, 0, 0])
    attitude_vector = np.array([45, 10, 0])
    center = np.array([5, 0, 0])

    velocity_field_function = geo.functions.velocity_field_function_generator(
        velocity_vector, rotation_vector, attitude_vector, center
    )

    x_vector = np.linspace(-10, 10, 5)
    y_vector = np.linspace(-10, 10, 5)
    z_vector = np.linspace(-10, 10, 5)

    x_list = []
    y_list = []
    z_list = []
    vector_list = []

    for x in x_vector:
        for y in y_vector:
            for z in z_vector:
                x_list.append(x)
                y_list.append(y)
                z_list.append(z)

                velocity_vector = velocity_field_function(np.array([x, y, z]))
                vector_list.append(velocity_vector)

    vector_list = np.array(vector_list)
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    z_list = np.array(z_list)

    print(vector_list)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", proj_type="persp")
    ax.quiver(
        x_list,
        y_list,
        z_list,
        vector_list[:, 0],
        vector_list[:, 1],
        vector_list[:, 2],
        length=0.01,
    )
    ax.scatter(center[0], center[1], center[2])
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    plt.show()


# ==================================================================================================
# TESTS

if __name__ == "__main__":

    print()
    print("==============================")
    print("= Testing geometry.functions =")
    print("==============================")
    print()
    # test_rotate_point()
    # test_velocity_vector()

    #    print("# discretization")
    #    test_discretization()
    #    print()
    #
    #    print("# replace_closest")
    #    test_replace_closest()
    #    print()
    #
    #    print("# grid_to_vector")
    #    test_grid_to_vector()
    #    print()

    print("# velocity_field_function_generator")
    test_velocity_field_function_generator()
    print()
