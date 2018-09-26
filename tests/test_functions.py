import numpy as np
from context import src

from src import functions as fc


def test_rotate_point():

    points_coord = np.array([[5, 5, 0], [0, 10, 0]]).transpose()
    axis = [1, 0, 0]
    rotation_angle = 45
    rotation_center = [0, 5, 0]

    rot_point = fc.rotate_point(points_coord, axis, rotation_center, rotation_angle, degrees=True)

    print(f"Rotated Point: {rot_point}")


def test_velocity_vector():

    true_airspeed = 200
    alpha = 5
    beta = 5
    gamma = 0

    v_vector = fc.velocity_vector(true_airspeed, alpha, beta, gamma)

    print("Velocity Vector:")
    print(v_vector)

if __name__ == "__main__":

    print()
    print("============================")
    print("= Testing functions module =")
    print("============================")
    print()
    test_rotate_point()
    test_velocity_vector()
