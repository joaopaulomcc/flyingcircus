from context import src

import numpy as np
from src import vortex_lattice as vlm


def test_vortex_line():

    first_point = [1, 5, 0]
    second_point = [3, 1, 0]
    target_point = [8, 5, 0]
    vicinity_point = [3, 1, 0.001]
    circulation = 1

    induced_velocity = vlm.vortex_line(first_point, second_point, target_point, circulation)
    induced_velocity_v = vlm.vortex_line(first_point, second_point, vicinity_point, circulation)

    if not (abs(induced_velocity[0] - 0) < 0.0000001 and abs(induced_velocity[1] - 0) < 0.0000001 and
       abs(induced_velocity[2] - 0.008347229344620727) < 0.0000001):

        print(f"# Vortex Line Test: FAIL")

    elif not (abs(induced_velocity_v[0] - 0) < 0.0000001 and abs(induced_velocity_v[1] - 0) < 0.0000001 and
         abs(induced_velocity_v[2] - 0) < 0.0000001):

        print(f"# Vortex Line Test: FAIL")

    else:

        print(f"# Vortex Line Test: PASS")


def test_vortex_ring():

    vertex_coordinates = np.array([[2, 0, 0],
                                   [0, 0, 0],
                                   [0, 2, 0],
                                   [2, 2, 0]]).transpose()

    target_point = [1, 1, 0]
    circulation = 1

    induced_velocity = vlm.vortex_ring(vertex_coordinates, target_point, circulation)

    if not (abs(induced_velocity[0] - 0) < 0.0000001 and abs(induced_velocity[1] - 0) < 0.0000001 and
       abs(induced_velocity[2] - (-0.45015816)) < 0.0000001):

        print(f"# Vortex Ring Test: FAIL")

    else:

        print(f"# Vortex Ring Test: PASS")


def test_vortex_horseshoe():

    vertex_coordinates = np.array([[2, 0, 0],
                                   [0, 0, 0],
                                   [0, 2, 0],
                                   [2, 2, 0]]).transpose()

    target_point = [1, 1, 0]
    circulation = 1

    induced_velocity, wake_induced_velocity = vlm.vortex_horseshoe(vertex_coordinates, target_point,
                                                                   circulation)

    # print(f"Induced velocity = {induced_velocity}")
    # print(f"Wake Induced Velocity = {wake_induced_velocity}")

    if not (abs(induced_velocity[0] - 0) < 0.0000001 and
            abs(induced_velocity[1] - 0) < 0.0000001 and
            abs(induced_velocity[2] - (-0.33761862)) < 0.0000001 and
            abs(wake_induced_velocity[0] - 0) < 0.0000001 and
            abs(wake_induced_velocity[1] - 0) < 0.0000001 and
            abs(wake_induced_velocity[2] - (-0.22507908)) < 0.0000001):

        print(f"# Vortex Ring Test: FAIL")

    else:

        print(f"# Vortex Ring Test: PASS")

if __name__ == "__main__":

    print()
    print("=================================")
    print("= Testing vortex_lattice module =")
    print("=================================")
    print()
    test_vortex_line()
    test_vortex_ring()
    test_vortex_horseshoe()
