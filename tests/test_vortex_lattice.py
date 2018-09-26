from context import src

import numpy as np
from src import vortex_lattice as vlm
from src import objects as obj
from src import mesh_generation as msh
from src import functions as fc


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


def test_lifting_line_horse_shoe():

    area = 20
    aspect_ratio = 5
    taper_ratio = 1
    sweep_quarter_chord = 0
    dihedral = 0
    incidence = 0
    torsion = 0
    position = [0, 0, 0]

    alpha = 5
    beta = 0
    gamma = 0
    attitude_vector = [alpha, beta, gamma]
    altitude = 5000
    simple_rectangular = obj.Wing(area, aspect_ratio, taper_ratio, sweep_quarter_chord, dihedral,
                                  incidence, torsion, position)

    true_airspeed = 100
    flow_velocity_vector = fc.velocity_vector(true_airspeed, alpha, beta, gamma)[:,0]
    infinity_mult = 25

    wing = simple_rectangular
    n_semi_wingspam_panels = 5
    n_chord_panels = 1
    wingspam_discretization_type = "linear"
    chord_discretization_type = "linear"

    gamma, exitCode = vlm.lifting_line_horse_shoe(simple_rectangular, attitude_vector,
                                                  true_airspeed, altitude,
                                                  n_semi_wingspam_panels, n_chord_panels,
                                                  wingspam_discretization_type,
                                                  chord_discretization_type,
                                                  infinity_mult)

    print(gamma)
    print(exitCode)



if __name__ == "__main__":

    print()
    print("=================================")
    print("= Testing vortex_lattice module =")
    print("=================================")
    print()
    # test_vortex_line()
    # test_vortex_ring()
    # test_vortex_horseshoe()
    # test_vortex_solver()
    test_lifting_line_horse_shoe()
