import numpy as np
from context import src
from src import aerodynamics

# ==================================================================================================
# FUNCTIONS


def test_horse_shoe_ind_vel():

    point_a = np.array([0, 0, 0])
    point_b = np.array([0, 1, 0])
    target_point = np.array([0.5, 0.5, 0])
    circulation = 1
    vortex_radius = 0.001

    induced_velocity = aerodynamics.functions.horse_shoe_ind_vel(point_a, point_b, target_point, circulation, vortex_radius)

    print(induced_velocity)


# ==================================================================================================
# TESTS

if __name__ == "__main__":

    print()
    print("==================================")
    print("= Testing aerodynamics.functions =")
    print("==================================")
    print()
    print("- Testing horse_shoe_ind_vel")
    test_horse_shoe_ind_vel()

