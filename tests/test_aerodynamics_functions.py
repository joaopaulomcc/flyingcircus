import numpy as np
from context import flyingcircus
from flyingcircus import aerodynamics

# ==================================================================================================
# FUNCTIONS


def test_horse_shoe_ind_vel():

    point_a = np.array([0, 0, 0])
    point_b = np.array([0, 1, 0])
    target_point = np.array([0.5, 0.5, 0])
    circulation = 1
    vortex_radius = 0.001

    induced_velocity = aerodynamics.functions.horse_shoe_ind_vel(
        point_a, point_b, target_point, circulation, vortex_radius
    )

    print(induced_velocity)


def test_horse_shoe_aero_force():

    point_a = np.array([0, 0, 0])
    point_b = np.array([0, 1, 0])
    circulation = 1
    alpha = np.radians(5)
    true_airspeed = 100
    flow_vector = np.array([np.cos(alpha) * true_airspeed, 0, np.sin(alpha) * true_airspeed])
    air_density = 1.225

    aero_force = aerodynamics.functions.horse_shoe_aero_force(
        point_a, point_b, circulation, flow_vector, air_density
    )

    print(aero_force)


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
    print()

    print("- Testing test_horse_shoe_aero_force")
    test_horse_shoe_aero_force()
    print()

