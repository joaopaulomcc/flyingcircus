"""
test_vortex_lattice_method.py

Testing suite for cortex_lattice_method module

Author: João Paulo Monteiro Cruvinel da Costa
email: joaopaulomcc@gmail.com / joao.cruvinel@embraer.com.br
github: joaopaulomcc
"""
# ==================================================================================================
# IMPORTS

import numpy as np
import scipy as sc
import timeit
import time

from numpy import sin, cos, tan, pi, dot, cross
from numpy.linalg import norm

from context import flyingcircus
from flyingcircus import vortex_lattice_method
from flyingcircus import mesh
from flyingcircus import basic_objects
from flyingcircus import geometry
from flyingcircus import atmosphere

from numba import jit

# ==============================================================================================


# --------------------------------------------------------------------------------------------------


def test_gamma_solver():

    gamma = vortex_lattice_method.gamma_solver(panel_vector, flow_velocity_vector,
                                               infinity_mult * wing.wing_span)

# --------------------------------------------------------------------------------------------------


def test_downwash_solver():

    downwash = vortex_lattice_method.downwash_solver(panel_vector, gamma)

# --------------------------------------------------------------------------------------------------


def test_lift_drag():

    lift, drag = vortex_lattice_method.lift_drag(panel_vector, gamma, downwash, true_airspeed, density)

# ==================================================================================================
# RUNNING TESTS

if __name__ == "__main__":

    print()
    print("========================================")
    print("= Testing vortex_lattice_method module =")
    print("========================================")
    print()
    area = 20
    aspect_ratio = 5
    taper_ratio = 1
    sweep_quarter_chord = 0
    dihedral = 0
    incidence = 0
    torsion = 0
    position = [0, 0, 0]

    n_semi_wingspam_panels = 10
    n_chord_panels = 4
    wingspam_discretization_type = "linear"
    chord_discretization_type = "linear"

    alpha = 5
    beta = 0
    gamma = 0
    attitude_vector = [alpha, beta, gamma]
    altitude = 5000

    density, pressure, temperature = atmosphere.ISA(altitude)


    true_airspeed = 100
    flow_velocity_vector = geometry.velocity_vector(true_airspeed, alpha, beta, gamma)[:,0]
    infinity_mult = 25

    wing = basic_objects.Wing(area, aspect_ratio, taper_ratio, sweep_quarter_chord, dihedral,
                                  incidence, torsion, position)

    xx, yy, zz = mesh.generate_mesh(wing, n_semi_wingspam_panels, n_chord_panels,
                                        wingspam_discretization_type, chord_discretization_type)
    end = time.time()
    print(f"Generating Mesh completed in {end - start} seconds")

    print("Generating Panel Matrix...")
    start = time.time()
    panel_matrix = mesh.generate_panel_matrix(xx, yy, zz, infinity_mult * wing.wing_span)
    end = time.time()
    print(f"Generating Panel Matrix completeted in {end - start} seconds")

    panel_vector = vortex_lattice_method.flatten(panel_matrix)

    print("Solving circulations...")
    start = time.time()

    gamma = vortex_lattice_method.gamma_solver(panel_vector, flow_velocity_vector,
                                               infinity_mult * wing.wing_span)
    end = time.time()
    print(f"Solving circulations completed in {end - start}")

    print("Solving downwash...")
    start = time.time()

    downwash = vortex_lattice_method.downwash_solver(panel_vector, gamma)

    end = time.time()
    print(f"Solving downwash completed in {end - start}")

    lift, drag = vortex_lattice_method.lift_drag(panel_vector, gamma, downwash, true_airspeed, density)

    print()
    print(f"# RESULTS")
    print(f"- Lift: = {lift.sum()}")
    print(f"- Cl: {lift.sum() / (0.5 * density * (true_airspeed) **2 * wing.area)}")
    print(f"- Drag: = {drag.sum()}")
    print(f"- Cd: {drag.sum() / (0.5 * density * (true_airspeed) **2 * wing.area)}")
    print()

    print("# Testing gamma_solver")
    execution_time = timeit.timeit(test_gamma_solver, setup='from __main__ import panel_vector, flow_velocity_vector, infinity_mult, wing', number=10) / 10
    print(f"- Total Execution Time: {execution_time} s")
    print()
    print("# Testing downwash_solver")
    execution_time = timeit.timeit(test_downwash_solver, setup='from __main__ import panel_vector, gamma', number=1) / 1
    print(f"- Total Execution Time: {execution_time} s")
    print()
    print("# Testing lift_drag")
    execution_time = timeit.timeit(test_downwash_solver, setup='from __main__ import panel_vector, gamma, downwash, true_airspeed, density', number=10) / 10
    print(f"- Total Execution Time: {execution_time} s")
