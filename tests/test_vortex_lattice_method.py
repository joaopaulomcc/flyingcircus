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
import matplotlib.pyplot as plt
import timeit

from numpy import sin, cos, tan, pi, dot, cross
from numpy.linalg import norm

from context import src
from src import vortex_lattice_method
from src import mesh
from src import basic_objects
from src import geometry
from samples import wing_simple

# ==============================================================================================


def test_vortex_solver():

    area = 20
    aspect_ratio = 5
    taper_ratio = 1
    sweep_quarter_chord = 0
    dihedral = 0
    incidence = 0
    torsion = 0
    position = [0, 0, 0]

    n_semi_wingspam_panels = 20
    n_chord_panels = 4
    wingspam_discretization_type = "linear"
    chord_discretization_type = "linear"

    alpha = 5
    beta = 0
    gamma = 0
    attitude_vector = [alpha, beta, gamma]
    altitude = 5000

    true_airspeed = 100
    flow_velocity_vector = geometry.velocity_vector(true_airspeed, alpha, beta, gamma)[:,0]
    infinity_mult = 25

    wing = basic_objects.Wing(area, aspect_ratio, taper_ratio, sweep_quarter_chord, dihedral,
                              incidence, torsion, position)

    xx, yy, zz = mesh.generate_mesh(wing, n_semi_wingspam_panels, n_chord_panels,
                                    wingspam_discretization_type, chord_discretization_type)

    panel_matrix = mesh.generate_panel_matrix(xx, yy, zz)

    panel_vector = vortex_lattice_method.flatten(panel_matrix)

    gamma = vortex_lattice_method.vortex_solver(panel_vector, flow_velocity_vector,
                                                infinity_mult * wing.wing_span)

    # gamma, exitCode = vlm.lifting_line_horse_shoe(simple_rectangular, attitude_vector,
    #                                               true_airspeed, altitude,
    #                                               n_semi_wingspam_panels, n_chord_panels,
    #                                               wingspam_discretization_type,
    #                                               chord_discretization_type,
    #                                               infinity_mult)

    # print(gamma)
    # print(f"Exit Code: {exitCode}")

# ==================================================================================================
# RUNNING TESTS

if __name__ == "__main__":

    print()
    print("========================================")
    print("= Testing vortex_lattice_method module =")
    print("========================================")
    print()
    print("Testing vortex_solver")
    print(f"Execution Time: {timeit.timeit(test_vortex_solver, number=1) / 1}")
