"""
test_visualization.py

Testing suite for visualization module

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

from context import src
from src import atmosphere
from src import vortex_lattice_method
from src import mesh
from src import basic_objects
from src import geometry
from src import visualization
from samples import wing_simple

from numba import jit
# ==============================================================================================

@jit
def test_plot_mesh():

    xx, yy, zz = mesh.generate_mesh(wing, n_semi_wingspam_panels, n_chord_panels,
                                    wingspam_discretization_type, chord_discretization_type)

    visualization.plot_mesh(xx, yy, zz)

# --------------------------------------------------------------------------------------------------


@jit
def test_plot_results():

    xx, yy, zz = mesh.generate_mesh(wing, n_semi_wingspam_panels, n_chord_panels,
                                    wingspam_discretization_type, chord_discretization_type)

    panel_matrix = mesh.generate_panel_matrix(xx, yy, zz, wing.wing_span)

    panel_vector = vortex_lattice_method.flatten(panel_matrix)

    gamma = vortex_lattice_method.gamma_solver(panel_vector, flow_velocity_vector,
                                               infinity_mult * wing.wing_span)

    downwash = vortex_lattice_method.downwash_solver(panel_vector, gamma)

    lift, drag = vortex_lattice_method.lift_drag(panel_vector, gamma, downwash, true_airspeed, density)

    gamma_grid = np.reshape(gamma, np.shape(panel_matrix))
    lift_grid = np.reshape(lift, np.shape(panel_matrix))
    drag_grid = np.reshape(drag, np.shape(panel_matrix))

    visualization.plot_results(xx, yy, zz, lift_grid)


# ==================================================================================================
# RUNNING TESTS

if __name__ == "__main__":

    print()
    print("================================")
    print("= Testing visualization module =")
    print("================================")
    print()

    area = 19.4
    aspect_ratio = 6.39
    taper_ratio = 0.42
    sweep_quarter_chord = 0
    dihedral = 7
    incidence = 0
    torsion = 0
    position = [0, 0, 0]

    n_semi_wingspam_panels = 50
    n_chord_panels = 6
    wingspam_discretization_type = "linear"
    chord_discretization_type = "linear"

    alpha = 3.6
    beta = 0
    gamma = 0
    attitude_vector = [alpha, beta, gamma]
    altitude = 0

    density, pressure, temperature = atmosphere.ISA(altitude)

    true_airspeed = 113.
    flow_velocity_vector = geometry.velocity_vector(true_airspeed, alpha, beta, gamma)[:,0]
    infinity_mult = 25.

    wing = basic_objects.Wing(area, aspect_ratio, taper_ratio, sweep_quarter_chord, dihedral,
                              incidence, torsion, position)

    print("Testing plot_results")
    #test_plot_mesh()
    test_plot_results()