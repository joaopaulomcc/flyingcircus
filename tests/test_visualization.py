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
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['backend.qt5']='PySide2'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import timeit
import time

from numpy import sin, cos, tan, pi, dot, cross
from numpy.linalg import norm

from context import src
from src import vortex_lattice_method
from src import mesh
from src import basic_objects
from src import geometry
from src import visualization
from samples import wing_simple

# ==============================================================================================



def test_plot_results():

    area = 20
    aspect_ratio = 5
    taper_ratio = 0.666
    sweep_quarter_chord = 0
    dihedral = 0
    incidence = 0
    torsion = 0
    position = [0, 0, 0]

    n_semi_wingspam_panels = 5
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

    print("Generating Mesh...")
    start = time.time()
    xx, yy, zz = mesh.generate_mesh(wing, n_semi_wingspam_panels, n_chord_panels,
                                    wingspam_discretization_type, chord_discretization_type)

    visualization.plot_results(xx, yy, zz, gamma_grid)

def test_plot_results():

    area = 20
    aspect_ratio = 5
    taper_ratio = 0.666
    sweep_quarter_chord = 0
    dihedral = 0
    incidence = 0
    torsion = 0
    position = [0, 0, 0]

    n_semi_wingspam_panels = 5
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

    print("Generating Mesh...")
    start = time.time()
    xx, yy, zz = mesh.generate_mesh(wing, n_semi_wingspam_panels, n_chord_panels,
                                    wingspam_discretization_type, chord_discretization_type)
    end = time.time()
    print(f"Generating Mesh completed in {end - start} seconds")

    print("Generating Panel Matrix...")
    start = time.time()
    panel_matrix = mesh.generate_panel_matrix(xx, yy, zz)
    end = time.time()
    print(f"Generating Panel Matrix completeted in {end - start} seconds")

    print("Generating Panel Vector...")
    start = time.time()
    panel_vector = vortex_lattice_method.flatten(panel_matrix)
    end = time.time()
    print(f"Generating Panel Vector completed in {end - start} seconds")

    print("Solving circulations...")
    start = time.time()
    gamma = vortex_lattice_method.vortex_solver(panel_vector, flow_velocity_vector,
                                                infinity_mult * wing.wing_span)
    end = time.time()
    print(f"Solving circulations completed in {end - start}")

    gamma_grid = np.reshape(gamma, np.shape(panel_matrix))

    visualization.plot_results(xx, yy, zz, gamma_grid)

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
    print("================================")
    print("= Testing visualization module =")
    print("================================")
    print()
    print("Testing plot_results")
    test_plot_results()
