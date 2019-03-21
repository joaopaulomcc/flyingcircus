"""
test_mesh.py

Testing suite for mesh module

Author: João Paulo Monteiro Cruvinel da Costa
email: joaopaulomcc@gmail.com / joao.cruvinel@embraer.com.br
github: joaopaulomcc
"""
# ==================================================================================================
# IMPORTS

import numpy as np
import scipy as sc
import timeit

import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['backend.qt5']='PySide2'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from numpy import sin, cos, tan, pi, dot, cross
from numpy.linalg import norm

from context import flyingcircus
from flyingcircus import mesh
from flyingcircus import basic_objects
from samples import wing_simple
# ==================================================================================================
# TESTS


def test_generate_mesh():

    area = 20
    aspect_ratio = 5
    taper_ratio = 0.66
    sweep_quarter_chord = 25
    dihedral = 25
    incidence = 0
    torsion = 0
    position = [0, 0, 0]

    wing = basic_objects.Wing(area, aspect_ratio, taper_ratio, sweep_quarter_chord, dihedral,
                              incidence, torsion, position)

    n_semi_wingspam_panels = 500
    n_chord_panels = 100
    wingspam_discretization_type = "linear"
    chord_discretization_type = "linear"

    xx, yy, zz = mesh.generate_mesh(wing, n_semi_wingspam_panels, n_chord_panels,
                                    wingspam_discretization_type, chord_discretization_type)

    return xx, yy, zz

# --------------------------------------------------------------------------------------------------


def plot_mesh(xx, yy, zz):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type="persp")
    ax.autoscale(False)
    ax.plot_wireframe(xx, yy, zz)
    plt.show()

# --------------------------------------------------------------------------------------------------


def test_generate_panel_matrix():

    wing = wing_simple
    n_semi_wingspam_panels = 500
    n_chord_panels = 100
    wingspam_discretization_type = "linear"
    chord_discretization_type = "linear"

    xx, yy, zz = mesh.generate_mesh(wing, n_semi_wingspam_panels, n_chord_panels,
                                    wingspam_discretization_type, chord_discretization_type)

    panel_matrix = mesh.generate_panel_matrix(xx, yy, zz)
    #print("Panel Matrix")
    #print(panel_matrix)

# --------------------------------------------------------------------------------------------------


def test_generate_col_points_matrix():

    area = 20
    aspect_ratio = 5
    taper_ratio = 1
    sweep_quarter_chord = 0
    dihedral = 0
    incidence = 0
    torsion = 0
    position = [0, 0, 0]

    simple_rectangular = basic_objects.Wing(area, aspect_ratio, taper_ratio, sweep_quarter_chord, dihedral,
                                  incidence, torsion, position)

    wing = simple_rectangular
    n_semi_wingspam_panels = 500
    n_chord_panels = 100
    wingspam_discretization_type = "linear"
    chord_discretization_type = "linear"

    xx, yy, zz = mesh.generate_mesh(wing, n_semi_wingspam_panels, n_chord_panels,
                                    wingspam_discretization_type, chord_discretization_type)

    col_points_matrix_xx, col_points_matrix_yy, col_points_matrix_zz = mesh.generate_col_points_matrix(xx,
    yy, zz)
    #print(col_points_matrix_xx)
    #print(col_points_matrix_yy)
    #print(col_points_matrix_zz)


if __name__ == "__main__":

    print()
    print("==================================")
    print("= Testing mesh_generation module =")
    print("==================================")
    print()

    print("generate_mesh time")
    print(timeit.timeit(test_generate_mesh, number=10) / 10)
    #xx, yy, zz = test_generate_mesh()
    #plot_mesh(xx, yy, zz)
    print()

    print("generate_panel_matrix time")
    print(timeit.timeit(test_generate_panel_matrix, number=10) / 10)
    print()

    print("generate_col_points_matrix time")
    print(timeit.timeit(test_generate_col_points_matrix, number=10) / 10)
    print()