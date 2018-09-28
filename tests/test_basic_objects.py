"""
test_basic_elements.py

Testing suite for basic_elements module

Author: João Paulo Monteiro Cruvinel da Costa
email: joaopaulomcc@gmail.com / joao.cruvinel@embraer.com.br
github: joaopaulomcc
"""
# ==================================================================================================
# IMPORTS

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from numpy import sin, cos, tan, pi, dot, cross
from numpy.linalg import norm

from context import src
from src import basic_objects

# ==================================================================================================
# TESTS


def test_wing():

    area = 20
    aspect_ratio = 5
    taper_ratio = 1
    sweep_quarter_chord = 0
    dihedral = 0
    incidence = 0
    torsion = 0
    position = [0, 0, 0]

    simple_rectangular = basic_objects.Wing(area, aspect_ratio, taper_ratio, sweep_quarter_chord,
                                            dihedral, incidence, torsion, position)

    print()
    print("TESTING Wing")
    print(f"area: {simple_rectangular.area}")
    print(f"AR: {simple_rectangular.AR}")
    print(f"taper_ratio: {simple_rectangular.taper_ratio}")
    print(f"sweep: {simple_rectangular.sweep}")
    print(f"sweep_rad: {simple_rectangular.sweep_rad}")
    print(f"dihedral: {simple_rectangular.dihedral}")
    print(f"dihedral_rad: {simple_rectangular.dihedral_rad}")
    print(f"incidence: {simple_rectangular.incidence}")
    print(f"incidence_rad: {simple_rectangular.incidence_rad}")
    print(f"torsion: {simple_rectangular.torsion}")
    print(f"torsion_rad: {simple_rectangular.torsion_rad}")
    print(f"position: {simple_rectangular.position}")
    print(f"wing_span: {simple_rectangular.wing_span}")
    print(f"semi_wing_span: {simple_rectangular.semi_wing_span}")
    print(f"root_chord: {simple_rectangular.root_chord}")
    print(f"tip_chord: {simple_rectangular.tip_chord}")

# --------------------------------------------------------------------------------------------------


def test_panel():
    x = np.array([0, 2])
    y = np.array([0, 2])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    zz = np.zeros((2, 2))

    target_point = np.array([1, 1, 0])
    circulation = 1
    infinity = 25

    P = basic_objects.Panel(xx, yy, zz)
    induced_velocity, wake_induced_velocity = P.hs_induced_velocity(target_point, circulation, infinity)

    print()
    print("TESTING Panel")
    print(f"Vector AC: {P.AC}")
    print(f"Vector BD: {P.BD}")
    print(f"Vector n: {P.n}")
    print(f"Area n: {P.area}")
    print(f"Collocation Point n: {P.col_point}")
    print(f"induced_velocity: {induced_velocity}")
    print(f"wake_induced_velocity: {wake_induced_velocity}")


def test_panel_4():
    x = np.array([0, 2])
    y = np.array([0, 2])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    zz = np.zeros((2, 2))

    target_point = np.array([1, 1, 0])
    circulation = 1
    infinity = 25

    P = basic_objects.Panel4(xx, yy, zz, infinity)
    induced_velocity, wake_induced_velocity = P.hs_induced_velocity(target_point, circulation)

    print()
    print("TESTING Panel")
    print(f"Vector AC: {P.AC}")
    print(f"Vector BD: {P.BD}")
    print(f"Vector n: {P.n}")
    print(f"Area n: {P.area}")
    print(f"Collocation Point n: {P.col_point}")
    print(f"induced_velocity: {induced_velocity}")
    print(f"wake_induced_velocity: {wake_induced_velocity}")


# ==================================================================================================
# RUN TESTS

if __name__ == "__main__":

    print()
    print("============================")
    print("= Testing objects module =")
    print("============================")
    print()
    test_wing()
    test_panel()
    test_panel_4()