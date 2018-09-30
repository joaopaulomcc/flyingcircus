"""
basic_objects.py

Collection of basic objects both aerodynamic and structural
Except where explicitly stated all units are S.I.

Author: João Paulo Monteiro Cruvinel da Costa
email: joaopaulomcc@gmail.com / joao.cruvinel@embraer.com.br
github: joaopaulomcc
"""
# ==================================================================================================
# IMPORTS
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from numpy import sin, cos, tan, pi
from numba import jit, jitclass

from . import basic_elements

from .fast_operations import dot, cross, norm, normalize
# ==================================================================================================


class Wing(object):
    """Description of wing properties"""

    def __init__(self, area, aspect_ratio, taper_ratio, sweep_quarter_chord, dihedral, incidence,
                 torsion, position):
        """Class to store wing geometrical data"""

        self.area = area
        self.AR = aspect_ratio
        self.taper_ratio = taper_ratio
        self.sweep = sweep_quarter_chord
        self.sweep_rad = np.radians(sweep_quarter_chord)
        self.dihedral = dihedral
        self.dihedral_rad = np.radians(dihedral)
        self.incidence = incidence
        self.incidence_rad = np.radians(incidence)
        self.torsion = torsion
        self.torsion_rad = np.radians(torsion)
        self.position = position
        self.wing_span = (self.area * self.AR) ** 0.5
        self.semi_wing_span = self.wing_span / 2
        self.root_chord = 2 * self.area / (self.wing_span * (1 + self.taper_ratio))
        self.tip_chord = self.root_chord * self.taper_ratio


#class Panel(object):
#    """Panel object"""
#
#    def __init__(self, xx, yy, zz):
#        """Args:
#            xx [[float]] = grid with panel points x coordinates
#            yy [[float]] = grid with panel points x coordinates
#            zz [[float]] = grid with panel points x coordinates
#        """
#        self.xx = xx
#        self.yy = yy
#        self.zz = zz
#        self.A = np.array([xx[1][0], yy[1][0], zz[1][0]])
#        self.B = np.array([xx[0][0], yy[0][0], zz[0][0]])
#        self.C = np.array([xx[0][1], yy[0][1], zz[0][1]])
#        self.D = np.array([xx[1][1], yy[1][1], zz[1][1]])
#        self.AC = self.C - self.A
#        self.BD = self.D - self.B
#        self.n = cross(self.BD, self.AC) / norm(cross(self.BD, self.AC))
#        self.area = dot(self.n, cross(self.BD, self.AC)) / 2
#        self.col_point = np.array([0.5 * (0.75 * self.A[0] + 0.25 * self.B[0] + 0.25 * self.C[0] + 0.75 * self.D[0]),
#                                  0.5 * (self.A[1] + self.D[1]),
#                                  0.5 * (self.A[2] + self.D[2])])
#
#    def horse_shoe(self, infinity):
#        hs_B = self.B + [0.25 * (self.A[0] - self.B[0]), 0, 0]
#        hs_A = hs_B + [infinity, 0, 0]
#        hs_C = self.C + [0.25 * (self.D[0] - self.C[0]), 0, 0]
#        hs_D = hs_C + [infinity, 0, 0]
#
#        hs_A = hs_A[np.newaxis]
#        hs_B = hs_B[np.newaxis]
#        hs_C = hs_C[np.newaxis]
#        hs_D = hs_D[np.newaxis]
#
#        horse_shoe_vertex = np.concatenate([hs_A, hs_B, hs_C, hs_D])
#
#        return horse_shoe_vertex.transpose()
#
#    def hs_induced_velocity(self, target_point, circulation, infinity):
#
#        vertex_coordinates = self.horse_shoe(infinity)
#
#        induced_velocity, wake_induced_velocity = basic_elements.vortex_horseshoe(vertex_coordinates,
#                                                                                  target_point,
#                                                                                  circulation)
#
#        return induced_velocity, wake_induced_velocity


class Panel(object):
    """Panel object"""

    def __init__(self, xx, yy, zz, infinity):
        """Args:
            xx [[float]] = grid with panel points x coordinates
            yy [[float]] = grid with panel points x coordinates
            zz [[float]] = grid with panel points x coordinates
        """
        self.xx = xx
        self.yy = yy
        self.zz = zz
        self.infinity = infinity
        self.A = np.array([xx[1][0], yy[1][0], zz[1][0]])
        self.B = np.array([xx[0][0], yy[0][0], zz[0][0]])
        self.C = np.array([xx[0][1], yy[0][1], zz[0][1]])
        self.D = np.array([xx[1][1], yy[1][1], zz[1][1]])
        self.AC = self.C - self.A
        self.BD = self.D - self.B

        self.l_chord = self.A - self.B
        self.l_chord_1_4 = self.B + 0.25 * self.l_chord
        self.l_chord_3_4 = self.B + 0.75 * self.l_chord

        self.r_chord = self.D - self.C
        self.r_chord_1_4 = self.C + 0.25 * self.r_chord
        self.r_chord_3_4 = self.C + 0.75 * self.r_chord

        self.l_edge = self.C - self.B
        self.l_edge_1_2 = self.B + 0.5 * self.l_edge

        self.t_edge = self.D - self.A
        self.t_edge_1_2 = self.A + 0.5 * self.t_edge

        self.col_point = 0.75 * (self.t_edge_1_2 - self.l_edge_1_2) + self.l_edge_1_2

        self.spam = dot(self.l_edge, np.array([0, 1, 0]))
        self.n = normalize(cross(self.BD, self.AC))
        self.area = dot(self.n, cross(self.BD, self.AC)) / 2

        hs_A = np.array([self.l_chord_1_4[0] + infinity, self.l_chord_1_4[1], self.l_chord_1_4[2]])
        hs_B = np.array([self.l_chord_1_4[0], self.l_chord_1_4[1], self.l_chord_1_4[2]])
        hs_C = np.array([self.r_chord_1_4[0], self.r_chord_1_4[1], self.r_chord_1_4[2]])
        hs_D = np.array([self.r_chord_1_4[0] + infinity, self.r_chord_1_4[1], self.r_chord_1_4[2]])
        hs_A = hs_A[np.newaxis]
        hs_B = hs_B[np.newaxis]
        hs_C = hs_C[np.newaxis]
        hs_D = hs_D[np.newaxis]

        self.horse_shoe_vortex = np.concatenate([hs_A, hs_B, hs_C, hs_D]).transpose()

    def hs_induced_velocity(self, target_point, circulation):

        induced_velocity, wake_induced_velocity = basic_elements.vortex_horseshoe(self.horse_shoe_vortex,
                                                                                  target_point,
                                                                                  circulation)

        return induced_velocity, wake_induced_velocity


