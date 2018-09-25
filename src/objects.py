import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from .vortex_lattice import vortex_horseshoe
#from .vortex_lattice import horse_shoe_vertex
from numpy import cos, sin, tan, dot, cross
from numpy.linalg import norm


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


class Panel(object):
    """Panel object"""

    def __init__(self, xx, yy, zz):
        """Args:
            xx [[float]] = grid with panel points x coordinates
            yy [[float]] = grid with panel points x coordinates
            zz [[float]] = grid with panel points x coordinates
        """
        self.xx = xx
        self.yy = yy
        self.zz = zz
        self.A = np.array([xx[1][0], yy[1][0], zz[1][0]])
        self.B = np.array([xx[0][0], yy[0][0], zz[0][0]])
        self.C = np.array([xx[0][1], yy[0][1], zz[0][1]])
        self.D = np.array([xx[1][1], yy[1][1], zz[1][1]])
        self.AC = self.C - self.A
        self.BD = self.D - self.B
        self.n = cross(self.BD, self.AC) / norm(cross(self.BD, self.AC))
        self.area = dot(self.n, cross(self.BD, self.AC)) / 2
        self.col_point = np.array([0.5 * (0.75 * self.A[0] + 0.25 * self.B[0] + 0.25 * self.C[0] + 0.75 * self.D[0]),
                                  0.5 * (self.A[1] + self.D[1]),
                                  0.5 * (self.A[2] + self.D[2])])

    def horse_shoe(self, infinity):
        hs_B = self.B + [0.25 * (self.A[0] - self.B[0]), 0, 0]
        hs_A = hs_B + [infinity, 0, 0]
        hs_C = self.C + [0.25 * (self.D[0] - self.C[0]), 0, 0]
        hs_D = hs_C + [infinity, 0, 0]

        hs_A = hs_A[np.newaxis]
        hs_B = hs_B[np.newaxis]
        hs_C = hs_C[np.newaxis]
        hs_D = hs_D[np.newaxis]

        horse_shoe_vertex = np.concatenate([hs_A, hs_B, hs_C, hs_D])

        return horse_shoe_vertex.transpose()

    def hs_induced_velocity(self, target_point, circulation, infinity):

        vertex_coordinates = self.horse_shoe(infinity)

        induced_velocity, wake_induced_velocity = vortex_horseshoe(vertex_coordinates,
                                                                   target_point,
                                                                   circulation)

        return induced_velocity, wake_induced_velocity


