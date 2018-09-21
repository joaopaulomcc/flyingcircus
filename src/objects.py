import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from numpy import cos, sin, tan
from numpy.linalg import norm


class Wing():
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

    def six_points(self):

        points = np.zeros((6, 2))



