"""
samples.py

Collection of sample objects for use in tests.

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

from context import flyingcircus
from flyingcircus import basic_objects

# --------------------------------------------------------------------------------------------------
# wing_simple: Simple rectangular wing with no dihedral or torsion

wing_simple = basic_objects.Wing(area=20,
                                 aspect_ratio=5,
                                 taper_ratio=1,
                                 sweep_quarter_chord=0,
                                 dihedral=0,
                                 incidence=0,
                                 torsion=0,
                                 position=[0, 0, 0])


def print_all_attributes(test_object):
    for key in test_object.__dict__:
        print(f"{key} = {test_object.__dict__[key]}")


if __name__ == "__main__":

    print()
    print("wing_simple:")
    print_all_attributes(wing_simple)
