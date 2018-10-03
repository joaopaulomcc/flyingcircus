"""
finite_elements_method.py

Implementation of the finite element method for 3D beam

Author: João Paulo Monteiro Cruvinel da Costa
email: joaopaulomcc@gmail.com / joao.cruvinel@embraer.com.br
github: joaopaulomcc
"""
# ==================================================================================================
# IMPORTS
import numpy as np
import scipy as sc

from numpy import sin, cos, tan, pi
from .fast_operations import dot, cross, norm
