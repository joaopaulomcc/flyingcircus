# IMPORTS

import numpy as np
import scipy as sc
#import matplotlib
#matplotlib.use('Qt5Agg')
#matplotlib.rcParams['backend.qt5']='PySide2'
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import axes3d

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

print(timeit.timeit('\
import numpy as np \n\
from context import src \n\
from src import vortex_lattice_method \n\
from src import mesh \n\
from src import basic_objects \n\
from src import geometry \n\
from src import visualization \n\
from samples import wing_simple \n\
area = 20 \n\
aspect_ratio = 5 \n\
taper_ratio = 0.666 \n\
sweep_quarter_chord = 0 \n\
dihedral = 0 \n\
incidence = 0 \n\
torsion = 0 \n\
position = [0, 0, 0] \n\
n_semi_wingspam_panels = 10 \n\
n_chord_panels = 4 \n\
wingspam_discretization_type = "linear" \n\
chord_discretization_type = "linear" \n\
alpha = 5 \n\
beta = 0 \n\
gamma = 0 \n\
attitude_vector = [alpha, beta, gamma] \n\
altitude = 5000 \n\
true_airspeed = 100 \n\
flow_velocity_vector = geometry.velocity_vector(true_airspeed, alpha, beta, gamma)[:,0] \n\
infinity_mult = 25 \n\
wing = basic_objects.Wing(area, aspect_ratio, taper_ratio, sweep_quarter_chord, dihedral, incidence, torsion, position) \n\
xx, yy, zz = mesh.generate_mesh(wing, n_semi_wingspam_panels, n_chord_panels, wingspam_discretization_type, chord_discretization_type) \n\
panel_matrix = mesh.generate_panel_matrix(xx, yy, zz, wing.wing_span) \n\
panel_vector = vortex_lattice_method.flatten(panel_matrix) \n\
gamma = vortex_lattice_method.gamma_solver(panel_vector, flow_velocity_vector, infinity_mult * wing.wing_span) \n\
downwash = vortex_lattice_method.downwash_solver(panel_vector, gamma) \n\
lift, drag = vortex_lattice_method.lift_drag(panel_vector, gamma, downwash, true_airspeed, 1.225) \n\
lift_grid = np.reshape(lift, np.shape(panel_matrix))', number=10)/10)


#visualization.plot_results(xx, yy, zz, gamma_grid)\