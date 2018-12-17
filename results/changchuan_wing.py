"""
====================================================================================================
Static aeroelastic analysis of very flexible wings based on non-planar vortex lattice method

Xie Changchuan, Wang Libo, Yang Chao, Liu Yi

School of Aeronautic Science and Engineering, Beihang University, Beijing 100191, China

http://dx.doi.org/10.1016/j.cja.2013.04.048

====================================================================================================

Comparisson of the results obtained by in the paper above with those generated by the tool developed
in this work

Author: João Paulo Monteiro Cruvinel da Costa
"""

# ==================================================================================================
# IMPORTS

# Import python scientific libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

# Import code sub packages
from context import src
from src import aerodynamics as aero
from src import aeroelasticity as aelast
from src import control
from src import flight_mechanics as flmec
from src import geometry as geo
from src import loads
from src import structures as struct
from src import visualization as vis

# ==================================================================================================
print()
print("============================================================")
print("= VALIDATION OF AEROELASTIC CALCULATION                    =")
print("= VALIDATION CASE: Static aeroelastic analysis of very     =")
print("= flexible wings based on non-planar vortex lattice method =")
print("= AUTHORS: Xie Changchuan, Wang Libo, Yang Chao, Liu Yi    =")
print("============================================================")
# ==================================================================================================
# GEOMETRY DEFINITION
print()
print("# Generating geometry...")
# Wing section

# Aifoil name, only informative
naca0015 = "NACA 0015"

# Material properties
spring_steel = struct.objects.Material(
    name="Spring Steel",
    density=7.6e3,
    elasticity_modulus=230e9,
    rigidity_modulus=89.3e9,
    poisson_ratio=0.25,
    yield_tensile_stress=350e6,
    ultimate_tensile_stress=420e6,
    yield_shear_stress=0,
    ultimate_shear_stress=0,
)

# Wing section properties
wing_section = geo.objects.Section(
    airfoil=naca0015,
    material=spring_steel,
    area=8.0142e-6,
    Iyy=8.679e-13,
    Izz=3.301e-11,
    J=3.117e-12,
    shear_center=0.5,
)

# --------------------------------------------------------------------------------------------------

# Wing surface

# Definition of the wing planform
wing_surface = geo.objects.Surface(
    identifier="main_wing",
    root_chord=0.06,
    root_section=wing_section,
    tip_chord=0.06,
    tip_section=wing_section,
    length=0.487,
    leading_edge_sweep_angle_deg=0,
    dihedral_angle_deg=0,
    tip_torsion_angle_deg=0,
    control_surface_hinge_position=None,
)

# Creation of the wing macrosurface
wing = geo.objects.MacroSurface(
    position=0,
    incidence=0,
    surface_list=[wing_surface],
    symmetry_plane=None,
    torsion_center=0.5,
)

# ==================================================================================================
# GRID CREATION
print()
print("# Generating aerodynamic and structural grid...")
# Number of panels and finite elements
n_chord_panels = 8
n_span_panels = 64
n_beam_elements_list = 64 * 2

wing_n_chord_panels = n_chord_panels
wing_n_span_panels_list = [n_span_panels]
wing_n_beam_elements_list = [n_beam_elements_list]

# Types of discretization to be used
wing_chord_discretization = "linear"
wing_span_discretization_list = ["linear"]
wing_torsion_function_list = ["linear"]
# Creation of the wing grids
wing_aero_grid, wing_struct_grid = wing.create_grids(
    wing_n_chord_panels,
    wing_n_span_panels_list,
    wing_n_beam_elements_list,
    wing_chord_discretization,
    wing_span_discretization_list,
    wing_torsion_function_list,
)

aero_grid = [wing_aero_grid]

# ==================================================================================================
# AERODYNAMIC LOADS CALCULATION
print()
print("# Setting flight conditions")
# Flight Conditions definition

# Translation velocities
V_x = 34.0
V_y = 0
V_z = 0
velocity_vector = np.array([V_x, V_y, V_z])

# Rotation velocities
rotation_vector = np.array([0, 0, 0])

# Attitude angles in degrees
alpha = 0.4
beta = 0
gamma = 0

attitude_vector = np.array([alpha, beta, gamma])

# Center of rotation is set to the origin
center = np.array([0, 0, 0])

# Altitude is set to sea level
altitude = 0

print()
print("# Running aeroelastic calculation:")
(
    components_force_vector,
    components_panel_vector,
    components_gamma_vector,
    components_force_grid,
    components_panel_grid,
    components_gamma_grid,
) = aero.vlm.aero_loads(
    aero_grid, velocity_vector, rotation_vector, attitude_vector, altitude, center
)

# ==================================================================================================
# PROCESSING RESULTS
components_delta_p_grids = []
components_force_mag_grids = []

for panels, forces in zip(components_panel_grid, components_force_grid):

    delta_p, force = aero.vlm.calc_panels_delta_pressure(panels, forces)
    components_delta_p_grids.append(delta_p)
    components_force_mag_grids.append(force)

ax, fig = vis.plot_3D.plot_results(
    aero_grid,
    components_delta_p_grids,
    title="Changchuan Wing",
    label="Delta Pressure [Pa]",
    colormap="coolwarm"
)

plt.show()
