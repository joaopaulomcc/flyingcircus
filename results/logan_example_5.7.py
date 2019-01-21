"""
Comparition between the results obtained using the finite element solver developed with those
presented in the textbook "A First Course in the Finite Element Method" - 5th edition by
Daryl L. Logan
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from context import src
from src import geometry as geo
from src import visualization as vis
from src import aerodynamics as aero
from src import loads
from src import structures as struct


# ==================================================================================================
print()
print("============================================================")
print("= VALIDATION OF FEM SOLVER                                 =")
print("= VALIDATION CASE: A First Course in the Finite Element    =")
print("= Method - 5th Edition - Chapter 5 - Example 5.7           =")
print("= VALIDATION CASE AUTHOR: Daryl L. Logan                   =")
print("============================================================")
# ==================================================================================================

POINT_1 = np.array([0.0, 0.0, 0.0])
POINT_2 = np.array([3.0, 4.0, 12.0])

material = struct.objects.Material(
    name="material",
    density=None,
    elasticity_modulus=30000,
    rigidity_modulus=10000,
    poisson_ratio=None,
    yield_tensile_stress=None,
    ultimate_tensile_stress=None,
    yield_shear_stress=None,
    ultimate_shear_stress=None,
)

section = geo.objects.Section(
    identifier="section",
    material=material,
    area=10,
    Iyy=100,
    Izz=100,
    J=50,
    shear_center=0.5,
)

# ==================================================================================================
# BEAMS

beam_property = struct.objects.ElementProperty(section=section, material=material)

beam_I = geo.objects.Beam(
    identifier="beam_I",
    root_point=POINT_1,
    tip_point=POINT_2,
    orientation_vector=np.array([-4, 3, 0]),
    ElementProperty=beam_property,
)

struct_components = [beam_I]

# ==================================================================================================
# CONNECTIONS

struct_connections = []

# ==================================================================================================
# FEM GRID GENERATION

N_BEAM_ELEMENTS = 1

beam_I_struct_grid = beam_I.create_grid(n_elements=N_BEAM_ELEMENTS)

struct_grid = [
    beam_I_struct_grid,
]

# Numbering nodes

struct.fem.number_nodes(struct_components, struct_grid, struct_connections)


# ==================================================================================================
# FEM ELEMENTS GENERATION

beam_I_fem_elements = struct.fem.generate_beam_fem_elements(
    beam=beam_I, beam_nodes_list=beam_I_struct_grid, prop_choice="ROOT"
)

struct_elements = [
    beam_I_fem_elements,
]

# ==================================================================================================
# LOADS GENERATION

beam_I_tip_load = struct.objects.Load(
    application_node=beam_I_struct_grid[-1],
    load=np.array([0.0, -50.0, 0.0, -1000.0, 0.0, 0.0]),
)

struct_loads = [beam_I_tip_load]

# ==================================================================================================
# CONSTRAINTS GENERATION

beam_I_fixation = struct.objects.Constraint(
    application_node=beam_I_struct_grid[0],
    dof_constraints=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
)

struct_constraints = [beam_I_fixation]

# ==================================================================================================
# PLOT STRUCTURE

vis.plot_3D.plot_structure(struct_elements)

# ==================================================================================================
# FEM SOLVER

deformations, internal_loads = struct.fem.structural_solver(
    struct_grid, struct_elements, struct_loads, struct_constraints
)

# deformed_grid, force_vector, deformations, node_vector = struct.fem.structural_solver(
#    struct_grid, struct_elements, struct_loads, struct_constraints
# )

# ==================================================================================================
# PLOT DEFORMED STRUCTURE

vis.plot_3D.plot_deformed_structure(
    struct_elements, deformations, scale_factor=5
)

print("FINISHED")
