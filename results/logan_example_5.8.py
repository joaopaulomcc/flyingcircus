"""
Comparition between the results obtained using the finite element solver developed with those
presented in the textbook "A First Course in the Finite Element Method" - 5th edition by
Daryl L. Logan
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from context import flyingcircus
from flyingcircus import geometry as geo
from flyingcircus import visualization as vis
from flyingcircus import aerodynamics as aero
from flyingcircus import loads
from flyingcircus import structures as struct


# ==================================================================================================
print()
print("============================================================")
print("= VALIDATION OF FEM SOLVER                                 =")
print("= VALIDATION CASE: A First Course in the Finite Element    =")
print("= Method - 5th Edition - Chapter 5 - Example 5.8           =")
print("= VALIDATION CASE AUTHOR: Daryl L. Logan                   =")
print("============================================================")
# ==================================================================================================

POINT_1 = np.array([100.0, 100.0, 100.0])
POINT_2 = np.array([0.0, 100.0, 100.0])
POINT_3 = np.array([100.0, 100.0, 0.0])
POINT_4 = np.array([100.0, 0.0,100.0])

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
    root_point=POINT_2,
    tip_point=POINT_1,
    orientation_vector=np.array([0.0, 1.0, 0.0]),
    ElementProperty=beam_property,
)

beam_II = geo.objects.Beam(
    identifier="beam_II",
    root_point=POINT_3,
    tip_point=POINT_1,
    orientation_vector=np.array([0.0, 1.0, 0.0]),
    ElementProperty=beam_property,
)

beam_III = geo.objects.Beam(
    identifier="beam_III",
    root_point=POINT_4,
    tip_point=POINT_1,
    orientation_vector=np.array([-1.0, 0.0, 0.0]),
    ElementProperty=beam_property,
)

struct_components = [beam_I, beam_II, beam_III]

# ==================================================================================================
# CONNECTIONS

I_to_II = struct.objects.Connection(beam_I, "TIP", beam_II, "TIP")

I_to_III = struct.objects.Connection(beam_I, "TIP", beam_III, "TIP")


struct_connections = [I_to_II, I_to_III]

# ==================================================================================================
# FEM GRID GENERATION

N_BEAM_ELEMENTS = 1

beam_I_struct_grid = beam_I.create_grid(n_elements=N_BEAM_ELEMENTS)
beam_II_struct_grid = beam_II.create_grid(n_elements=N_BEAM_ELEMENTS)
beam_III_struct_grid = beam_III.create_grid(n_elements=N_BEAM_ELEMENTS)


struct_grid = [
    beam_I_struct_grid,
    beam_II_struct_grid,
    beam_III_struct_grid,
]

# Numbering nodes

struct.fem.number_nodes(struct_components, struct_grid, struct_connections)


# ==================================================================================================
# FEM ELEMENTS GENERATION

beam_I_fem_elements = struct.fem.generate_beam_fem_elements(
    beam=beam_I, beam_nodes_list=beam_I_struct_grid, prop_choice="ROOT"
)
beam_II_fem_elements = struct.fem.generate_beam_fem_elements(
    beam=beam_II, beam_nodes_list=beam_II_struct_grid, prop_choice="ROOT"
)
beam_III_fem_elements = struct.fem.generate_beam_fem_elements(
    beam=beam_III, beam_nodes_list=beam_III_struct_grid, prop_choice="ROOT"
)

struct_elements = [
    beam_I_fem_elements,
    beam_II_fem_elements,
    beam_III_fem_elements,
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

beam_II_fixation = struct.objects.Constraint(
    application_node=beam_II_struct_grid[0],
    dof_constraints=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
)

beam_III_fixation = struct.objects.Constraint(
    application_node=beam_III_struct_grid[0],
    dof_constraints=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
)

struct_constraints = [beam_I_fixation, beam_II_fixation, beam_III_fixation]

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
