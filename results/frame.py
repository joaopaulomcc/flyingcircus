import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from context import src
from src import geometry as geo
from src import visualization as vis
from src import aerodynamics as aero
from src import loads
from src import structures as struct


POINT_1 = np.array([0.0, 0.0, 0.0])
POINT_2 = np.array([3.0, 0.0, 0.0])
POINT_3 = np.array([3.0, 3.0, 0.0])
POINT_4 = np.array([0.0, 0.0, 3.0])
POINT_5 = np.array([3.0, 0.0, 3.0])
POINT_6 = np.array([3.0, 3.0, 3.0])

material = struct.objects.Material(
    name="Aluminum 7075-T6",
    density=2810,
    elasticity_modulus=7.17e10,
    rigidity_modulus=2.69e10,
    poisson_ratio=0.33,
    yield_tensile_stress=5.03e08,
    ultimate_tensile_stress=5.72e08,
    yield_shear_stress=3.31e08,
    ultimate_shear_stress=0,
)

section = geo.objects.Section(
    identifier="rectangle",
    material=material,
    area=0.032,
    Iyy=0.0000017067,
    Izz=4.267e-7,
    J=0.0000011719,
    shear_center=0.5,
)

# ==================================================================================================
# BEAMS

beam_property = struct.objects.ElementProperty(section=section, material=material)

beam_I = geo.objects.Beam(
    identifier="beam_I",
    root_point=POINT_1,
    tip_point=POINT_4,
    orientation_vector=np.array([0.0, 1.0, 0.0]),
    ElementProperty=beam_property,
)

beam_II = geo.objects.Beam(
    identifier="beam_II",
    root_point=POINT_2,
    tip_point=POINT_5,
    orientation_vector=np.array([0.0, 1.0, 0.0]),
    ElementProperty=beam_property,
)

beam_III = geo.objects.Beam(
    identifier="beam_III",
    root_point=POINT_3,
    tip_point=POINT_6,
    orientation_vector=np.array([0.0, 1.0, 0.0]),
    ElementProperty=beam_property,
)

beam_IV = geo.objects.Beam(
    identifier="beam_IV",
    root_point=POINT_4,
    tip_point=POINT_5,
    orientation_vector=np.array([0.0, 1.0, 0.0]),
    ElementProperty=beam_property,
)

beam_V = geo.objects.Beam(
    identifier="beam_V",
    root_point=POINT_5,
    tip_point=POINT_6,
    orientation_vector=np.array([-1.0, 0.0, 0.0]),
    ElementProperty=beam_property,
)

beam_VI = geo.objects.Beam(
    identifier="beam_VI",
    root_point=POINT_1,
    tip_point=POINT_6,
    orientation_vector=np.array([-1.0, 1.0, 0.0]),
    ElementProperty=struct.objects.RigidConnection(),
)

struct_components = [beam_I, beam_II, beam_III, beam_IV, beam_V, beam_VI]

# ==================================================================================================
# CONNECTIONS

I_to_IV = struct.objects.Connection(beam_I, "TIP", beam_IV, "ROOT")

II_to_IV = struct.objects.Connection(beam_II, "TIP", beam_IV, "TIP")

II_to_V = struct.objects.Connection(beam_II, "TIP", beam_V, "ROOT")

III_to_V = struct.objects.Connection(beam_III, "TIP", beam_V, "TIP")

III_to_VI = struct.objects.Connection(beam_III, "TIP", beam_VI, "TIP")

I_to_VI = struct.objects.Connection(beam_I, "ROOT", beam_VI, "ROOT")

struct_connections = [I_to_IV, II_to_IV, II_to_V, III_to_V, III_to_VI, I_to_VI]

# ==================================================================================================
# FEM GRID GENERATION

N_BEAM_ELEMENTS = 5

beam_I_struct_grid = beam_I.create_grid(n_elements=N_BEAM_ELEMENTS)
beam_II_struct_grid = beam_II.create_grid(n_elements=N_BEAM_ELEMENTS)
beam_III_struct_grid = beam_III.create_grid(n_elements=N_BEAM_ELEMENTS)
beam_IV_struct_grid = beam_IV.create_grid(n_elements=N_BEAM_ELEMENTS)
beam_V_struct_grid = beam_V.create_grid(n_elements=N_BEAM_ELEMENTS)
beam_VI_struct_grid = beam_VI.create_grid(n_elements=N_BEAM_ELEMENTS)

struct_grid = [
    beam_I_struct_grid,
    beam_II_struct_grid,
    beam_III_struct_grid,
    beam_IV_struct_grid,
    beam_V_struct_grid,
    beam_VI_struct_grid,
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
beam_IV_fem_elements = struct.fem.generate_beam_fem_elements(
    beam=beam_IV, beam_nodes_list=beam_IV_struct_grid, prop_choice="ROOT"
)
beam_V_fem_elements = struct.fem.generate_beam_fem_elements(
    beam=beam_V, beam_nodes_list=beam_V_struct_grid, prop_choice="ROOT"
)
beam_VI_fem_elements = struct.fem.generate_beam_fem_elements(
    beam=beam_VI, beam_nodes_list=beam_VI_struct_grid, prop_choice="ROOT"
)

struct_elements = [
    beam_I_fem_elements,
    beam_II_fem_elements,
    beam_III_fem_elements,
    beam_IV_fem_elements,
    beam_V_fem_elements,
    beam_VI_fem_elements,
]

# ==================================================================================================
# LOADS GENERATION

beam_I_tip_load = struct.objects.Load(
    application_node=beam_I_struct_grid[-1],
    load=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1000.0]),
)

beam_II_tip_load = struct.objects.Load(
    application_node=beam_II_struct_grid[-1],
    load=np.array([1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
)

beam_III_tip_load = struct.objects.Load(
    application_node=beam_III_struct_grid[-1],
    load=np.array([0.0, 2000.0, 0.0, 0.0, 0.0, 0.0]),
)

struct_loads = [beam_I_tip_load, beam_II_tip_load, beam_III_tip_load]

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

deformed_struct_grid, struct_internal_loads, struct_strains, node_vector, deformations = struct.fem.structural_solver(
    struct_grid, struct_elements, struct_loads, struct_constraints
)

# deformed_grid, force_vector, deformations, node_vector = struct.fem.structural_solver(
#    struct_grid, struct_elements, struct_loads, struct_constraints
# )

# ==================================================================================================
# PLOT DEFORMED STRUCTURE

vis.plot_3D.plot_deformed_structure(
    struct_elements, node_vector, deformations, scale_factor=5
)

print("FINISHED")
