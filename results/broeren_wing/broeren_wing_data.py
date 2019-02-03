"""
====================================================================================================
Comparisson between results found in the literature and those obtained by using Flying Circus.
Definition of the geometrical data

Author: João Paulo Monteiro Cruvinel da Costa

Literature results:

NACA Technical Note No.1270 - EXPERIMENTAL AND CALCULATED CHARACTERISTICS OF SEVERALNACA 44-SERIES
WINGS WITH ASPECT RATIOS OF 8, 10, AND 12 AND TAPER RATIOS OF 2.5 AND 3.5

Authors: Robert H. Neely, Thomas V. Bollech, Gertrude C. Westrick, and Robert R. Graham

Langley Memorial Aeronautical Laboratory
Langley Field, Va.

Washington, May 1947
====================================================================================================
"""

# ==================================================================================================
# IMPORTS

import numpy as np

# Import code sub packages
from context import src
from src import geometry as geo
from src import structures as struct

# ==================================================================================================
# STRUCTURAL PROPERTIES

# In this case, since only a rigid structure simulation will be done, the material and section
# properties are throw away values that will not be used.

MATERIAL = struct.objects.Material(
    name="material",
    density=1,
    elasticity_modulus=1,
    rigidity_modulus=1,
    poisson_ratio=1,
    yield_tensile_stress=1,
    ultimate_tensile_stress=1,
    yield_shear_stress=1,
    ultimate_shear_stress=1,
)

# Wing section properties
WING_SECTION = geo.objects.Section(
    identifier="section",
    material=MATERIAL,
    area=1,
    Iyy=1,
    Izz=1,
    J=1,
    shear_center=0.5,
)

# ==================================================================================================
# BROEREN WING I: 2.5-08-4416

# Wing I surface

WING_I_ROOT_CHORD = 0.816428571
WING_I_TIP_CHORD = 0.326571429
SEMI_WING_I_LENGTH = 4.572 / 2
WING_I_SWEEP_ANGLE = 3.066485501
WING_I_DIHEDRAL = 0
WING_I_TIP_TORSION_ANGLE = -4.5


# Definition of the wing planform
left_wing_i_surface = geo.objects.Surface(
    identifier="left_wing",
    root_chord=WING_I_ROOT_CHORD,
    root_section=WING_SECTION,
    tip_chord=WING_I_TIP_CHORD,
    tip_section=WING_SECTION,
    length=SEMI_WING_I_LENGTH,
    leading_edge_sweep_angle_deg=WING_I_SWEEP_ANGLE,
    dihedral_angle_deg=WING_I_DIHEDRAL,
    tip_torsion_angle_deg=WING_I_TIP_TORSION_ANGLE,
    control_surface_hinge_position=None,
)

right_wing_i_surface = geo.objects.Surface(
    identifier="right_wing",
    root_chord=WING_I_ROOT_CHORD,
    root_section=WING_SECTION,
    tip_chord=WING_I_TIP_CHORD,
    tip_section=WING_SECTION,
    length=SEMI_WING_I_LENGTH,
    leading_edge_sweep_angle_deg=WING_I_SWEEP_ANGLE,
    dihedral_angle_deg=WING_I_DIHEDRAL,
    tip_torsion_angle_deg=WING_I_TIP_TORSION_ANGLE,
    control_surface_hinge_position=None,
)


# Creation of the wing macrosurface
# Since no information is given about the torsion center used in the application of the washout
# angle it is assumed that the torsion center is at the middle of the chord

broeren_wing_i_wing = geo.objects.MacroSurface(
    position=np.array([0, 0, 0]),
    incidence=0,
    surface_list=[left_wing_i_surface, right_wing_i_surface],
    symmetry_plane="XZ",
    torsion_center=0.5,
)

# Aircraft definition

broeren_wing_i = geo.objects.Aircraft(
    name="Broeren Wing I - 2.5-08-4416",
    macrosurfaces=[broeren_wing_i_wing],
    inertial_properties=geo.objects.MaterialPoint(),
    ref_area=2.612898,
    mean_aero_chord=0.606552,
)
