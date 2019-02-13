"""
====================================================================================================
Flight Dynamics of Flexible Aircraft Using General Body Axes: A Theoretical and Computational Study

Author: Antônio Bernardo Guimarães Neto

====================================================================================================
"""

# ==================================================================================================
# IMPORTS

import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

# Import code sub packages
from context import src
from src import geometry as geo
from src import structures as struct
from src import visualization as vis

# ==================================================================================================
# MATERIAL DEFINITION

# Since NETO only provides the EI, EIyy, EIzz and GJ values all materials properties are set to 1
# so the beam elements have the correct stiffness

MATERIAL = struct.objects.Material(
    name="material",
    density=0.75,
    elasticity_modulus=1,
    rigidity_modulus=1,
    poisson_ratio=1,
    yield_tensile_stress=1,
    ultimate_tensile_stress=1,
    yield_shear_stress=1,
    ultimate_shear_stress=1,
)

# ==================================================================================================
# WING DEFINITION

# --------------------------------------------------------------------------------------------------
# Data

# Region I Data
REGION_I_ROOT_CHORD = 6.03
REGION_I_TIP_CHORD = 6.03
REGION_I_LENGTH = 1.85
REGION_I_SWEEP_ANGLE = 0
REGION_I_DIHEDRAL_ANGLE = 0
REGION_I_TIP_TORSION_ANGLE = 0

REGION_I_ROOT_AREA = 3.7e9
REGION_I_ROOT_IYY = 2.97e8
REGION_I_ROOT_IZZ = 3.7e9
REGION_I_ROOT_J = 1.54e8
REGION_I_ROOT_SHEAR_CENTER = 0.33

REGION_I_TIP_AREA = 3.7e9
REGION_I_TIP_IYY = 2.97e8
REGION_I_TIP_IZZ = 3.7e9
REGION_I_TIP_J = 1.54e8
REGION_I_TIP_SHEAR_CENTER = 0.33

# Region II Data
REGION_II_ROOT_CHORD = 6.03
REGION_II_TIP_CHORD = 3.836
REGION_II_LENGTH = 3.829
REGION_II_SWEEP_ANGLE = 30
REGION_II_DIHEDRAL_ANGLE = 7
REGION_II_TIP_TORSION_ANGLE = 1.6 - 3

REGION_II_ROOT_AREA = 7762790697.67442
REGION_II_ROOT_IYY = 282440258.57016945
REGION_II_ROOT_IZZ = 10294853949.753887
REGION_II_ROOT_J = 851851851.851852
REGION_II_ROOT_SHEAR_CENTER = 0.33

REGION_II_TIP_AREA = 4246511627.90697
REGION_II_TIP_IYY = 124881092.19769526
REGION_II_TIP_IZZ = 3977362051.241332
REGION_II_TIP_J = 130099218.695983
REGION_II_TIP_SHEAR_CENTER = 0.4

# Region III Data
REGION_III_ROOT_CHORD = 3.836
REGION_III_TIP_CHORD = 1.697
REGION_III_LENGTH = 10.760
REGION_III_SWEEP_ANGLE = 30
REGION_III_DIHEDRAL_ANGLE = 4.5
REGION_III_TIP_TORSION_ANGLE = -2 - 1.6

REGION_III_ROOT_AREA = 3454932013.51788
REGION_III_ROOT_IYY = 91075661.89229631
REGION_III_ROOT_IZZ = 3097502153.3193293
REGION_III_ROOT_J = 96200924.9897938
REGION_III_ROOT_SHEAR_CENTER = 0.4

REGION_III_TIP_AREA = 346595556.332642
REGION_III_TIP_IYY = 2873238.3121461496
REGION_III_TIP_IZZ = 58250277.306790605
REGION_III_TIP_J = 1642021.27805322
REGION_III_TIP_SHEAR_CENTER = 0.505

# --------------------------------------------------------------------------------------------------
# Section Objects Creation

region_I_root_section = geo.objects.Section(
    identifier="region_I_root_section",
    material=MATERIAL,
    area=REGION_I_ROOT_AREA,
    Iyy=REGION_I_ROOT_IYY,
    Izz=REGION_I_ROOT_IZZ,
    J=REGION_I_ROOT_J,
    shear_center=REGION_I_ROOT_SHEAR_CENTER,
)

region_I_tip_section = geo.objects.Section(
    identifier="region_I_root_section",
    material=MATERIAL,
    area=REGION_I_TIP_AREA,
    Iyy=REGION_I_TIP_IYY,
    Izz=REGION_I_TIP_IZZ,
    J=REGION_I_TIP_J,
    shear_center=REGION_I_TIP_SHEAR_CENTER,
)

region_II_root_section = geo.objects.Section(
    identifier="region_II_root_section",
    material=MATERIAL,
    area=REGION_II_ROOT_AREA,
    Iyy=REGION_II_ROOT_IYY,
    Izz=REGION_II_ROOT_IZZ,
    J=REGION_II_ROOT_J,
    shear_center=REGION_II_ROOT_SHEAR_CENTER,
)

region_II_tip_section = geo.objects.Section(
    identifier="region_II_root_section",
    material=MATERIAL,
    area=REGION_II_TIP_AREA,
    Iyy=REGION_II_TIP_IYY,
    Izz=REGION_II_TIP_IZZ,
    J=REGION_II_TIP_J,
    shear_center=REGION_II_TIP_SHEAR_CENTER,
)

region_III_root_section = geo.objects.Section(
    identifier="region_III_root_section",
    material=MATERIAL,
    area=REGION_III_ROOT_AREA,
    Iyy=REGION_III_ROOT_IYY,
    Izz=REGION_III_ROOT_IZZ,
    J=REGION_III_ROOT_J,
    shear_center=REGION_III_ROOT_SHEAR_CENTER,
)

region_III_tip_section = geo.objects.Section(
    identifier="region_III_root_section",
    material=MATERIAL,
    area=REGION_III_TIP_AREA,
    Iyy=REGION_III_TIP_IYY,
    Izz=REGION_III_TIP_IZZ,
    J=REGION_III_TIP_J,
    shear_center=REGION_III_TIP_SHEAR_CENTER,
)

# --------------------------------------------------------------------------------------------------
# Surface Objects Creation

left_region_i_surface = geo.objects.Surface(
    identifier="left_region_i",
    root_chord=REGION_I_ROOT_CHORD,
    root_section=region_I_root_section,
    tip_chord=REGION_I_TIP_CHORD,
    tip_section=region_I_tip_section,
    length=REGION_I_LENGTH,
    leading_edge_sweep_angle_deg=REGION_I_SWEEP_ANGLE,
    dihedral_angle_deg=REGION_I_DIHEDRAL_ANGLE,
    tip_torsion_angle_deg=REGION_I_TIP_TORSION_ANGLE,
    control_surface_hinge_position=None,
)

right_region_i_surface = geo.objects.Surface(
    identifier="right_region_i",
    root_chord=REGION_I_ROOT_CHORD,
    root_section=region_I_root_section,
    tip_chord=REGION_I_TIP_CHORD,
    tip_section=region_I_tip_section,
    length=REGION_I_LENGTH,
    leading_edge_sweep_angle_deg=REGION_I_SWEEP_ANGLE,
    dihedral_angle_deg=REGION_I_DIHEDRAL_ANGLE,
    tip_torsion_angle_deg=REGION_I_TIP_TORSION_ANGLE,
    control_surface_hinge_position=None,
)

left_region_ii_surface = geo.objects.Surface(
    identifier="left_region_ii",
    root_chord=REGION_II_ROOT_CHORD,
    root_section=region_II_root_section,
    tip_chord=REGION_II_TIP_CHORD,
    tip_section=region_II_tip_section,
    length=REGION_II_LENGTH,
    leading_edge_sweep_angle_deg=REGION_II_SWEEP_ANGLE,
    dihedral_angle_deg=REGION_II_DIHEDRAL_ANGLE,
    tip_torsion_angle_deg=REGION_II_TIP_TORSION_ANGLE,
    control_surface_hinge_position=None,
)

right_region_ii_surface = geo.objects.Surface(
    identifier="right_region_ii",
    root_chord=REGION_II_ROOT_CHORD,
    root_section=region_II_root_section,
    tip_chord=REGION_II_TIP_CHORD,
    tip_section=region_II_tip_section,
    length=REGION_II_LENGTH,
    leading_edge_sweep_angle_deg=REGION_II_SWEEP_ANGLE,
    dihedral_angle_deg=REGION_II_DIHEDRAL_ANGLE,
    tip_torsion_angle_deg=REGION_II_TIP_TORSION_ANGLE,
    control_surface_hinge_position=None,
)

left_region_iii_surface = geo.objects.Surface(
    identifier="left_region_iii",
    root_chord=REGION_III_ROOT_CHORD,
    root_section=region_III_root_section,
    tip_chord=REGION_III_TIP_CHORD,
    tip_section=region_III_tip_section,
    length=REGION_III_LENGTH,
    leading_edge_sweep_angle_deg=REGION_III_SWEEP_ANGLE,
    dihedral_angle_deg=REGION_III_DIHEDRAL_ANGLE,
    tip_torsion_angle_deg=REGION_III_TIP_TORSION_ANGLE,
    control_surface_hinge_position=None,
)

right_region_iii_surface = geo.objects.Surface(
    identifier="right_region_iii",
    root_chord=REGION_III_ROOT_CHORD,
    root_section=region_III_root_section,
    tip_chord=REGION_III_TIP_CHORD,
    tip_section=region_III_tip_section,
    length=REGION_III_LENGTH,
    leading_edge_sweep_angle_deg=REGION_III_SWEEP_ANGLE,
    dihedral_angle_deg=REGION_III_DIHEDRAL_ANGLE,
    tip_torsion_angle_deg=REGION_III_TIP_TORSION_ANGLE,
    control_surface_hinge_position=None,
)

# --------------------------------------------------------------------------------------------------
# Macrosurface Object Creation

WING_POSITION = np.array([14.545, 0, -0.880])
WING_INCIDENCE = 3
WING_SURFACE_LIST = [
    left_region_iii_surface,
    left_region_ii_surface,
    left_region_i_surface,
    right_region_i_surface,
    right_region_ii_surface,
    right_region_iii_surface,
]
WING_SIMMETRY_PLANE = "XZ"
WING_TORSION_CENTER = 0.25

# Creation of the wing macrosurface
wing_macrosurface = geo.objects.MacroSurface(
    position=WING_POSITION,
    incidence=WING_INCIDENCE,
    surface_list=WING_SURFACE_LIST,
    symmetry_plane=WING_SIMMETRY_PLANE,
    torsion_center=WING_TORSION_CENTER,
)

# ==================================================================================================
# HORIZONTAL TAIL DEFINITION

# --------------------------------------------------------------------------------------------------
# Data

# Horizontal Tail Data
HTAIL_ROOT_CHORD = 0.933 / 0.28
HTAIL_TIP_CHORD = 0.933
HTAIL_SEMI_LENGTH = 5.8228
HTAIL_SWEEP_ANGLE = 33
HTAIL_DIHEDRAL_ANGLE = 7
HTAIL_TIP_TORSION_ANGLE = 0

HTAIL_ROOT_AREA = 857407553.979751
HTAIL_ROOT_IYY = 21517990.1270589
HTAIL_ROOT_IZZ = 205632292.951332
HTAIL_ROOT_J = 19633425.3508166
HTAIL_ROOT_SHEAR_CENTER = 0.42

HTAIL_TIP_AREA = 242598292.353536
HTAIL_TIP_IYY = 802982.238884449
HTAIL_TIP_IZZ = 10321170.14946
HTAIL_TIP_J = 1009661.83574879
HTAIL_TIP_SHEAR_CENTER = 0.42

# --------------------------------------------------------------------------------------------------
# Section Objects Creation

htail_root_section = geo.objects.Section(
    identifier="htail_root_section",
    material=MATERIAL,
    area=HTAIL_ROOT_AREA,
    Iyy=HTAIL_ROOT_IYY,
    Izz=HTAIL_ROOT_IZZ,
    J=HTAIL_ROOT_J,
    shear_center=HTAIL_ROOT_SHEAR_CENTER,
)

htail_tip_section = geo.objects.Section(
    identifier="htail_tip_section",
    material=MATERIAL,
    area=HTAIL_TIP_AREA,
    Iyy=HTAIL_TIP_IYY,
    Izz=HTAIL_TIP_IZZ,
    J=HTAIL_TIP_J,
    shear_center=HTAIL_TIP_SHEAR_CENTER,
)

# --------------------------------------------------------------------------------------------------
# Surface Objects Creation

left_htail_surface = geo.objects.Surface(
    identifier="left_htail_surface",
    root_chord=HTAIL_ROOT_CHORD,
    root_section=htail_root_section,
    tip_chord=HTAIL_TIP_CHORD,
    tip_section=htail_tip_section,
    length=HTAIL_SEMI_LENGTH,
    leading_edge_sweep_angle_deg=HTAIL_SWEEP_ANGLE,
    dihedral_angle_deg=HTAIL_DIHEDRAL_ANGLE,
    tip_torsion_angle_deg=HTAIL_TIP_TORSION_ANGLE,
    control_surface_hinge_position=None,
)

right_htail_surface = geo.objects.Surface(
    identifier="right_htail_surface",
    root_chord=HTAIL_ROOT_CHORD,
    root_section=htail_root_section,
    tip_chord=HTAIL_TIP_CHORD,
    tip_section=htail_tip_section,
    length=HTAIL_SEMI_LENGTH,
    leading_edge_sweep_angle_deg=HTAIL_SWEEP_ANGLE,
    dihedral_angle_deg=HTAIL_DIHEDRAL_ANGLE,
    tip_torsion_angle_deg=HTAIL_TIP_TORSION_ANGLE,
    control_surface_hinge_position=None,
)

# --------------------------------------------------------------------------------------------------
# Macrosurface Object Creation

HTAIL_POSITION = np.array([34.761, 0, 1.041])
HTAIL_INCIDENCE = 0
HTAIL_SURFACE_LIST = [left_htail_surface, right_htail_surface]
HTAIL_SIMMETRY_PLANE = "XZ"
HTAIL_TORSION_CENTER = 0.25

# Creation of the wing macrosurface
htail_macrosurface = geo.objects.MacroSurface(
    position=HTAIL_POSITION,
    incidence=HTAIL_INCIDENCE,
    surface_list=HTAIL_SURFACE_LIST,
    symmetry_plane=HTAIL_SIMMETRY_PLANE,
    torsion_center=HTAIL_TORSION_CENTER,
)

# ==================================================================================================
# VERTICAL TAIL DEFINITION

# --------------------------------------------------------------------------------------------------
# Data

# Vertical Tail Data
VTAIL_ROOT_CHORD = 5.221
VTAIL_TIP_CHORD = 1.723
VTAIL_LENGTH = 5.902
VTAIL_SWEEP_ANGLE = 39
VTAIL_DIHEDRAL_ANGLE = 90
VTAIL_TIP_TORSION_ANGLE = 0

VTAIL_ROOT_AREA = 1196944308.54533
VTAIL_ROOT_IYY = 83325410.2452924
VTAIL_ROOT_IZZ = 531183112.420225
VTAIL_ROOT_J = 71869055.0828481
VTAIL_ROOT_SHEAR_CENTER = 0.37

VTAIL_TIP_AREA = 288168623.613829
VTAIL_TIP_IYY = 3811086.96752677
VTAIL_TIP_IZZ = 26818851.2518416
VTAIL_TIP_J = 3544827.58620689
VTAIL_TIP_SHEAR_CENTER = 0.37

# --------------------------------------------------------------------------------------------------
# Section Objects Creation

vtail_root_section = geo.objects.Section(
    identifier="vtail_root_section",
    material=MATERIAL,
    area=VTAIL_ROOT_AREA,
    Iyy=VTAIL_ROOT_IYY,
    Izz=VTAIL_ROOT_IZZ,
    J=VTAIL_ROOT_J,
    shear_center=VTAIL_ROOT_SHEAR_CENTER,
)

vtail_tip_section = geo.objects.Section(
    identifier="vtail_tip_section",
    material=MATERIAL,
    area=VTAIL_TIP_AREA,
    Iyy=VTAIL_TIP_IYY,
    Izz=VTAIL_TIP_IZZ,
    J=VTAIL_TIP_J,
    shear_center=VTAIL_TIP_SHEAR_CENTER,
)

# --------------------------------------------------------------------------------------------------
# Surface Objects Creation

vtail_surface = geo.objects.Surface(
    identifier="vtail_surface",
    root_chord=VTAIL_ROOT_CHORD,
    root_section=htail_root_section,
    tip_chord=VTAIL_TIP_CHORD,
    tip_section=htail_tip_section,
    length=VTAIL_LENGTH,
    leading_edge_sweep_angle_deg=VTAIL_SWEEP_ANGLE,
    dihedral_angle_deg=VTAIL_DIHEDRAL_ANGLE,
    tip_torsion_angle_deg=VTAIL_TIP_TORSION_ANGLE,
    control_surface_hinge_position=None,
)

# --------------------------------------------------------------------------------------------------
# Macrosurface Object Creation

VTAIL_POSITION = np.array([34.761, 0, 1.562])
VTAIL_INCIDENCE = 0
VTAIL_SURFACE_LIST = [vtail_surface]
VTAIL_SIMMETRY_PLANE = "XZ"
VTAIL_TORSION_CENTER = 0.25

# Creation of the wing macrosurface
vtail_macrosurface = geo.objects.MacroSurface(
    position=VTAIL_POSITION,
    incidence=VTAIL_INCIDENCE,
    surface_list=VTAIL_SURFACE_LIST,
    symmetry_plane=VTAIL_SIMMETRY_PLANE,
    torsion_center=VTAIL_TORSION_CENTER,
)

## --------------------------------------------------------------------------------------------------
## FUSELAGE AND TAIL BOOM DEFINITION
#
#POINT_1 = np.array([0.5, 0.0, 0.0])
#POINT_2 = np.array([0.93, 0.0, 0.0])
#POINT_3 = np.array([11.0, 0.0, 0.0])
#
#beam_property = struct.objects.ElementProperty(section=SECTION, material=MATERIAL)
#
#fuselage = geo.objects.Beam(
#    identifier="fuselage",
#    root_point=POINT_1,
#    tip_point=POINT_2,
#    orientation_vector=np.array([0.0, 1.0, 0.0]),
#    ElementProperty=beam_property,
#)
#
#tail_boom = geo.objects.Beam(
#    identifier="tail_boom",
#    root_point=POINT_2,
#    tip_point=POINT_3,
#    orientation_vector=np.array([0.0, 1.0, 0.0]),
#    ElementProperty=beam_property,
#)
#
#aircraft_beams = [fuselage, tail_boom]
#
# --------------------------------------------------------------------------------------------------
# AIRCRAFT CG DEFINITION

AIRCRAFT_MASS = 4000
CG_POSITION = np.array([16.0, 0, 0])
IXX = 1
IYY = 1
IZZ = 1
IXY = 1
IXZ = 1
IYZ = 1

aircraft_cg = geo.objects.MaterialPoint(
    identifier="aircraft_cg",
    orientation_quaternion=Quaternion(),
    mass=AIRCRAFT_MASS,
    position=CG_POSITION,
    Ixx=IXX,
    Iyy=IYY,
    Izz=IZZ,
    Ixy=IXY,
    Ixz=IXZ,
    Iyz=IYZ,
)

# --------------------------------------------------------------------------------------------------
## AIRCRAFT STRUCTURE CONNECTIONS
#
#wing_to_fuselage = struct.objects.Connection(
#    left_wing_surface, "ROOT", fuselage, "ROOT"
#)
#
#fuselage_to_cg = struct.objects.Connection(fuselage, "TIP", aircraft_cg, "ROOT")
#
#fuselage_to_tail_boom = struct.objects.Connection(fuselage, "TIP", tail_boom, "ROOT")
#
#tail_boom_to_tail = struct.objects.Connection(
#    tail_boom, "TIP", left_elevator_surface, "ROOT"
#)
#
#aircraft_struct_connections = [
#    wing_to_fuselage,
#    fuselage_to_tail_boom,
#    tail_boom_to_tail,
#]
#
# --------------------------------------------------------------------------------------------------

# Aircraft definition

AIRCRAFT_NAME = "GNBA aircraft"
AIRCRAFT_MACROSURFACES = [wing_macrosurface, htail_macrosurface, vtail_macrosurface]
AIRCRAFT_BEAMS = []
AIRCRAFT_INERTIAL_PROPERTIES = aircraft_cg
AIRCRAFT_STRUCT_CONNECTIONS = []
AIRCRAFT_REF_AREA = 116
AIRCRAFT_REF_MAC = 3.862

gnba_aircraft = geo.objects.Aircraft(
    name=AIRCRAFT_NAME,
    macrosurfaces=AIRCRAFT_MACROSURFACES,
    beams=AIRCRAFT_BEAMS,
    inertial_properties=AIRCRAFT_INERTIAL_PROPERTIES,
    connections=AIRCRAFT_STRUCT_CONNECTIONS,
    ref_area=AIRCRAFT_REF_AREA,
    mean_aero_chord=AIRCRAFT_REF_MAC,
)

# Draw Aircraft
aircraft_ax, aircraft_fig = vis.plot_3D2.generate_aircraft_plot(
    gnba_aircraft, title=AIRCRAFT_NAME
)

