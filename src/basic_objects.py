"""
basic_objects.py

Collection of basic objects both aerodynamic and structural
Except where explicitly stated all units are S.I.

Author: João Paulo Monteiro Cruvinel da Costa
email: joaopaulomcc@gmail.com / joao.cruvinel@embraer.com.br
github: joaopaulomcc
"""
# ==================================================================================================
# IMPORTS
import numpy as np
import scipy as sc

from numpy import sin, cos, tan, pi
from numba import jit, jitclass

from . import basic_elements
from . import geometry

from .fast_operations import dot, cross, norm, normalize
# ==================================================================================================


class Wing(object):
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

# --------------------------------------------------------------------------------------------------


class Panel(object):
    """Panel object"""

    def __init__(self, xx, yy, zz, infinity):
        """Args:
            xx [[float]] = grid with panel points x coordinates
            yy [[float]] = grid with panel points x coordinates
            zz [[float]] = grid with panel points x coordinates
        """
        self.xx = xx
        self.yy = yy
        self.zz = zz
        self.infinity = infinity
        self.A = np.array([xx[1][0], yy[1][0], zz[1][0]])
        self.B = np.array([xx[0][0], yy[0][0], zz[0][0]])
        self.C = np.array([xx[0][1], yy[0][1], zz[0][1]])
        self.D = np.array([xx[1][1], yy[1][1], zz[1][1]])
        self.AC = self.C - self.A
        self.BD = self.D - self.B

        self.l_chord = self.A - self.B
        self.l_chord_1_4 = self.B + 0.25 * self.l_chord
        self.l_chord_3_4 = self.B + 0.75 * self.l_chord

        self.r_chord = self.D - self.C
        self.r_chord_1_4 = self.C + 0.25 * self.r_chord
        self.r_chord_3_4 = self.C + 0.75 * self.r_chord

        self.l_edge = self.C - self.B
        self.l_edge_1_2 = self.B + 0.5 * self.l_edge

        self.t_edge = self.D - self.A
        self.t_edge_1_2 = self.A + 0.5 * self.t_edge

        self.col_point = 0.75 * (self.t_edge_1_2 - self.l_edge_1_2) + self.l_edge_1_2

        self.span = dot(self.l_edge, np.array([0, 1, 0]))
        self.n = normalize(cross(self.BD, self.AC))
        self.area = dot(self.n, cross(self.BD, self.AC)) / 2

        hs_A = np.array([self.l_chord_1_4[0] + infinity, self.l_chord_1_4[1], self.l_chord_1_4[2]])
        hs_B = np.array([self.l_chord_1_4[0], self.l_chord_1_4[1], self.l_chord_1_4[2]])
        hs_C = np.array([self.r_chord_1_4[0], self.r_chord_1_4[1], self.r_chord_1_4[2]])
        hs_D = np.array([self.r_chord_1_4[0] + infinity, self.r_chord_1_4[1], self.r_chord_1_4[2]])
        hs_A = hs_A[np.newaxis]
        hs_B = hs_B[np.newaxis]
        hs_C = hs_C[np.newaxis]
        hs_D = hs_D[np.newaxis]

        self.horse_shoe_vortex = np.concatenate([hs_A, hs_B, hs_C, hs_D]).transpose()

    def hs_induced_velocity(self, target_point, circulation):

        induced_velocity, wake_induced_velocity = basic_elements.vortex_horseshoe(self.horse_shoe_vortex,
                                                                                  target_point,
                                                                                  circulation)

        return induced_velocity, wake_induced_velocity

# --------------------------------------------------------------------------------------------------


class BeamElement(object):

    def __init__(self, point_A_index, point_B_index, rotation, E, A, G, J, Iy, Iz):

        self.point_A_index = point_A_index
        self.point_B_index = point_B_index
        self.rotation = rotation
        self.E = E
        self.A = A
        self.G = G
        self.J = J
        self.Iy = Iy
        self.Iz = Iz

    def calc_rotation_matrix(self, grid):
        
        point_A = grid[self.point_A_index]
        point_B = grid[self.point_B_index]

        N = point_B - point_A
        Nxy = np.array([N[0], N[1], 0])

        X = np.array([1., 0., 0.])
        Y = np.array([0., 1., 0.])
        Z = np.array([0., 0., 1.])
        origin = np.zeros(3)

        global_coord_system = np.eye(3)
        local_coord_sys = np.eye(3)

        elev = pi / 2 - geometry.angle_between(N, Z)

        # Solution for the case in witch the vector is parallel to the Z axis
        if norm(Nxy) == 0:
            azim = 0
        else:
            azim = geometry.angle_between(Nxy, X)

        # Rotation around X
        local_coord_sys = geometry.rotate_point(local_coord_sys, X, origin, self.rotation)

        # Rotation around Y
        local_coord_sys = geometry.rotate_point(local_coord_sys, Y, origin, elev)

        # Rotation around Z
        local_coord_sys = geometry.rotate_point(local_coord_sys, Z, origin, azim)

        # Local coordinate system
        #x = local_coord_sys[:, 0]
        #y = local_coord_sys[:, 1]
        #z = local_coord_sys[:, 2]

        # Rotation matrix
        rotation_matrix = basic_elements.beam_3D_rot(global_coord_system, local_coord_sys)

        return rotation_matrix


    def calc_K_local(self, grid):

        point_A = grid[self.point_A_index]
        point_B = grid[self.point_B_index]

        N = point_B - point_A
        L = norm(N)

        # Local Stiffness Matrix
        K_local = basic_elements.beam_3D_stiff(self.E, self.A, L, self.G, self.J, self.Iy, self.Iz)

        return K_local

    
    def calc_K_global(self, grid):

        rotation_matrix = self.calc_rotation_matrix(grid)
        K_local = self.calc_K_local(grid)

        # Global Stiffness Matrix
        K_global = rotation_matrix.transpose() @ (K_local @ rotation_matrix)

        return K_global

# --------------------------------------------------------------------------------------------------


class Structure():

    def __init__(self, points, beams, loads, constraints):

        self.points = points
        self.beams = beams
        self.loads = loads
        self.constraints = constraints

# --------------------------------------------------------------------------------------------------


class Beam():

    def __init__(self, structure_points, point_A_index, point_B_index, section, material, n_elements):

        self.structure_points = structure_points
        self.point_A_index = point_A_index
        self.point_B_index = point_B_index
        self.point_A = structure_points[point_A_index]
        self.point_B = structure_points[point_B_index]
        self.section = section
        self.material = material
        self.n_elements = n_elements
        self.vector = self.point_B - self.point_A
        self.L = norm(self.vector)

    def mesh(self, n_elements):
        
        delta = self.vector / n_elements

        mesh_points = []

        for i in range(n_elements + 1):

            mesh_points.append(self.point_A + i * delta)

        mesh_points = np.array(mesh_points)
        return mesh_points


# --------------------------------------------------------------------------------------------------


class Material():

    def __init__(self, name, young_modulus, shear_modulus, poisson_ratio, density, yield_strength, ultimate_strength):

        self.name = name
        self.density = density
        self.young_modulus = young_modulus
        self.shear_modulus = shear_modulus
        self.poisson_ratio = poisson_ratio
        self.yield_strength = yield_strength
        self.ultimate_strength = ultimate_strength


# --------------------------------------------------------------------------------------------------


class Section():

    def __init__(self, area, rotation, m_inertia_y, m_inertia_z, polar_moment):

        self.area = area
        self.rotation = rotation
        self.m_inertia_y = m_inertia_y
        self.m_inertia_z = m_inertia_z
        self.polar_moment = polar_moment


# --------------------------------------------------------------------------------------------------


class Load():

    def __init__(self, application_point_index, components):

        self.application_point_index = application_point_index
        self.components = components

# --------------------------------------------------------------------------------------------------


class Constraint():

    def __init__(self, application_point_index, dof_constraints):

        self.application_point_index = application_point_index
        self.dof_constraints = dof_constraints
