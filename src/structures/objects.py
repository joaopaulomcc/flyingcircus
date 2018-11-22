import numpy as np
import scipy as sc

from numpy import sin, cos, tan, pi
from . import functions as f
from .. import mathematics as m
from .. import geometry as geo

# ==================================================================================================
# Objects


class Node(object):
    """Node object, defines a node in the FEM mesh.

    Args:
        x (float): node x position in the 3D space.
        y (float): node y position in the 3D space.
        z (float): node z position in the 3D space.
        rx (float): node rotation around the x axis.
        ry (float): node rotation around the y axis.
        rz (float): node rotation around the z axis.

    Attributes:
        x (float): node x position in the 3D space.
        y (float): node y position in the 3D space.
        z (float): node z position in the 3D space.
        rx (float): node rotation around the x axis.
        ry (float): node rotation around the y axis.
        rz (float): node rotation around the z axis.
        xyz (np.array(dtype=float)): numpy array with x, y, z coordinates
        rxyz (np.array(dtype=float)): numpy array with rx, ry, rz rotations
    """

    def __init__(self, x, y, z, rx=0, ry=0, rz=0):

        self.x = x
        self.y = y
        self.z = z
        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.xyz = np.array([x, y, z])
        self.rxyz = np.array([rx, ry, ry])


# --------------------------------------------------------------------------------------------------


class Material:
    """Defines a material and it's properties

    Args:
        name (string) = material's name
        density (float) = density of the material [kg/m³]
        elasticity_modulus (float) = elastic modulus of the material [Pa]
        rigidity_modulus (float) = rigidity modulus of the material [Pa]
        poisson_ratio (float) = poisson's ratio of the material
        yield_tensile_stress (float) = yield stress of the material in tension [Pa]
        ultimate_tensile_stress (float) = ultimate stress of the material in tension [Pa]
        yield_shear_stress (float) = yield stress of the material in shear [Pa]
        ultimate_shear_stress (float) = ultimate stress of the material in shear [Pa]

    Attributes:
        name (string) = material's name
        density (float) = density of the material [kg/m³]
        elasticity_modulus (float) = elastic modulus of the material [Pa]
        rigidity_modulus (float) = rigidity modulus of the material [Pa]
        poisson_ratio (float) = poisson's ratio of the material
        yield_tensile_stress (float) = yield stress of the material in tension [Pa]
        ultimate_tensile_stress (float) = ultimate stress of the material in tension [Pa]
        yield_shear_stress (float) = yield stress of the material in shear [Pa]
        ultimate_shear_stress (float) = ultimate stress of the material in shear [Pa]
    """

    def __init__(
        self,
        name,
        density,
        elasticity_modulus,
        rigidity_modulus,
        poisson_ratio=0,
        yield_tensile_stress=0,
        ultimate_tensile_stress=0,
        yield_shear_stress=0,
        ultimate_shear_stress=0,
    ):

        self.name = name
        self.density = density
        self.elasticity_modulus = elasticity_modulus
        self.rigidity_modulus = rigidity_modulus
        self.poisson_ratio = poisson_ratio
        self.yield_tensile_stress = yield_tensile_stress
        self.ultimate_tensile_stress = ultimate_tensile_stress
        self.yield_shear_stress = yield_shear_stress
        self.ultimate_shear_stress = ultimate_shear_stress


# --------------------------------------------------------------------------------------------------


class Property(object):
    """ Property stores the properties of a beam finite element

    Args:
        section (object): A Section object from the Geometry module.
        material (object): A Material object

    Attributes:
        section (object): A Section object from the Geometry module.
        material (object): A Material object
        A (float): Area of the section
        J (float): Polar moment of inertia of the section
        Iyy (float): Moment of inertia of the section in the Y axis
        Izz (float): Moment of inertia of the section in the Z axis
        E (float): Elastic modulus of the material
        G (float): Shear modulus of the material
    """

    def __init__(self, section, material):

        self.section = section
        self.material = material
        self.A = section.area
        self.J = section.J
        self.Iyy = section.Iyy
        self.Izz = section.Izz
        self.E = material.elasticity_modulus
        self.G = material.rigidity_modulus


# --------------------------------------------------------------------------------------------------


class BeamElement(object):
    """ Defines a beam finite element, saves it's properties and calculates it's stiffness matrices

    Args:
        node_A_index (int): index of the first node of the element in the FEM grid
        node_B_index (int): index of the second node of the element in the FEM grid
        rotation (float): rotation of the element section in relation to it's own axis
        propertie (object): propertie object containing section and material characteristics

    Attributes:
        node_A_index (int): index of the first node of the element in the FEM grid
        node_B_index (int): index of the second node of the element in the FEM grid
        A (float): Area of the section
        J (float): Polar moment of inertia of the section
        Iyy (float): Moment of inertia of the section in the Y axis
        Izz (float): Moment of inertia of the section in the Z axis
        E (float): Elastic modulus of the material
        G (float): Shear modulus of the material
        correlation_vector (np.array): Vector with the indexes of the global DOF that correspond
                                       to the element local DOF
    """

    def __init__(self, node_A_index, node_B_index, rotation, element_property):

        self.node_A_index = int(node_A_index)
        self.node_B_index = int(node_B_index)
        self.rotation = rotation
        self.A = element_property.A
        self.J = element_property.J
        self.Iyy = element_property.Iyy
        self.Izz = element_property.Izz
        self.E = element_property.E
        self.G = element_property.G

        A_index = 6 * self.node_A_index
        B_index = 6 * self.node_B_index
        self.correlation_vector = np.array(
            [
                A_index,
                A_index + 1,
                A_index + 2,
                A_index + 3,
                A_index + 4,
                A_index + 5,
                B_index,
                B_index + 1,
                B_index + 2,
                B_index + 3,
                B_index + 4,
                B_index + 5,
            ]
        )

    def calc_rotation_matrix(self, grid):
        """ Calculates the rotation matrix of the element.

        Args:
            grid (list): list with all the node elements of the FEM mesh

        Returns:
            rotation_matrix (np.array): transformation matrix from the local coordinate system
                                        to the global coordinate system.
        """

        # Global Coordinate System
        x_global = np.array([1.0, 0.0, 0.0])
        y_global = np.array([0.0, 1.0, 0.0])
        z_global = np.array([0.0, 0.0, 1.0])
        origin = np.zeros(3)

        # Calculates the element local coordinate system before rotation
        point_A = grid[self.point_A_index]
        point_B = grid[self.point_B_index]

        x_local = m.normalize(point_B - point_A)
        y_local = np.array([0, -1, 0])
        z_local = m.cross(x_local, y_local)

        # Apply rotation to the local coordinate system
        y_local = geo.functions.rotate_point(y_local, x_local, origin, self.rotation)
        z_local = geo.functions.rotate_point(z_local, x_local, origin, self.rotation)

        # Calculate rotation matrix
        zero = np.zeros((3, 3))
        r = np.zeros((3, 3))

        for i, local_axis in [x_local, y_local, z_local]:
            for j, global_axis in [x_global, y_global, z_global]:
                r[i][j] = geo.functions.cos_between(local_axis, global_axis)

        rotation_matrix = np.block([[r, zero, zero, zero],
                                    [zero, r, zero, zero],
                                    [zero, zero, r, zero],
                                    [zero, zero, zero, r]])

        return rotation_matrix

    def calc_K_local(self, grid):
        """ Calculates the local stiffness matrix of the element.

        Args:
            grid (list): list with all the node elements of the FEM mesh

        Returns:
            K_local (np.array(dtype=float)): element local stiffness matrix
        """

        # Calculate element length
        point_A = grid[self.point_A_index].xyz
        point_B = grid[self.point_B_index].xyz

        N = point_B - point_A
        L = m.norm(N)

        # Calculate Local Stiffness Matrix
        K_local = f.beam_3D_stiff(
            self.E, self.A, L, self.G, self.J, self.Iy, self.Iz
        )

        return K_local

    def calc_K_global(self, grid):
        """ Apply the rotation matrix to the local stiffness matrix and calculate the element
            global stiffness matrix.

        Args:
            grid (list): list with all the node elements of the FEM mesh

        Returns:
            K_global (np.array(dtype=float)): element global stiffness matrix
        """

        rotation_matrix = self.calc_rotation_matrix(grid)
        K_local = self.calc_K_local(grid)

        # Global Stiffness Matrix
        K_global = rotation_matrix.transpose() @ (K_local @ rotation_matrix)

        return K_global


# --------------------------------------------------------------------------------------------------


class Structure:
    def __init__(self, points, beams):

        self.points = points
        self.beams = beams


# --------------------------------------------------------------------------------------------------


class Beam:
    def __init__(
        self,
        structure_points,
        point_A_index,
        point_B_index,
        section,
        material,
        n_elements,
    ):

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


class Section:
    def __init__(self, area, rotation, m_inertia_y, m_inertia_z, polar_moment):

        self.area = area
        self.rotation = rotation
        self.m_inertia_y = m_inertia_y
        self.m_inertia_z = m_inertia_z
        self.polar_moment = polar_moment


# --------------------------------------------------------------------------------------------------


class Load:
    def __init__(self, application_point_index, components):

        self.application_point_index = application_point_index
        self.components = components


# --------------------------------------------------------------------------------------------------


class Constraint:
    def __init__(self, application_point_index, dof_constraints):

        self.application_point_index = application_point_index
        self.dof_constraints = dof_constraints
