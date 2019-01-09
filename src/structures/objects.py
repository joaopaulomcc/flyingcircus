"""
DOCSTRING
"""

import numpy as np
import scipy as sc

from numpy import sin, cos, tan, pi
from pyquaternion import Quaternion

from . import functions as f
from .. import mathematics as m
from .. import geometry as geo

# ==================================================================================================
# Objects


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


# ==================================================================================================


class ElementProperty(object):
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


# ==================================================================================================


class RigidConnection(object):
    """ Property stores the properties of a rigid connection element

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

    def __init__(self):

        self.section = "rigid_connection"
        self.material = "rigid_connection"
        self.A = 1
        self.J = 1
        self.Iyy = 1
        self.Izz = 1
        self.E = 1e100
        self.G = 1e100


# ==================================================================================================


class EulerBeamElement(object):
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

    def __init__(self, node_A, node_B, A, Iyy, Izz, J, E, G, prop_choice="MIDDLE"):

        self.node_A = node_A
        self.node_B = node_B
        self.A = A
        self.Iyy = Iyy
        self.Izz = Izz
        self.J = J
        self.E = E
        self.G = G
        self.L = m.norm(node_B.xyz - node_A.xyz)
        self.prop_choice = prop_choice

        A_index = 6 * self.node_A.number
        B_index = 6 * self.node_B.number
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

    def calc_rotation_matrix(self):
        """ Calculates the rotation matrix of the element.

        Args:
            grid (list): list with all the node elements of the FEM mesh

        Returns:
            rotation_matrix (np.array): transformation matrix from the local coordinate system
                                        to the global coordinate system.
        """

        if self.prop_choice == "ROOT":

            orientation_node = self.node_A

        elif self.prop_choice == "TIP":

            orientation_node = self.node_B

        elif self.prop_choice == "MIDDLE":

            # Interpolate root and tip nodes to find a node in the middle
            nodes_prop = geo.functions.interpolate_nodes(self.node_A, self.node_B, 3)
            middle_node_prop = nodes_prop[1]
            orientation_node = geo.objects.Node(
                middle_node_prop[0], middle_node_prop[1]
            )

        # Global Coordinate System
        x_global = np.array([1.0, 0.0, 0.0])
        y_global = np.array([0.0, 1.0, 0.0])
        z_global = np.array([0.0, 0.0, 1.0])

        # Calculate rotation matrix
        zero = np.zeros((3, 3))
        r = np.zeros((3, 3))

        for i, local_axis in enumerate([
            orientation_node.x_axis,
            orientation_node.y_axis,
            orientation_node.z_axis,
        ]):
            for j, global_axis in enumerate([x_global, y_global, z_global]):
                r[i][j] = geo.functions.cos_between(local_axis, global_axis)

        rotation_matrix = np.block(
            [
                [r, zero, zero, zero],
                [zero, r, zero, zero],
                [zero, zero, r, zero],
                [zero, zero, zero, r],
            ]
        )

        return rotation_matrix

    def calc_K_local(self):
        """ Calculates the local stiffness matrix of the element.

        Args:
            grid (list): list with all the node elements of the FEM mesh

        Returns:
            K_local (np.array(dtype=float)): element local stiffness matrix
        """

        # Calculate Local Stiffness Matrix
        K_local = f.euler_beam_stiff(self.E, self.A, self.L, self.G, self.J, self.Iyy, self.Izz)

        return K_local

    def calc_K_global(self):
        """ Apply the rotation matrix to the local stiffness matrix and calculate the element
            global stiffness matrix.

        Args:
            grid (list): list with all the node elements of the FEM mesh

        Returns:
            K_global (np.array(dtype=float)): element global stiffness matrix
        """

        rotation_matrix = self.calc_rotation_matrix()
        K_local = self.calc_K_local()

        # Global Stiffness Matrix
        K_global = rotation_matrix.transpose() @ (K_local @ rotation_matrix)

        return K_global


# ==================================================================================================


class Structure:
    def __init__(self, points, beams):

        self.points = points
        self.beams = beams


# ==================================================================================================


# ==================================================================================================


class Section:
    def __init__(self, area, rotation, m_inertia_y, m_inertia_z, polar_moment):

        self.area = area
        self.rotation = rotation
        self.m_inertia_y = m_inertia_y
        self.m_inertia_z = m_inertia_z
        self.polar_moment = polar_moment


# ==================================================================================================


class Load:
    def __init__(self, application_node, load):

        self.application_node = application_node
        self.load = load



# ==================================================================================================


class Constraint:
    def __init__(self, application_node, dof_constraints):

        self.application_node = application_node
        self.dof_constraints = dof_constraints


# ==================================================================================================


class Connection(object):
    """Describes a connection between two components, in other words, set their connecting node
    as a single node in the global matrix.

    Args:
        component1 (object): an object able to generate nodes
        component1_node (string): "ROOT" or "TIP"
        component2 (object): an object able to generate nodes
        component2_node (string): "ROOT" or "TIP"

    Attibutes:
        descriptor (list(string)): a list with two strings describing the conection, the strings are
                                   composed by concatenating the component identifier with the node
                                   string with an hifen between
    """

    def __init__(self, component1, component1_node, component2, component2_node):

        self.component1 = component1
        self.component1_node = component1_node
        self.component2 = component2
        self.component2_node = component2_node

        self.descriptor = [
            component1.identifier + "-" + component1_node,
            component2.identifier + "-" + component2_node,
        ]

