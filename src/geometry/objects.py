import numpy as np
import scipy as sc

from numpy import sin, cos, tan, pi
from pyquaternion import Quaternion

from . import functions as f
from .. import mathematics as m

# ==================================================================================================
# OBJECTS


class Aircraft(object):
    def __init__(self, name, components, engines, inertial_properties):
        self.name = name
        self.components = components
        self.engines = engines
        self.inertial_properties = inertial_properties


# ==================================================================================================


class Engine(object):
    def __init__(self, position, inertial_properties, thrust_vector, thrust_function):
        self.position = position
        self.inertial_properties = inertial_properties
        self.thrust_vector = thrust_vector
        self.thrust_function = thrust_function

    def thrust(self, throttle, parameters):

        thrust_force = self.thrust_function(throttle, parameters) * self.thrust_vector

        return thrust_force


# ==================================================================================================


class MaterialPoint(object):
    def __init__(self, mass, cg_position, Ixx, Iyy, Izz, Ixy, Ixz, Iyz):
        self.mass = mass
        self.cg_position = cg_position
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.Ixy = Ixy
        self.Ixz = Ixz
        self.Iyz = Iyz


# ==================================================================================================


class Node(object):
    """Node object, defines a node in the FEM mesh.

    Args:
        xyz (np.array([3], dtype=float): array with x, y and z coordinates of the node in 3D space
        quaternion (pyquaternion.Quaternion): quaternion object with node orientation information

    Attributes:
        x (float): node x position in the 3D space.
        y (float): node y position in the 3D space.
        z (float): node z position in the 3D space.
        xyz (np.array([3], dtype=float): array with x, y and z coordinates of the node in 3D space
        quaternion (pyquaternion.Quaternion): quaternion object with node orientation information
        x_axis (np.array([3], dtype=float): numpy array with the node's x_axis
        y_axis (np.array([3], dtype=float): numpy array with the node's y_axis
        z_axis (np.array([3], dtype=float): numpy array with the node's z_axis
        x_cos (float): Direction cossine of the node's x_axis in relation to the global x_axis
        y_cos (float): Direction cossine of the node's y_axis in relation to the global y_axis
        z_cos (float): Direction cossine of the node's z_axis in relation to the global z_axis
        direction_cos (np.array([3], dtype=float): array with x_axis, y_axis and z_axis direction
                                                   cossines
    """

    def __init__(self, xyz, quaternion):

        self.x = xyz[0]
        self.y = xyz[1]
        self.z = xyz[2]
        self.xyz = np.array([self.x, self.y, self.z])
        self.quaternion = quaternion

        orig_x_axis = np.array([1, 0, 0])
        orig_y_axis = np.array([0, 1, 0])
        orig_z_axis = np.array([0, 0, 1])

        self.x_axis = quaternion.rotate(orig_x_axis)
        self.y_axis = quaternion.rotate(orig_y_axis)
        self.z_axis = quaternion.rotate(orig_z_axis)

        self.x_cos = f.cos_between(orig_x_axis, self.x_axis)
        self.y_cos = f.cos_between(orig_y_axis, self.y_axis)
        self.z_cos = f.cos_between(orig_z_axis, self.z_axis)
        self.direction_cos = np.array([self.x_cos, self.y_cos, self.z_cos])

    def translate(self, translation_vector):
        """Applies a translation to the node, returns a new Node object with the transformed
        coordinates

        Args:
            translation_vector (np.array([3], dtype=float): numpy array with the translation vector

        Returns:
            Node (geometry.objects.Node): Node object with a new node with the required
                                          transformation
        """

        new_xyz = self.xyz + translation_vector

        return Node(new_xyz, self.quaternion)

    def rotate(self, rotation_quaternion, rotation_center=np.array([0, 0, 0])):
        """Applies a rotation to the node, returns a new Node object with the transformed
        coordinates

        Args:
            rotation_quaternion (pyquaternion.Quaternion): quaternion object with the rotation to be
                                                           applied
            rotation_center (np.array([3], dtype=float): point around which the node will be rotated

        Returns:
            Node (geometry.objects.Node): Node object with a new node with the required
                                          transformation
        """

        # Coordinate transformation to move rotation center to origin
        temp_xyz = self.xyz - rotation_center

        # Rotate Node
        new_quaternion = rotation_quaternion * self.quaternion
        rot_xyz = rotation_quaternion.rotate(temp_xyz)

        # Coordinate transformation to return rotation center to it's original position
        new_xyz = rot_xyz + rotation_center

        return Node(new_xyz, new_quaternion)


# ==================================================================================================


class Airfoil(object):
    def __init__(self, upper_spline, lower_spline, cl_alpha, cd_alpha, cm_alpha):
        self.upper_spline = upper_spline
        self.lower_spline = lower_spline
        self.cl_alpha_spline = cl_alpha
        self.cd_alpha_spline = cd_alpha
        self.cm_alpha_spline = cm_alpha


# ==================================================================================================


class Section(object):
    def __init__(self, airfoil, material, area, Iyy, Izz, J, shear_center):
        self.airfoil = airfoil
        self.material = material
        self.area = area
        self.Iyy = Iyy
        self.Izz = Izz
        self.J = J
        self.shear_center = shear_center


# ==================================================================================================


class Panel(object):
    """Panel object"""

    def __init__(self, xx, yy, zz):
        """Args:
            xx [[float]] = grid with panel points x coordinates
            yy [[float]] = grid with panel points x coordinates
            zz [[float]] = grid with panel points x coordinates
        """
        self.xx = xx
        self.yy = yy
        self.zz = zz
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
        self.aero_center = 0.25 * (self.t_edge_1_2 - self.l_edge_1_2) + self.l_edge_1_2

        self.span = m.dot(self.l_edge, np.array([0, 1, 0]))
        self.n = m.normalize(m.cross(self.BD, self.AC))
        self.area = m.dot(self.n, m.cross(self.BD, self.AC)) / 2

        infinity = 100000
        hs_A = np.array(
            [self.l_chord_1_4[0] + infinity, self.l_chord_1_4[1], self.l_chord_1_4[2]]
        )
        hs_B = np.array([self.l_chord_1_4[0], self.l_chord_1_4[1], self.l_chord_1_4[2]])
        hs_C = np.array([self.r_chord_1_4[0], self.r_chord_1_4[1], self.r_chord_1_4[2]])
        hs_D = np.array(
            [self.r_chord_1_4[0] + infinity, self.r_chord_1_4[1], self.r_chord_1_4[2]]
        )
        hs_A = hs_A[np.newaxis]
        hs_B = hs_B[np.newaxis]
        hs_C = hs_C[np.newaxis]
        hs_D = hs_D[np.newaxis]


# ==================================================================================================


class Surface(object):
    def __init__(
        self,
        identifier,
        root_chord,
        root_section,
        tip_chord,
        tip_section,
        length,
        leading_edge_sweep_angle_deg,
        dihedral_angle_deg,
        tip_torsion_angle_deg,
        control_surface_hinge_position=None,
    ):

        self.identifier = identifier
        self.root_section = root_section
        self.root_chord = root_chord
        self.tip_section = tip_section
        self.tip_chord = tip_chord
        self.tip_section = tip_section
        self.length = length
        self.leading_edge_sweep_angle_deg = leading_edge_sweep_angle_deg
        self.dihedral_angle_deg = dihedral_angle_deg
        self.tip_torsion_angle_deg = tip_torsion_angle_deg
        self.control_surface_hinge_position = control_surface_hinge_position

        self.leading_edge_sweep_angle_rad = np.radians(leading_edge_sweep_angle_deg)
        self.dihedral_angle_rad = np.radians(dihedral_angle_deg)
        self.tip_torsion_angle_rad = np.radians(tip_torsion_angle_deg)

        # Avoid division by zero when the section is rectangular
        if root_chord == tip_chord:
            self.quarter_chord_sweep_ang_rad = self.leading_edge_sweep_angle_rad
        else:
            self.quarter_chord_sweep_ang_rad = np.arctan(
                length
                / (
                    length * tan(self.leading_edge_sweep_angle_rad)
                    + 0.25 * tip_chord
                    - 0.25 * root_chord
                )
            )

        self.quarter_chord_sweep_ang_deg = np.degrees(self.quarter_chord_sweep_ang_rad)

        self.span = length * cos(self.dihedral_angle_rad)
        self.ref_area = self.span * (root_chord + tip_chord) / 2
        self.true_area = length * (root_chord + tip_chord) / 2
        self.taper_ratio = tip_chord / root_chord
        self.aspect_ratio = (self.span ** 2) / self.ref_area

    # ----------------------------------------------------------------------------------------------

    def generate_aero_mesh(
        self,
        n_span_panels,
        n_chord_panels,
        control_surface_deflection=0,
        chord_discretization="linear",
        span_discretization="linear",
        apply_torsion=True,
        torsion_function="linear",
        torsion_center=0.0,
    ):

        n_chord_points = n_chord_panels + 1
        n_span_points = n_span_panels + 1

        chord_points, hinge_index = f.discretization(
            chord_discretization,
            n_chord_points,
            control_surface_hinge_position=self.control_surface_hinge_position,
        )

        # Generate a torsion linear torsion function when not supplied with one
        if torsion_function == "linear":
            torsion_function = (
                lambda span_position: span_position * self.tip_torsion_angle_rad
            )

        # Find root points by simple scaling
        root_chord_points_x = chord_points * self.root_chord
        root_chord_points_y = np.repeat(0, n_chord_points)

        # Find tip points by scaling and translation
        tip_chord_points_x = chord_points * self.tip_chord + self.length * tan(
            self.leading_edge_sweep_angle_rad
        )
        tip_chord_points_y = np.repeat(self.length, n_chord_points)

        # Find span points by simple scaling
        span_points_y = self.length * f.discretization(
            span_discretization, n_span_points
        )

        # Generate root and tip grids for simple calculation of the planar mesh points
        root_points_xx = np.repeat(
            root_chord_points_x[np.newaxis].transpose(), n_span_points, axis=1
        )
        root_points_yy = np.repeat(
            root_chord_points_y[np.newaxis].transpose(), n_span_points, axis=1
        )
        tip_points_xx = np.repeat(
            tip_chord_points_x[np.newaxis].transpose(), n_span_points, axis=1
        )
        tip_points_yy = np.repeat(
            tip_chord_points_y[np.newaxis].transpose(), n_span_points, axis=1
        )

        # Calculate planar mesh points
        planar_mesh_points_yy = np.repeat(
            span_points_y[np.newaxis], n_chord_points, axis=0
        )
        planar_mesh_points_xx = root_points_xx + (tip_points_xx - root_points_xx) * (
            planar_mesh_points_yy - root_points_yy
        ) / (tip_points_yy - root_points_yy)
        planar_mesh_points_zz = np.zeros((n_chord_points, n_span_points))

        # Apply control surface rotation
        if self.control_surface_hinge_position is not None:

            # Hinge Vector and point
            control_surface_hinge_axis = m.normalize(
                np.array(
                    [
                        tip_chord_points_x[hinge_index]
                        - root_chord_points_x[hinge_index],
                        self.length,
                        0,
                    ]
                )
            )
            # Hinge Point
            hinge_point = np.array([root_chord_points_x[hinge_index], 0, 0])

            # Slicing control surface grid
            control_surface_points_xx = planar_mesh_points_xx[(hinge_index + 1) :, :]
            control_surface_points_yy = planar_mesh_points_yy[(hinge_index + 1) :, :]
            control_surface_points_zz = planar_mesh_points_zz[(hinge_index + 1) :, :]

            # Converting grid to points list
            control_surface_points = f.grid_to_vector(
                control_surface_points_xx,
                control_surface_points_yy,
                control_surface_points_zz,
            )

            # Rotate control surface points around hinge axis
            rot_control_surface_points = f.rotate_point(
                control_surface_points,
                control_surface_hinge_axis,
                hinge_point,
                control_surface_deflection,
                degrees=True,
            )

            # Converting points list do grid
            shape = np.shape(control_surface_points_xx)
            control_surface_points_xx, control_surface_points_yy, control_surface_points_zz = f.vector_to_grid(
                rot_control_surface_points, shape
            )

            # Replacing planar points by rotate control surface points
            planar_mesh_points_xx[(hinge_index + 1) :, :] = control_surface_points_xx
            planar_mesh_points_yy[(hinge_index + 1) :, :] = control_surface_points_yy
            planar_mesh_points_zz[(hinge_index + 1) :, :] = control_surface_points_zz

        # Apply wing dihedral

        # Convert grid to list
        mesh_points = f.grid_to_vector(
            planar_mesh_points_xx, planar_mesh_points_yy, planar_mesh_points_zz
        )

        # Calculate rotation characteristics and apply rotation
        rot_angle = self.dihedral_angle_rad
        rot_axis = np.array([1, 0, 0])  # X axis
        rot_center = np.array([0, 0, 0])

        rot_mesh_points = f.rotate_point(mesh_points, rot_axis, rot_center, rot_angle)

        # Convert mesh_points from list to grid
        shape = (n_chord_points, n_span_points)
        mesh_points_xx, mesh_points_yy, mesh_points_zz = f.vector_to_grid(
            rot_mesh_points, shape
        )

        # Generate definitive mesh array
        t_mesh_points_xx = np.zeros(np.shape(planar_mesh_points_xx))
        t_mesh_points_yy = np.zeros(np.shape(planar_mesh_points_yy))
        t_mesh_points_zz = np.zeros(np.shape(planar_mesh_points_zz))

        if apply_torsion:
            # Applying wing torsion
            for i in range(n_span_points):
                # Extract section points from grid
                section_points_x = mesh_points_xx[:, i]
                section_points_y = mesh_points_yy[:, i]
                section_points_z = mesh_points_zz[:, i]

                # Convert points from grid to list
                section_points = f.grid_to_vector(
                    section_points_x, section_points_y, section_points_z
                )

                # Calculate rotation characteristics and apply rotation
                rot_angle = torsion_function(
                    section_points_y[0]
                    / (self.length * np.cos(self.dihedral_angle_rad))
                )
                rot_axis = np.array([0, 1, 0])  # Y axis

                # Calculate Rotation center
                section_point_1 = section_points[:, 0]
                section_point_2 = section_points[:, 1]
                section_vector = m.normalize(section_point_2 - section_point_1)
                rot_center = (
                    section_point_1 + torsion_center * section_vector * self.tip_chord
                )

                # rot_center = section_points_x.min() + torsion_center * (
                #    section_points_x.max() - section_points_x.min()
                # )

                rot_section_points = f.rotate_point(
                    section_points, rot_axis, rot_center, rot_angle
                )

                # Convert section points from list to grid
                shape = (n_chord_points, 1)
                rot_section_points_x, rot_section_points_y, rot_section_points_z = f.vector_to_grid(
                    rot_section_points, shape
                )

                # Paste rotated section into grid
                t_mesh_points_xx[:, i] = rot_section_points_x[:, 0]
                t_mesh_points_yy[:, i] = rot_section_points_y[:, 0]
                t_mesh_points_zz[:, i] = rot_section_points_z[:, 0]

        else:
            t_mesh_points_xx = mesh_points_xx
            t_mesh_points_yy = mesh_points_yy
            t_mesh_points_zz = mesh_points_zz

        return t_mesh_points_xx, t_mesh_points_yy, t_mesh_points_zz

    # ----------------------------------------------------------------------------------------------

    def generate_structure_nodes(self, n_nodes, torsion_center=0.0, mirror=False):

        # Find the positions of the root and tip nodes n the planform
        root_node_xyz = np.array(
            [self.root_chord * self.root_section.shear_center, 0, 0]
        )

        tip_x_position = self.length * tan(self.leading_edge_sweep_angle_rad)

        tip_node_xyz = np.array(
            [
                tip_x_position + self.tip_chord * self.tip_section.shear_center,
                self.length,
                0,
            ]
        )

        if mirror:
            tip_node_xyz[1] = -tip_node_xyz[1]

        # Calculate rotation due to wing sweep
        z_rotation = 0.5 * np.pi - np.arctan(
            (tip_node_xyz - root_node_xyz)[0] / self.length
        )

        if mirror:
            z_rotation = np.pi - z_rotation

        root_quaternion = Quaternion(axis=[0, 0, 1], angle=z_rotation)
        tip_quaternion = Quaternion(axis=[0, 0, 1], angle=z_rotation)

        # Create nodes in the planform
        root_node = Node(root_node_xyz, root_quaternion)
        tip_node = Node(tip_node_xyz, tip_quaternion)

        # Apply dihedral angle, rotate around wing root in the x axis
        rotation_center = np.array([0, 0, 0])
        x_rotation = self.dihedral_angle_rad

        if mirror:
            x_rotation = -x_rotation

        rotation_quaternion = Quaternion(axis=[1, 0, 0], angle=x_rotation)

        root_node = root_node.rotate(
            rotation_quaternion=rotation_quaternion, rotation_center=rotation_center
        )
        tip_node = tip_node.rotate(
            rotation_quaternion=rotation_quaternion, rotation_center=rotation_center
        )

        # Apply torsion to the wing tip
        # The locatio
        tip_x_position = (
            tip_node.xyz[0] - self.tip_section.shear_center * self.tip_chord
        )
        tip_torsion_center = np.array(
            [tip_x_position + self.tip_chord * torsion_center, tip_node.y, tip_node.z]
        )

        rotation_center = tip_torsion_center
        rotation_axis = np.array([0, 1, 0])
        rotation_quaternion = Quaternion(
            axis=[0, 1, 0], angle=self.tip_torsion_angle_rad
        )

        tip_node_location = tip_node.rotate(
            rotation_quaternion=rotation_quaternion, rotation_center=rotation_center
        )

        yaw_pitch_row = f.decompose_rotation(
            rotation_axis,
            rotation_quaternion.angle,
            tip_node.x_axis,
            tip_node.y_axis,
            tip_node.z_axis,
        )

        rotation_quaternion = Quaternion(axis=tip_node.x_axis, angle=yaw_pitch_row[2])
        rotation_center = tip_node.xyz

        tip_node_rotation = tip_node.rotate(
            rotation_quaternion=rotation_quaternion, rotation_center=rotation_center
        )

        tip_node = Node(tip_node_location.xyz, tip_node_rotation.quaternion)

        structure_nodes_props = f.interpolate_nodes(root_node, tip_node, n_nodes)

        structure_nodes = []

        for node_prop in structure_nodes_props:

            structure_nodes.append(Node(node_prop[0], node_prop[1]))

        return structure_nodes


# ==================================================================================================


class MacroSurface(object):
    def __init__(
        self,
        position,
        incidence,
        surface_list,
        symmetry_plane=None,
        torsion_center=0.25,
    ):

        self.position = position
        self.incidence_degrees = incidence
        self.incidence_rad = np.radians(incidence)
        self.surface_list = surface_list
        self.symmetry_plane = symmetry_plane
        self.torsion_center = torsion_center

        control_surfaces = []

        for surface in surface_list:
            if surface.control_surface_hinge_position is not None:
                control_surfaces.append(surface.identifier)

        self.control_surfaces = control_surfaces

    # ----------------------------------------------------------------------------------------------
    def create_mesh(
        self,
        n_chord_panels,
        n_span_panels_list,
        chord_discretization,
        span_discretization_list,
        torsion_function_list,
        control_surface_deflection_dict=dict(),
    ):

        if self.symmetry_plane == "XZ" or self.symmetry_plane == "xz":

            middle_index = int(len(self.surface_list) / 2)

            left_side = self.surface_list[:middle_index]
            left_side_n_span_panels_list = n_span_panels_list[:middle_index]
            left_side_span_discretization_list = span_discretization_list[:middle_index]
            left_side_torsion_function_list = torsion_function_list[:middle_index]

            right_side = self.surface_list[middle_index:]
            right_side_n_span_panels_list = n_span_panels_list[:middle_index]
            right_side_span_discretization_list = span_discretization_list[
                :middle_index
            ]
            right_side_torsion_function_list = torsion_function_list[:middle_index]

            # If the surface is on the left side of the aircraft mirror it's mesh
            left_side = np.flip(left_side)
            left_side_n_span_panels_list = np.flip(left_side_n_span_panels_list)
            left_side_span_discretization_list = np.flip(
                left_side_span_discretization_list
            )
            left_side_torsion_function_list = np.flip(left_side_torsion_function_list)

            left_side_meshs = f.connect_surface_grid(
                left_side,
                self.incidence_rad,
                self.position,
                n_chord_panels,
                left_side_n_span_panels_list,
                chord_discretization,
                left_side_span_discretization_list,
                left_side_torsion_function_list,
                self.torsion_center,
                control_surface_deflection_dict,
            )

            mirrored_grids = []
            for mesh in left_side_meshs:
                grid_xx = mesh["xx"]
                grid_yy = mesh["yy"]
                grid_zz = mesh["zz"]
                mirror_plane = self.symmetry_plane

                grid_xx, grid_yy, grid_zz = f.mirror_grid(
                    grid_xx, grid_yy, grid_zz, mirror_plane
                )
                mirrored_grids.append({"xx": grid_xx, "yy": grid_yy, "zz": grid_zz})

            left_side_meshs = list(np.flip(mirrored_grids))

            # On the right side no modification is needed
            right_side_meshs = f.connect_surface_grid(
                right_side,
                self.incidence_rad,
                self.position,
                n_chord_panels,
                right_side_n_span_panels_list,
                chord_discretization,
                right_side_span_discretization_list,
                right_side_torsion_function_list,
                self.torsion_center,
                control_surface_deflection_dict,
            )

            macro_surface_mesh = left_side_meshs + right_side_meshs

        else:

            macro_surface_mesh = f.connect_surface_grid(
                self.surface_list,
                self.incidence_rad,
                self.position,
                n_chord_panels,
                n_span_panels_list,
                chord_discretization,
                span_discretization_list,
                torsion_function_list,
                self.torsion_center,
                control_surface_deflection_dict,
            )

        return macro_surface_mesh

    def create_struct_mesh(self, n_elements_list):

        if self.symmetry_plane == "XZ" or self.symmetry_plane == "xz":

            middle_index = int(len(self.surface_list) / 2)

            left_side = self.surface_list[:middle_index]
            left_side_n_elements_list = n_elements_list[:middle_index]

            right_side = self.surface_list[middle_index:]
            right_side_n_elements_list = n_elements_list[:middle_index]

            # Flip left side so the first surface is the one at the root
            left_side = np.flip(left_side)
            left_side_n_elements_list = np.flip(left_side_n_elements_list)

            left_side_nodes = f.connect_surface_nodes(
                left_side,
                left_side_n_elements_list,
                self.position,
                self.incidence_rad,
                self.torsion_center,
                mirror=True,
            )

            right_side_nodes = f.connect_surface_nodes(
                right_side,
                right_side_n_elements_list,
                self.position,
                self.incidence_rad,
                self.torsion_center,
                mirror=False,
            )

            left_side_nodes = list(np.flip(left_side_nodes))

            macro_surface_nodes = left_side_nodes + right_side_nodes

        else:

            macro_surface_nodes = f.connect_surface_nodes(
                self.surface_list,
                n_elements_list,
                self.position,
                self.incidence_rad,
                self.torsion_center,
                mirror=False,
            )

        return macro_surface_nodes


# ==================================================================================================


class Surface2(object):
    """The surface object is a wing (or similar structure) section.

    Args:
        identifier (string): Name of the surface
        root_chord (float): length of the root chord of the surface [m]
        root_section (geometry.Section): the section object that describes the root section of the
                                         surface
        tip_chord (float): length of the tip chord of the surface [m]
        tip_section (geometry.Section): the section object that describes the root section of the
                                        surface
        length (float): the length of the surface [m]
        leading_edge_sweep_angle_deg (float): the sweep angle of the leading edge of the surface [º]
        dihedral_angle_deg (float): the surface dihedral angle [º]
        tip_torsion_angle_deg (float): the tip torsion angle in relation to the root [º]
        control_surface_hinge_position (float): relative position of control surface in the chord,
                                                must be between 0 and 1.

    Attributes:
        identifier (string): Name of the surface
        root_chord (float): length of the root chord of the surface [m]
        root_section (geometry.Section): the section object that describes the root section of the
                                         surface
        tip_chord (float): length of the tip chord of the surface [m]
        tip_section (geometry.Section): the section object that describes the root section of the
                                        surface
        length (float): the length of the surface [m]
        leading_edge_sweep_angle_deg (float): the sweep angle of the leading edge of the surface [º]
        leading_edge_sweep_angle_rad (float): the sweep angle of the leading edge of the surface
                                              [rad]
        quarter_chord_sweep_angle_deg (float): the sweep angle of the surface at quarter chord [º]
        quarter_chord_sweep_angle_rad (float): the sweep angle of the surface at quarter chord [rad]
        dihedral_angle_deg (float): the surface dihedral angle [º]
        dihedral_angle_rad (float): the surface dihedral angle [rad]
        tip_torsion_angle_deg (float): the tip torsion angle in relation to the root [º]
        tip_torsion_angle_rad (float): the tip torsion angle in relation to the root [rad]
        control_surface_hinge_position (float): relative position of control surface in the chord,
                                                must be between 0 and 1.
        span (float): projection of the surface length in the XY plane [m]
        ref_area (float): area of the surface projection in the XY plane [m^2]
        true_area (float): real area of the surface [m^2]
        taper_ratio (float): surface's taper ration
        aspect_ratio (float): surface's aspect ratio
    """

    def __init__(
        self,
        identifier,
        root_chord,
        root_section,
        tip_chord,
        tip_section,
        length,
        leading_edge_sweep_angle_deg,
        dihedral_angle_deg,
        tip_torsion_angle_deg,
        control_surface_hinge_position=None,
    ):

        self.identifier = identifier
        self.root_section = root_section
        self.root_chord = float(root_chord)
        self.tip_section = tip_section
        self.tip_chord = float(tip_chord)
        self.tip_section = tip_section
        self.length = float(length)
        self.leading_edge_sweep_angle_deg = float(leading_edge_sweep_angle_deg)
        self.dihedral_angle_deg = float(dihedral_angle_deg)
        self.tip_torsion_angle_deg = float(tip_torsion_angle_deg)
        self.control_surface_hinge_position = control_surface_hinge_position

        self.leading_edge_sweep_angle_rad = np.radians(leading_edge_sweep_angle_deg)
        self.dihedral_angle_rad = np.radians(dihedral_angle_deg)
        self.tip_torsion_angle_rad = np.radians(tip_torsion_angle_deg)

        # Calculation of the sweep angle at the surface's quarter chord, the formula fails if the
        # wing's taper ratio is equal to 1 so an if clause is used
        if root_chord == tip_chord:
            # the wing is a paralelogram, so the sweep is the same for the whole chord
            self.quarter_chord_sweep_ang_rad = self.leading_edge_sweep_angle_rad

        else:
            self.quarter_chord_sweep_ang_rad = np.arctan(
                length
                / (
                    length * tan(self.leading_edge_sweep_angle_rad)
                    + 0.25 * tip_chord
                    - 0.25 * root_chord
                )
            )

        self.quarter_chord_sweep_ang_deg = np.degrees(self.quarter_chord_sweep_ang_rad)

        self.span = length * cos(self.dihedral_angle_rad)
        self.ref_area = self.span * (root_chord + tip_chord) / 2
        self.true_area = length * (root_chord + tip_chord) / 2
        self.taper_ratio = tip_chord / root_chord
        self.aspect_ratio = (self.span ** 2) / self.ref_area

    # ----------------------------------------------------------------------------------------------

    def generate_aero_grid(
        self,
        n_span_panels,
        n_chord_panels,
        apply_torsion=True,
        torsion_center=0.0,
        torsion_function="linear",
        mirror=False,
        control_surface_deflection=0,
        chord_discretization="linear",
        span_discretization="linear",
    ):
        """Generates the surface's aerodynamic and structural grids.

            The aerodynamic grid is returned as a dictionary with with three keywords: xx, yy and zz
            they each contais a 2 dimensional array with the x, y or z coordinates of each of the
            grid points arranged in space, for example, the upper right element in the xx matrix
            contains the x coordinate of the leading edge of the wing tip. The easiest way to think
            about this is: as if you are looking at the surface from above imagine that the a
            propertie of each point in the grid is written in the surface, than copy this
            information column by column in a matrix.

            The structure grid is returned as a list of node objects, with the first node being
            the node at the root of the surface and so on. The difference between a node and a point
            in space is that the node has an orientation, this is needed for the contruction of
            the beam finite element.

            The surface is defined as if the root leading edge is located in the origin of the
            coordinate system and the wing tip is located in the POSITIVE side of the y axis. If the
            mirror option is set to TRUE the wing tip will be located at the NEGATIVE side of the
            y axis.

            The structure nodes are oriented so that its x axis points from the root of the surface
            to the tip and its y axis is parallel to the section chord.

        Args:
            n_span_panels (int): number of panels to be created in the surface span direction
            n_chord_panels (int): number of panels to be created in the surface chord direction,
                                  must be equal or greater than 2 if a control surface is present
            n_beam_elements (int): number of beam elements to be created in the surface
            apply_torsion (bool): if True geometrical torsion will be applied to the surface, if
                                  False only the dihedral will be applied
            torsion_center (float): position relative to the chord around wich the section will be
                                    rotated, must be a number between 0 and 1
            torsion_function (function): a function that receives a span position, between 0 and 1,
                                         and returns a number between 0 and 1 that will be
                                         multiplied by the tip rotation to calculate the section
                                         rotation, if "linear" is suplied a linear function will be
                                         created and applied.
            mirror (bool): If false wing tip will be in the positive side of the y axis, if True
                           wing tip will be in the negative side of the y axis.
            control_surface_deflection (float): deflection of the control surface, positive in the
                                                root->tip direction, negative if Mirro is True [º]
            chord_discretization (string): type of chord discretization, available types are
                                           linear, cos, sin e cos_sim
            span_discretization (string): type of span discretization, available types are
                                          linear, cos, sin e cos_sim
        """
        # Corrects input types
        n_span_panels = int(n_span_panels)
        n_chord_panels = int(n_chord_panels)
        apply_torsion = bool(apply_torsion)
        torsion_center = float(torsion_center)
        mirror = bool(mirror)
        control_surface_deflection = float(control_surface_deflection)

        # Checks if the number of chord panels is valid, if it isn't fix it
        if (self.control_surface_hinge_position is not None) and (n_chord_panels < 2):
            n_chord_panels = 2
            print(
                "WARINING: Invalid number of chord panels, number of chord panels set to 2."
            )

        # Caculate number of grid points
        n_chord_points = n_chord_panels + 1
        n_span_points = n_span_panels + 1

        # Generate discretization of chord, a list of numbers from 0 to 1
        chord_points, hinge_index = f.discretization(
            chord_discretization, n_chord_points, self.control_surface_hinge_position
        )

        # Generate a torsion linear torsion function when not supplied with one
        if torsion_function == "linear":
            torsion_function = (
                lambda span_position: span_position * self.tip_torsion_angle_rad
            )

        # Find root points by scaling by the root chord
        root_chord_points_x = chord_points * self.root_chord
        root_chord_points_y = np.repeat(0, n_chord_points)

        # Find tip points by scaling and translation
        tip_chord_points_x = chord_points * self.tip_chord + self.length * tan(
            self.leading_edge_sweep_angle_rad
        )
        tip_chord_points_y = np.repeat(self.length, n_chord_points)

        # Find span points by scaling by the surface's length
        span_points, _ = f.discretization(span_discretization, n_span_points)
        span_points_y = self.length * span_points

        # Generate root and tip grids for simple calculation of the planar mesh points
        root_points_xx = np.repeat(
            root_chord_points_x[np.newaxis].transpose(), n_span_points, axis=1
        )
        root_points_yy = np.repeat(
            root_chord_points_y[np.newaxis].transpose(), n_span_points, axis=1
        )
        tip_points_xx = np.repeat(
            tip_chord_points_x[np.newaxis].transpose(), n_span_points, axis=1
        )
        tip_points_yy = np.repeat(
            tip_chord_points_y[np.newaxis].transpose(), n_span_points, axis=1
        )

        # Calculate planar mesh points
        planar_mesh_points_yy = np.repeat(
            span_points_y[np.newaxis], n_chord_points, axis=0
        )
        planar_mesh_points_xx = root_points_xx + (tip_points_xx - root_points_xx) * (
            planar_mesh_points_yy - root_points_yy
        ) / (tip_points_yy - root_points_yy)
        planar_mesh_points_zz = np.zeros((n_chord_points, n_span_points))

        # Apply control surface rotation
        if self.control_surface_hinge_position is not None:

            # Hinge Vector and point
            control_surface_hinge_axis = m.normalize(
                np.array(
                    [
                        tip_chord_points_x[hinge_index]
                        - root_chord_points_x[hinge_index],
                        self.length,
                        0,
                    ]
                )
            )
            # Hinge Point
            hinge_point = np.array([root_chord_points_x[hinge_index], 0, 0])

            # Slicing control surface grid
            control_surface_points_xx = planar_mesh_points_xx[(hinge_index + 1) :, :]
            control_surface_points_yy = planar_mesh_points_yy[(hinge_index + 1) :, :]
            control_surface_points_zz = planar_mesh_points_zz[(hinge_index + 1) :, :]

            # Converting grid to points list
            control_surface_points = f.grid_to_vector(
                control_surface_points_xx,
                control_surface_points_yy,
                control_surface_points_zz,
            )

            # Rotate control surface points around hinge axis
            rot_control_surface_points = f.rotate_point(
                control_surface_points,
                control_surface_hinge_axis,
                hinge_point,
                control_surface_deflection,
                degrees=True,
            )

            # Converting points list do grid
            shape = np.shape(control_surface_points_xx)
            control_surface_points_xx, control_surface_points_yy, control_surface_points_zz = f.vector_to_grid(
                rot_control_surface_points, shape
            )

            # Replacing planar points by rotate control surface points
            planar_mesh_points_xx[(hinge_index + 1) :, :] = control_surface_points_xx
            planar_mesh_points_yy[(hinge_index + 1) :, :] = control_surface_points_yy
            planar_mesh_points_zz[(hinge_index + 1) :, :] = control_surface_points_zz

        # Apply wing dihedral

        # Convert grid to list
        mesh_points = f.grid_to_vector(
            planar_mesh_points_xx, planar_mesh_points_yy, planar_mesh_points_zz
        )

        # Calculate rotation characteristics and apply rotation
        rot_angle = self.dihedral_angle_rad
        rot_axis = np.array([1, 0, 0])  # X axis
        rot_center = np.array([0, 0, 0])

        rot_mesh_points = f.rotate_point(mesh_points, rot_axis, rot_center, rot_angle)

        # Convert mesh_points from list to grid
        shape = (n_chord_points, n_span_points)
        mesh_points_xx, mesh_points_yy, mesh_points_zz = f.vector_to_grid(
            rot_mesh_points, shape
        )

        grid_dict = {"xx": mesh_points_xx, "yy": mesh_points_yy, "zz": mesh_points_zz}

        # Apply torsion to surface grid
        if apply_torsion:

            grid_dict = f.apply_torsion_to_grid(
                grid_dict, torsion_center, torsion_function, self
            )

        # Mirror grid if needed
        if mirror:

            xx, yy, zz = f.mirror_grid(
                grid_dict["xx"], grid_dict["yy"], grid_dict["zz"], "XZ"
            )
            grid_dict = {"xx": xx, "yy": yy, "zz": zz}

        return grid_dict

    # ----------------------------------------------------------------------------------------------

    def generate_structure_nodes(
        self,
        n_beam_elements,
        apply_torsion=True,
        torsion_center=0.0,
        torsion_function="linear",
        mirror=False,
    ):
        """Generates the surface's structural grids.

        The structure grid is returned as a list of node objects, with the first node being
        the node at the root of the surface and so on. The difference between a node and a point
        in space is that the node has an orientation, this is needed for the contruction of
        the beam finite element.

        The surface is defined as if the root leading edge is located in the origin of the
        coordinate system and the wing tip is located in the POSITIVE side of the y axis. If the
        mirror option is set to TRUE the wing tip will be located at the NEGATIVE side of the
        y axis.

        The structure nodes are oriented so that its x axis points from the root of the surface
        to the tip and its y axis is parallel to the section chord.

        Args:
            n_beam_elements (int): number of beam elements to be created in the surface
            apply_torsion (bool): if True geometrical torsion will be applied to the surface, if
                                    False only the dihedral will be applied
            torsion_center (float): position relative to the chord around wich the section will be
                                    rotated, must be a number between 0 and 1
            torsion_function (function): a function that receives a span position, between 0 and 1,
                                            and returns a rotation angle in radians
            mirror (bool): If false wing tip will be in the positive side of the y axis, if True
                           wing tip will be in the negative side of the y axis.
        """

        n_nodes = n_beam_elements + 1

        if torsion_function == "linear":
            torsion_function = (
                lambda span_position: span_position * self.tip_torsion_angle_rad
            )

        # Find the positions of the root and tip nodes n the planform
        root_node_xyz = np.array(
            [self.root_chord * self.root_section.shear_center, 0, 0]
        )

        tip_x_position = self.length * tan(self.leading_edge_sweep_angle_rad)

        tip_node_xyz = np.array(
            [
                tip_x_position + self.tip_chord * self.tip_section.shear_center,
                self.length,
                0,
            ]
        )

        if mirror:
            tip_node_xyz[1] = -tip_node_xyz[1]

        # Calculate rotation due to wing sweep
        z_rotation = 0.5 * np.pi - np.arctan(
            (tip_node_xyz - root_node_xyz)[0] / self.length
        )

        if mirror:
            z_rotation = np.pi - z_rotation

        root_quaternion = Quaternion(axis=[0, 0, 1], angle=z_rotation)
        tip_quaternion = Quaternion(axis=[0, 0, 1], angle=z_rotation)

        # Create nodes in the planform
        root_node = Node(root_node_xyz, root_quaternion)
        tip_node = Node(tip_node_xyz, tip_quaternion)

        # Apply dihedral angle, rotate around wing root in the x axis
        rotation_center = np.array([0, 0, 0])
        x_rotation = self.dihedral_angle_rad

        if mirror:
            x_rotation = -x_rotation

        rotation_quaternion = Quaternion(axis=[1, 0, 0], angle=x_rotation)

        root_node = root_node.rotate(
            rotation_quaternion=rotation_quaternion, rotation_center=rotation_center
        )
        tip_node = tip_node.rotate(
            rotation_quaternion=rotation_quaternion, rotation_center=rotation_center
        )

        # Interpolate root and tip nodes to create grid
        structure_nodes_props = f.interpolate_nodes(root_node, tip_node, n_nodes)

        structure_nodes = []

        for node_prop in structure_nodes_props:
            structure_nodes.append(Node(node_prop[0], node_prop[1]))

        # Apply Node Rotation
        if apply_torsion:
            structure_nodes_props = f.apply_torsion_to_nodes(
                structure_nodes, torsion_center, torsion_function, self
            )

            structure_nodes = []

            for node_prop in structure_nodes_props:
                structure_nodes.append(Node(node_prop[0], node_prop[1]))

        return structure_nodes


# ==================================================================================================


class MacroSurface2(object):
    """Defines a macrosurface composed of surfaces, is used to create wings, horizontal and vertical
    stabilizers.

    Args:
        position (float): position of the leading edge of the macrosurface root [m]
        incidence (float): angle of incidence of the macrosurface [º]
        surface_list (list[surface]): list of surface objects tha compose the macrosurface, ordered
                                      from left to right
        symetry_plane (string): plane of symmetry of the surface, for example XZ for a wing, none if
                                the surface is not symmetrical
        torsion_center (float): position in the chord around wich geometric torsion is applyed

    Attributes:
        position (np.array(3)): x, y, z position of the leading edge of the macrosurface root [m]
        incidence_degrees (float): angle of incidence of the macrosurface [º]
        incidence_rad (float): angle of incidence of the macrosurface [rad]
        surface_list (list[surface]): list of surface objects tha compose the macrosurface, ordered
                                      from left to right
        symetry_plane (string): plane of symmetry of the surface, for example XZ for a wing, none if
                                the surface is not symmetrical
        torsion_center (float): position in the chord around wich geometric torsion is applyed
        control_surfaces (list[string]): list with the identifiers of all control surfaces present
                                         in the macro surface.
    """

    def __init__(
        self,
        position,
        incidence,
        surface_list,
        symmetry_plane=None,
        torsion_center=0.25,
    ):

        self.position = position
        self.incidence_degrees = float(incidence)
        self.incidence_rad = np.radians(incidence)
        self.surface_list = surface_list
        self.symmetry_plane = symmetry_plane
        self.torsion_center = float(torsion_center)

        control_surfaces = []

        for surface in surface_list:
            if surface.control_surface_hinge_position is not None:
                control_surfaces.append(surface.identifier)

        self.control_surfaces = control_surfaces

    # ----------------------------------------------------------------------------------------------
    def create_aero_grid(
        self,
        n_chord_panels,
        n_span_panels_list,
        chord_discretization,
        span_discretization_list,
        torsion_function_list,
        control_surface_deflection_dict=dict(),
    ):

        macro_surface_mesh = []

        if self.symmetry_plane == "XZ" or self.symmetry_plane == "xz":

            middle_index = int(len(self.surface_list) / 2)

        else:
            middle_index = 0

        right_side = self.surface_list[middle_index:]

        translation_vector_list = [self.position]
        incidence_angle_list = [self.incidence_rad]

        for i, surface in enumerate(right_side):

            leading_edge_x = surface.length * tan(surface.leading_edge_sweep_angle_rad)
            leading_edge_y = surface.span
            leading_edge_z = surface.span * tan(surface.dihedral_angle_rad)

            position = translation_vector_list[i] + np.array(
                [leading_edge_x, leading_edge_y, leading_edge_z]
            )

            incidence = incidence_angle_list[i] + surface.tip_torsion_angle_rad

            if i < len(right_side) - 1:
                translation_vector_list.append(position)
                incidence_angle_list.append(incidence)

        if self.symmetry_plane == "XZ" or self.symmetry_plane == "xz":

            # Create a mirror image of the translation_vector_list
            mirror_translation_vector_list = np.flip(translation_vector_list, axis=0)

            for vector in mirror_translation_vector_list:
                vector[1] = -vector[1]

            translation_vector_list = np.concatenate([
                mirror_translation_vector_list, translation_vector_list]
            )
            incidence_angle_list = np.concatenate([
                np.flip(incidence_angle_list), incidence_angle_list]
            )

        for i, surface in enumerate(self.surface_list):

            if i < middle_index:
                mirror = True
            else:
                mirror = False

            if surface.identifier in control_surface_deflection_dict:
                control_surface_deflection = control_surface_deflection_dict[
                    surface.identifier
                ]
            else:
                control_surface_deflection = 0

            # Generate planar mesh
            aero_grid_dict = surface.generate_aero_grid(
                n_span_panels=n_span_panels_list[i],
                n_chord_panels=n_chord_panels,
                apply_torsion=False,
                mirror=mirror,
                control_surface_deflection=control_surface_deflection,
                chord_discretization=chord_discretization,
                span_discretization=span_discretization_list[i],
            )

            # Create torsion function
            torsion_function = (
                lambda span_position: incidence_angle_list[i]
                + span_position * surface.tip_torsion_angle_rad
            )

            # Apply torsion
            aero_grid_dict = f.apply_torsion_to_grid(
                aero_grid_dict, self.torsion_center, torsion_function, surface
            )

            # Translate grid
            xx, yy, zz = f.translate_grid(
                aero_grid_dict["xx"],
                aero_grid_dict["yy"],
                aero_grid_dict["zz"],
                translation_vector_list[i],
                start_point=np.array([0, 0, 0]),
            )
            aero_grid_dict = {"xx": xx, "yy": yy, "zz": zz}

            macro_surface_mesh.append(aero_grid_dict)

        # Rotate macro surface around leading edge to apply

        return macro_surface_mesh

    def create_struct_grid(self, n_elements_list):

        if self.symmetry_plane == "XZ" or self.symmetry_plane == "xz":

            middle_index = int(len(self.surface_list) / 2)

            left_side = self.surface_list[:middle_index]
            left_side_n_elements_list = n_elements_list[:middle_index]

            right_side = self.surface_list[middle_index:]
            right_side_n_elements_list = n_elements_list[:middle_index]

            # Flip left side so the first surface is the one at the root
            left_side = np.flip(left_side)
            left_side_n_elements_list = np.flip(left_side_n_elements_list)

            left_side_nodes = f.connect_surface_nodes(
                left_side,
                left_side_n_elements_list,
                self.position,
                self.incidence_rad,
                self.torsion_center,
                mirror=True,
            )

            right_side_nodes = f.connect_surface_nodes(
                right_side,
                right_side_n_elements_list,
                self.position,
                self.incidence_rad,
                self.torsion_center,
                mirror=False,
            )

            left_side_nodes = list(np.flip(left_side_nodes))

            macro_surface_nodes = left_side_nodes + right_side_nodes

        else:

            macro_surface_nodes = f.connect_surface_nodes(
                self.surface_list,
                n_elements_list,
                self.position,
                self.incidence_rad,
                self.torsion_center,
                mirror=False,
            )

        return macro_surface_nodes
