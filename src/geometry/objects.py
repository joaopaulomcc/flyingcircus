import numpy as np
import scipy as sc

from numpy import sin, cos, tan, pi
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
        hs_A = np.array([self.l_chord_1_4[0] + infinity, self.l_chord_1_4[1], self.l_chord_1_4[2]])
        hs_B = np.array([self.l_chord_1_4[0], self.l_chord_1_4[1], self.l_chord_1_4[2]])
        hs_C = np.array([self.r_chord_1_4[0], self.r_chord_1_4[1], self.r_chord_1_4[2]])
        hs_D = np.array([self.r_chord_1_4[0] + infinity, self.r_chord_1_4[1], self.r_chord_1_4[2]])
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
        torsion_function="linear",
        torsion_center=0.0,
    ):

        n_chord_points = n_chord_panels + 1
        n_span_points = n_span_panels + 1

        chord_points = f.discretization(chord_discretization, n_chord_points)

        # Generate a torsion linear torsion function when not supplied with one
        if torsion_function == "linear":
            torsion_function = (
                lambda span_position: span_position * self.tip_torsion_angle_rad
            )

        # Change discretization in order to put panel division at control surface hinge line
        if self.control_surface_hinge_position is not None:
            chord_points, hinge_index = f.replace_closest(
                chord_points, self.control_surface_hinge_position
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

        rot_mesh_points = f.rotate_point(
            mesh_points, rot_axis, rot_center, rot_angle
        )

        # Convert mesh_points from list to grid
        shape = (n_chord_points, n_span_points)
        mesh_points_xx, mesh_points_yy, mesh_points_zz = f.vector_to_grid(
            rot_mesh_points, shape
        )

        # Generate definitive mesh array
        t_mesh_points_xx = np.zeros(np.shape(planar_mesh_points_xx))
        t_mesh_points_yy = np.zeros(np.shape(planar_mesh_points_yy))
        t_mesh_points_zz = np.zeros(np.shape(planar_mesh_points_zz))

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
            rot_angle = torsion_function(section_points_y[0] / (self.length * np.cos(self.dihedral_angle_rad)))
            rot_axis = np.array([0, 1, 0])  # Y axis

            # Calculate Rotation center
            section_point_1 = section_points[:,0]
            section_point_2 = section_points[:,n_chord_points - 1]
            section_vector = section_point_2 - section_point_1
            rot_center = section_point_1 + torsion_center * section_vector

            #rot_center = section_points_x.min() + torsion_center * (
            #    section_points_x.max() - section_points_x.min()
            #)

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

        return t_mesh_points_xx, t_mesh_points_yy, t_mesh_points_zz



# ==================================================================================================


class MacroSurface(object):
    def __init__(self, position, incidence, surface_list, symmetry_plane=None, torsion_center=0.25):

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
            right_side_span_discretization_list = span_discretization_list[:middle_index]
            right_side_torsion_function_list = torsion_function_list[:middle_index]

            # If the surface is on the left side of the aircraft mirror it's mesh
            left_side = np.flip(left_side)
            left_side_n_span_panels_list = np.flip(left_side_n_span_panels_list)
            left_side_span_discretization_list = np.flip(left_side_span_discretization_list)
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
