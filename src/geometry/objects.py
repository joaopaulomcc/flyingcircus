import numpy as np
import scipy as sc

from numpy import sin, cos, tan, pi
from . import functions as f
from .. import mathematics as m


class Airfoil(object):
    def __init__(self, upper_spline, lower_spline, cl_alpha, cd_alpha, cm_alpha):
        self.upper_spline = upper_spline
        self.lower_spline = lower_spline
        self.cl_alpha_spline = cl_alpha
        self.cd_alpha_spline = cd_alpha
        self.cm_alpha_spline = cm_alpha


class Section(object):
    def __init__(self, airfoil, material, area, Iyy, Izz, J, shear_center):
        self.airfoil = airfoil
        self.material = material
        self.area = area
        self.Iyy = Iyy
        self.Izz = Izz
        self.J = J
        self.shear_center = shear_center


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

    def generate_aero_mesh(
        self,
        n_span_panels,
        n_chord_panels,
        control_surface_deflection=0,
        chord_discretization="linear",
        span_discretization="linear",
        torsion_function="linear",
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

        # TODO fix bug when control_surface_hinge_position is None
        
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

            # Generate definitive mesh array
            mesh_points_xx = np.zeros(np.shape(planar_mesh_points_xx))
            mesh_points_yy = np.zeros(np.shape(planar_mesh_points_yy))
            mesh_points_zz = np.zeros(np.shape(planar_mesh_points_zz))

            # Applying wing torsion
            for i in range(n_span_points):
                # Extract section points from grid
                section_points_x = planar_mesh_points_xx[:, i]
                section_points_y = planar_mesh_points_yy[:, i]
                section_points_z = planar_mesh_points_zz[:, i]

                # Convert points from grid to list
                section_points = f.grid_to_vector(
                    section_points_x, section_points_y, section_points_z
                )

                # Calculate rotation characteristics and apply rotation
                rot_angle = torsion_function(section_points_y[0] / self.length)
                rot_axis = np.array([0, 1, 0])  # Y axis
                rot_center = section_points_x.min() + 0.25 * (
                    section_points_x.max() - section_points_x.min()
                )

                rot_section_points = f.rotate_point(
                    section_points, rot_axis, rot_center, rot_angle
                )

                # Convert section points from list to grid
                shape = (n_chord_points, 1)
                rot_section_points_x, rot_section_points_y, rot_section_points_z = f.vector_to_grid(
                    rot_section_points, shape
                )

                # Paste rotated section into grid
                mesh_points_xx[:, i] = rot_section_points_x[:, 0]
                mesh_points_yy[:, i] = rot_section_points_y[:, 0]
                mesh_points_zz[:, i] = rot_section_points_z[:, 0]

            # Apply wing dihedral

            # Convert grid to list
            mesh_points = f.grid_to_vector(
                mesh_points_xx, mesh_points_yy, mesh_points_zz
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

        return mesh_points_xx, mesh_points_yy, mesh_points_zz


# --------------------------------------------------------------------------------------------------


class MacroSurface(object):
    def __init__(self, position, incidence, surface_list, symmetry_plane=None):

        self.position = position
        self.incidence_degrees = incidence
        self.incidence_rad = np.radians(incidence)
        self.surface_list = surface_list
        self.symmetry_plane = symmetry_plane

        control_surfaces = []

        for surface in surface_list:
            if surface.control_surface_hinge_position is not None:
                control_surfaces.append(surface.identifier)

        self.control_surfaces = control_surfaces

    def create_mesh(
        self,
        n_chord_panels,
        n_span_panels,
        control_surface_deflection_dict=dict(),
        chord_discretization="linear",
        span_discretization="linear",
        torsion_function="linear",
    ):

        if self.symmetry_plane == "XZ" or self.symmetry_plane == "xz":

            middle_index = int(len(self.surface_list) / 2)
            left_side = self.surface_list[:middle_index]
            right_side = self.surface_list[middle_index:]

            # If the surface is on the left side of the aircraft mirror it's mesh
            left_side = np.flip(left_side)

            left_side_meshs = f.connect_surface_grid(
                left_side,
                self.incidence_rad,
                self.position,
                n_span_panels,
                n_chord_panels,
                control_surface_deflection_dict,
                chord_discretization,
                span_discretization,
            )

            mirrored_grids = []
            for mesh in left_side_meshs:
                grid_xx = mesh[0]
                grid_yy = mesh[1]
                grid_zz = mesh[2]
                mirror_plane = self.symmetry_plane

                grid_xx, grid_yy, grid_zz = f.mirror_grid(
                    grid_xx, grid_yy, grid_zz, mirror_plane
                )
                mirrored_grids.append([grid_xx, grid_yy, grid_zz])

            left_side_meshs = mirrored_grids

            # On the right side no modification is needed
            right_side_meshs = f.connect_surface_grid(
                right_side,
                self.incidence_rad,
                self.position,
                n_span_panels,
                n_chord_panels,
                control_surface_deflection_dict,
                chord_discretization,
                span_discretization,
            )

            macro_surface_mesh = left_side_meshs + right_side_meshs

        else:

            macro_surface_mesh = f.connect_surface_grid(
                self.surface_list,
                self.incidence_rad,
                self.position,
                n_span_panels,
                n_chord_panels,
                control_surface_deflection_dict,
                chord_discretization,
                span_discretization,
            )

        return macro_surface_mesh

