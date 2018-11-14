import numpy as np
from numpy import sin, cos, tan, arccos, arcsin, arctan

from .. import mathematics as m
from numba import jit

# --------------------------------------------------------------------------------------------------


@jit(nopython=True)
def distance_point_to_line(line_point_1, line_point_2, point):
    """
    reference: http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    """
    x0 = point
    x1 = line_point_1
    x2 = line_point_2

    distance = m.norm(m.cross((x0 - x1), (x0 - x2))) / m.norm(x2 - x1)

    return distance


# --------------------------------------------------------------------------------------------------


def discretization(discretization_type, n_points):

    if discretization_type == "linear":
        points = np.linspace(0, 1, n_points)

    elif discretization_type == "cos":
        angles = np.linspace(np.pi, np.pi / 2, n_points)
        points = cos(angles) + 1

    elif discretization_type == "sin":
        angles = np.linspace(0, np.pi / 2, n_points)
        points = sin(angles)

    elif discretization_type == "cos_sim":
        angles = np.linspace(np.pi, 0, n_points)
        points = cos(angles) / 2 + 0.5

    return points


# --------------------------------------------------------------------------------------------------


def replace_closest(array, value):

    closest_index = (np.abs(array - value)).argmin()
    new_array = array.copy()
    new_array[closest_index] = value

    return new_array, closest_index


# --------------------------------------------------------------------------------------------------


def grid_to_vector(x_grid, y_grid, z_grid):

    x_vector = np.reshape(x_grid, x_grid.size)[np.newaxis]
    y_vector = np.reshape(y_grid, y_grid.size)[np.newaxis]
    z_vector = np.reshape(z_grid, z_grid.size)[np.newaxis]

    points_vector = np.concatenate((x_vector, y_vector, z_vector), axis=0)

    return points_vector


# --------------------------------------------------------------------------------------------------


def vector_to_grid(points_vector, shape):

    x_grid = np.reshape(points_vector[0, :], shape)
    y_grid = np.reshape(points_vector[1, :], shape)
    z_grid = np.reshape(points_vector[2, :], shape)

    return x_grid, y_grid, z_grid


# --------------------------------------------------------------------------------------------------


def rotate_point(point_coord, rot_axis, rot_center, rot_angle, degrees=False):
    """Rotates a point around an axis

    Args:
        point_coord [[float, float, float]]: x, y and z coordinates of the points, every column is a point
        rot_axis [float, float, float]: vector that will be used as rotation axis
        rot_center [float, float, float]: point that will be used as rotation center
        rot_angle [float]: angle of rotation in radians (default) or degrees if degrees = True
        degrees [bool]: True if the user wants to use angles in degrees

    Returns:
        point [float, float, float]: coordinates of the rotated point
    """

    # Converts inputs to numpy arrays, normalizes axis vector
    rot_center = (rot_center[np.newaxis]).transpose()
    U = m.normalize(rot_axis)

    if degrees:
        theta = np.radians(rot_angle)
    else:
        theta = rot_angle

    u0 = U[0]
    u1 = U[1]
    u2 = U[2]

    # Calculating rotation matrix
    # reference: https://en.wikipedia.org/wiki/Rotation_matrix - "Rotation matrix from axis and angle"

    # Identity matrix
    I = np.identity(3)

    # Cross product matrix
    CPM = np.array([[0.0, -u2, u1], [u2, 0.0, -u0], [-u1, u0, 0.0]])

    # Tensor product U X U, this is NOT a cross product
    TP = np.tensordot(U, U, axes=0)

    # Rotation Matrix
    R = cos(theta) * I + sin(theta) * CPM + (1 - cos(theta)) * TP

    # Calculating rotated point

    # Translates points so rotation center is the origin of the coordinate system
    point_coord = point_coord - rot_center

    # Rotates all points
    rotated_points = R @ point_coord

    # Undo translation
    rotated_points = rotated_points + rot_center

    return rotated_points


# --------------------------------------------------------------------------------------------------


def mirror_grid(grid_xx, grid_yy, grid_zz, mirror_plane):

    if mirror_plane == "XY" or mirror_plane == "xy":

        new_grid_zz = -grid_zz

        new_grid_xx = grid_xx
        new_grid_yy = grid_yy

    elif mirror_plane == "XZ" or mirror_plane == "xz":

        new_grid_yy = np.flip(-grid_yy, axis=1)

        new_grid_xx = np.flip(grid_xx, axis=1)
        new_grid_zz = np.flip(grid_zz, axis=1)

    elif mirror_plane == "YZ" or mirror_plane == "yz":

        new_grid_xx = np.flip(-grid_xx, axis=0)

        new_grid_yy = np.flip(grid_yy, axis=0)
        new_grid_zz = np.flip(grid_zz, axis=0)

    else:
        print("ERROR: Mirror plane not recognized")
        return None

    return new_grid_xx, new_grid_yy, new_grid_zz


# --------------------------------------------------------------------------------------------------


def translate_grid(
    grid_xx, grid_yy, grid_zz, final_point, start_point=np.array([0, 0, 0])
):

    translation_vector = final_point - start_point
    x_translation = translation_vector[0]
    y_translation = translation_vector[1]
    z_translation = translation_vector[2]

    new_grid_xx = grid_xx + x_translation
    new_grid_yy = grid_yy + y_translation
    new_grid_zz = grid_zz + z_translation

    return new_grid_xx, new_grid_yy, new_grid_zz


# --------------------------------------------------------------------------------------------------


def rotate_grid(grid_xx, grid_yy, grid_zz, rot_axis, rot_center, rot_angle):

    points = grid_to_vector(grid_xx, grid_yy, grid_zz)

    rot_points = rotate_point(points, rot_axis, rot_center, rot_angle)

    shape = np.shape(grid_xx)
    new_grid_xx, new_grid_yy, new_grid_zz = vector_to_grid(rot_points, shape)

    return new_grid_xx, new_grid_yy, new_grid_zz


# --------------------------------------------------------------------------------------------------


def connect_surface_grid(
    surface_list,
    marco_surface_incidence,
    macro_surface_position,
    n_chord_panels,
    n_span_panels_list,
    chord_discretization,
    span_discretization_list,
    torsion_function_list,
    torsion_center,
    control_surface_dictionary,
):

    connected_grids = []

    for i, surface in enumerate(surface_list):

        n_span_panels = n_span_panels_list[i]
        span_discretization = span_discretization_list[i]
        torsion_function = torsion_function_list[i]

        if surface.identifier in control_surface_dictionary:
            control_surface_deflection = control_surface_dictionary[surface.identifier]
        else:
            control_surface_deflection = 0

        # Generates surface mesh, with torsion and dihedral
        surface_mesh_xx, surface_mesh_yy, surface_mesh_zz = surface.generate_aero_mesh(
            n_span_panels,
            n_chord_panels,
            control_surface_deflection,
            chord_discretization,
            span_discretization,
            torsion_function,
            torsion_center,
        )

        # When the surface is not at the root
        if i != 0:

            last_surface = connected_grids[i - 1]
            shape = np.shape(last_surface["xx"])

            # Get tip line segment from last surface
            tip_xx = last_surface["xx"][:, shape[1] - 1]
            tip_yy = last_surface["yy"][:, shape[1] - 1]
            tip_zz = last_surface["zz"][:, shape[1] - 1]

            tip_lead_edge = np.array([tip_xx[0], tip_yy[0], tip_zz[0]])
            tip_trai_edge = np.array([tip_xx[1], tip_yy[1], tip_zz[1]])

            tip_vector = tip_trai_edge - tip_lead_edge

            # Translate surface so it's root leading edge contacts the tip leading edge of the last 
            # surface
            final_point = tip_lead_edge
            surface_mesh_xx, surface_mesh_yy, surface_mesh_zz = translate_grid(
                surface_mesh_xx, surface_mesh_yy, surface_mesh_zz, final_point
            )

            # Get root line segment from current surface
            root_xx = surface_mesh_xx[:, 0]
            root_yy = surface_mesh_yy[:, 0]
            root_zz = surface_mesh_zz[:, 0]

            root_lead_edge = np.array([root_xx[0], root_yy[0], root_zz[0]])
            root_trai_edge = np.array([root_xx[1], root_yy[1], root_zz[1]])

            root_vector = root_trai_edge - root_lead_edge

            # use cross vector to find rot axis
            rot_axis = m.cross(root_vector, tip_vector)

            # find by wich angle the surcafe needs to be rotates
            rot_angle = angle_between(root_vector, tip_vector)

            # Rotates surface around tip_lead_edge so tip_vector and root_vector match
            rot_center = tip_lead_edge

            surface_mesh_xx, surface_mesh_yy, surface_mesh_zz = rotate_grid(
                surface_mesh_xx,
                surface_mesh_yy,
                surface_mesh_zz,
                rot_axis,
                rot_center,
                rot_angle,
            ) 

        else:
            # Translates grid to correct position
            final_point = macro_surface_position
            surface_mesh_xx, surface_mesh_yy, surface_mesh_zz = translate_grid(
                surface_mesh_xx, surface_mesh_yy, surface_mesh_zz, final_point
            )

            # Apply macro surface incidence angle, other surfaces will automatically have this incidence
            rot_axis = np.array([0, 1, 0])  # Y axis
            rot_center = macro_surface_position
            rot_angle = marco_surface_incidence
            surface_mesh_xx, surface_mesh_yy, surface_mesh_zz = rotate_grid(
                surface_mesh_xx,
                surface_mesh_yy,
                surface_mesh_zz,
                rot_axis,
                rot_center,
                rot_angle,
            )

        connected_grids.append(
            {"xx": surface_mesh_xx, "yy": surface_mesh_yy, "zz": surface_mesh_zz}
        )

    return connected_grids


# --------------------------------------------------------------------------------------------------


def velocity_field_function_generator(
    velocity_vector, rotation_vector, attitude_vector, center
):

    # This is a horrible hack

    v_x = velocity_vector[0]
    v_y = velocity_vector[0]
    v_z = velocity_vector[0]

    r_x = rotation_vector[0]
    r_y = rotation_vector[1]
    r_z = rotation_vector[2]

    alpha = attitude_vector[0]
    beta = attitude_vector[1]
    gamma = attitude_vector[2]

    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    origin = np.array([0.0, 0.0, 0.0])

    true_airspeed = velocity_vector[0]

    cg_velocity = np.array([true_airspeed, 0, 0])[np.newaxis].transpose()

    # Rotate around y for alfa
    cg_velocity = rotate_point(cg_velocity, y_axis, origin, -alpha, degrees=True)

    # Rotate around z for beta
    cg_velocity = rotate_point(cg_velocity, z_axis, origin, -beta, degrees=True)

    # Rotate around x for gamma
    cg_velocity = rotate_point(cg_velocity, x_axis, origin, -gamma, degrees=True)

    cg_velocity = cg_velocity.transpose()[0]

    def velocity_field_function(point_location):

        r = point_location - center
        tangential_velocity = -m.cross(rotation_vector, r)

        flow_velocity = cg_velocity + tangential_velocity

        return flow_velocity

    return velocity_field_function


# --------------------------------------------------------------------------------------------------


def angle_between(vector_1, vector_2):

    cos_theta = m.dot(vector_1, vector_2) / (m.norm(vector_1) * m.norm(vector_2))
    theta = np.arccos(cos_theta)

    return theta


# --------------------------------------------------------------------------------------------------


def cos_between(vector_1, vector_2):

    cos_theta = m.dot(vector_1, vector_2) / (m.norm(vector_1) * m.norm(vector_2))

    return cos_theta


# --------------------------------------------------------------------------------------------------

