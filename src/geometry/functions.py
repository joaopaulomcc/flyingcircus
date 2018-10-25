import numpy as np
from numpy import sin, cos, tan, arccos, arcsin, arctan

from .. import mathematics as m

# --------------------------------------------------------------------------------------------------


def discretization(discretization_type, n_points):

    if discretization_type == "linear":
        points = np.linspace(0, 1, n_points)

    elif discretization_type == "cos":
        angles = np.linspace(np.pi / 2, np.pi, n_points)
        points = cos(angles) + 1

    elif discretization_type == "sin":
        angles = np.linspace(0, np.pi / 2, n_points)
        points = sin(angles)

    elif discretization_type == "cos_sim":
        angles = np.linspace(0, np.pi, n_points)
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
    CPM = np.array([[ 0., -u2,  u1],
                    [ u2,  0., -u0],
                    [-u1,  u0,  0.]])

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

# ==================================================================================================
# TESTS

if __name__ == "__main__":

    print()
    print("######################")
    print("# GEOMETRY functions #")
    print("######################")
    print()

    print("# discretization")
    print(f"- linear: {discretization('linear', 11)}")
    print(f"- cos: {discretization('cos', 11)}")
    print(f"- sin: {discretization('sin', 11)}")
    print(f"- cos_sim: {discretization('cos_sim', 11)}")
    print()

    print("# replace_closest")
    original = np.linspace(0, 1, 11)
    value = 0.75
    new_array, closest_index = replace_closest(original, value)
    print(f"- original: {original}")
    print(f"- new_array: {new_array}")
    print(f"- closest_index: {closest_index}")
    print()

    print("# grid_to_vector")
    x_grid = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    y_grid = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    z_grid = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]])
    points_vector = grid_to_vector(x_grid, y_grid, z_grid)
    print("- grid_to_vector")
    print(points_vector)
    print()
