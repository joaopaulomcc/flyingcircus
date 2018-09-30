"""
mesh.py

Mesh generation routines

Author: João Paulo Monteiro Cruvinel da Costa
email: joaopaulomcc@gmail.com / joao.cruvinel@embraer.com.br
github: joaopaulomcc
"""
# ==================================================================================================
# IMPORTS
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from numpy import sin, cos, tan, pi

from . import geometry
from . import basic_objects
from .fast_operations import dot, cross, norm, normalize
# ==================================================================================================
# FUNCTIONS


def generate_mesh(wing, n_semi_wingspam_panels, n_chord_panels,
                  wingspam_discretization_type="linear", chord_discretization_type="linear"):
    """ Generates a wing mesh

        Args:
            wing [Wing Object]: Wing object containing wing geometry information
            n_semi_wingspam_panels [int]: number of panels to be used in the wingspam direction
            n_chord_panels [int]: number of panes to be used in the chord direction
            wingspam_discretization_type [str]: type of discretization, "liner" or "cos"
            chord_discretization_type [str]: type of discretization, "liner" or "cos"
    """

    n_chord_points = n_chord_panels + 1
    n_semi_wingspam_points = n_semi_wingspam_panels + 1

    # Find points in the root chord:
    # TODO Refazer calculo para considera asa no ponto 0,0,0, transladar depois

    if chord_discretization_type == "linear":
        root_points_x = np.linspace(0, wing.root_chord, n_chord_points)

    elif chord_discretization_type == "cos":
        cosine = np.cos(np.linspace(0, np.pi, n_chord_points))
        root_points_x = 0.5 * wing.root_chord * (1 - cosine)

    root_points_y = np.repeat(0, n_chord_points)
    # root_points_z = np.repeat(0, n_chord_points)

    # Find leading edge x coordinates of the wing tip
    tip_leading_edge_x = (0.25 * wing.root_chord + wing.semi_wing_span * np.tan(wing.sweep_rad)
                          - 0.25 * wing.tip_chord)

    # Find points in the tip chord:

    if chord_discretization_type == "linear":
        tip_points_x = np.linspace(tip_leading_edge_x, tip_leading_edge_x + wing.tip_chord, n_chord_points)

    elif chord_discretization_type == "cos":
        cosine = np.cos(np.linspace(0, np.pi, n_chord_points))
        tip_points_x = tip_leading_edge_x + (0.5 * wing.tip_chord * (1 - cosine))

    tip_points_y = np.repeat(wing.semi_wing_span, n_chord_points)
    #tip_points_z = np.repeat(wing.semi_wing_span, n_chord_points)

    # Spanwise points

    if wingspam_discretization_type == "linear":
        span_points_y = np.linspace(0, wing.semi_wing_span, n_semi_wingspam_points)

    elif wingspam_discretization_type == "cos":
        cosine = np.cos(np.linspace(np.pi/2, 0, n_semi_wingspam_points))
        span_points_y = wing.semi_wing_span * cosine

    # Mesh calculation
    root_points_xx = np.repeat(root_points_x[np.newaxis].transpose(), n_semi_wingspam_points, axis=1)
    root_points_yy = np.repeat(root_points_y[np.newaxis].transpose(), n_semi_wingspam_points, axis=1)
    tip_points_xx = np.repeat(tip_points_x[np.newaxis].transpose(), n_semi_wingspam_points, axis=1)
    tip_points_yy = np.repeat(tip_points_y[np.newaxis].transpose(), n_semi_wingspam_points, axis=1)

    span_points_yy = np.repeat(span_points_y[np.newaxis], n_chord_points, axis=0)
    # span_points_yy = span_points_yy.transpose()

    span_points_xx = root_points_xx + (tip_points_xx - root_points_xx) * (span_points_yy - root_points_yy) / (tip_points_yy - root_points_yy)
    span_points_zz = np.zeros((n_chord_points, n_semi_wingspam_points,))

    span_points_x_vector = np.reshape(span_points_xx, span_points_xx.size)[np.newaxis]
    span_points_y_vector = np.reshape(span_points_yy, span_points_yy.size)[np.newaxis]
    span_points_z_vector = np.reshape(span_points_zz, span_points_zz.size)[np.newaxis]

    span_points = np.concatenate((span_points_x_vector, span_points_y_vector, span_points_z_vector),
                                 axis=0)

    rot_axis = np.array([1, 0, 0])    # x axis
    rot_center = np.array([0, 0, 0])  # origin
    rot_angle = wing.dihedral_rad

    span_points_rot = geometry.rotate_point(span_points, rot_axis, rot_center, rot_angle, degrees=False)

    span_points_xx = np.reshape(span_points_rot[0], (n_chord_points, n_semi_wingspam_points))
    span_points_yy = np.reshape(span_points_rot[1], (n_chord_points, n_semi_wingspam_points))
    span_points_zz = np.reshape(span_points_rot[2], (n_chord_points, n_semi_wingspam_points))

    # Mirror right wing to the left side,
    span_points_xx_mirror = np.flip(span_points_xx, axis=1)
    span_points_yy_mirror = -np.flip(span_points_yy, axis=1)
    span_points_zz_mirror = np.flip(span_points_zz, axis=1)

    # Join original array with mirror image
    span_points_xx = np.delete(span_points_xx, 0, 1)
    span_points_yy = np.delete(span_points_yy, 0, 1)
    span_points_zz = np.delete(span_points_zz, 0, 1)

    span_points_xx = np.concatenate((span_points_xx_mirror, span_points_xx), axis=1)
    span_points_yy = np.concatenate((span_points_yy_mirror, span_points_yy), axis=1)
    span_points_zz = np.concatenate((span_points_zz_mirror, span_points_zz), axis=1)

    return span_points_xx, span_points_yy, span_points_zz


def generate_panel_matrix(xx, yy, zz, infinity):
    n_x = np.shape(xx)[0] - 1
    n_y = np.shape(xx)[1] - 1
    panel_matrix = [[None for x in range(n_y)] for y in range(n_x)]

    for i in range(n_x):
        for j in range(n_y):
            xx_slice = xx[i:i + 2, j:j + 2]
            yy_slice = yy[i:i + 2, j:j + 2]
            zz_slice = zz[i:i + 2, j:j + 2]
            panel_matrix[i][j] = basic_objects.Panel(xx_slice, yy_slice, zz_slice, infinity)

    return panel_matrix


def generate_col_points_matrix(xx, yy, zz, infinity):
    n_x = np.shape(xx)[0] - 1
    n_y = np.shape(xx)[1] - 1
    col_points_matrix_xx = np.zeros((n_x, n_y))
    col_points_matrix_yy = np.zeros((n_x, n_y))
    col_points_matrix_zz = np.zeros((n_x, n_y))
    panel_matrix = generate_panel_matrix(xx, yy, zz, infinity)

    for i in range(n_x):
        for j in range(n_y):
            col_points_matrix_xx[i][j] = panel_matrix[i][j].col_point[0]
            col_points_matrix_yy[i][j] = panel_matrix[i][j].col_point[1]
            col_points_matrix_zz[i][j] = panel_matrix[i][j].col_point[2]

    return col_points_matrix_xx, col_points_matrix_yy, col_points_matrix_zz
