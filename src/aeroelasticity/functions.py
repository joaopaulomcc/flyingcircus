"""
Aeroelasticity submodule
"""
import numpy as np

from .. import geometry as geo
from .. import aerodynamics as aero

# ==================================================================================================


def loads_to_nodes_weight_matrix(
    macrosurface_aero_grid, macrosurface_struct_grid, algorithm="closest"
):

    node_vector = geo.functions.create_macrosurface_node_vector(
        macrosurface_struct_grid
    )
    panel_grid = aero.vlm.create_panel_grid(macrosurface_aero_grid)
    panel_vector = aero.vlm.flatten(panel_grid)

    weight_matrix = np.zeros((len(node_vector), len(panel_vector)))

    if algorithm == "closest":

        closest_node = None
        min_distance = float("inf")

        for j, panel in enumerate(panel_vector):

            for i, node in enumerate(node_vector):

                distance = geo.functions.distance_between_points(
                    panel.aero_center, node.xyz
                )

                if distance <= min_distance:

                    closest_node_index = i
                    min_distance = distance

            weight_matrix[closest_node_index][j] = 1

            closest_node = None
            min_distance = float("inf")

        return weight_matrix


# ==================================================================================================


def deformation_to_aero_grid_weight_matrix(
    macrosurface_aero_grid, macrosurface_struct_grid, algorithm="closest"
):

    node_vector = geo.functions.create_macrosurface_node_vector(
        macrosurface_struct_grid
    )

    aero_grid = geo.functions.macrosurface_aero_grid_to_single_grid(
        macrosurface_aero_grid
    )

    aero_points_vector = geo.functions.grid_to_vector(
        aero_grid["xx"], aero_grid["yy"], aero_grid["zz"]
    ).transpose()

    weight_matrix = np.zeros((len(aero_points_vector), len(node_vector)))

    if algorithm == "closest":

        closest_node = None
        min_distance = float("inf")

        for i, point in enumerate(aero_points_vector):

            for j, node in enumerate(node_vector):

                distance = geo.functions.distance_between_points(
                    point, node.xyz
                )

                if distance <= min_distance:

                    closest_node_index = j
                    min_distance = distance

            weight_matrix[i][closest_node_index] = 1

            closest_node = None
            min_distance = float("inf")

        return weight_matrix


# ==================================================================================================


def generated_aero_loads(components_force_grid, macrosurface_struct_grid):
    pass


# ==================================================================================================


def deform_aero_grid(macrofurface_aero_grid, struct_grid_deformations):
    pass
