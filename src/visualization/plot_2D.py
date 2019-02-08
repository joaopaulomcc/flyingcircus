"""
visualization.py

Routines for results visualization

Author: João Paulo Monteiro Cruvinel da Costa
email: joaopaulomcc@gmail.com / joao.cruvinel@embraer.com.br
github: joaopaulomcc
"""
# ==================================================================================================
# IMPORTS
import numpy as np
import scipy as sc
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d
from numpy import sin, cos, tan, pi

from .. import mathematics as m

# ==================================================================================================
# FUNCTIONS

def plot_deformation(
    struct_elements,
    struct_deformations,
    ax,
    plot_axis="Y"
    scale_factor=1,
    show_nodes=True,
    line_color="k",
    alpha=1,
):

    for component_elements in struct_elements:
        for beam_element in component_elements:

            node_A = beam_element.node_A
            node_B = beam_element.node_B

            point_A = [
                node_A.x + scale_factor * struct_deformations[node_A.number][0],
                node_A.y + scale_factor * struct_deformations[node_A.number][1],
                node_A.z + scale_factor * struct_deformations[node_A.number][2],
            ]

            point_B = [
                node_B.x + scale_factor * struct_deformations[node_B.number][0],
                node_B.y + scale_factor * struct_deformations[node_B.number][1],
                node_B.z + scale_factor * struct_deformations[node_B.number][2],
            ]

            x = [point_A[0], point_B[0]]
            y = [point_A[1], point_B[1]]
            z = [point_A[2], point_B[2]]

            if show_nodes:
                ax.plot(
                    x, y, z, color=line_color, marker="o", markersize=2, alpha=alpha
                )
            else:
                ax.plot(x, y, z, color=line_color, alpha=alpha)

    return ax