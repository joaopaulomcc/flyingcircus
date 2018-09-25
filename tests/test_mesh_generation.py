import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from context import src

from src import mesh_generation as msh
from src import objects as obj


def test_generate_mesh():

    area = 20
    aspect_ratio = 5
    taper_ratio = 0.66
    sweep_quarter_chord = 25
    dihedral = 25
    incidence = 0
    torsion = 0
    position = [0, 0, 0]

    simple_rectangular = obj.Wing(area, aspect_ratio, taper_ratio, sweep_quarter_chord, dihedral,
                                  incidence, torsion, position)

    wing = simple_rectangular
    n_semi_wingspam_panels = 5
    n_chord_panels = 4
    wingspam_discretization_type = "linear"
    chord_discretization_type = "linear"

    xx, yy, zz = msh.generate_mesh(wing, n_semi_wingspam_panels, n_chord_panels,
                                   wingspam_discretization_type, chord_discretization_type)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type="persp")
    ax.autoscale(False)
    ax.plot_wireframe(xx, yy, zz)
    plt.show()

if __name__ == "__main__":

    print()
    print("==================================")
    print("= Testing mesh_generation module =")
    print("==================================")
    print()
    test_generate_mesh()