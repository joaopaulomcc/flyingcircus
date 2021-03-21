import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

import geometry as geo
import plotting as plot

from pathlib import Path


def test_plot_airfoil():

    airfoil_folder = Path(r"D:\OneDrive\Projects\flyingcircus\flyingcircus\data")

    aifoil_files = airfoil_folder.glob("*.dat")

    for airfoil_file in aifoil_files:

        print(airfoil_file.name)

        try:

            name, x, y = geo.read_airfoil_data(airfoil_file)

            print(name)
            print()

            for x_value, y_value in zip(x, y):
                print(f"{x_value:10.3}\t{y_value:10.3}")

            print()

            plot.plot_airfoil(name, x, y, show_points=False)

        except Exception:
            print("Oops!")


def test_plot_3d_lines():

    n_points = 10000

    theta = np.linspace(0, 10 * np.pi, n_points)
    z = np.linspace(0, np.pi, n_points)
    x0 = np.cos(theta) * np.sin(z)
    y0 = np.sin(theta) * np.sin(z)

    x1 = np.cos(theta + 2 * np.pi / 3) * np.sin(z)
    y1 = np.sin(theta + 2 * np.pi / 3) * np.sin(z)

    x2 = np.cos(theta + 4 * np.pi / 3) * np.sin(z)
    y2 = np.sin(theta + 4 * np.pi / 3) * np.sin(z)

    colors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    plot.plot_3d_lines([x0, x1, x2], [y0, y1, y2], [z, z, z], colors=colors)


def plot_wing():

    airfoil_file_path = Path(
        r"D:\OneDrive\Projects\flyingcircus\flyingcircus\data\k3311sm.dat"
    )
    name, x, y = geo.read_airfoil_data(airfoil_file_path)

    wingspam = 10
    n_airfoils = 11
    wing_x_position = 0

    z = np.linspace(-wingspam / 2, wingspam / 2, n_airfoils)

    xs0 = []
    ys0 = []
    zs0 = []

    for i in range(n_airfoils):

        xs0.append(x + wing_x_position)
        ys0.append(y)
        zs0.append(np.full(len(x), z[i]))

    for i in range(n_airfoils - 1):

        for j in range(len(x)):

            n_x = np.array([xs0[i][j], xs0[i + 1][j]])
            n_y = np.array([ys0[i][j], ys0[i + 1][j]])
            n_z = np.array([zs0[i][j], zs0[i + 1][j]])

            xs0.append(n_x)
            ys0.append(n_y)
            zs0.append(n_z)

    wingspam = 5
    n_airfoils = 5
    wing_x_position = 5
    wing_y_position = 0.5

    z = np.linspace(-wingspam / 2, wingspam / 2, n_airfoils)

    xs1 = []
    ys1 = []
    zs1 = []

    for i in range(n_airfoils):

        xs1.append(x + wing_x_position)
        ys1.append(y + wing_y_position)
        zs1.append(np.full(len(x), z[i]))

    for i in range(n_airfoils - 1):

        for j in range(len(x)):

            n_x = np.array([xs1[i][j], xs1[i + 1][j]])
            n_y = np.array([ys1[i][j], ys1[i + 1][j]])
            n_z = np.array([zs1[i][j], zs1[i + 1][j]])

            xs1.append(n_x)
            ys1.append(n_y)
            zs1.append(n_z)

    xs = xs0 + xs1
    ys = ys0 + ys1
    zs = zs0 + zs1

    plot.plot_3d_lines(xs, ys, zs)


# ======================================================================================

if __name__ == "__main__":

    # test_plot_airfoil()
    # test_plot_3d_lines()
    plot_wing()
