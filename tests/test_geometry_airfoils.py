import context

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from flyingcircus.geometry import airfoils


def test_read_selig_airfoil_file() -> None:

    airfoil_filepath = Path("data/airfoils/naca_0012_selig.dat")
    airfoil_name, all_points, upper_points, lower_points = airfoils.read_selig_airfoil_file(
        airfoil_filepath
    )

    fig, ax = plt.subplots()
    ax.plot(all_points[:, 0], all_points[:, 1])
    ax.scatter(upper_points[:, 0], upper_points[:, 1], marker="x", color="red")
    ax.scatter(lower_points[:, 0], lower_points[:, 1], marker="x", color="black")
    ax.set(title=airfoil_name, xlabel="chord", ylabel="thickness", aspect="equal")

    plt.show()


def test_read_lednicer_airfoil_file() -> None:

    airfoil_filepath = Path("data/airfoils/naca_0012_lednicer.dat")
    airfoil_name, all_points, upper_points, lower_points = airfoils.read_lednicer_airfoil_file(
        airfoil_filepath
    )

    fig, ax = plt.subplots()
    ax.plot(all_points[:, 0], all_points[:, 1])
    ax.scatter(upper_points[:, 0], upper_points[:, 1], marker="x", color="red")
    ax.scatter(lower_points[:, 0], lower_points[:, 1], marker="x", color="black")
    ax.set(title=airfoil_name, xlabel="chord", ylabel="thickness", aspect="equal")

    plt.show()


def test_create_airfoil_spline() -> None:

    airfoil_filepath = Path("data/airfoils/naca_0012_selig.dat")
    airfoil_name, all_points, upper_points, lower_points = airfoils.read_selig_airfoil_file(
        airfoil_filepath
    )

    airfoil_spline = airfoils.create_airfoil_spline(
        upper_points,
        lower_points,
        smoothing=0.0,
        spline_order=3,
    )

    x_coord = np.linspace(-1, 1, 201)
    interpolated_x_coord, interpolated_y_coord = airfoil_spline(x_coord)
    interpolated_x_coord = np.abs(interpolated_x_coord)

    fig, ax = plt.subplots()
    ax.plot(interpolated_x_coord, interpolated_y_coord)
    ax.scatter(upper_points[:, 0], upper_points[:, 1], marker="x", color="black")
    ax.scatter(lower_points[:, 0], lower_points[:, 1], marker="x", color="black")
    ax.set(title=airfoil_name, xlabel="chord", ylabel="thickness", aspect="equal")

    plt.show()


def test_generate_airfoil_points() -> None:

    airfoil_filepath = Path("data/airfoils/NASA_SC(2)-0714_selig.dat")
    airfoil_name, all_points, upper_points, lower_points = airfoils.read_selig_airfoil_file(
        airfoil_filepath
    )

    airfoil_spline = airfoils.create_airfoil_spline(
        upper_points,
        lower_points,
        smoothing=0.0,
        spline_order=3,
    )

    n_upper_points = len(upper_points)
    n_lower_points = len(lower_points)

    interpolated_upper_points, interpolated_lower_points = airfoils.generate_airfoil_points(
        airfoil_spline, n_upper_points, n_lower_points, distribution="cos**2"
    )

    fig, ax = plt.subplots()
    ax.plot(upper_points[:, 0], upper_points[:, 1], marker="x", color="black")
    ax.plot(lower_points[:, 0], lower_points[:, 1], marker="x", color="black")
    ax.plot(
        interpolated_upper_points[:, 0],
        interpolated_upper_points[:, 1],
        marker="x",
        color="firebrick",
    )
    ax.plot(
        interpolated_lower_points[:, 0],
        interpolated_lower_points[:, 1],
        marker="x",
        color="firebrick",
    )

    ax.set(title=airfoil_name, xlabel="chord", ylabel="thickness", aspect="equal")

    plt.show()


if __name__ == "__main__":
    test_read_selig_airfoil_file()
    test_read_lednicer_airfoil_file()
    test_create_airfoil_spline()
    test_generate_airfoil_points()
