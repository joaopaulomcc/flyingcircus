from typing import AsyncIterable
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from pathlib import Path


def read_selig_airfoil_data(airfoil_data_filepath):
    """Reads a selig style airfoil file and returns its points as two numpy arrays

    Args
        airfoil_data_filepath: string or Path object to the airfoil data file

    Returns
        name: name of the airfoil according to the data file
        x, y: numpy arrays containing the airfoil points from the upper trailing edge to
              to the lower trainling edge
    """

    with open(airfoil_data_filepath, "r") as airfoil_data_file:

        lines = airfoil_data_file.readlines()

        name = lines[0].strip()

        x = np.zeros(len(lines) - 1)
        y = np.zeros(len(lines) - 1)

        for i, line in enumerate(lines[1:]):
            x_data, y_data = line.split()

            x[i] = float(x_data)
            y[i] = float(y_data)

    return name, x, y


def read_lednicer_airfoil_data(airfoil_data_filepath):
    """Reads a lednicer style airfoil file and returns its points as two numpy arrays

    Args
        airfoil_data_filepath: string or Path object to the airfoil data file

    Returns
        name: name of the airfoil according to the data file
        x, y: numpy arrays containing the airfoil points from the upper trailing edge to
              to the lower trainling edge
    """

    with open(airfoil_data_filepath, "r") as airfoil_data_file:

        lines = airfoil_data_file.readlines()

        name = lines[0].strip()

        n_upper_points, n_lower_points = map(float, lines[1].split())
        n_upper_points = int(n_upper_points)
        n_lower_points = int(n_lower_points)

        x = np.zeros(n_upper_points + n_lower_points - 1)
        y = np.zeros(n_upper_points + n_lower_points - 1)

        upper_x = np.zeros(n_upper_points)
        upper_y = np.zeros(n_lower_points)

        lower_x = np.zeros(n_upper_points)
        lower_y = np.zeros(n_lower_points)

        for i in range(n_upper_points):
            x_data, y_data = lines[3 + i].split()

            upper_x[i] = float(x_data)
            upper_y[i] = float(y_data)

        for i in range(n_lower_points):
            x_data, y_data = lines[4 + n_upper_points + i].split()

            lower_x[i] = float(x_data)
            lower_y[i] = float(y_data)

        for i, (x_value, y_value) in enumerate(zip(np.flip(upper_x), np.flip(upper_y))):
            x[i] = x_value
            y[i] = y_value

        for i, (x_value, y_value) in enumerate(
            zip(lower_x, lower_y), start=(n_upper_points - 1)
        ):
            x[i] = x_value
            y[i] = y_value

    return name, x, y


def read_airfoil_data(airfoil_data_filepath):
    """Reads an aifoil data file and returns the points as numpy arrays

        The supported file formats are Selig and Lednicer, the data is returned in
        the selig format

    Args
        airfoil_data_filepath: string or Path object to the airfoil data file

    Returns
        name: name of the airfoil according to the data file
        x, y: numpy arrays containing the airfoil points from the upper trailing edge to
              to the lower trainling edge
    """

    airfoil_data_filepath = Path(airfoil_data_filepath)

    if not airfoil_data_filepath.is_file():
        raise OSError("The airfoil file does not exist.")

    with open(airfoil_data_filepath, "r") as airfoil_data_file:

        lines = airfoil_data_file.readlines()

        first_data_line, _ = map(float, lines[1].split())

        try:

            if first_data_line == 1.0:

                name, x, y = read_selig_airfoil_data(airfoil_data_filepath)

            else:

                name, x, y = read_lednicer_airfoil_data(airfoil_data_filepath)

        except Exception:

            print("Airfoil file format is not supported")
            return None

    return name, x, y


def plot_airfoil(name, x, y, show_points=False):

    fig, ax = plt.subplots()
    # Move the left and bottom spines to x = 0 and y = 0, respectively.
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))
    # Hide the top and right spines.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Draw arrows (as black triangles: ">k"/"^k") at the end of the axes.  In each
    # case, one of the coordinates (0) is a data coordinate (i.e., y = 0 or x = 0,
    # respectively) and the other one (1) is an axes coordinate (i.e., at the very
    # right/top of the axes).  Also, disable clipping (clip_on=False) as the marker
    # actually spills out of the axes.
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    ax.plot(x, y)
    ax.set_title(name)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.axis("equal")

    if show_points:
        ax.scatter(x, y)

    plt.show()


if __name__ == "__main__":

    airfoil_folder = Path("data")

    aifoil_files = airfoil_folder.glob("*.dat")

    for airfoil_file in aifoil_files:

        print(airfoil_file.name)

        name, x, y = read_airfoil_data(airfoil_file)

        print(name)
        print()

        for x_value, y_value in zip(x, y):
            print(f"{x_value:10.3}\t{y_value:10.3}")

        print()

        plot_airfoil(name, x, y, show_points=False)
