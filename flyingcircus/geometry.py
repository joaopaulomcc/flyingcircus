import numpy as np
import scipy as sc

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

        try:

            lines = airfoil_data_file.readlines()

            name = lines[0].strip()

            x = np.zeros(len(lines) - 1)
            y = np.zeros(len(lines) - 1)

            for i, line in enumerate(lines[1:]):

                x_data, y_data = line.split()

                if i == 0:
                    if float(x_data) != 1.0:
                        raise ValueError

                if i == len(lines) - 1:
                    if float(x_data) != 1.0:
                        raise ValueError

                x[i] = float(x_data)
                y[i] = float(y_data)

        except Exception:
            print("Airfoil file does not conform to the Selig format")
            return None

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

        try:

            lines = airfoil_data_file.readlines()

            name = lines[0].strip()

            n_upper_points, n_lower_points = map(float, lines[1].split())
            n_upper_points = int(n_upper_points)
            n_lower_points = int(n_lower_points)

            if (n_upper_points <= 1) or (n_lower_points <= 1):
                raise ValueError

            x = np.zeros(n_upper_points + n_lower_points - 1)
            y = np.zeros(n_upper_points + n_lower_points - 1)

            upper_x = np.zeros(n_upper_points)
            upper_y = np.zeros(n_lower_points)

            lower_x = np.zeros(n_upper_points)
            lower_y = np.zeros(n_lower_points)

            if (lines[2].strip() != "") or (lines[3 + n_upper_points].strip() != ""):
                raise ValueError

            for i in range(n_upper_points):
                x_data, y_data = lines[3 + i].split()

                upper_x[i] = float(x_data)
                upper_y[i] = float(y_data)

            for i in range(n_lower_points):
                x_data, y_data = lines[4 + n_upper_points + i].split()

                lower_x[i] = float(x_data)
                lower_y[i] = float(y_data)

            for i, (x_value, y_value) in enumerate(
                zip(np.flip(upper_x), np.flip(upper_y))
            ):
                x[i] = x_value
                y[i] = y_value

            for i, (x_value, y_value) in enumerate(
                zip(lower_x, lower_y), start=(n_upper_points - 1)
            ):
                x[i] = x_value
                y[i] = y_value

        except Exception:
            print("Airfoil file does not conform to the Lednicer format")
            return None

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
