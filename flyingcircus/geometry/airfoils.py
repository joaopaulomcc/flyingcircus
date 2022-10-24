# IMPORTS ------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Callable
from functools import partial

import numpy as np
import numpy.typing as npt
from scipy import interpolate

# CLASSES ------------------------------------------------------------------------------------------


# FUNCTIONS


def read_selig_airfoil_file(
    airfoil_filepath: Path,
) -> tuple[str, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:

    file_contents = airfoil_filepath.read_text().split("\n")

    airfoil_name = file_contents[0].strip()

    all_points_list = []
    leading_edge_idx = 0

    for i, line in enumerate(file_contents[1:]):

        if line != "":

            line_contents = line.split()
            all_points_list.append([float(line_contents[0]), float(line_contents[1])])

            if float(line_contents[0]) == 0.0:
                leading_edge_idx = i

    all_points_array = np.array(all_points_list)

    upper_points_array = np.flip(all_points_array[: (leading_edge_idx + 1)], axis=0)
    lower_points_array = all_points_array[leading_edge_idx:]

    return airfoil_name, all_points_array, upper_points_array, lower_points_array


def read_lednicer_airfoil_file(
    airfoil_filepath: Path,
) -> tuple[str, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:

    file_contents = airfoil_filepath.read_text().split("\n")

    airfoil_name = file_contents[0].strip()
    n_upper_points, n_lower_points = [int(float(number)) for number in file_contents[1].split()]

    upper_points_list = []

    for i in range(n_upper_points):

        line_contents = file_contents[i + 3].split()
        upper_points_list.append([float(line_contents[0]), float(line_contents[1])])

    upper_points_array = np.array(upper_points_list)

    lower_points_list = []

    for i in range(n_lower_points):

        line_contents = file_contents[i + n_upper_points + 4].split()
        lower_points_list.append([float(line_contents[0]), float(line_contents[1])])

    lower_points_array = np.array(lower_points_list)

    all_points_array = np.concatenate([np.flip(upper_points_array, axis=0), lower_points_array[1:]])

    return airfoil_name, all_points_array, upper_points_array, lower_points_array


def create_airfoil_spline(
    airfoil_upper_points: npt.NDArray[np.float64],
    airfoil_lower_points: npt.NDArray[np.float64],
    smoothing: float = 0.0,
    spline_order: int = 3,
) -> Callable:

    modified_lower_points = np.flip(airfoil_lower_points.copy(), axis=0)
    modified_lower_points[:, 0] = -modified_lower_points[:, 0]

    airfoil_points = np.concatenate([modified_lower_points, airfoil_upper_points[1:]])

    x_coordinates = airfoil_points[:, 0]
    y_coordinates = airfoil_points[:, 1]
    spline_parameter = airfoil_points[:, 0]

    tck, u = interpolate.splprep(
        [x_coordinates, y_coordinates], s=smoothing, k=spline_order, u=spline_parameter
    )

    return partial(interpolate.splev, tck=tck)


def generate_airfoil_points(
    airfoil_spline: Callable, n_upper_points: int, n_lower_points: int, distribution="cos**2"
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:

    if distribution in ["cos", "cos**2"]:

        upper_parametric = np.linspace(0, np.pi / 2, n_upper_points)
        lower_parametric = np.linspace(0, np.pi / 2, n_lower_points)

        if distribution == "cos":
            upper_x_coord = 1 - np.cos(upper_parametric)
            lower_x_coord = np.cos(lower_parametric) - 1

        else:
            upper_x_coord = 1 - np.cos(upper_parametric) ** 2
            lower_x_coord = np.cos(lower_parametric) ** 2 - 1

    elif distribution == "linear":

        upper_x_coord = np.linspace(0, 1, n_upper_points)
        lower_x_coord = np.linspace(0, -1, n_upper_points)

    else:

        raise ValueError(
            (
                f"{distribution} is not a valid distribution value.",
                "Supported distributions are 'linear', 'cos', and 'cos**2'.",
            )
        )

    interpolated_upper_x_coord, interpolated_upper_y_coord = airfoil_spline(upper_x_coord)
    interpolated_lower_x_coord, interpolated_lower_y_coord = airfoil_spline(lower_x_coord)

    upper_points = np.column_stack([interpolated_upper_x_coord, interpolated_upper_y_coord])
    lower_points = np.column_stack([-interpolated_lower_x_coord, interpolated_lower_y_coord])

    return upper_points, lower_points
