from context import src
from src import atmos

import numpy as np
# from src import vortex_lattice as vlm
# from src import objects as obj
# from src import mesh_generation as msh
# from src import functions as fc


def test_atmos():

    altitude = 5000
    density, pressure, temperature = atmos.ISA(altitude)

    print("TESTING atmos")
    print(f"Density: {density}")
    print(f"Pressure: {pressure}")
    print(f"Temperature: {temperature}")
    print()


if __name__ == "__main__":

    print()
    print("========================")
    print("= Testing atmos module =")
    print("========================")
    print()
    test_atmos()