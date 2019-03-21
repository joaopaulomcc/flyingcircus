"""
test_atmosphere.py

Testing suite for atmosphere module

Author: João Paulo Monteiro Cruvinel da Costa
email: joaopaulomcc@gmail.com / joao.cruvinel@embraer.com.br
github: joaopaulomcc
"""
# ==================================================================================================
# IMPORTS

from context import flyingcircus
from flyingcircus import atmosphere

# ==================================================================================================
# TESTS


def test_ISA():

    altitude = 5000
    density, pressure, temperature = atmosphere.ISA(altitude)

    print("TESTING ISA")
    print(f"Density: {density}")
    print(f"Pressure: {pressure}")
    print(f"Temperature: {temperature}")
    print()

# ==================================================================================================
# RUNNING TESTS
if __name__ == "__main__":

    print()
    print("=============================")
    print("= Testing atmosphere module =")
    print("=============================")
    print()
    test_ISA()
