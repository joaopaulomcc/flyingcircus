# IMPORTS

# Import python scientific libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import sys

# Import code sub packages
from context import src
from src import aerodynamics as aero
from src import aeroelasticity as aelast
from src import control
from src import flight_mechanics as flmec
from src import geometry as geo
from src import loads
from src import structures as struct
from src import visualization as vis

# ==================================================================================================
#grid_file.write()
#grid_file.write("============================================================")
#grid_file.write("= VALIDATION OF AEROELASTIC CALCULATION                    =")
#grid_file.write("= VALIDATION CASE: CFD-Based Analysis of Nonlinear         =")
#grid_file.write("= Aeroelastic Behavior of High-Aspect Ratio Wings          =")
#grid_file.write("= AUTHORS: M. J. Smith, M. J. Patil, D. H. Hodges          =")
#grid_file.write("============================================================")
# ==================================================================================================
# EXECUTE CALCULATION

from hale_aircraft_data import hale_aircraft

#from hale_aircraft_simulation import results, iteration_results

import pickle

#f = open("results\\hale_aircraft\\results\\results_no_deflection.pckl", 'wb')
#pickle.dump([results, iteration_results], f)
#f.close()

f = open("results\\hale_aircraft\\results\\results_no_deflection.pckl", "rb")
results, iteration_results = pickle.load(f)
f.close()

macrosurfaces_struct_grids = results["aircraft_original_grids"]["macrosurfaces_struct_grids"]
macrosurfaces_fem_elements = results["aircraft_struct_fem_elements"]["macrosurfaces_fem_elements"]

beams_struct_grids = results["aircraft_original_grids"]["beams_struct_grids"]
beams_fem_elements = results["aircraft_struct_fem_elements"]["beams_fem_elements"]


grid_file = open("results\\hale_aircraft\\results\\grids.txt", "w")

grid_file.write("\n")
grid_file.write("#########\n")
grid_file.write("# GRIDS #\n")
grid_file.write("#########\n")
grid_file.write("\n")

for macrosurface_grid, macrosurface in zip(macrosurfaces_struct_grids, hale_aircraft.macrosurfaces):

    grid_file.write("\n")
    grid_file.write("MACROSURFACE\n")
    grid_file.write("\n")

    for surface_grid, surface in zip(macrosurface_grid, macrosurface.surface_list):


        grid_file.write("\n")
        grid_file.write(f"{surface.identifier}\n")
        grid_file.write(f"node_nu|X      |Y      |Z      |\n")
        for node in surface_grid:
            grid_file.write(f"{node.number:<8d}{node.x:<8.3f}{node.y:<8.3f}{node.z:<8.3f}\n")

for beam_grid, beam in zip(beams_struct_grids, hale_aircraft.beams):

    grid_file.write("\n")
    grid_file.write(f"{beam.identifier}\n")
    grid_file.write(f"node_nu|X      |Y      |Z      |\n")
    for node in beam_grid:
        grid_file.write(f"{node.number:<8d}{node.x:<8.3f}{node.y:<8.3f}{node.z:<8.3f}\n")

grid_file.write("\n")
grid_file.write("################\n")
grid_file.write("# FEM ELEMENTS #\n")
grid_file.write("################\n")
grid_file.write("\n")

element_counter = 0
for macrosurface_fem_elements, macrosurface in zip(macrosurfaces_fem_elements, hale_aircraft.macrosurfaces):

    grid_file.write("\n")
    grid_file.write("MACROSURFACE\n")
    grid_file.write("\n")

    for surface_fem_elements, surface in zip(macrosurface_fem_elements, macrosurface.surface_list):


        grid_file.write("\n")
        grid_file.write(f"{surface.identifier}\n")
        grid_file.write(f"ele nu |root_n |tip_n  |\n")
        for element in surface_fem_elements:

            grid_file.write(f"{element_counter:<8d}{element.node_A.number:<8d}{element.node_B.number:<8d}\n")
            element_counter += 1

for beam_fem_elements, beam in zip(beams_fem_elements, hale_aircraft.beams):

    grid_file.write("\n")
    grid_file.write(f"{beam.identifier}\n")
    grid_file.write(f"ele nu |root_n |tip_n  |\n")
    for element in beam_fem_elements:
        grid_file.write(f"{element_counter:<8d}{element.node_A.number:<8d}{element.node_B.number:<8d}\n")
        element_counter += 1

grid_file.close()