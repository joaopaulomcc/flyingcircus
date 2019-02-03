"""
====================================================================================================
Comparisson between results found in the literature and those obtained by using Flying Circus.
Definition of the simulation parameters

Author: João Paulo Monteiro Cruvinel da Costa

Literature results:

NACA Technical Note No.1270 - EXPERIMENTAL AND CALCULATED CHARACTERISTICS OF SEVERALNACA 44-SERIES
WINGS WITH ASPECT RATIOS OF 8, 10, AND 12 AND TAPER RATIOS OF 2.5 AND 3.5

Authors: Robert H. Neely, Thomas V. Bollech, Gertrude C. Westrick, and Robert R. Graham

Langley Memorial Aeronautical Laboratory
Langley Field, Va.

Washington, May 1947
====================================================================================================
"""

# ==================================================================================================
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

# EXECUTE CALCULATION

from broeren_wing_data import broeren_wing_i
from broeren_wing_simulation import broeren_wing_i_results

# ==================================================================================================
# PROCESSING RESULTS

components_delta_p_grids = []
components_force_mag_grids = []

for panels, forces in zip(
    broeren_wing_i_results["aircraft_macrosurfaces_panels"],
    broeren_wing_i_results["aircraft_force_grid"],
):

    delta_p, force = aero.vlm.calc_panels_delta_pressure(panels, forces)
    components_delta_p_grids.append(delta_p)
    components_force_mag_grids.append(force)

ax, fig = vis.plot_3D.plot_results(
    broeren_wing_i_results["aircraft_original_grids"]["macrosurfaces_aero_grids"],
    components_delta_p_grids,
    title="Smith Wing - alpha: 2º - 10 Iterations",
    label="Delta Pressure [Pa]",
    colormap="coolwarm",
)

plt.show()

sys.exit()
