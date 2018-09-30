from context import src
from src import atmosphere
from src import basic_elements
from src import basic_objects
from src import finite_element_method
from src import geometry
from src import mesh
from src import translator
from src import visualization
from src import vortex_lattice_method

import cProfile
import pstats

cProfile.run("test_vizualization", "test_visualization_profile.profile")

p = pstats.Stats("test_visualization_profile.profile")
p.strip_dirs().sort_stats("cumulative").print_stats(50)