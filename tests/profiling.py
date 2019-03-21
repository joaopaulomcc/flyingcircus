from context import flyingcircus
from flyingcircus import atmosphere
from flyingcircus import basic_elements
from flyingcircus import basic_objects
from flyingcircus import finite_element_method
from flyingcircus import geometry
from flyingcircus import mesh
from flyingcircus import translator
from flyingcircus import visualization
from flyingcircus import vortex_lattice_method

import cProfile
import pstats

cProfile.run("test_vizualization", "test_visualization_profile.profile")

p = pstats.Stats("test_visualization_profile.profile")
p.strip_dirs().sort_stats("cumulative").print_stats(50)