import sys
from pathlib import Path

parent = str(Path(__file__).parents[1])
sys.path.insert(0, parent)

import flyingcircus
