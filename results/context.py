"""
context.py

Simple python code that adds the source code in the path in order to allow the test code to
import the necessary modules from src

"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src
