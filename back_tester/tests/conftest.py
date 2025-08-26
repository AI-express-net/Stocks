"""
Pytest configuration file for back tester tests.

This file sets up the Python path so that all tests can import from the back_tester package.
"""

import os
import sys
import pytest

# Add the back_tester directory to the Python path
back_tester_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if back_tester_dir not in sys.path:
    sys.path.insert(0, back_tester_dir)

# Also add the parent directory (stocks) to the path for stocks imports
parent_dir = os.path.dirname(back_tester_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
