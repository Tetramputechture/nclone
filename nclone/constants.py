"""
Legacy constants file - now imports from centralized physics_constants.

This module maintains backward compatibility by importing all constants
from the new centralized constants package.
"""

# Import all constants from the centralized location
from .constants.physics_constants import *