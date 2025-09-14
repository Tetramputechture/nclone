"""
Constants and enums for hierarchical reachability analysis.

This module contains shared constants to avoid circular imports.
"""

from enum import Enum


class ResolutionLevel(Enum):
    """Resolution levels for hierarchical analysis."""
    REGION = 96    # Strategic planning level
    TILE = 24      # Standard movement level  
    SUBCELL = 6    # High-precision level