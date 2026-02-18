"""Simulation-based PBRS phi grid computation.

.. deprecated::
    Prefer :mod:`nclone.value_function` for velocity- and jump-aware 5D
    value function PBRS. This package provides 2D position-only phi grids
    and is kept for backward compatibility when value function files
    are not precomputed.

This package replaces the analytical graph-based PBRS potential computation
with simulation-based frame-count distance fields. The physics simulator
directly enumerates transitions, backward Dijkstra computes minimum-frame
distances from goals, and the result is stored as phi lookup grids for
O(1) runtime access.

Key components:
- GridClassifier: Classify level cells as valid/invalid at 6px resolution
- TransitionEnumerator: Enumerate physics transitions with multi-velocity sweep
- BackwardDijkstra: Compute frame-count distance fields from goals
- SimulationPhiGridBuilder: Orchestrate full pipeline with caching

The phi grids produced here are drop-in replacements for the BFS-based
phi grids in LevelBasedPathDistanceCache. Everything downstream
(PBRSCalculator, RewardCalculator) stays unchanged.
"""

from .grid_classifier import GridClassifier, GRID_RESOLUTION, GRID_WIDTH, GRID_HEIGHT
from .transition_enumerator import TransitionEnumerator, INITIAL_VELOCITIES
from .backward_dijkstra import backward_dijkstra
from .phi_grid_builder import SimulationPhiGridBuilder
from .native_sim_adapter import NativeSimAdapter, create_native_sim_from_map_path

__all__ = [
    "GridClassifier",
    "TransitionEnumerator",
    "backward_dijkstra",
    "SimulationPhiGridBuilder",
    "NativeSimAdapter",
    "create_native_sim_from_map_path",
    "GRID_RESOLUTION",
    "GRID_WIDTH",
    "GRID_HEIGHT",
    "INITIAL_VELOCITIES",
]
