"""Analytical physics module for trajectory computation and jump hulls.

Provides exact analytical solutions to the N++ physics recurrence relations,
hull-based jump reachability, and PBRS potential evaluation.
"""

from .analytical_trajectory import (
    S1,
    S2,
    Sx2,
    velocity_at,
    position_y_at,
    position_x_at,
    jump_trajectory,
    solve_landing_frame,
    find_cap_frame,
)

__all__ = [
    "S1",
    "S2",
    "Sx2",
    "velocity_at",
    "position_y_at",
    "position_x_at",
    "jump_trajectory",
    "solve_landing_frame",
    "find_cap_frame",
]

from .trajectory_catalog import (
    ArcPoint,
    TrajectoryEntry,
    WallTrajectoryEntry,
    AirborneTrajectoryEntry,
    TrajectoryCatalog,
)
from .jump_hull import (
    JumpHullCollection,
    WallHullCollection,
    classify_surface,
    get_or_build_hulls,
    get_or_build_wall_hulls,
)
from .reachability_graph import ReachabilityGraph
from .reachability_pbrs import ReachabilityPBRS
from .reachability_builder import build_reachability_pbrs

__all__ += [
    "ArcPoint",
    "TrajectoryEntry",
    "WallTrajectoryEntry",
    "AirborneTrajectoryEntry",
    "TrajectoryCatalog",
    "JumpHullCollection",
    "WallHullCollection",
    "classify_surface",
    "get_or_build_hulls",
    "get_or_build_wall_hulls",
    "ReachabilityGraph",
    "ReachabilityPBRS",
    "build_reachability_pbrs",
]
