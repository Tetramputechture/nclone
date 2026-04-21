"""Programmatic optimal-route planner for trivial N++ levels.

Public entry point :func:`plan_optimal_route`. Given a raw ``map_data``
list (as produced by :class:`nclone.map_generation.map.Map`), the planner
attempts to classify the map into one of two supported "trivial" families,
computes the minimum-frame action sequence for that family, and
(optionally) verifies that the plan completes the level in the real
simulator.

Supported topologies
--------------------

* ``wall_jump_shaft`` - single-tile-wide vertical chamber (one empty column)
  with the exit above (or below) the spawn. Planned via
  :class:`WallJumpShaftPlanner`.
* ``flat_run_corridor`` - wide, short chamber with the switch and exit
  door on the same floor as the spawn. Planned via
  :class:`FlatRunCorridorPlanner` (evaluates ``direct_run``,
  ``running_jump_cruise`` and ``wall_sling`` candidates).

Returns ``None`` for maps that do not match either family (including any
map with non-flat walls, mines, or locked/trap doors inside the ninja's
chamber).
"""

from __future__ import annotations

from typing import List, Optional

from .flat_run_corridor import FlatRunCorridorPlanner
from .geometry import (
    LevelGeometry,
    TOPOLOGY_FLAT_RUN_CORRIDOR,
    TOPOLOGY_WALL_JUMP_SHAFT,
    classify,
    extract_level_geometry,
)
from .physics import NinjaState, step
from .types import ActionFrame, OptimalRoute, OptimalRoutePlanner
from .verifier import replay_and_verify
from .wall_jump_shaft import WallJumpShaftPlanner


__all__ = [
    "ActionFrame",
    "FlatRunCorridorPlanner",
    "LevelGeometry",
    "NinjaState",
    "OptimalRoute",
    "OptimalRoutePlanner",
    "TOPOLOGY_FLAT_RUN_CORRIDOR",
    "TOPOLOGY_WALL_JUMP_SHAFT",
    "WallJumpShaftPlanner",
    "classify",
    "extract_level_geometry",
    "plan_optimal_route",
    "replay_and_verify",
    "step",
]


def plan_optimal_route(
    map_data: List[int], verify: bool = True
) -> Optional[OptimalRoute]:
    """Compute and (optionally) verify an optimal route through ``map_data``.

    Args:
        map_data: Raw map-data list (length >= 1235), as produced by
            :meth:`nclone.map_generation.map.Map.map_data`.
        verify: If ``True`` (default), the returned
            :class:`~nclone.planning.optimal_route.types.OptimalRoute` has
            :attr:`~OptimalRoute.verified_in_sim` / :attr:`~OptimalRoute.verified_frames`
            set by replaying the action sequence through a real
            :class:`nclone.nsim.Simulator`. Disable for speed when planning
            a large batch of levels.

    Returns:
        An :class:`OptimalRoute` when the map is a supported trivial
        topology; ``None`` when the map cannot be parsed or does not match
        any supported family (non-flat walls, mines, locked doors, or
        unusual chamber shapes).
    """
    geom = extract_level_geometry(map_data)
    if geom is None:
        return None

    topology = classify(geom)
    if topology is None:
        return None

    planner: OptimalRoutePlanner
    if topology == TOPOLOGY_WALL_JUMP_SHAFT:
        planner = WallJumpShaftPlanner()
    elif topology == TOPOLOGY_FLAT_RUN_CORRIDOR:
        planner = FlatRunCorridorPlanner()
    else:
        return None

    route = planner.plan(geom)

    if verify and route.action_sequence:
        won, frames = replay_and_verify(map_data, route.action_sequence)
        return OptimalRoute(
            map_type=route.map_type,
            strategy=route.strategy,
            optimal_frames=route.optimal_frames,
            action_sequence=route.action_sequence,
            verified_in_sim=won,
            verified_frames=frames,
            notes=route.notes,
        )

    return route
