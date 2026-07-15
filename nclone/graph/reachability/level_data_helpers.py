"""
Helper functions for extracting goal positions and mine state signatures from LevelData.

These functions are used for cache invalidation and goal identification
in pathfinding systems.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from ..level_data import LevelData
from ...constants.entity_types import EntityType


def extract_goal_positions(level_data: LevelData) -> List[Tuple[Tuple[int, int], str]]:
    """Extract Layer 1 pathfinding goals from level data.

    Layer 1 scope (toggle-mines chunk): only the exit switch and exit door are
    first-class goals. ``LOCKED_DOOR_SWITCH`` is intentionally excluded — locked
    doors are not part of the toggle-mines-world policy and have no associated
    A* goal here. If locked doors come back into scope they should be reintroduced
    behind a feature flag rather than by resurrecting this branch in place.

    Returns:
        List of ``(position, goal_id)`` tuples where position is ``(x, y)`` in
        pixels and ``goal_id`` is a stable per-goal string used by the A*
        distance cache.
    """
    goals: List[Tuple[Tuple[int, int], str]] = []

    # Exit switches: only include if still active (not collected).
    exit_switches = level_data.get_entities_by_type(EntityType.EXIT_SWITCH)
    for i, switch in enumerate(exit_switches):
        is_active = switch.get("active", True)
        if not is_active:
            continue
        x = switch.get("x", 0)
        y = switch.get("y", 0)
        if x == 0 and y == 0:
            raise ValueError(
                f"EXIT SWITCH at (0, 0) - entity not properly loaded! "
                f"Switch index: {i}, Entity data: {switch}. "
                f"This should have been caught in entity_extractor.py. "
                f"Level file may be corrupted or entity initialization failed."
            )
        goals.append(((int(x), int(y)), f"exit_switch_{i}"))

    # Exit doors: always include.
    exit_doors = level_data.get_entities_by_type(EntityType.EXIT_DOOR)
    for i, door in enumerate(exit_doors):
        x = door.get("x", 0)
        y = door.get("y", 0)
        if x == 0 and y == 0:
            raise ValueError(
                f"EXIT DOOR at (0, 0) - entity not properly loaded! "
                f"Door index: {i}, Entity data: {door}. "
                f"This should have been caught in entity_extractor.py. "
                f"Level file may be corrupted or entity initialization failed."
            )
        goals.append(((int(x), int(y)), f"exit_door_{i}"))

    return goals


def _extract_mine_state(mine: Dict[str, Any]) -> int:
    """Resolve a toggle mine's discrete state (0=toggled/deadly, 1=untoggled,
    2=toggling). Prefers the live ``entity_ref.state`` (native/python entity),
    falling back to the serialized ``state`` field or the entity-type default."""
    # Live entity reference is authoritative (always present on the native backend).
    ref = mine.get("entity_ref")
    if ref is not None and hasattr(ref, "state"):
        return int(ref.state)

    entity_type = mine.get("type", EntityType.TOGGLE_MINE)
    if "state" in mine:
        state_val = mine["state"]
        if isinstance(state_val, (int, float)):
            if state_val in (0, 1, 2):
                return int(state_val)
            if state_val == 1.0:
                return 0 if entity_type == EntityType.TOGGLE_MINE_TOGGLED else 1
            if state_val == 0.0:
                return 0
    # Entity-type default: TOGGLE_MINE_TOGGLED (21) starts toggled (deadly).
    return 0 if entity_type == EntityType.TOGGLE_MINE_TOGGLED else 1


def get_mine_state_signature(level_data: LevelData) -> Tuple:
    """
    Get a signature of current mine states for cache invalidation.

    Mine POSITIONS are static for a given level (callers detect level changes via
    ``level_id`` before comparing this signature), so the signature only needs to
    capture the per-mine STATE vector in the stable index order. Reading the
    states into a compact ``int8`` array and returning its bytes is ~14x faster
    than building and sorting ``(x, y, state)`` tuples — which dominated per-step
    reward-cache invalidation on mine-dense levels (13k+ mines ~= 18ms/call, run
    twice per step). The signature is used only for equality comparison, so this
    is a faithful drop-in: identical states -> identical signature; any toggle ->
    a different signature.

    Args:
        level_data: LevelData instance containing entities

    Returns:
        ``(mine_count, state_bytes)`` — hashable and comparable for equality.
    """
    # Extract toggle mines (both TOGGLE_MINE and TOGGLE_MINE_TOGGLED)
    toggle_mines = level_data.get_all_toggle_mines()
    n = len(toggle_mines)
    if n == 0:
        return (0, b"")

    states = np.fromiter(
        (_extract_mine_state(mine) for mine in toggle_mines),
        dtype=np.int8,
        count=n,
    )
    return (n, states.tobytes())
