"""
Helper functions for extracting goal positions and mine state signatures from LevelData.

These functions are used for cache invalidation and goal identification
in pathfinding systems.
"""

from typing import List, Tuple
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


def get_mine_state_signature(level_data: LevelData) -> Tuple:
    """
    Get a signature of current mine states for cache invalidation.

    Args:
        level_data: LevelData instance containing entities

    Returns:
        Tuple of sorted (mine_position, mine_state) tuples
    """
    mines = []

    # Extract toggle mines (both TOGGLE_MINE and TOGGLE_MINE_TOGGLED)
    toggle_mines = level_data.get_all_toggle_mines()

    for mine in toggle_mines:
        x = mine.get("x", 0)
        y = mine.get("y", 0)
        # Get mine state: 0=toggled, 1=untoggled, 2=toggling
        # For TOGGLE_MINE_TOGGLED (type 21), initial state is 0 (toggled)
        # For TOGGLE_MINE (type 1), initial state is 1 (untoggled)
        entity_type = mine.get("type", EntityType.TOGGLE_MINE)

        # Try to get state from entity_ref if available (more accurate)
        state = None
        if "entity_ref" in mine and hasattr(mine["entity_ref"], "state"):
            state = mine["entity_ref"].state
        elif "state" in mine:
            # Check if state is the actual mine state (0, 1, or 2) or converted (0.0/1.0)
            state_val = mine["state"]
            if isinstance(state_val, (int, float)):
                if state_val in (0, 1, 2):
                    state = int(state_val)
                elif state_val == 1.0:
                    # This might be "active" converted to state - check entity type
                    if entity_type == EntityType.TOGGLE_MINE_TOGGLED:
                        state = 0  # Starts toggled
                    else:
                        state = 1  # Default to untoggled
                elif state_val == 0.0:
                    state = 0  # Toggled

        # Fallback based on entity type
        if state is None:
            if entity_type == EntityType.TOGGLE_MINE_TOGGLED:
                state = 0  # Starts toggled
            else:
                state = 1  # Default to untoggled

        mines.append(((int(x), int(y)), state))

    # Return sorted tuple for consistent comparison
    return tuple(sorted(mines))
