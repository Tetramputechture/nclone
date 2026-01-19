"""
Helper functions for extracting goal positions and mine state signatures from LevelData.

These functions are used for cache invalidation and goal identification
in pathfinding systems.
"""

from typing import List, Tuple
from ..level_data import LevelData
from ...constants.entity_types import EntityType


def extract_goal_positions(level_data: LevelData) -> List[Tuple[Tuple[int, int], str]]:
    """
    Extract goal positions from level data.

    Matches the exact logic used in debug_overlay_renderer.py visualization:
    - Exit switches: Only include if active=True (not collected)
    - Exit doors: Always include
    - Locked door switches: Always include

    Args:
        level_data: LevelData instance containing entities

    Returns:
        List of (position, goal_id) tuples where position is (x, y) in pixels
    """
    import logging
    _logger = logging.getLogger(__name__)
    
    goals = []

    # Extract exit switches (only active ones - matches visualization logic)
    exit_switches = level_data.get_entities_by_type(EntityType.EXIT_SWITCH)
    for i, switch in enumerate(exit_switches):
        # Check if switch is still active (not collected)
        # In nclone: active=True means NOT collected, active=False means collected
        is_active = switch.get("active", True)
        if is_active:
            x = switch.get("x", 0)
            y = switch.get("y", 0)
            # CRITICAL: Validate goal position - should never be (0, 0)
            if x == 0 and y == 0:
                raise ValueError(
                    f"EXIT SWITCH at (0, 0) - entity not properly loaded! "
                    f"Switch index: {i}, Entity data: {switch}. "
                    f"This should have been caught in entity_extractor.py. "
                    f"Level file may be corrupted or entity initialization failed."
                )
            goal_id = f"exit_switch_{i}"
            goals.append(((int(x), int(y)), goal_id))

    # Extract exit doors (always include - matches visualization logic)
    exit_doors = level_data.get_entities_by_type(EntityType.EXIT_DOOR)
    for i, door in enumerate(exit_doors):
        x = door.get("x", 0)
        y = door.get("y", 0)
        # CRITICAL: Validate goal position - should never be (0, 0)
        if x == 0 and y == 0:
            raise ValueError(
                f"EXIT DOOR at (0, 0) - entity not properly loaded! "
                f"Door index: {i}, Entity data: {door}. "
                f"This should have been caught in entity_extractor.py. "
                f"Level file may be corrupted or entity initialization failed."
            )
        goal_id = f"exit_door_{i}"
        goals.append(((int(x), int(y)), goal_id))

    # Extract locked door switches (always include)
    locked_door_switches = level_data.get_entities_by_type(
        EntityType.LOCKED_DOOR_SWITCH
    )
    for i, switch in enumerate(locked_door_switches):
        x = switch.get("x", 0)
        y = switch.get("y", 0)
        # CRITICAL: Validate goal position - should never be (0, 0)
        if x == 0 and y == 0:
            raise ValueError(
                f"LOCKED DOOR SWITCH at (0, 0) - entity not properly loaded! "
                f"Switch index: {i}, Entity data: {switch}. "
                f"This should have been caught in entity_extractor.py. "
                f"Level file may be corrupted or entity initialization failed."
            )
        goal_id = f"locked_door_switch_{i}"
        goals.append(((int(x), int(y)), goal_id))

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
