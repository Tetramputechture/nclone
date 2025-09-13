"""
Game mechanics handling for reachability analysis.

This module handles all game-specific mechanics including:
- Switch activation detection and state management
- Door blocking checks and unlocking logic
- Subgoal identification for hierarchical planning
- Entity-based traversability modifications
"""

from typing import List, Tuple

from ..common import SUB_CELL_SIZE
from ...constants.physics_constants import TILE_PIXEL_SIZE
from ...constants.entity_types import EntityType


class GameMechanics:
    """Handles game-specific mechanics for reachability analysis."""

    def __init__(self, debug: bool = False):
        """
        Initialize game mechanics handler.

        Args:
            debug: Enable debug output
        """
        self.debug = debug

    def check_switch_activation(
        self, level_data, sub_row: int, sub_col: int, reachability_state
    ):
        """
        Check if player can activate any switches at this position.

        Updates the reachability state with newly activated switches.

        Args:
            level_data: Level data containing entities
            sub_row: Current sub-grid row
            sub_col: Current sub-grid column
            reachability_state: Current reachability state (modified in-place)
        """
        pixel_x = sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        pixel_y = sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2

        for entity in level_data.entities:
            entity_type = entity.get("type")
            if entity_type in [
                EntityType.LOCKED_DOOR,
                EntityType.TRAP_DOOR,
                EntityType.EXIT_SWITCH,
            ]:
                # Check if player is close enough to activate switch
                switch_x = entity.get("x", 0)
                switch_y = entity.get("y", 0)

                # Activation range (within ~1 tile)
                if (
                    abs(pixel_x - switch_x) < TILE_PIXEL_SIZE
                    and abs(pixel_y - switch_y) < TILE_PIXEL_SIZE
                ):
                    entity_id = id(entity)
                    if not reachability_state.switch_states.get(entity_id, False):
                        # Activate the switch
                        reachability_state.switch_states[entity_id] = True
                        if self.debug:
                            print(
                                f"DEBUG: Activated switch at ({switch_x}, {switch_y}) - "
                                f"type {entity_type}"
                            )

    def is_position_blocked_by_door(
        self, level_data, pixel_x: float, pixel_y: float, reachability_state
    ) -> bool:
        """
        Check if position is blocked by a locked door.

        Args:
            level_data: Level data containing entities
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels
            reachability_state: Current reachability state

        Returns:
            True if position is blocked by a locked door
        """
        for entity in level_data.entities:
            if entity.get("type") == EntityType.LOCKED_DOOR:
                # Check if this door is still locked
                entity_id = id(entity)  # Use object id as unique identifier
                if not reachability_state.switch_states.get(entity_id, False):
                    # Door is locked, check if position overlaps with door
                    door_x = entity.get("door_x", entity.get("x", 0))
                    door_y = entity.get("door_y", entity.get("y", 0))

                    # Simple overlap check (could be refined)
                    half_tile = TILE_PIXEL_SIZE // 2
                    if (
                        abs(pixel_x - door_x) < half_tile
                        and abs(pixel_y - door_y) < half_tile
                    ):
                        return True
        return False

    def identify_subgoals(self, level_data, reachability_state):
        """
        Identify key subgoals for hierarchical navigation.

        Subgoals are entities that unlock new areas or represent key objectives:
        - Switches that unlock doors
        - Exit switches
        - Exit doors

        Updates the reachability state with identified subgoals.

        Args:
            level_data: Level data containing entities
            reachability_state: Current reachability state (modified in-place)
        """
        reachability_state.subgoals.clear()

        for entity in level_data.entities:
            entity_type = entity.get("type")
            entity_x = entity.get("x", 0)
            entity_y = entity.get("y", 0)

            # Convert to sub-grid coordinates
            sub_row = int(entity_y // SUB_CELL_SIZE)
            sub_col = int(entity_x // SUB_CELL_SIZE)

            # Check if entity is in reachable area
            if (sub_row, sub_col) in reachability_state.reachable_positions:
                subgoal_type = self._get_subgoal_type(entity_type)
                if subgoal_type:
                    reachability_state.subgoals.append((sub_row, sub_col, subgoal_type))

        # Sort subgoals by priority (switches before exits)
        priority_order = {
            "locked_door_switch": 1,
            "trap_door_switch": 2,
            "exit_switch": 3,
            "exit": 4,
        }
        reachability_state.subgoals.sort(key=lambda x: priority_order.get(x[2], 999))

        if self.debug:
            print(
                f"DEBUG: Identified {len(reachability_state.subgoals)} subgoals: "
                f"{[g[2] for g in reachability_state.subgoals]}"
            )

    def _get_subgoal_type(self, entity_type: int) -> str:
        """
        Map entity type to subgoal type string.

        Args:
            entity_type: Entity type ID

        Returns:
            Subgoal type string, or None if not a subgoal
        """
        subgoal_mapping = {
            EntityType.LOCKED_DOOR: "locked_door_switch",
            EntityType.TRAP_DOOR: "trap_door_switch",
            EntityType.EXIT_SWITCH: "exit_switch",
            EntityType.EXIT_DOOR: "exit",
        }
        return subgoal_mapping.get(entity_type, None)

    def get_unlocked_areas_by_switches(
        self, level_data, reachability_state
    ) -> List[Tuple[int, int]]:
        """
        Get list of areas that are unlocked by currently activated switches.

        Args:
            level_data: Level data containing entities
            reachability_state: Current reachability state

        Returns:
            List of (sub_row, sub_col) positions unlocked by switches
        """
        unlocked_areas = []

        for entity in level_data.entities:
            if entity.get("type") == EntityType.LOCKED_DOOR:
                entity_id = id(entity)
                # If switch is activated, the door area is unlocked
                if reachability_state.switch_states.get(entity_id, False):
                    door_x = entity.get("door_x", entity.get("x", 0))
                    door_y = entity.get("door_y", entity.get("y", 0))

                    # Convert door position to sub-grid coordinates
                    sub_row = int(door_y // SUB_CELL_SIZE)
                    sub_col = int(door_x // SUB_CELL_SIZE)
                    unlocked_areas.append((sub_row, sub_col))

        return unlocked_areas

    def is_entity_interactive(self, entity_type: int) -> bool:
        """
        Check if an entity type is interactive (can be activated by player).

        Args:
            entity_type: Entity type ID

        Returns:
            True if entity can be interacted with
        """
        interactive_types = {
            EntityType.LOCKED_DOOR,  # Switch can be activated
            EntityType.TRAP_DOOR,  # Switch can be activated
            EntityType.EXIT_SWITCH,  # Can be activated
            EntityType.EXIT_DOOR,  # Can be entered (goal)
        }
        return entity_type in interactive_types

    def is_entity_blocking(
        self, entity_type: int, entity: dict, reachability_state
    ) -> bool:
        """
        Check if an entity blocks movement at its position.

        Args:
            entity_type: Entity type ID
            entity: Entity data dictionary
            reachability_state: Current reachability state

        Returns:
            True if entity blocks movement
        """
        if entity_type == EntityType.LOCKED_DOOR:
            # Door blocks movement only if switch is not activated
            entity_id = id(entity)
            return not reachability_state.switch_states.get(entity_id, False)

        # Add other blocking entity logic here as needed
        # For now, most entities don't block movement
        return False
