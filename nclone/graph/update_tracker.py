"""
State change detection system for selective graph updates.

This module tracks entity state changes to determine when graph features
need updating. Only critical entities are tracked:
- Toggle mines (state changes)
- Locked doors (opened/closed)
- Exit switch (activated)
- Exit door (accessible)
"""

from dataclasses import dataclass, field
from typing import Dict, Set, Any, List
from ..constants.entity_types import EntityType


@dataclass
class GraphUpdateInfo:
    """Tracks which graph elements need updating."""

    needs_full_rebuild: bool = False
    needs_reachability_update: bool = False
    updated_entity_ids: Set[str] = field(default_factory=set)
    updated_node_indices: Set[int] = field(default_factory=set)
    connectivity_changed: bool = False  # Doors opened/closed


class StateChangeDetector:
    """Detects entity state changes requiring graph updates."""

    def __init__(self):
        self.previous_entity_states: Dict[str, Any] = {}

    def detect_changes(
        self,
        current_entities: List[Dict],
        entity_node_indices: Dict[str, int],
    ) -> GraphUpdateInfo:
        """
        Compare current state with previous to determine updates needed.

        Only tracks critical entities:
        - Toggle mines (state changes)
        - Locked doors (opened/closed)
        - Exit switch (activated)
        - Exit door (accessible)

        Args:
            current_entities: Current entity list from level_data
            entity_node_indices: Mapping from entity_id to node_index

        Returns:
            GraphUpdateInfo indicating what needs updating
        """
        update_info = GraphUpdateInfo()

        # Build current state dict
        current_states = {}
        for entity in current_entities:
            entity_id = entity.get("entity_id")
            if not entity_id:
                continue

            entity_type = entity.get("type")

            if entity_type in [EntityType.TOGGLE_MINE, EntityType.TOGGLE_MINE_TOGGLED]:
                current_states[entity_id] = entity.get("state", 1)
            elif entity_type == EntityType.LOCKED_DOOR:
                current_states[entity_id] = entity.get("closed", True)
            elif entity_type == EntityType.EXIT_SWITCH:
                current_states[entity_id] = entity.get("active", True)
            elif entity_type == EntityType.EXIT_DOOR:
                current_states[entity_id] = entity.get("switch_hit", False)

        # First run - initialize
        if not self.previous_entity_states:
            self.previous_entity_states = current_states
            return update_info

        # Detect changes
        for entity_id, current_state in current_states.items():
            previous_state = self.previous_entity_states.get(entity_id)

            if previous_state != current_state:
                update_info.updated_entity_ids.add(entity_id)

                # Map to node indices
                if entity_id in entity_node_indices:
                    node_idx = entity_node_indices[entity_id]
                    update_info.updated_node_indices.add(node_idx)

                # Check if connectivity changed (doors)
                # Locked door opened/closed affects reachability
                if entity_id.startswith("locked_"):
                    update_info.connectivity_changed = True
                    update_info.needs_reachability_update = True

        # Update stored state
        self.previous_entity_states = current_states

        return update_info

    def reset(self):
        """Reset state tracking (e.g., when starting a new level)."""
        self.previous_entity_states.clear()
