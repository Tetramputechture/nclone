"""
Entity extraction utilities for N++ environment.

This module contains logic for extracting and processing entities from the game state
for use in graph construction and reachability analysis.
"""

from typing import List, Dict, Any
from ..constants.entity_types import EntityType
from ..constants.physics_constants import NINJA_RADIUS

# Entity classes
from ..entity_classes.entity_exit_switch import EntityExitSwitch
from ..entity_classes.entity_exit import EntityExit
from ..entity_classes.entity_door_locked import EntityDoorLocked


class EntityExtractor:
    """
    Handles extraction of entities from the game state.

    This class provides centralized logic for extracting entities from the N++
    game environment for use in graph construction and reachability analysis.
    """

    def __init__(self, nplay_headless):
        """
        Initialize the entity extractor.

        Args:
            nplay_headless: The NPlayHeadless instance to extract entities from
        """
        self.nplay_headless = nplay_headless

    def extract_graph_entities(self) -> List[Dict[str, Any]]:
        """
        Extract entities for graph construction.

        This is the centralized entity extraction logic used by both
        graph observations and debug visualization to ensure consistency.

        Entity Structure:
        - Switch entities (locked/trap doors): positioned at switch locations
        - Door segment entities: positioned at door geometry centers
        - Regular doors: positioned at door centers (proximity activated)
        - Exit doors/switches: positioned at their respective locations
        - One-way platforms: positioned at platform centers

        This ensures functional edges connect switches to door segments correctly.

        Returns:
            List of entity dictionaries with type, position, and state
        """
        entities = []

        # Add ninja as a special entity node
        ninja_pos = self.nplay_headless.ninja_position()
        entities.append(
            {
                "type": EntityType.NINJA,
                "radius": NINJA_RADIUS,
                "x": ninja_pos[0],
                "y": ninja_pos[1],
                "active": True,
                "state": 0.0,
                "entity_id": "ninja",
            }
        )

        # Exit doors and switches using direct entity relationships
        entities.extend(self._extract_exit_entities())

        # Locked doors using direct entity access
        entities.extend(self._extract_locked_doors())

        # Mines (toggle mines and regular mines)
        entities.extend(self._extract_mines())

        return entities

    def _extract_exit_entities(self) -> List[Dict[str, Any]]:
        """Extract exit doors and switches."""
        entities = []

        # Get exit entities from entity_dic key 3 (contains both EntityExit and EntityExitSwitch)
        if hasattr(self.nplay_headless.sim, "entity_dic"):
            exit_entities = self.nplay_headless.sim.entity_dic.get(3, [])

            # Find exit switch and door pairs
            exit_switches = [
                e for e in exit_entities if type(e).__name__ == "EntityExitSwitch"
            ]
            exit_doors = [e for e in exit_entities if type(e).__name__ == "EntityExit"]

            # Create switch-door pairs with matching entity IDs
            for i, switch in enumerate(exit_switches):
                switch_id = f"exit_pair_{i}"

                # Add exit switch entity
                entities.append(
                    {
                        "type": EntityType.EXIT_SWITCH,
                        "radius": EntityExitSwitch.RADIUS,
                        "x": switch.xpos,
                        "y": switch.ypos,
                        "active": getattr(switch, "active", True),
                        "state": 1.0 if getattr(switch, "active", True) else 0.0,
                        "entity_id": switch_id,
                        "entity_ref": switch,
                    }
                )

                # Find corresponding door
                if i < len(exit_doors):
                    door = exit_doors[i]
                    entities.append(
                        {
                            "type": EntityType.EXIT_DOOR,
                            "radius": EntityExit.RADIUS,
                            "x": door.xpos,
                            "y": door.ypos,
                            "active": getattr(door, "active", True),
                            "state": 1.0 if getattr(door, "active", True) else 0.0,
                            "entity_id": f"exit_door_{i}",
                            "switch_entity_id": switch_id,  # Link to switch
                            "entity_ref": door,
                        }
                    )

        return entities



    def _extract_locked_doors(self) -> List[Dict[str, Any]]:
        """Extract locked doors using direct entity access."""
        entities = []

        for i, locked_door in enumerate(self.nplay_headless.locked_doors()):
            segment = getattr(locked_door, "segment", None)
            if segment:
                door_x = (segment.x1 + segment.x2) * 0.5
                door_y = (segment.y1 + segment.y2) * 0.5
            else:
                door_x, door_y = 0.0, 0.0

            entity_id = f"locked_{i}"

            # Add locked door switch part
            entities.append(
                {
                    "type": EntityType.LOCKED_DOOR,
                    "x": locked_door.xpos,  # Switch position (where entity is positioned)
                    "y": locked_door.ypos,  # Switch position
                    "sw_xcoord": locked_door.xpos,  # Switch coordinates for reachability
                    "sw_ycoord": locked_door.ypos,  # Switch coordinates for reachability
                    "door_x": door_x,  # Door segment position
                    "door_y": door_y,  # Door segment position
                    "active": getattr(locked_door, "active", True),
                    "closed": getattr(locked_door, "closed", True),
                    "radius": EntityDoorLocked.RADIUS,
                    "state": 0.0 if getattr(locked_door, "active", True) else 1.0,
                    "entity_id": entity_id,
                    "is_door_part": False,  # This is the switch part
                    "entity_ref": locked_door,
                }
            )

            # Add locked door door part (at door segment position)
            entities.append(
                {
                    "type": EntityType.LOCKED_DOOR,
                    "radius": EntityDoorLocked.RADIUS,
                    "x": door_x,  # Door segment position
                    "y": door_y,  # Door segment position
                    "sw_xcoord": locked_door.xpos,  # Switch coordinates for reachability
                    "sw_ycoord": locked_door.ypos,  # Switch coordinates for reachability
                    "door_x": door_x,  # Door segment position
                    "door_y": door_y,  # Door segment position
                    "active": getattr(locked_door, "active", True),
                    "closed": getattr(locked_door, "closed", True),
                    "state": 0.0 if getattr(locked_door, "active", True) else 1.0,
                    "entity_id": entity_id,
                    "is_door_part": True,  # This is the door part
                    "entity_ref": locked_door,
                }
            )

        return entities



    def _extract_mines(self) -> List[Dict[str, Any]]:
        """Extract mines (toggle mines and regular mines)."""
        entities = []

        # Extract toggle mines (type 1)
        if hasattr(self.nplay_headless.sim, "entity_dic"):
            toggle_mines = self.nplay_headless.sim.entity_dic.get(1, [])
            for i, mine in enumerate(toggle_mines):
                entities.append(
                    {
                        "type": EntityType.TOGGLE_MINE,
                        "radius": 0.5,  # Standard mine radius
                        "x": getattr(mine, "xpos", 0.0),
                        "y": getattr(mine, "ypos", 0.0),
                        "active": getattr(mine, "active", True),
                        "state": 1.0 if getattr(mine, "active", True) else 0.0,
                        "entity_id": f"mine_{i}",
                        "entity_ref": mine,
                    }
                )

        return entities
