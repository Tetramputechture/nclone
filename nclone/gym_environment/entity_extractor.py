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
from ..entity_classes.entity_door_regular import EntityDoorRegular
from ..entity_classes.entity_door_locked import EntityDoorLocked
from ..entity_classes.entity_door_trap import EntityDoorTrap
from ..entity_classes.entity_one_way_platform import EntityOneWayPlatform


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

        # Regular doors (proximity activated)
        entities.extend(self._extract_regular_doors())

        # Locked doors using direct entity access
        entities.extend(self._extract_locked_doors())

        # Trap doors using direct entity access
        entities.extend(self._extract_trap_doors())

        # One-way platforms
        entities.extend(self._extract_one_way_platforms())

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

    def _extract_regular_doors(self) -> List[Dict[str, Any]]:
        """Extract regular doors (proximity activated)."""
        entities = []

        for d in self.nplay_headless.regular_doors():
            segment = getattr(d, "segment", None)
            if segment:
                door_x = (segment.x1 + segment.x2) * 0.5
                door_y = (segment.y1 + segment.y2) * 0.5
            else:
                door_x, door_y = d.xpos, d.ypos

            entities.append(
                {
                    "type": EntityType.REGULAR_DOOR,
                    "radius": EntityDoorRegular.RADIUS,
                    "x": door_x,  # Regular doors use door center as entity position
                    "y": door_y,
                    "active": getattr(d, "active", True),
                    "closed": getattr(d, "closed", True),
                    "state": 0.0,
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
                    "door_x": door_x,  # Door segment position
                    "door_y": door_y,  # Door segment position
                    "active": getattr(locked_door, "active", True),
                    "closed": getattr(locked_door, "closed", True),
                    "radius": EntityDoorRegular.RADIUS,
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

    def _extract_trap_doors(self) -> List[Dict[str, Any]]:
        """Extract trap doors using direct entity access."""
        entities = []

        for i, trap_door in enumerate(self.nplay_headless.trap_doors()):
            segment = getattr(trap_door, "segment", None)
            if segment:
                door_x = (segment.x1 + segment.x2) * 0.5
                door_y = (segment.y1 + segment.y2) * 0.5
            else:
                door_x, door_y = 0.0, 0.0

            entity_id = f"trap_{i}"

            # Add trap door switch part
            entities.append(
                {
                    "type": EntityType.TRAP_DOOR,
                    "radius": EntityDoorTrap.RADIUS,
                    "x": trap_door.xpos,  # Switch position (where entity is positioned)
                    "y": trap_door.ypos,  # Switch position
                    "door_x": door_x,  # Door segment position
                    "door_y": door_y,  # Door segment position
                    "active": getattr(trap_door, "active", True),
                    "closed": getattr(
                        trap_door, "closed", False
                    ),  # Trap doors start open
                    "state": 0.0 if getattr(trap_door, "active", True) else 1.0,
                    "entity_id": entity_id,
                    "is_door_part": False,  # This is the switch part
                    "entity_ref": trap_door,
                }
            )

            # Add trap door door part (at door segment position)
            entities.append(
                {
                    "type": EntityType.TRAP_DOOR,
                    "radius": EntityDoorTrap.RADIUS,
                    "x": door_x,  # Door segment position
                    "y": door_y,  # Door segment position
                    "door_x": door_x,  # Door segment position
                    "door_y": door_y,  # Door segment position
                    "active": getattr(trap_door, "active", True),
                    "closed": getattr(
                        trap_door, "closed", False
                    ),  # Trap doors start open
                    "state": 0.0 if getattr(trap_door, "active", True) else 1.0,
                    "entity_id": entity_id,
                    "is_door_part": True,  # This is the door part
                    "entity_ref": trap_door,
                }
            )

        return entities

    def _extract_one_way_platforms(self) -> List[Dict[str, Any]]:
        """Extract one-way platforms."""
        entities = []

        if hasattr(self.nplay_headless.sim, "entity_dic"):
            one_ways = self.nplay_headless.sim.entity_dic.get(11, [])
            for ow in one_ways:
                entities.append(
                    {
                        "type": EntityType.ONE_WAY,
                        "radius": EntityOneWayPlatform.SEMI_SIDE,
                        "x": getattr(ow, "xpos", 0.0),
                        "y": getattr(ow, "ypos", 0.0),
                        "orientation": getattr(ow, "orientation", 0),
                        "active": getattr(ow, "active", True),
                        "state": 0.0,
                    }
                )

        return entities
