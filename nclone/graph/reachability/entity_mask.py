"""
Entity-based dynamic masking for path-aware reward shaping.

Handles dynamic entity states (doors, mines) that change during gameplay,
applying them as lightweight masks over the static tile connectivity graph.
"""

from typing import Dict, Set, Tuple, List, Any, Optional
from ...constants.entity_types import EntityType

# Hardcoded cell size as per N++ constants
CELL_SIZE = 24


class EntityMask:
    """
    Manages dynamic entity states and generates blocking masks.

    Handles:
    - Locked doors (open/closed based on switch collection)
    - Toggle mines (deadly/safe/toggling states)

    Key insight: Entity POSITIONS never change during a run, only STATES change.
    This allows efficient caching of position-based structures.
    """

    def __init__(self, level_data: Dict[str, Any]):
        """
        Initialize entity mask from level data.

        Args:
            level_data: Level data containing tiles, entities, switch_states
        """
        self.entities = level_data.get("entities", [])
        self.switch_states = level_data.get("switch_states", {})

        # Parse entities by type (positions are cached - never change!)
        self.doors = self._extract_doors()
        self.mines = self._extract_mines()

        # Build switch->door mapping (static per level)
        self.switch_to_door = self._build_switch_door_mapping()

    def _extract_doors(self) -> List[Dict]:
        """Extract locked door entities only."""
        doors = []
        for entity in self.entities:
            entity_type = entity.get("type")
            if entity_type == EntityType.LOCKED_DOOR:
                doors.append(entity)
        return doors

    def _extract_mines(self) -> List[Dict]:
        """Extract toggle mine entities."""
        mines = []
        for entity in self.entities:
            entity_type = entity.get("type")
            # Toggle mines: TOGGLE_MINE (untoggled) and TOGGLE_MINE_TOGGLED (toggled)
            if entity_type in [EntityType.TOGGLE_MINE, EntityType.TOGGLE_MINE_TOGGLED]:
                mines.append(entity)
        return mines

    def _build_switch_door_mapping(self) -> Dict[str, Dict]:
        """
        Build switch ID -> door mapping.
        One switch always corresponds to one locked door (1:1 relationship).
        """
        mapping = {}

        for door in self.doors:
            if door.get("type") == EntityType.LOCKED_DOOR:
                switch_id = door.get("controlled_by") or door.get("switch_id")
                if switch_id:
                    mapping[switch_id] = door

        return mapping

    def get_blocked_positions(self) -> Set[Tuple[int, int]]:
        """
        Get set of tile positions currently blocked by entities.

        Returns:
            Set of (tile_x, tile_y) positions that are blocked
        """
        blocked = set()

        # Check doors
        for door in self.doors:
            if self._is_door_closed(door):
                door_tiles = self._get_door_tile_positions(door)
                blocked.update(door_tiles)

        # Check mines (only deadly state blocks)
        for mine in self.mines:
            if self._is_mine_deadly(mine):
                mine_tile = self._get_entity_tile_position(mine)
                if mine_tile:
                    blocked.add(mine_tile)

        return blocked

    def get_blocked_pixel_positions(
        self, all_nodes: Set[Tuple[int, int]]
    ) -> Set[Tuple[int, int]]:
        """
        Get set of pixel positions (sub-nodes) blocked by entities.

        Uses radius-based blocking for mines:
        - Finds all graph nodes within (mine_radius + ninja_radius) of deadly mines
        - Blocks entire tiles for closed doors

        Args:
            all_nodes: Set of all graph node positions in pixel coordinates

        Returns:
            Set of (pixel_x, pixel_y) positions that are blocked
        """
        from ...constants.physics_constants import NINJA_RADIUS

        blocked = set()

        # Block doors (tile-based, all 4 sub-nodes per tile)
        blocked_tiles = set()
        for door in self.doors:
            if self._is_door_closed(door):
                door_tiles = self._get_door_tile_positions(door)
                blocked_tiles.update(door_tiles)

        # Convert blocked tiles to sub-node pixel positions
        sub_offsets = [(6, 6), (18, 6), (6, 18), (18, 18)]
        for tile_x, tile_y in blocked_tiles:
            for offset_x, offset_y in sub_offsets:
                pixel_x = tile_x * CELL_SIZE + offset_x
                pixel_y = tile_y * CELL_SIZE + offset_y
                blocked.add((pixel_x, pixel_y))

        # Block mines (radius-based, node-by-node)
        for mine in self.mines:
            if self._is_mine_deadly(mine):
                mine_pos = self._get_entity_pixel_position(mine)
                if mine_pos is None:
                    continue

                mine_x, mine_y = mine_pos
                mine_radius = mine.get("radius", 4.0)

                # Block all nodes within collision distance
                # Collision occurs when: distance(ninja_center, mine_center) < ninja_radius + mine_radius
                block_radius = NINJA_RADIUS + mine_radius
                block_radius_sq = block_radius * block_radius

                for node_pos in all_nodes:
                    nx, ny = node_pos
                    dist_sq = (nx - mine_x) ** 2 + (ny - mine_y) ** 2

                    if dist_sq < block_radius_sq:
                        blocked.add(node_pos)

        return blocked

    def get_blocked_edges(self) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Get set of edges blocked by entity orientations.

        Returns:
            Set of ((tile_x1, tile_y1), (tile_x2, tile_y2)) edge tuples
        """
        # For now, only nodes are blocked, not edges
        # Could be extended in future for directional blocking
        return set()

    def _is_door_closed(self, door: Dict) -> bool:
        """Check if locked door is currently closed (blocking)."""
        # Only handling EntityType.LOCKED_DOOR
        switch_id = door.get("controlled_by") or door.get("switch_id")
        if switch_id:
            # Door open if switch collected (one-way: never closes again)
            return not self.switch_states.get(switch_id, False)
        return door.get("closed", True)  # Default closed

    def _is_mine_deadly(self, mine: Dict) -> bool:
        """
        Check if toggle mine is in deadly (toggled) state.

        State mapping:
        - 0: Toggled (deadly) - blocks movement
        - 1: Untoggled (safe) - passable
        - 2: Toggling (transitioning) - passable but preparing

        TOGGLE_MINE_TOGGLED entities start toggled and NEVER become untoggled.
        """
        entity_type = mine.get("type")

        # TOGGLE_MINE_TOGGLED are permanently toggled (deadly)
        if entity_type == EntityType.TOGGLE_MINE_TOGGLED:
            return True

        # TOGGLE_MINE entities can change state
        state = mine.get("state", 1)  # Default untoggled (safe)

        # Handle both numeric and float states
        if isinstance(state, (int, float)):
            return int(state) == 0  # Only state 0 is deadly

        return False

    def _get_door_tile_positions(self, door: Dict) -> List[Tuple[int, int]]:
        """Get all tile positions occupied by door segments in inner tile space.

        Door positions from entities are in world space (with 1-tile padding).
        Convert to inner space by subtracting NODE_WORLD_COORD_OFFSET.
        """
        from .pathfinding_utils import NODE_WORLD_COORD_OFFSET

        pos = door.get("position", (0, 0))
        orientation = door.get("orientation", 0)

        # Convert from world space to inner space
        inner_x = pos[0] - NODE_WORLD_COORD_OFFSET
        inner_y = pos[1] - NODE_WORLD_COORD_OFFSET

        # Door length is CELL_SIZE pixels (1 tile)
        # Doors can be horizontal (orientation 0) or vertical (orientation 1)
        tile_x = int(inner_x // CELL_SIZE)
        tile_y = int(inner_y // CELL_SIZE)

        positions = [(tile_x, tile_y)]

        # Doors may span adjacent tile depending on precise position
        if orientation == 0:  # Horizontal
            # Check if extends to adjacent tile
            if (
                inner_x % CELL_SIZE
            ) + CELL_SIZE > CELL_SIZE:  # Extends past tile boundary
                positions.append((tile_x + 1, tile_y))
        else:  # Vertical
            if (inner_y % CELL_SIZE) + CELL_SIZE > CELL_SIZE:
                positions.append((tile_x, tile_y + 1))

        return positions

    def _get_entity_tile_position(self, entity: Dict) -> Optional[Tuple[int, int]]:
        """Get tile position for point entity (mine, switch, etc.) in inner tile space.

        Entities from simulator are in world space (with 1-tile padding).
        Tiles array is in inner space (without padding).
        Convert by subtracting NODE_WORLD_COORD_OFFSET (24px) before dividing by CELL_SIZE.
        """
        from .pathfinding_utils import NODE_WORLD_COORD_OFFSET

        # Handle both formats: dict with 'x'/'y' keys or 'position' tuple
        if "position" in entity:
            pos = entity["position"]
            tile_x = int((pos[0] - NODE_WORLD_COORD_OFFSET) // CELL_SIZE)
            tile_y = int((pos[1] - NODE_WORLD_COORD_OFFSET) // CELL_SIZE)
        elif "x" in entity and "y" in entity:
            tile_x = int((entity["x"] - NODE_WORLD_COORD_OFFSET) // CELL_SIZE)
            tile_y = int((entity["y"] - NODE_WORLD_COORD_OFFSET) // CELL_SIZE)
        else:
            return None

        return (tile_x, tile_y)

    def _get_entity_pixel_position(self, entity: Dict) -> Optional[Tuple[int, int]]:
        """Get pixel position for point entity in graph coordinate space.

        Entities from simulator are in world space (with 1-tile padding).
        Graph nodes are in inner space (without padding).
        Convert by subtracting NODE_WORLD_COORD_OFFSET (24px).
        """
        from .pathfinding_utils import NODE_WORLD_COORD_OFFSET

        if "position" in entity:
            pos = entity["position"]
            return (
                int(pos[0]) - NODE_WORLD_COORD_OFFSET,
                int(pos[1]) - NODE_WORLD_COORD_OFFSET,
            )
        elif "x" in entity and "y" in entity:
            return (
                int(entity["x"]) - NODE_WORLD_COORD_OFFSET,
                int(entity["y"]) - NODE_WORLD_COORD_OFFSET,
            )
        return None

    def update_switch_state(self, switch_id: str, activated: bool):
        """
        Update switch state (affects corresponding door states).

        Args:
            switch_id: Switch identifier
            activated: True if switch collected, False otherwise
        """
        self.switch_states[switch_id] = activated

    def update_mine_state(self, mine_id: str, state: int):
        """
        Update toggle mine state.

        Args:
            mine_id: Mine identifier
            state: New state (0=toggled/deadly, 1=untoggled/safe, 2=toggling)
        """
        # Find mine and update its state
        for mine in self.mines:
            if mine.get("id") == mine_id or mine.get("entity_id") == mine_id:
                mine["state"] = state
                break

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about entities and current states."""
        total_doors = len(self.doors)
        closed_doors = sum(1 for door in self.doors if self._is_door_closed(door))

        total_mines = len(self.mines)
        deadly_mines = sum(1 for mine in self.mines if self._is_mine_deadly(mine))

        blocked_positions = self.get_blocked_positions()
        blocked_edges = self.get_blocked_edges()

        return {
            "total_doors": total_doors,
            "closed_doors": closed_doors,
            "total_mines": total_mines,
            "deadly_mines": deadly_mines,
            "blocked_tile_count": len(blocked_positions),
            "blocked_edge_count": len(blocked_edges),
            "switch_count": len(self.switch_states),
            "active_switches": sum(1 for v in self.switch_states.values() if v),
        }
