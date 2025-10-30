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
        self.entities = level_data.get('entities', [])
        self.switch_states = level_data.get('switch_states', {})
        
        # Parse entities by type (positions are cached - never change!)
        self.doors = self._extract_doors()
        self.mines = self._extract_mines()
        
        # Build switch->door mapping (static per level)
        self.switch_to_door = self._build_switch_door_mapping()
    
    def _extract_doors(self) -> List[Dict]:
        """Extract locked door entities only."""
        doors = []
        for entity in self.entities:
            entity_type = entity.get('type')
            if entity_type == EntityType.LOCKED_DOOR:
                doors.append(entity)
        return doors
    
    def _extract_mines(self) -> List[Dict]:
        """Extract toggle mine entities."""
        mines = []
        for entity in self.entities:
            entity_type = entity.get('type')
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
            if door.get('type') == EntityType.LOCKED_DOOR:
                switch_id = door.get('controlled_by') or door.get('switch_id')
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
        switch_id = door.get('controlled_by') or door.get('switch_id')
        if switch_id:
            # Door open if switch collected (one-way: never closes again)
            return not self.switch_states.get(switch_id, False)
        return door.get('closed', True)  # Default closed
    
    def _is_mine_deadly(self, mine: Dict) -> bool:
        """
        Check if toggle mine is in deadly (toggled) state.
        
        State mapping:
        - 0: Toggled (deadly) - blocks movement
        - 1: Untoggled (safe) - passable
        - 2: Toggling (transitioning) - passable but preparing
        
        TOGGLE_MINE_TOGGLED entities start toggled and NEVER become untoggled.
        """
        entity_type = mine.get('type')
        
        # TOGGLE_MINE_TOGGLED are permanently toggled (deadly)
        if entity_type == EntityType.TOGGLE_MINE_TOGGLED:
            return True
        
        # TOGGLE_MINE entities can change state
        state = mine.get('state', 1)  # Default untoggled (safe)
        return state == 0  # Only state 0 is deadly
    
    def _get_door_tile_positions(self, door: Dict) -> List[Tuple[int, int]]:
        """Get all tile positions occupied by door segments."""
        pos = door.get('position', (0, 0))
        orientation = door.get('orientation', 0)
        
        # Door length is CELL_SIZE pixels (1 tile)
        # Doors can be horizontal (orientation 0) or vertical (orientation 1)
        tile_x = int(pos[0] // CELL_SIZE)
        tile_y = int(pos[1] // CELL_SIZE)
        
        positions = [(tile_x, tile_y)]
        
        # Doors may span adjacent tile depending on precise position
        if orientation == 0:  # Horizontal
            # Check if extends to adjacent tile
            if (pos[0] % CELL_SIZE) + CELL_SIZE > CELL_SIZE:  # Extends past tile boundary
                positions.append((tile_x + 1, tile_y))
        else:  # Vertical
            if (pos[1] % CELL_SIZE) + CELL_SIZE > CELL_SIZE:
                positions.append((tile_x, tile_y + 1))
        
        return positions
    
    def _get_entity_tile_position(self, entity: Dict) -> Optional[Tuple[int, int]]:
        """Get tile position for point entity (mine, switch, etc.)."""
        pos = entity.get('position')
        if pos is None:
            return None
        
        tile_x = int(pos[0] // CELL_SIZE)
        tile_y = int(pos[1] // CELL_SIZE)
        return (tile_x, tile_y)
    
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
            if mine.get('id') == mine_id or mine.get('entity_id') == mine_id:
                mine['state'] = state
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
            'total_doors': total_doors,
            'closed_doors': closed_doors,
            'total_mines': total_mines,
            'deadly_mines': deadly_mines,
            'blocked_tile_count': len(blocked_positions),
            'blocked_edge_count': len(blocked_edges),
            'switch_count': len(self.switch_states),
            'active_switches': sum(1 for v in self.switch_states.values() if v)
        }
