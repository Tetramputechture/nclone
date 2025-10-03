"""
Mine State Processor for tracking mine states and danger levels.

This module provides functionality for tracking toggle mine states
and calculating safe/dangerous areas for hierarchical navigation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time

from ..constants.physics_constants import (
    TOGGLE_MINE_RADIUS_TOGGLED,
    TOGGLE_MINE_RADIUS_UNTOGGLED,
    TOGGLE_MINE_RADIUS_TOGGLING,
)
from ..constants.entity_types import EntityType


class MineState:
    """
    Represents the state of a single toggle mine.
    
    States:
    - 0: Toggled (deadly, 4px radius)
    - 1: Untoggled (safe, 3.5px radius)
    - 2: Toggling (transitioning, 4.5px radius)
    """
    
    TOGGLED = 0
    UNTOGGLED = 1
    TOGGLING = 2
    
    def __init__(
        self,
        mine_id: int,
        position: Tuple[float, float],
        state: int = UNTOGGLED,
    ):
        """
        Initialize mine state.
        
        Args:
            mine_id: Unique identifier for the mine
            position: (x, y) position in pixels
            state: Initial state (0=toggled, 1=untoggled, 2=toggling)
        """
        self.mine_id = mine_id
        self.position = position
        self.state = state
        self.last_update = time.time()
        self.time_in_state = 0.0
    
    @property
    def radius(self) -> float:
        """Get current mine radius based on state."""
        if self.state == self.TOGGLED:
            return TOGGLE_MINE_RADIUS_TOGGLED
        elif self.state == self.UNTOGGLED:
            return TOGGLE_MINE_RADIUS_UNTOGGLED
        else:  # TOGGLING
            return TOGGLE_MINE_RADIUS_TOGGLING
    
    @property
    def is_dangerous(self) -> bool:
        """Check if mine is currently dangerous."""
        return self.state in (self.TOGGLED, self.TOGGLING)
    
    @property
    def is_safe(self) -> bool:
        """Check if mine is safe to touch."""
        return self.state == self.UNTOGGLED
    
    def update_state(self, new_state: int) -> None:
        """
        Update mine state.
        
        Args:
            new_state: New state value
        """
        if new_state != self.state:
            self.state = new_state
            self.last_update = time.time()
            self.time_in_state = 0.0
        else:
            self.time_in_state = time.time() - self.last_update
    
    def distance_to(self, position: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance to a position.
        
        Args:
            position: (x, y) position in pixels
            
        Returns:
            Distance in pixels
        """
        dx = self.position[0] - position[0]
        dy = self.position[1] - position[1]
        return np.sqrt(dx * dx + dy * dy)
    
    def is_blocking_path(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        safety_margin: float = 2.0,
    ) -> bool:
        """
        Check if mine blocks a straight-line path.
        
        Args:
            start: Starting position (x, y)
            end: Ending position (x, y)
            safety_margin: Safety margin multiplier for mine radius
            
        Returns:
            True if mine blocks the path
        """
        if not self.is_dangerous:
            return False
        
        # Calculate distance from mine to line segment
        x0, y0 = self.position
        x1, y1 = start
        x2, y2 = end
        
        # Vector from start to end
        dx = x2 - x1
        dy = y2 - y1
        
        # If start and end are the same, just check distance
        if dx == 0 and dy == 0:
            return self.distance_to(start) < self.radius * safety_margin
        
        # Parameter t for closest point on line
        t = max(0, min(1, ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)))
        
        # Closest point on line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Distance from mine to closest point
        dist = self.distance_to((closest_x, closest_y))
        
        return dist < self.radius * safety_margin
    
    def to_vector(self, ninja_position: Tuple[float, float]) -> np.ndarray:
        """
        Convert mine state to feature vector.
        
        Returns:
            7D vector: [x, y, state, radius, time_in_state, distance, is_blocking]
        """
        distance = self.distance_to(ninja_position)
        
        return np.array([
            self.position[0],
            self.position[1],
            float(self.state),
            self.radius,
            self.time_in_state,
            distance,
            0.0,  # is_blocking will be set by context
        ], dtype=np.float32)


class MineStateProcessor:
    """
    Processes and tracks mine states for hierarchical navigation.
    
    This processor maintains a registry of all mines in the level,
    tracks their states, and provides queries for safe path planning.
    """
    
    def __init__(self, safety_radius: float = 2.0):
        """
        Initialize mine state processor.
        
        Args:
            safety_radius: Safety margin multiplier for mine danger radius
        """
        self.mine_states: Dict[int, MineState] = {}
        self.safety_radius = safety_radius
        self._last_update_time = 0.0
    
    def update_mine_states(self, entities: List[Any]) -> None:
        """
        Update mine states from entity list.
        
        Args:
            entities: List of game entities
        """
        current_time = time.time()
        self._last_update_time = current_time
        
        # Track which mines we've seen this update
        seen_mines = set()
        
        for entity in entities:
            # Check if entity is a toggle mine
            if not hasattr(entity, 'entity_type'):
                continue
            
            entity_type = entity.entity_type
            if entity_type not in (EntityType.TOGGLE_MINE, 1):  # 1 is TOGGLE_MINE value
                continue
            
            # Get mine ID (use position as ID if not available)
            mine_id = getattr(entity, 'id', None)
            if mine_id is None:
                # Use position tuple as ID
                x = getattr(entity, 'x', 0.0)
                y = getattr(entity, 'y', 0.0)
                mine_id = (int(x), int(y))
            
            seen_mines.add(mine_id)
            
            # Get mine position and state
            position = (
                getattr(entity, 'x', 0.0),
                getattr(entity, 'y', 0.0),
            )
            
            # Get mine state (0=toggled, 1=untoggled, 2=toggling)
            # Default to untoggled if state not available
            state = getattr(entity, 'state', MineState.UNTOGGLED)
            
            # Update or create mine state
            if mine_id in self.mine_states:
                self.mine_states[mine_id].update_state(state)
                # Update position in case mine moved (shouldn't happen but be safe)
                self.mine_states[mine_id].position = position
            else:
                self.mine_states[mine_id] = MineState(mine_id, position, state)
        
        # Remove mines that weren't seen (no longer in level)
        mines_to_remove = set(self.mine_states.keys()) - seen_mines
        for mine_id in mines_to_remove:
            del self.mine_states[mine_id]
    
    def get_dangerous_mines(
        self,
        ninja_position: Tuple[float, float],
        max_distance: Optional[float] = None,
    ) -> List[MineState]:
        """
        Get list of dangerous mines near the ninja.
        
        Args:
            ninja_position: Current ninja position (x, y)
            max_distance: Maximum distance to consider (None = all mines)
            
        Returns:
            List of dangerous mine states
        """
        dangerous = []
        
        for mine in self.mine_states.values():
            if not mine.is_dangerous:
                continue
            
            if max_distance is not None:
                distance = mine.distance_to(ninja_position)
                if distance > max_distance:
                    continue
            
            dangerous.append(mine)
        
        return dangerous
    
    def get_nearest_dangerous_mine(
        self,
        ninja_position: Tuple[float, float],
    ) -> Optional[MineState]:
        """
        Get the nearest dangerous mine to ninja position.
        
        Args:
            ninja_position: Current ninja position (x, y)
            
        Returns:
            Nearest dangerous mine or None if no dangerous mines
        """
        dangerous_mines = self.get_dangerous_mines(ninja_position)
        
        if not dangerous_mines:
            return None
        
        return min(
            dangerous_mines,
            key=lambda m: m.distance_to(ninja_position)
        )
    
    def get_mine_proximity_score(
        self,
        ninja_position: Tuple[float, float],
    ) -> float:
        """
        Calculate mine proximity threat score (0=safe, 1=very close).
        
        Args:
            ninja_position: Current ninja position (x, y)
            
        Returns:
            Proximity score in range [0, 1]
        """
        nearest_mine = self.get_nearest_dangerous_mine(ninja_position)
        
        if nearest_mine is None:
            return 0.0
        
        distance = nearest_mine.distance_to(ninja_position)
        danger_radius = nearest_mine.radius * self.safety_radius
        
        if distance >= danger_radius:
            return 0.0
        
        # Linear proximity score
        return 1.0 - (distance / danger_radius)
    
    def is_path_safe(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> bool:
        """
        Check if a straight-line path is safe from dangerous mines.
        
        Args:
            start: Starting position (x, y)
            end: Ending position (x, y)
            
        Returns:
            True if path is safe
        """
        for mine in self.mine_states.values():
            if mine.is_blocking_path(start, end, self.safety_radius):
                return False
        
        return True
    
    def get_mines_blocking_path(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> List[MineState]:
        """
        Get list of mines blocking a path.
        
        Args:
            start: Starting position (x, y)
            end: Ending position (x, y)
            
        Returns:
            List of mines blocking the path
        """
        blocking_mines = []
        
        for mine in self.mine_states.values():
            if mine.is_blocking_path(start, end, self.safety_radius):
                blocking_mines.append(mine)
        
        return blocking_mines
    
    def get_mine_features(
        self,
        ninja_position: Tuple[float, float],
        target_position: Optional[Tuple[float, float]] = None,
        max_mines: int = 5,
    ) -> np.ndarray:
        """
        Get feature vector for nearest mines.
        
        Args:
            ninja_position: Current ninja position (x, y)
            target_position: Optional target position for path blocking check
            max_mines: Maximum number of mines to include
            
        Returns:
            Feature array of shape (max_mines, 7) with mine features
        """
        # Get all mines sorted by distance
        all_mines = list(self.mine_states.values())
        all_mines.sort(key=lambda m: m.distance_to(ninja_position))
        
        # Take nearest max_mines
        nearest_mines = all_mines[:max_mines]
        
        # Create feature array
        features = np.zeros((max_mines, 7), dtype=np.float32)
        
        for i, mine in enumerate(nearest_mines):
            features[i] = mine.to_vector(ninja_position)
            
            # Set is_blocking flag if target provided
            if target_position is not None:
                is_blocking = mine.is_blocking_path(
                    ninja_position,
                    target_position,
                    self.safety_radius
                )
                features[i, 6] = float(is_blocking)
        
        return features
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics about mine states.
        
        Returns:
            Dictionary with mine statistics
        """
        total_mines = len(self.mine_states)
        dangerous_count = sum(1 for m in self.mine_states.values() if m.is_dangerous)
        safe_count = sum(1 for m in self.mine_states.values() if m.is_safe)
        
        return {
            'total_mines': total_mines,
            'dangerous_mines': dangerous_count,
            'safe_mines': safe_count,
            'last_update_time': self._last_update_time,
        }
    
    def reset(self) -> None:
        """Reset processor state (clear all mines)."""
        self.mine_states.clear()
        self._last_update_time = 0.0
