"""
Shared utilities for the planning system.

This module provides common functionality used across multiple planning components
to avoid code duplication and improve maintainability.
"""

from typing import Dict, Tuple, Optional
from ..constants.entity_types import EntityType


def is_switch_activated_authoritative(switch_id: str, level_data, switch_states: Dict) -> bool:
    """
    Check switch activation using authoritative simulation data first.
    Falls back to passed switch_states if simulation data unavailable.
    
    Uses actual NppEnvironment data structures from nclone.
    """
    # Method 1: Check level_data.entities for switch with matching entity_id
    if hasattr(level_data, 'entities') and level_data.entities:
        for entity in level_data.entities:
            if (entity.get('entity_id') == switch_id and 
                entity.get('type') == EntityType.EXIT_SWITCH):
                # For exit switches, activated means active=False (inverted logic in nclone)
                return not entity.get('active', True)
    
    # Method 2: Check if level_data has direct switch state info (from environment observation)
    if hasattr(level_data, 'switch_activated'):
        # This is the direct boolean from NppEnvironment observation
        return level_data.switch_activated
    
    # Method 3: Fall back to passed switch_states (legacy compatibility)
    return switch_states.get(switch_id, False)


def find_switch_id_by_position(position: Tuple[float, float], level_data) -> Optional[str]:
    """Find switch ID by position using actual NppEnvironment data structures."""
    # Check level_data.entities for switches near the position
    if hasattr(level_data, 'entities') and level_data.entities:
        for entity in level_data.entities:
            if entity.get('type') == EntityType.EXIT_SWITCH:
                entity_x = entity.get('x', 0)
                entity_y = entity.get('y', 0)
                
                # Check if this entity is at the target position (within radius)
                if (abs(entity_x - position[0]) < 12.0 and 
                    abs(entity_y - position[1]) < 12.0):
                    return entity.get('entity_id')
    
    return None


def calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two positions."""
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5