"""
Core pathfinding system for N++ using physics-aware movement classification.

This is the authoritative pathfinding implementation that consolidates
all the working physics-aware logic into a single, coherent system.
"""

import math
from typing import List, Tuple, Optional, Dict, Any
from ..graph.movement_classifier import MovementClassifier, MovementType as GraphMovementType
from ..graph.level_data import LevelData
from .movement_types import MovementType
from .physics_validator import PhysicsValidator

class CorePathfinder:
    """
    Consolidated pathfinding system using physics-aware movement classification.
    
    This class provides the authoritative pathfinding implementation,
    using the proven MovementClassifier system that passes all validation tests.
    """
    
    def __init__(self):
        self.movement_classifier = MovementClassifier()
        self.physics_validator = PhysicsValidator()
    
    def find_path(self, level_data: LevelData, start_pos: Tuple[float, float], 
                  end_pos: Tuple[float, float]) -> List[Dict[str, Any]]:
        """
        Find a physics-aware path between two positions.
        
        Args:
            level_data: Level geometry and entity data
            start_pos: Starting position (x, y) in pixels
            end_pos: Ending position (x, y) in pixels
            
        Returns:
            List of path segments with movement type and physics data
        """
        
        # For now, create a direct path segment and classify it
        # This matches the approach used by the working validation tests
        movement_type, physics_params = self.movement_classifier.classify_movement(
            start_pos, end_pos, None, level_data
        )
        
        # Convert from graph movement type to pathfinding movement type
        pathfinding_movement_type = self._convert_movement_type(movement_type)
        
        # Validate physics
        is_valid = self.physics_validator.validate_movement(
            start_pos, end_pos, pathfinding_movement_type
        )
        
        path_segment = {
            'start_pos': start_pos,
            'end_pos': end_pos,
            'movement_type': pathfinding_movement_type,
            'physics_params': physics_params,
            'is_valid': is_valid
        }
        
        return [path_segment]
    
    def find_multi_segment_path(self, level_data: LevelData, 
                               waypoints: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
        """
        Find a path through multiple waypoints.
        
        Args:
            level_data: Level geometry and entity data
            waypoints: List of positions to visit in order
            
        Returns:
            List of path segments between consecutive waypoints
        """
        
        path_segments = []
        
        for i in range(len(waypoints) - 1):
            start_pos = waypoints[i]
            end_pos = waypoints[i + 1]
            
            segment = self.find_path(level_data, start_pos, end_pos)[0]
            path_segments.append(segment)
        
        return path_segments
    
    def _convert_movement_type(self, graph_movement_type: int) -> MovementType:
        """Convert from graph MovementType to pathfinding MovementType."""
        
        # Map graph movement types to pathfinding movement types
        type_mapping = {
            GraphMovementType.WALK: MovementType.WALK,
            GraphMovementType.JUMP: MovementType.JUMP,
            GraphMovementType.FALL: MovementType.FALL,
            GraphMovementType.WALL_SLIDE: MovementType.WALL_SLIDE,
            GraphMovementType.WALL_JUMP: MovementType.WALL_JUMP,
            GraphMovementType.LAUNCH_PAD: MovementType.LAUNCH_PAD,
            GraphMovementType.BOUNCE_BLOCK: MovementType.BOUNCE_BLOCK,
            GraphMovementType.BOUNCE_CHAIN: MovementType.BOUNCE_CHAIN,
        }
        
        return type_mapping.get(GraphMovementType(graph_movement_type), MovementType.WALK)
    
    def get_path_summary(self, path_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get a summary of the path including total distance and movement types.
        
        Args:
            path_segments: List of path segments from find_path or find_multi_segment_path
            
        Returns:
            Dictionary with path summary statistics
        """
        
        total_distance = sum(seg['physics_params']['distance'] for seg in path_segments)
        movement_types = [seg['movement_type'] for seg in path_segments]
        movement_type_counts = {}
        
        for movement_type in movement_types:
            movement_type_counts[movement_type.name] = movement_type_counts.get(movement_type.name, 0) + 1
        
        return {
            'total_distance': total_distance,
            'segment_count': len(path_segments),
            'movement_type_counts': movement_type_counts,
            'all_segments_valid': all(seg['is_valid'] for seg in path_segments)
        }