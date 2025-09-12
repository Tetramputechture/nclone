"""
Movement segment consolidation for N++ navigation.

This module consolidates consecutive movements of the same type to eliminate
micro-movements and create more natural, flowing movement sequences.
"""

import math
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from nclone.graph.movement_classifier import MovementType
from nclone.constants.physics_constants import TILE_PIXEL_SIZE


@dataclass
class MovementSegment:
    """Represents a consolidated movement segment."""
    movement_type: MovementType
    start_pos: Tuple[float, float]
    end_pos: Tuple[float, float]
    waypoints: List[Tuple[float, float]]
    total_distance: float
    total_time: float
    energy_cost: float
    risk_factor: float
    physics_params: Dict[str, Any]


class SegmentConsolidator:
    """Consolidates movement segments for more natural navigation."""
    
    def __init__(self):
        self.min_segment_distance = TILE_PIXEL_SIZE // 4  # 6 pixels minimum
        self.max_consolidation_distance = TILE_PIXEL_SIZE * 8  # 8 tiles maximum
        
    def consolidate_path(
        self,
        path_nodes: List[Tuple[float, float]],
        movement_types: List[MovementType],
        physics_params: List[Dict[str, Any]],
    ) -> List[MovementSegment]:
        """
        Consolidate a path into efficient movement segments.
        
        Args:
            path_nodes: List of waypoint positions
            movement_types: Movement type for each segment
            physics_params: Physics parameters for each segment
            
        Returns:
            List of consolidated movement segments
        """
        if len(path_nodes) < 2:
            return []
        
        if len(movement_types) != len(path_nodes) - 1:
            raise ValueError("Movement types must match number of path segments")
        
        consolidated_segments = []
        current_segment_start = 0
        
        i = 0
        while i < len(movement_types):
            # Find the end of the current segment type
            current_type = movement_types[i]
            segment_end = self._find_segment_end(
                movement_types, i, current_type, path_nodes
            )
            
            # Create consolidated segment
            segment = self._create_consolidated_segment(
                path_nodes[current_segment_start:segment_end + 2],  # +2 because we need end node
                current_type,
                physics_params[i:segment_end + 1]
            )
            
            if segment:
                consolidated_segments.append(segment)
            
            # Move to next segment
            current_segment_start = segment_end + 1
            i = segment_end + 1
        
        # Post-process to merge very short segments
        return self._merge_short_segments(consolidated_segments)
    
    def _find_segment_end(
        self,
        movement_types: List[MovementType],
        start_idx: int,
        segment_type: MovementType,
        path_nodes: List[Tuple[float, float]],
    ) -> int:
        """Find the end index of a segment of the same movement type."""
        current_idx = start_idx
        total_distance = 0
        
        while (current_idx < len(movement_types) and 
               movement_types[current_idx] == segment_type):
            
            # Calculate distance for this sub-segment
            if current_idx + 1 < len(path_nodes):
                x0, y0 = path_nodes[current_idx]
                x1, y1 = path_nodes[current_idx + 1]
                distance = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
                total_distance += distance
                
                # Stop consolidation if segment becomes too long
                if total_distance > self.max_consolidation_distance:
                    break
            
            current_idx += 1
        
        return current_idx - 1
    
    def _create_consolidated_segment(
        self,
        segment_nodes: List[Tuple[float, float]],
        movement_type: MovementType,
        segment_physics: List[Dict[str, Any]],
    ) -> Optional[MovementSegment]:
        """Create a consolidated segment from multiple waypoints."""
        if len(segment_nodes) < 2:
            return None
        
        start_pos = segment_nodes[0]
        end_pos = segment_nodes[-1]
        
        # Calculate total distance
        total_distance = 0
        for i in range(len(segment_nodes) - 1):
            x0, y0 = segment_nodes[i]
            x1, y1 = segment_nodes[i + 1]
            total_distance += math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        
        # Skip very short segments unless they're critical waypoints
        if (total_distance < self.min_segment_distance and 
            not self._is_critical_waypoint(start_pos, end_pos, movement_type)):
            return None
        
        # Aggregate physics parameters
        total_time = sum(params.get('time_estimate', 0) for params in segment_physics)
        total_energy = sum(params.get('energy_cost', 0) for params in segment_physics)
        max_risk = max(params.get('difficulty', 0) for params in segment_physics)
        
        # Create consolidated physics parameters
        consolidated_physics = {
            'total_distance': total_distance,
            'total_time': total_time,
            'total_energy': total_energy,
            'max_risk': max_risk,
            'segment_count': len(segment_nodes) - 1,
            'waypoint_count': len(segment_nodes),
        }
        
        # Add movement-specific parameters
        if movement_type == MovementType.JUMP:
            consolidated_physics.update(self._consolidate_jump_params(segment_physics))
        elif movement_type == MovementType.WALK:
            consolidated_physics.update(self._consolidate_walk_params(segment_physics))
        elif movement_type == MovementType.FALL:
            consolidated_physics.update(self._consolidate_fall_params(segment_physics))
        
        return MovementSegment(
            movement_type=movement_type,
            start_pos=start_pos,
            end_pos=end_pos,
            waypoints=segment_nodes[1:-1],  # Intermediate waypoints only
            total_distance=total_distance,
            total_time=total_time,
            energy_cost=total_energy,
            risk_factor=max_risk,
            physics_params=consolidated_physics
        )
    
    def _is_critical_waypoint(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        movement_type: MovementType,
    ) -> bool:
        """Check if this is a critical waypoint that shouldn't be consolidated."""
        # Direction changes are critical
        # Entity interactions are critical
        # Movement type changes are critical (handled by caller)
        
        # For now, keep all segments - more sophisticated logic could be added
        return True
    
    def _consolidate_jump_params(self, segment_physics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate physics parameters for jump segments."""
        if not segment_physics:
            return {}
        
        # For jumps, we care about maximum velocity and total airtime
        max_velocity = max(
            params.get('required_velocity', 0) for params in segment_physics
        )
        total_airtime = sum(
            params.get('airtime', 0) for params in segment_physics
        )
        
        return {
            'max_required_velocity': max_velocity,
            'total_airtime': total_airtime,
            'jump_type': 'consolidated_jump'
        }
    
    def _consolidate_walk_params(self, segment_physics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate physics parameters for walk segments."""
        if not segment_physics:
            return {}
        
        # For walking, we care about total ground time and average speed
        total_ground_time = sum(
            params.get('time_estimate', 0) for params in segment_physics
        )
        
        return {
            'total_ground_time': total_ground_time,
            'walk_type': 'consolidated_walk'
        }
    
    def _consolidate_fall_params(self, segment_physics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate physics parameters for fall segments."""
        if not segment_physics:
            return {}
        
        # For falls, we care about total fall time and landing velocity
        total_fall_time = sum(
            params.get('time_estimate', 0) for params in segment_physics
        )
        max_landing_velocity = max(
            params.get('landing_velocity', 0) for params in segment_physics
        )
        
        return {
            'total_fall_time': total_fall_time,
            'max_landing_velocity': max_landing_velocity,
            'fall_type': 'consolidated_fall'
        }
    
    def _merge_short_segments(
        self, segments: List[MovementSegment]
    ) -> List[MovementSegment]:
        """Merge very short segments with adjacent compatible segments."""
        if len(segments) <= 1:
            return segments
        
        merged_segments = []
        i = 0
        
        while i < len(segments):
            current_segment = segments[i]
            
            # Check if current segment is very short
            if (current_segment.total_distance < self.min_segment_distance and 
                i + 1 < len(segments)):
                
                next_segment = segments[i + 1]
                
                # Try to merge with next segment if compatible
                if self._can_merge_segments(current_segment, next_segment):
                    merged_segment = self._merge_two_segments(current_segment, next_segment)
                    merged_segments.append(merged_segment)
                    i += 2  # Skip both segments
                    continue
            
            merged_segments.append(current_segment)
            i += 1
        
        return merged_segments
    
    def _can_merge_segments(
        self, segment1: MovementSegment, segment2: MovementSegment
    ) -> bool:
        """Check if two segments can be merged."""
        # Can merge if same movement type
        if segment1.movement_type == segment2.movement_type:
            return True
        
        # Can merge WALK and FALL in some cases
        if ({segment1.movement_type, segment2.movement_type} == 
            {MovementType.WALK, MovementType.FALL}):
            return True
        
        return False
    
    def _merge_two_segments(
        self, segment1: MovementSegment, segment2: MovementSegment
    ) -> MovementSegment:
        """Merge two compatible segments."""
        # Use the more complex movement type
        movement_type = segment1.movement_type
        if segment2.movement_type != segment1.movement_type:
            # Prefer JUMP > FALL > WALK for mixed segments
            type_priority = {
                MovementType.JUMP: 3,
                MovementType.FALL: 2,
                MovementType.WALK: 1,
                MovementType.COMBO: 4,
            }
            if type_priority.get(segment2.movement_type, 0) > type_priority.get(segment1.movement_type, 0):
                movement_type = segment2.movement_type
        
        # Combine waypoints
        all_waypoints = [segment1.start_pos] + segment1.waypoints + [segment1.end_pos]
        all_waypoints.extend(segment2.waypoints + [segment2.end_pos])
        
        # Remove duplicate waypoints
        unique_waypoints = [all_waypoints[0]]
        for waypoint in all_waypoints[1:]:
            if waypoint != unique_waypoints[-1]:
                unique_waypoints.append(waypoint)
        
        # Calculate merged parameters
        total_distance = segment1.total_distance + segment2.total_distance
        total_time = segment1.total_time + segment2.total_time
        total_energy = segment1.energy_cost + segment2.energy_cost
        max_risk = max(segment1.risk_factor, segment2.risk_factor)
        
        merged_physics = {
            'total_distance': total_distance,
            'total_time': total_time,
            'total_energy': total_energy,
            'max_risk': max_risk,
            'merged_from': [segment1.movement_type, segment2.movement_type],
        }
        
        return MovementSegment(
            movement_type=movement_type,
            start_pos=segment1.start_pos,
            end_pos=segment2.end_pos,
            waypoints=unique_waypoints[1:-1],  # Intermediate waypoints
            total_distance=total_distance,
            total_time=total_time,
            energy_cost=total_energy,
            risk_factor=max_risk,
            physics_params=merged_physics
        )