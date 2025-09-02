"""
Precise tile collision detection system for graph traversability.

This module implements segment-based collision detection using the detailed
tile geometry definitions from tile_definitions.py, replacing the simplified
boolean tile checking with accurate physics-based collision testing.
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import IntEnum

from ..constants.physics_constants import TILE_PIXEL_SIZE, NINJA_RADIUS
from ..tile_definitions import (
    TILE_GRID_EDGE_MAP, TILE_SEGMENT_ORTHO_MAP, 
    TILE_SEGMENT_DIAG_MAP, TILE_SEGMENT_CIRCULAR_MAP
)


class SegmentType(IntEnum):
    """Types of collision segments."""
    ORTHOGONAL = 0  # Horizontal or vertical line segments
    DIAGONAL = 1    # Diagonal line segments
    CIRCULAR = 2    # Circular arc segments


@dataclass
class CollisionSegment:
    """Represents a collision segment in world coordinates."""
    segment_type: SegmentType
    start_x: float
    start_y: float
    end_x: float = 0.0
    end_y: float = 0.0
    center_x: float = 0.0
    center_y: float = 0.0
    radius: float = 0.0
    normal_x: float = 0.0
    normal_y: float = 0.0
    is_convex: bool = True  # For circular segments


@dataclass
class CollisionResult:
    """Result of collision detection."""
    collision: bool
    time_of_impact: float = 1.0
    contact_x: float = 0.0
    contact_y: float = 0.0
    normal_x: float = 0.0
    normal_y: float = 0.0


class PreciseTileCollision:
    """
    Precise tile collision detection using segment-based geometry.
    
    This class converts tile definitions to world-space collision segments
    and performs swept-circle collision detection for accurate pathfinding.
    """
    
    def __init__(self):
        """Initialize precise collision detector."""
        # Cache for converted tile segments (tile_id -> List[CollisionSegment])
        self._tile_segment_cache = {}
        # Cache for level tile segments (level_id -> Dict[(tile_x, tile_y), List[CollisionSegment]])
        self._level_segment_cache = {}
        self._current_level_id = None
    
    def is_path_traversable(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        level_data: Dict[str, Any],
        ninja_radius: float = NINJA_RADIUS
    ) -> bool:
        """
        Check if a path is traversable using precise tile collision detection.
        
        Args:
            src_x: Source x coordinate
            src_y: Source y coordinate
            tgt_x: Target x coordinate
            tgt_y: Target y coordinate
            level_data: Level tile data and structure
            ninja_radius: Ninja collision radius
            
        Returns:
            True if path is traversable, False if blocked by tile geometry
        """
        # Calculate movement vector
        dx = tgt_x - src_x
        dy = tgt_y - src_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 1e-6:  # No movement
            return True
        
        # Get all tile segments that could intersect the movement path
        segments = self._get_segments_for_path(src_x, src_y, tgt_x, tgt_y, level_data)
        
        # Test swept circle collision against all segments
        return self._test_swept_circle_vs_segments(
            src_x, src_y, dx, dy, ninja_radius, segments
        )
    
    def _get_segments_for_path(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        level_data: Dict[str, Any]
    ) -> List[CollisionSegment]:
        """
        Get all tile segments that could intersect with the movement path.
        
        Args:
            src_x: Source x coordinate
            src_y: Source y coordinate
            tgt_x: Target x coordinate
            tgt_y: Target y coordinate
            level_data: Level tile data
            
        Returns:
            List of collision segments in world coordinates
        """
        # Cache level segments for performance
        level_id = level_data.get('level_id', id(level_data))
        if self._current_level_id != level_id:
            self._cache_level_segments(level_data)
            self._current_level_id = level_id
        
        # Calculate bounding box of movement path with ninja radius padding
        min_x = min(src_x, tgt_x) - NINJA_RADIUS
        max_x = max(src_x, tgt_x) + NINJA_RADIUS
        min_y = min(src_y, tgt_y) - NINJA_RADIUS
        max_y = max(src_y, tgt_y) + NINJA_RADIUS
        
        # Convert to tile coordinates
        min_tile_x = int(min_x // TILE_PIXEL_SIZE)
        max_tile_x = int(max_x // TILE_PIXEL_SIZE) + 1
        min_tile_y = int(min_y // TILE_PIXEL_SIZE)
        max_tile_y = int(max_y // TILE_PIXEL_SIZE) + 1
        
        # Collect segments from tiles in the bounding box
        segments = []
        for tile_y in range(min_tile_y, max_tile_y):
            for tile_x in range(min_tile_x, max_tile_x):
                tile_segments = self._level_segment_cache.get((tile_x, tile_y), [])
                segments.extend(tile_segments)
        
        return segments
    
    def _cache_level_segments(self, level_data: Dict[str, Any]) -> None:
        """
        Cache all tile segments for the current level.
        
        Args:
            level_data: Level tile data
        """
        self._level_segment_cache = {}
        tiles = level_data.get('tiles', None)
        
        if tiles is None:
            return
        
        # Handle different tile data formats
        if hasattr(tiles, 'shape') and len(tiles.shape) == 2:
            # NumPy array format
            height, width = tiles.shape
            for tile_y in range(height):
                for tile_x in range(width):
                    tile_id = tiles[tile_y, tile_x]
                    if tile_id != 0:  # Non-empty tile
                        segments = self._get_tile_segments(tile_id, tile_x, tile_y)
                        if segments:
                            self._level_segment_cache[(tile_x, tile_y)] = segments
        elif isinstance(tiles, (list, tuple)):
            # List/tuple format
            for tile_y, row in enumerate(tiles):
                if hasattr(row, '__getitem__'):
                    for tile_x, tile_id in enumerate(row):
                        if tile_id != 0:  # Non-empty tile
                            segments = self._get_tile_segments(tile_id, tile_x, tile_y)
                            if segments:
                                self._level_segment_cache[(tile_x, tile_y)] = segments
        elif isinstance(tiles, dict):
            # Dictionary format
            for (tile_x, tile_y), tile_id in tiles.items():
                if tile_id != 0:
                    segments = self._get_tile_segments(tile_id, tile_x, tile_y)
                    if segments:
                        self._level_segment_cache[(tile_x, tile_y)] = segments
    
    def _get_tile_segments(
        self,
        tile_id: int,
        tile_x: int,
        tile_y: int
    ) -> List[CollisionSegment]:
        """
        Get collision segments for a specific tile in world coordinates.
        
        Args:
            tile_id: Tile type ID (0-37)
            tile_x: Tile x coordinate
            tile_y: Tile y coordinate
            
        Returns:
            List of collision segments in world coordinates
        """
        # Check cache first
        if tile_id in self._tile_segment_cache:
            cached_segments = self._tile_segment_cache[tile_id]
        else:
            cached_segments = self._convert_tile_definition_to_segments(tile_id)
            self._tile_segment_cache[tile_id] = cached_segments
        
        # Convert to world coordinates
        world_segments = []
        tile_world_x = tile_x * TILE_PIXEL_SIZE
        tile_world_y = tile_y * TILE_PIXEL_SIZE
        
        for segment in cached_segments:
            world_segment = CollisionSegment(
                segment_type=segment.segment_type,
                start_x=segment.start_x + tile_world_x,
                start_y=segment.start_y + tile_world_y,
                end_x=segment.end_x + tile_world_x,
                end_y=segment.end_y + tile_world_y,
                center_x=segment.center_x + tile_world_x,
                center_y=segment.center_y + tile_world_y,
                radius=segment.radius,
                normal_x=segment.normal_x,
                normal_y=segment.normal_y,
                is_convex=segment.is_convex
            )
            world_segments.append(world_segment)
        
        return world_segments
    
    def _convert_tile_definition_to_segments(self, tile_id: int) -> List[CollisionSegment]:
        """
        Convert tile definition data to collision segments in tile-local coordinates.
        
        Args:
            tile_id: Tile type ID (0-37)
            
        Returns:
            List of collision segments in tile-local coordinates
        """
        segments = []
        
        # Process orthogonal segments
        if tile_id in TILE_SEGMENT_ORTHO_MAP:
            ortho_data = TILE_SEGMENT_ORTHO_MAP[tile_id]
            segments.extend(self._create_orthogonal_segments(ortho_data))
        
        # Process diagonal segments
        if tile_id in TILE_SEGMENT_DIAG_MAP:
            diag_data = TILE_SEGMENT_DIAG_MAP[tile_id]
            segments.append(self._create_diagonal_segment(diag_data))
        
        # Process circular segments
        if tile_id in TILE_SEGMENT_CIRCULAR_MAP:
            circular_data = TILE_SEGMENT_CIRCULAR_MAP[tile_id]
            segments.append(self._create_circular_segment(circular_data))
        
        return segments
    
    def _create_orthogonal_segments(self, ortho_data: List[int]) -> List[CollisionSegment]:
        """
        Create orthogonal line segments from tile definition data.
        
        Args:
            ortho_data: List of 12 integers representing orthogonal segments
            
        Returns:
            List of orthogonal collision segments
        """
        segments = []
        
        # Horizontal segments (first 6 values)
        # Top edge segments
        if ortho_data[0] != 0:  # Left half of top edge
            segments.append(CollisionSegment(
                segment_type=SegmentType.ORTHOGONAL,
                start_x=0, start_y=0, end_x=12, end_y=0,
                normal_x=0, normal_y=-1 if ortho_data[0] == -1 else 1
            ))
        if ortho_data[1] != 0:  # Right half of top edge
            segments.append(CollisionSegment(
                segment_type=SegmentType.ORTHOGONAL,
                start_x=12, start_y=0, end_x=24, end_y=0,
                normal_x=0, normal_y=-1 if ortho_data[1] == -1 else 1
            ))
        
        # Bottom edge segments
        if ortho_data[4] != 0:  # Left half of bottom edge
            segments.append(CollisionSegment(
                segment_type=SegmentType.ORTHOGONAL,
                start_x=0, start_y=24, end_x=12, end_y=24,
                normal_x=0, normal_y=-1 if ortho_data[4] == -1 else 1
            ))
        if ortho_data[5] != 0:  # Right half of bottom edge
            segments.append(CollisionSegment(
                segment_type=SegmentType.ORTHOGONAL,
                start_x=12, start_y=24, end_x=24, end_y=24,
                normal_x=0, normal_y=-1 if ortho_data[5] == -1 else 1
            ))
        
        # Vertical segments (last 6 values)
        # Left edge segments
        if ortho_data[6] != 0:  # Top half of left edge
            segments.append(CollisionSegment(
                segment_type=SegmentType.ORTHOGONAL,
                start_x=0, start_y=0, end_x=0, end_y=12,
                normal_x=-1 if ortho_data[6] == -1 else 1, normal_y=0
            ))
        if ortho_data[7] != 0:  # Bottom half of left edge
            segments.append(CollisionSegment(
                segment_type=SegmentType.ORTHOGONAL,
                start_x=0, start_y=12, end_x=0, end_y=24,
                normal_x=-1 if ortho_data[7] == -1 else 1, normal_y=0
            ))
        
        # Right edge segments
        if ortho_data[10] != 0:  # Top half of right edge
            segments.append(CollisionSegment(
                segment_type=SegmentType.ORTHOGONAL,
                start_x=24, start_y=0, end_x=24, end_y=12,
                normal_x=-1 if ortho_data[10] == -1 else 1, normal_y=0
            ))
        if ortho_data[11] != 0:  # Bottom half of right edge
            segments.append(CollisionSegment(
                segment_type=SegmentType.ORTHOGONAL,
                start_x=24, start_y=12, end_x=24, end_y=24,
                normal_x=-1 if ortho_data[11] == -1 else 1, normal_y=0
            ))
        
        return segments
    
    def _create_diagonal_segment(self, diag_data: Tuple[Tuple[int, int], Tuple[int, int]]) -> CollisionSegment:
        """
        Create diagonal line segment from tile definition data.
        
        Args:
            diag_data: Tuple of ((x1, y1), (x2, y2)) coordinates
            
        Returns:
            Diagonal collision segment
        """
        (x1, y1), (x2, y2) = diag_data
        
        # Calculate normal vector (perpendicular to segment, pointing outward)
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            normal_x = -dy / length  # Perpendicular vector
            normal_y = dx / length
        else:
            normal_x = normal_y = 0
        
        return CollisionSegment(
            segment_type=SegmentType.DIAGONAL,
            start_x=x1, start_y=y1, end_x=x2, end_y=y2,
            normal_x=normal_x, normal_y=normal_y
        )
    
    def _create_circular_segment(
        self, 
        circular_data: Tuple[Tuple[int, int], Tuple[int, int], bool]
    ) -> CollisionSegment:
        """
        Create circular arc segment from tile definition data.
        
        Args:
            circular_data: Tuple of ((center_x, center_y), (quadrant_x, quadrant_y), is_convex)
            
        Returns:
            Circular collision segment
        """
        (center_x, center_y), (quad_x, quad_y), is_convex = circular_data
        
        return CollisionSegment(
            segment_type=SegmentType.CIRCULAR,
            center_x=center_x, center_y=center_y,
            radius=12.0,  # Quarter-tile radius (half of 24-pixel tile)
            normal_x=quad_x, normal_y=quad_y,  # Quadrant direction
            is_convex=is_convex
        )
    
    def _test_swept_circle_vs_segments(
        self,
        start_x: float,
        start_y: float,
        dx: float,
        dy: float,
        radius: float,
        segments: List[CollisionSegment]
    ) -> bool:
        """
        Test swept circle collision against a list of segments.
        
        Args:
            start_x: Starting x position
            start_y: Starting y position
            dx: Movement delta x
            dy: Movement delta y
            radius: Circle radius
            segments: List of collision segments to test
            
        Returns:
            True if path is clear, False if collision detected
        """
        for segment in segments:
            collision_result = self._test_swept_circle_vs_segment(
                start_x, start_y, dx, dy, radius, segment
            )
            if collision_result.collision and collision_result.time_of_impact < 1.0:
                return False  # Path is blocked
        
        return True  # Path is clear
    
    def _test_swept_circle_vs_segment(
        self,
        start_x: float,
        start_y: float,
        dx: float,
        dy: float,
        radius: float,
        segment: CollisionSegment
    ) -> CollisionResult:
        """
        Test swept circle collision against a single segment.
        
        Args:
            start_x: Starting x position
            start_y: Starting y position
            dx: Movement delta x
            dy: Movement delta y
            radius: Circle radius
            segment: Collision segment to test
            
        Returns:
            CollisionResult with collision information
        """
        if segment.segment_type == SegmentType.ORTHOGONAL:
            return self._test_circle_vs_line_segment(
                start_x, start_y, dx, dy, radius,
                segment.start_x, segment.start_y, segment.end_x, segment.end_y
            )
        elif segment.segment_type == SegmentType.DIAGONAL:
            return self._test_circle_vs_line_segment(
                start_x, start_y, dx, dy, radius,
                segment.start_x, segment.start_y, segment.end_x, segment.end_y
            )
        elif segment.segment_type == SegmentType.CIRCULAR:
            return self._test_circle_vs_arc_segment(
                start_x, start_y, dx, dy, radius, segment
            )
        else:
            return CollisionResult(collision=False)
    
    def _test_circle_vs_line_segment(
        self,
        start_x: float,
        start_y: float,
        dx: float,
        dy: float,
        radius: float,
        seg_x1: float,
        seg_y1: float,
        seg_x2: float,
        seg_y2: float
    ) -> CollisionResult:
        """
        Test swept circle collision against a line segment.
        
        This implements the swept circle vs line segment collision algorithm
        similar to get_time_of_intersection_circle_vs_lineseg in physics.py.
        
        Args:
            start_x: Circle starting x position
            start_y: Circle starting y position
            dx: Circle movement delta x
            dy: Circle movement delta y
            radius: Circle radius
            seg_x1: Line segment start x
            seg_y1: Line segment start y
            seg_x2: Line segment end x
            seg_y2: Line segment end y
            
        Returns:
            CollisionResult with collision information
        """
        # Vector from segment start to circle start
        to_circle_x = start_x - seg_x1
        to_circle_y = start_y - seg_y1
        
        # Segment vector
        seg_dx = seg_x2 - seg_x1
        seg_dy = seg_y2 - seg_y1
        seg_length_sq = seg_dx*seg_dx + seg_dy*seg_dy
        
        if seg_length_sq < 1e-6:  # Degenerate segment (point)
            # Test circle vs point
            dist_sq = to_circle_x*to_circle_x + to_circle_y*to_circle_y
            if dist_sq <= radius*radius:
                return CollisionResult(collision=True, time_of_impact=0.0)
            else:
                return CollisionResult(collision=False)
        
        # Project circle position onto segment
        t = (to_circle_x*seg_dx + to_circle_y*seg_dy) / seg_length_sq
        t = max(0.0, min(1.0, t))  # Clamp to segment
        
        # Closest point on segment
        closest_x = seg_x1 + t * seg_dx
        closest_y = seg_y1 + t * seg_dy
        
        # Vector from closest point to circle
        to_circle_x = start_x - closest_x
        to_circle_y = start_y - closest_y
        
        # Solve quadratic equation for swept circle collision
        # ||(start + t*velocity) - closest_point|| = radius
        a = dx*dx + dy*dy
        b = 2.0 * (to_circle_x*dx + to_circle_y*dy)
        c = to_circle_x*to_circle_x + to_circle_y*to_circle_y - radius*radius
        
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            return CollisionResult(collision=False)  # No collision
        
        if abs(a) < 1e-6:  # No movement
            if c <= 0:
                return CollisionResult(collision=True, time_of_impact=0.0)
            else:
                return CollisionResult(collision=False)
        
        # Calculate collision time
        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2*a)
        t2 = (-b + sqrt_discriminant) / (2*a)
        
        # Use earliest positive collision time
        collision_time = None
        if 0 <= t1 <= 1.0:
            collision_time = t1
        elif 0 <= t2 <= 1.0:
            collision_time = t2
        
        if collision_time is not None:
            # Calculate contact point and normal
            contact_x = start_x + collision_time * dx
            contact_y = start_y + collision_time * dy
            
            # Normal points from segment to circle center
            normal_x = contact_x - closest_x
            normal_y = contact_y - closest_y
            normal_length = math.sqrt(normal_x*normal_x + normal_y*normal_y)
            if normal_length > 0:
                normal_x /= normal_length
                normal_y /= normal_length
            
            return CollisionResult(
                collision=True,
                time_of_impact=collision_time,
                contact_x=contact_x,
                contact_y=contact_y,
                normal_x=normal_x,
                normal_y=normal_y
            )
        
        return CollisionResult(collision=False)
    
    def _test_circle_vs_arc_segment(
        self,
        start_x: float,
        start_y: float,
        dx: float,
        dy: float,
        radius: float,
        segment: CollisionSegment
    ) -> CollisionResult:
        """
        Test swept circle collision against a circular arc segment.
        
        This implements collision detection for quarter-circle tile segments
        like those found in quarter-pipes and quarter-moons.
        
        Args:
            start_x: Circle starting x position
            start_y: Circle starting y position
            dx: Circle movement delta x
            dy: Circle movement delta y
            radius: Circle radius
            segment: Circular collision segment
            
        Returns:
            CollisionResult with collision information
        """
        # Vector from arc center to circle start
        to_circle_x = start_x - segment.center_x
        to_circle_y = start_y - segment.center_y
        
        # For convex arcs (quarter-pipes), collision occurs when circle gets too close
        # For concave arcs (quarter-moons), collision occurs when circle enters the arc
        if segment.is_convex:
            # Convex arc - collision when circle approaches from outside
            target_distance = segment.radius + radius
        else:
            # Concave arc - collision when circle enters from inside
            target_distance = abs(segment.radius - radius)
        
        # Solve quadratic equation for circle-circle collision
        # ||(start + t*velocity) - arc_center|| = target_distance
        a = dx*dx + dy*dy
        b = 2.0 * (to_circle_x*dx + to_circle_y*dy)
        c = to_circle_x*to_circle_x + to_circle_y*to_circle_y - target_distance*target_distance
        
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            return CollisionResult(collision=False)  # No collision
        
        if abs(a) < 1e-6:  # No movement
            if (segment.is_convex and c <= 0) or (not segment.is_convex and c >= 0):
                return CollisionResult(collision=True, time_of_impact=0.0)
            else:
                return CollisionResult(collision=False)
        
        # Calculate collision time
        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2*a)
        t2 = (-b + sqrt_discriminant) / (2*a)
        
        # Use appropriate collision time based on arc type
        collision_time = None
        if segment.is_convex:
            # For convex arcs, use earliest positive time
            if 0 <= t1 <= 1.0:
                collision_time = t1
            elif 0 <= t2 <= 1.0:
                collision_time = t2
        else:
            # For concave arcs, use latest positive time (exit collision)
            if 0 <= t2 <= 1.0:
                collision_time = t2
            elif 0 <= t1 <= 1.0:
                collision_time = t1
        
        if collision_time is not None:
            # Check if collision point is within the arc quadrant
            contact_x = start_x + collision_time * dx
            contact_y = start_y + collision_time * dy
            
            # Vector from arc center to contact point
            to_contact_x = contact_x - segment.center_x
            to_contact_y = contact_y - segment.center_y
            
            # Check if contact point is in the correct quadrant
            if ((segment.normal_x > 0 and to_contact_x >= 0) or (segment.normal_x < 0 and to_contact_x <= 0)) and \
               ((segment.normal_y > 0 and to_contact_y >= 0) or (segment.normal_y < 0 and to_contact_y <= 0)):
                
                # Calculate normal vector
                contact_distance = math.sqrt(to_contact_x*to_contact_x + to_contact_y*to_contact_y)
                if contact_distance > 0:
                    if segment.is_convex:
                        normal_x = to_contact_x / contact_distance
                        normal_y = to_contact_y / contact_distance
                    else:
                        normal_x = -to_contact_x / contact_distance
                        normal_y = -to_contact_y / contact_distance
                else:
                    normal_x = normal_y = 0
                
                return CollisionResult(
                    collision=True,
                    time_of_impact=collision_time,
                    contact_x=contact_x,
                    contact_y=contact_y,
                    normal_x=normal_x,
                    normal_y=normal_y
                )
        
        return CollisionResult(collision=False)