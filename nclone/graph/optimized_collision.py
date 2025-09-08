"""
Optimized collision detection system for tile-based games with complex tile shapes.

This module implements efficient collision detection between circles (ninja) and 
complex tile geometries using spatial optimization and segment caching.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from ..constants.physics_constants import TILE_PIXEL_SIZE
from ..utils.tile_segment_factory import TileSegmentFactory
from ..physics import overlap_circle_vs_segment


@dataclass
class TileCollisionData:
    """Cached collision data for a single tile type."""
    tile_id: int
    has_segments: bool
    linear_segments: List[Tuple[float, float, float, float]]  # (x1, y1, x2, y2)
    circular_segments: List[Tuple[float, float, int, int, bool, float]]  # (cx, cy, hor, ver, convex, radius)
    bounding_box: Optional[Tuple[float, float, float, float]]  # (min_x, min_y, max_x, max_y)


class OptimizedTileCollisionCache:
    """
    High-performance collision detection cache for tile-based games.
    
    This class pre-computes and caches collision geometry for all tile types,
    enabling fast runtime collision queries without expensive segment generation.
    """
    
    def __init__(self):
        self._tile_cache: Dict[int, TileCollisionData] = {}
        self._initialized = False
    
    def initialize(self, unique_tile_ids: Set[int]):
        """
        Initialize the cache with collision data for all unique tile types.
        
        Args:
            unique_tile_ids: Set of all tile IDs that appear in the level
        """
        if self._initialized:
            return
            
        print(f"Initializing collision cache for {len(unique_tile_ids)} unique tile types...")
        
        for tile_id in unique_tile_ids:
            self._cache_tile_collision_data(tile_id)
        
        self._initialized = True
        print(f"Collision cache initialized with {len(self._tile_cache)} tile types")
    
    def _cache_tile_collision_data(self, tile_id: int):
        """Pre-compute and cache collision data for a single tile type."""
        if tile_id == 0:
            # Empty tile - no collision
            self._tile_cache[tile_id] = TileCollisionData(
                tile_id=tile_id,
                has_segments=False,
                linear_segments=[],
                circular_segments=[],
                bounding_box=None
            )
            return
        
        if tile_id == 1 or tile_id > 33:
            # Solid tile - full tile collision, no segments needed
            self._tile_cache[tile_id] = TileCollisionData(
                tile_id=tile_id,
                has_segments=False,
                linear_segments=[],
                circular_segments=[],
                bounding_box=(0, 0, TILE_PIXEL_SIZE, TILE_PIXEL_SIZE)
            )
            return
        
        # Shaped tile (2-33) - generate segments
        single_tile = {(0, 0): tile_id}  # Use (0,0) as reference position
        segment_dict = TileSegmentFactory.create_segment_dictionary(single_tile)
        
        linear_segments = []
        circular_segments = []
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        if (0, 0) in segment_dict:
            for segment in segment_dict[(0, 0)]:
                if hasattr(segment, 'x1') and hasattr(segment, 'y1'):
                    # Linear segment
                    seg_data = (segment.x1, segment.y1, segment.x2, segment.y2)
                    linear_segments.append(seg_data)
                    
                    # Update bounding box
                    min_x = min(min_x, segment.x1, segment.x2)
                    max_x = max(max_x, segment.x1, segment.x2)
                    min_y = min(min_y, segment.y1, segment.y2)
                    max_y = max(max_y, segment.y1, segment.y2)
                    
                elif hasattr(segment, 'xpos') and hasattr(segment, 'ypos'):
                    # Circular segment
                    seg_data = (segment.xpos, segment.ypos, segment.hor, segment.ver, 
                               segment.convex, segment.radius)
                    circular_segments.append(seg_data)
                    
                    # Update bounding box (approximate for circular segments)
                    min_x = min(min_x, segment.xpos - segment.radius)
                    max_x = max(max_x, segment.xpos + segment.radius)
                    min_y = min(min_y, segment.ypos - segment.radius)
                    max_y = max(max_y, segment.ypos + segment.radius)
        
        # Set bounding box
        bounding_box = None
        if linear_segments or circular_segments:
            bounding_box = (min_x, min_y, max_x, max_y)
        
        self._tile_cache[tile_id] = TileCollisionData(
            tile_id=tile_id,
            has_segments=bool(linear_segments or circular_segments),
            linear_segments=linear_segments,
            circular_segments=circular_segments,
            bounding_box=bounding_box
        )
    
    def get_tile_collision_data(self, tile_id: int) -> Optional[TileCollisionData]:
        """Get cached collision data for a tile type."""
        return self._tile_cache.get(tile_id)


class OptimizedCollisionDetector:
    """
    High-performance collision detector for circle vs. complex tile geometries.
    
    Uses spatial optimization and cached segment data for efficient collision queries.
    """
    
    def __init__(self):
        self.cache = OptimizedTileCollisionCache()
    
    def initialize_for_level(self, tiles: np.ndarray):
        """Initialize collision detection for a specific level."""
        unique_tile_ids = set(np.unique(tiles))
        self.cache.initialize(unique_tile_ids)
    
    def is_circle_position_clear(self, x: float, y: float, radius: float, tiles: np.ndarray) -> bool:
        """
        Check if a circle position is clear of collisions with tiles.
        
        Args:
            x, y: Circle center position (padded coordinates)
            radius: Circle radius
            tiles: Tile array (unpadded coordinates)
            
        Returns:
            True if position is clear, False if collision detected
        """
        # Convert to unpadded coordinates for tile array access
        unpadded_x = x - TILE_PIXEL_SIZE
        unpadded_y = y - TILE_PIXEL_SIZE
        
        # Calculate tile range that could intersect with circle
        min_tile_x = int(math.floor((unpadded_x - radius) / TILE_PIXEL_SIZE))
        max_tile_x = int(math.ceil((unpadded_x + radius) / TILE_PIXEL_SIZE))
        min_tile_y = int(math.floor((unpadded_y - radius) / TILE_PIXEL_SIZE))
        max_tile_y = int(math.ceil((unpadded_y + radius) / TILE_PIXEL_SIZE))
        
        # Check each tile in range
        for tile_y in range(min_tile_y, max_tile_y + 1):
            for tile_x in range(min_tile_x, max_tile_x + 1):
                # Skip out-of-bounds tiles
                if (tile_x < 0 or tile_x >= tiles.shape[1] or 
                    tile_y < 0 or tile_y >= tiles.shape[0]):
                    continue
                
                tile_id = tiles[tile_y, tile_x]
                
                # Skip empty tiles
                if tile_id == 0:
                    continue
                
                # Check collision with this tile
                if self._check_circle_tile_collision(unpadded_x, unpadded_y, radius, 
                                                   tile_x, tile_y, tile_id):
                    return False
        
        return True
    
    def _check_circle_tile_collision(self, circle_x: float, circle_y: float, radius: float,
                                   tile_x: int, tile_y: int, tile_id: int) -> bool:
        """
        Check collision between circle and a specific tile.
        
        Args:
            circle_x, circle_y: Circle center (unpadded coordinates)
            radius: Circle radius
            tile_x, tile_y: Tile coordinates in tile array
            tile_id: Tile type ID
            
        Returns:
            True if collision detected, False otherwise
        """
        collision_data = self.cache.get_tile_collision_data(tile_id)
        if not collision_data:
            # Unknown tile type - assume solid for safety
            return self._check_circle_solid_tile_collision(circle_x, circle_y, radius, tile_x, tile_y)
        
        if tile_id == 0:
            # Empty tile
            return False
        
        if tile_id == 1 or tile_id > 33:
            # Solid tile
            return self._check_circle_solid_tile_collision(circle_x, circle_y, radius, tile_x, tile_y)
        
        # Shaped tile - use cached segments
        return self._check_circle_shaped_tile_collision(circle_x, circle_y, radius, 
                                                      tile_x, tile_y, collision_data)
    
    def _check_circle_solid_tile_collision(self, circle_x: float, circle_y: float, radius: float,
                                         tile_x: int, tile_y: int) -> bool:
        """Check collision between circle and solid tile using AABB."""
        # Tile bounds in world coordinates
        tile_left = tile_x * TILE_PIXEL_SIZE
        tile_right = tile_left + TILE_PIXEL_SIZE
        tile_top = tile_y * TILE_PIXEL_SIZE
        tile_bottom = tile_top + TILE_PIXEL_SIZE
        
        # Find closest point on tile to circle center
        closest_x = max(tile_left, min(circle_x, tile_right))
        closest_y = max(tile_top, min(circle_y, tile_bottom))
        
        # Check distance to closest point
        dx = circle_x - closest_x
        dy = circle_y - closest_y
        distance_squared = dx * dx + dy * dy
        
        return distance_squared < (radius * radius)
    
    def _check_circle_shaped_tile_collision(self, circle_x: float, circle_y: float, radius: float,
                                          tile_x: int, tile_y: int, collision_data: TileCollisionData) -> bool:
        """Check collision between circle and shaped tile using cached segments."""
        if not collision_data.has_segments:
            return False
        
        # Early exit using bounding box
        if collision_data.bounding_box:
            bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = collision_data.bounding_box
            
            # Transform bounding box to world coordinates
            tile_world_x = tile_x * TILE_PIXEL_SIZE
            tile_world_y = tile_y * TILE_PIXEL_SIZE
            world_min_x = tile_world_x + bbox_min_x
            world_max_x = tile_world_x + bbox_max_x
            world_min_y = tile_world_y + bbox_min_y
            world_max_y = tile_world_y + bbox_max_y
            
            # Check if circle is too far from bounding box
            if (circle_x + radius < world_min_x or circle_x - radius > world_max_x or
                circle_y + radius < world_min_y or circle_y - radius > world_max_y):
                return False
        
        # Check collision with linear segments
        tile_world_x = tile_x * TILE_PIXEL_SIZE
        tile_world_y = tile_y * TILE_PIXEL_SIZE
        
        for seg_x1, seg_y1, seg_x2, seg_y2 in collision_data.linear_segments:
            # Transform segment to world coordinates
            world_x1 = tile_world_x + seg_x1
            world_y1 = tile_world_y + seg_y1
            world_x2 = tile_world_x + seg_x2
            world_y2 = tile_world_y + seg_y2
            
            if overlap_circle_vs_segment(circle_x, circle_y, radius, world_x1, world_y1, world_x2, world_y2):
                return True
        
        # Check collision with circular segments
        for seg_cx, seg_cy, hor, ver, convex, seg_radius in collision_data.circular_segments:
            # Transform segment center to world coordinates
            world_cx = tile_world_x + seg_cx
            world_cy = tile_world_y + seg_cy
            
            if self._check_circle_vs_circular_segment(circle_x, circle_y, radius,
                                                    world_cx, world_cy, hor, ver, convex, seg_radius):
                return True
        
        return False
    
    def _check_circle_vs_circular_segment(self, circle_x: float, circle_y: float, circle_radius: float,
                                        seg_cx: float, seg_cy: float, hor: int, ver: int, 
                                        convex: bool, seg_radius: float) -> bool:
        """Check collision between circle and circular segment (quarter-circle)."""
        # Distance from circle center to arc center
        dx = circle_x - seg_cx
        dy = circle_y - seg_cy
        distance = math.sqrt(dx * dx + dy * dy)
        
        # Check if we're in the right quadrant
        in_quadrant = (dx * hor >= 0) and (dy * ver >= 0)
        
        if not in_quadrant:
            return False
        
        if convex:
            # Convex arc (quarter-pipe) - collision if inside the arc
            return distance < (seg_radius + circle_radius)
        else:
            # Concave arc (quarter-moon) - collision if between inner and outer radius
            return (seg_radius - circle_radius) < distance < (seg_radius + circle_radius)


# Global instance for reuse
_collision_detector = OptimizedCollisionDetector()


def get_collision_detector() -> OptimizedCollisionDetector:
    """Get the global collision detector instance."""
    return _collision_detector