"""
Level collision data integration layer.

This module provides a unified interface for all collision optimization
structures, managing their lifecycle and coordinating cache usage.
"""

from typing import Optional
from .spatial_segment_index import SpatialSegmentIndex
from .tile_segment_cache import TileSegmentCache
from .collision_query_cache import CollisionQueryCache


class LevelCollisionData:
    """Unified collision data manager for a level.
    
    Coordinates all collision optimization structures:
    - Spatial segment index for fast segment queries
    - Tile segment cache for template-based segment creation
    - Collision query cache for hot query results
    
    Built once per level load, shared across all systems.
    """
    
    def __init__(self):
        """Initialize empty collision data structures."""
        self.segment_index: Optional[SpatialSegmentIndex] = None
        self.tile_cache: TileSegmentCache = TileSegmentCache()
        self.query_cache: CollisionQueryCache = CollisionQueryCache(max_size=10000)
        self.level_hash: Optional[str] = None
        self.is_built: bool = False
    
    def build(self, simulator, level_hash: str):
        """Build all collision optimization structures.
        
        Args:
            simulator: Simulator instance with populated segment_dic
            level_hash: Unique hash identifying this level's tile configuration
        """
        self.level_hash = level_hash
        
        # Initialize tile segment cache (lazy, will populate on first access)
        self.tile_cache.initialize()
        
        # Build spatial segment index from simulator's segment dictionary
        self.segment_index = SpatialSegmentIndex()
        self.segment_index.build(simulator.segment_dic, level_hash)
        
        # Clear query cache for new level
        self.query_cache.clear()
        
        self.is_built = True
    
    def get_segments_in_region(self, x1: float, y1: float, x2: float, y2: float):
        """Get segments in a rectangular region.
        
        Args:
            x1, y1, x2, y2: Query rectangle bounds
            
        Returns:
            List of segments in the region
        """
        if not self.is_built or self.segment_index is None:
            return []
        
        return self.segment_index.query_region(x1, y1, x2, y2)
    
    def get_closest_point_cached(self, sim, xpos: float, ypos: float, radius: float):
        """Get closest point with query caching.
        
        This wraps get_single_closest_point with an LRU cache layer.
        
        Args:
            sim: Simulator instance
            xpos, ypos: Query position
            radius: Query radius
            
        Returns:
            (result, closest_point) tuple
        """
        # Check cache first
        cached = self.query_cache.get(xpos, ypos, radius)
        if cached is not None:
            return cached
        
        # Cache miss - compute and store
        from ..physics import get_single_closest_point
        result, closest_point = get_single_closest_point(sim, xpos, ypos, radius)
        
        self.query_cache.put(xpos, ypos, radius, result, closest_point)
        return result, closest_point
    
    def clear(self):
        """Clear all caches (called on level unload)."""
        if self.segment_index:
            self.segment_index = None
        self.query_cache.clear()
        self.is_built = False
    
    def get_stats(self) -> dict:
        """Get statistics from all optimization structures.
        
        Returns:
            Dictionary with comprehensive statistics
        """
        stats = {
            'level_hash': self.level_hash,
            'is_built': self.is_built
        }
        
        if self.segment_index:
            stats['spatial_index'] = self.segment_index.get_stats()
        
        stats['query_cache'] = self.query_cache.get_stats()
        
        stats['tile_cache_initialized'] = self.tile_cache._initialized
        
        return stats

