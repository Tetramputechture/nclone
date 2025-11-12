"""
Collision query cache for expensive get_single_closest_point results.

This module provides an LRU cache for collision queries, trading memory
for speed by caching expensive segment distance calculations.
"""

from typing import Tuple, Optional
from collections import OrderedDict


class CollisionQueryCache:
    """LRU cache for collision query results.
    
    Caches results of get_single_closest_point() calls using quantized
    position/radius keys. Provides significant speedup for terminal
    velocity simulations which repeat similar queries.
    """
    
    def __init__(self, max_size: int = 10000, quantization: float = 0.1):
        """Initialize collision query cache.
        
        Args:
            max_size: Maximum number of cached entries (~1MB for 10K entries)
            quantization: Position quantization for cache keys (0.1 pixel precision)
        """
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.quantization = quantization
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _quantize(self, value: float) -> float:
        """Quantize a float value to reduce cache key space.
        
        Args:
            value: Float value to quantize
            
        Returns:
            Quantized value rounded to nearest quantization step
        """
        return round(value / self.quantization) * self.quantization
    
    def _make_key(self, xpos: float, ypos: float, radius: float) -> Tuple[float, float, float]:
        """Create cache key from position and radius.
        
        Args:
            xpos, ypos: Query position
            radius: Query radius
            
        Returns:
            Tuple key for cache lookup
        """
        return (
            self._quantize(xpos),
            self._quantize(ypos),
            self._quantize(radius)
        )
    
    def get(self, xpos: float, ypos: float, radius: float) -> Optional[Tuple]:
        """Get cached result for a query.
        
        Args:
            xpos, ypos: Query position
            radius: Query radius
            
        Returns:
            Cached (result, closest_point) tuple, or None if not cached
        """
        key = self._make_key(xpos, ypos, radius)
        
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, xpos: float, ypos: float, radius: float, result: int, closest_point: Tuple):
        """Cache a query result.
        
        Args:
            xpos, ypos: Query position
            radius: Query radius
            result: Result value from get_single_closest_point
            closest_point: Closest point tuple from get_single_closest_point
        """
        key = self._make_key(xpos, ypos, radius)
        
        # Add/update cache entry
        self.cache[key] = (result, closest_point)
        self.cache.move_to_end(key)
        
        # Evict oldest entry if cache is full
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
            self.evictions += 1
    
    def clear(self):
        """Clear all cached entries (called on level change)."""
        self.cache.clear()
        # Keep statistics
    
    def get_stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache performance metrics
        """
        total_queries = self.hits + self.misses
        hit_rate = self.hits / total_queries if total_queries > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': hit_rate,
            'total_queries': total_queries
        }
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0

