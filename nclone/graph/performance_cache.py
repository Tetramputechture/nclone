"""
Performance cache for graph visualization and construction.

This module provides caching mechanisms to improve graph performance by:
1. Caching static level geometry that doesn't change during gameplay
2. Caching rendered graph overlays that can be reused
3. Providing efficient invalidation when dynamic elements change
"""

import pygame
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from .common import GraphData

@dataclass
class CacheEntry:
    """Represents a cached graph or rendering entry."""
    data: Any
    timestamp: float
    level_hash: str
    ninja_position: Optional[Tuple[float, float]] = None
    entities_hash: Optional[str] = None

class GraphPerformanceCache:
    """
    High-performance cache for graph construction and visualization.
    
    Since tiles never move during a level (only entities do), we can cache
    the static part of the graph and only update dynamic elements.
    """
    
    def __init__(self, max_entries: int = 10):
        """Initialize performance cache."""
        self.max_entries = max_entries
        
        # Static graph cache (level geometry only)
        self._static_graph_cache: Dict[str, CacheEntry] = {}
        
        # Rendered overlay cache
        self._overlay_cache: Dict[str, CacheEntry] = {}
        
        # Entity state cache for change detection
        self._entity_state_cache: Dict[str, str] = {}
        
        # Performance stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_invalidations = 0
    
    def get_level_hash(self, level_data: Any) -> str:
        """Generate hash for level data to detect changes."""
        if hasattr(level_data, 'tiles'):
            # Hash the tile array
            if isinstance(level_data.tiles, np.ndarray):
                return str(hash(level_data.tiles.tobytes()))
        elif isinstance(level_data, np.ndarray):
            return str(hash(level_data.tobytes()))
        elif isinstance(level_data, dict):
            # Hash dictionary representation
            return str(hash(frozenset(level_data.items())))
        else:
            # Fallback to object id
            return str(id(level_data))
    
    def get_entities_hash(self, entities: List[Dict[str, Any]]) -> str:
        """Generate hash for entity states to detect changes."""
        if not entities:
            return "empty"
        
        # Create a stable hash based on entity positions and states
        entity_data = []
        for entity in entities:
            entity_data.append((
                entity.get('type', 0),
                entity.get('x', 0.0),
                entity.get('y', 0.0),
                entity.get('state', 0)
            ))
        
        return str(hash(tuple(sorted(entity_data))))
    
    def get_cached_static_graph(self, level_data: Any) -> Optional[GraphData]:
        """Get cached static graph for level geometry."""
        level_hash = self.get_level_hash(level_data)
        
        if level_hash in self._static_graph_cache:
            self.cache_hits += 1
            return self._static_graph_cache[level_hash].data
        
        self.cache_misses += 1
        return None
    
    def cache_static_graph(self, level_data: Any, graph_data: GraphData):
        """Cache static graph for level geometry."""
        level_hash = self.get_level_hash(level_data)
        
        # Remove oldest entries if cache is full
        if len(self._static_graph_cache) >= self.max_entries:
            oldest_key = min(self._static_graph_cache.keys(), 
                           key=lambda k: self._static_graph_cache[k].timestamp)
            del self._static_graph_cache[oldest_key]
        
        import time
        self._static_graph_cache[level_hash] = CacheEntry(
            data=graph_data,
            timestamp=time.time(),
            level_hash=level_hash
        )
    
    def get_cached_overlay(
        self, 
        level_data: Any, 
        ninja_position: Tuple[float, float],
        entities: List[Dict[str, Any]],
        surface_size: Tuple[int, int]
    ) -> Optional[pygame.Surface]:
        """Get cached overlay surface if available and valid."""
        level_hash = self.get_level_hash(level_data)
        entities_hash = self.get_entities_hash(entities)
        
        cache_key = f"{level_hash}_{entities_hash}_{surface_size[0]}x{surface_size[1]}"
        
        if cache_key in self._overlay_cache:
            cached_entry = self._overlay_cache[cache_key]
            
            # Check if ninja position hasn't changed significantly (within 5 pixels)
            if cached_entry.ninja_position:
                pos_diff = (
                    abs(ninja_position[0] - cached_entry.ninja_position[0]) +
                    abs(ninja_position[1] - cached_entry.ninja_position[1])
                )
                if pos_diff < 5.0:  # Ninja hasn't moved much
                    self.cache_hits += 1
                    return cached_entry.data
        
        self.cache_misses += 1
        return None
    
    def cache_overlay(
        self, 
        level_data: Any, 
        ninja_position: Tuple[float, float],
        entities: List[Dict[str, Any]],
        surface: pygame.Surface
    ):
        """Cache overlay surface."""
        level_hash = self.get_level_hash(level_data)
        entities_hash = self.get_entities_hash(entities)
        surface_size = surface.get_size()
        
        cache_key = f"{level_hash}_{entities_hash}_{surface_size[0]}x{surface_size[1]}"
        
        # Remove oldest entries if cache is full
        if len(self._overlay_cache) >= self.max_entries:
            oldest_key = min(self._overlay_cache.keys(), 
                           key=lambda k: self._overlay_cache[k].timestamp)
            del self._overlay_cache[oldest_key]
        
        import time
        self._overlay_cache[cache_key] = CacheEntry(
            data=surface.copy(),  # Make a copy to avoid modification
            timestamp=time.time(),
            level_hash=level_hash,
            ninja_position=ninja_position,
            entities_hash=entities_hash
        )
    
    def invalidate_dynamic_caches(self, entities: List[Dict[str, Any]]):
        """Invalidate caches when dynamic elements change significantly."""
        entities_hash = self.get_entities_hash(entities)
        
        # Check if entities have changed significantly
        if hasattr(self, '_last_entities_hash') and self._last_entities_hash != entities_hash:
            # Clear overlay cache as entities have changed
            keys_to_remove = [k for k in self._overlay_cache.keys() if entities_hash not in k]
            for key in keys_to_remove:
                del self._overlay_cache[key]
            self.cache_invalidations += len(keys_to_remove)
        
        self._last_entities_hash = entities_hash
    
    def clear_all_caches(self):
        """Clear all caches."""
        self._static_graph_cache.clear()
        self._overlay_cache.clear()
        self._entity_state_cache.clear()
        self.cache_invalidations += 1
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests) * 100 if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': hit_rate,
            'cache_invalidations': self.cache_invalidations,
            'static_graph_entries': len(self._static_graph_cache),
            'overlay_entries': len(self._overlay_cache)
        }

# Global cache instance
_global_cache = GraphPerformanceCache()

def get_performance_cache() -> GraphPerformanceCache:
    """Get the global performance cache instance."""
    return _global_cache
