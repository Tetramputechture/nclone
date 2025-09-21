"""
Intelligent caching system for reachability analysis.

This module provides an LRU cache with configurable size limits and cache invalidation
for efficient repeated reachability queries during RL training.
"""

import hashlib
import time
from collections import OrderedDict
from typing import Dict, Any, Tuple, Set
from dataclasses import dataclass



@dataclass
class CacheEntry:
    """Represents a cached reachability analysis result."""
    
    result  # Union[ReachabilityApproximation, ReachabilityResult]
    timestamp: float
    access_count: int
    ninja_position: Tuple[float, float]
    switch_states: Dict[int, bool]
    level_id: str


class ReachabilityCache:
    """
    Intelligent LRU cache for reachability analysis results.
    
    Features:
    - LRU eviction policy with configurable size limits
    - Cache invalidation based on game state changes
    - Approximate matching for similar states
    - Hit rate tracking for performance monitoring
    - Persistent caching across episodes
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0):
        """
        Initialize the reachability cache.
        
        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
        # LRU cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        self.invalidation_count = 0
        
        # Cache warming data
        self.common_scenarios: Set[str] = set()
        
    def _generate_cache_key(
        self, 
        ninja_position: Tuple[float, float], 
        switch_states: Dict[int, bool], 
        level_id: str,
        precision: int = 10
    ) -> str:
        """
        Generate a cache key for the given state.
        
        Args:
            ninja_position: Ninja position (x, y)
            switch_states: Current switch states
            level_id: Level identifier
            precision: Position precision for approximate matching
            
        Returns:
            Cache key string
        """
        # Round position for approximate matching
        rounded_x = round(ninja_position[0] / precision) * precision
        rounded_y = round(ninja_position[1] / precision) * precision
        
        # Create deterministic switch state representation
        switch_repr = tuple(sorted(switch_states.items()))
        
        # Generate hash
        key_data = f"{level_id}_{rounded_x}_{rounded_y}_{switch_repr}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(
        self, 
        ninja_position: Tuple[float, float], 
        switch_states: Dict[int, bool], 
        level_id: str
    ):  # -> Optional[Union[ReachabilityApproximation, ReachabilityResult]]
        """
        Retrieve cached reachability result if available.
        
        Args:
            ninja_position: Ninja position (x, y)
            switch_states: Current switch states
            level_id: Level identifier
            
        Returns:
            Cached reachability result or None if not found/expired
        """
        cache_key = self._generate_cache_key(ninja_position, switch_states, level_id)
        
        if cache_key not in self.cache:
            self.miss_count += 1
            return None
        
        entry = self.cache[cache_key]
        
        # Check TTL
        if time.time() - entry.timestamp > self.ttl_seconds:
            del self.cache[cache_key]
            self.miss_count += 1
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(cache_key)
        entry.access_count += 1
        self.hit_count += 1
        
        return entry.result
    
    def put(
        self, 
        ninja_position: Tuple[float, float], 
        switch_states: Dict[int, bool], 
        level_id: str, 
        result  # Union[ReachabilityApproximation, ReachabilityResult]
    ):
        """
        Store reachability result in cache.
        
        Args:
            ninja_position: Ninja position (x, y)
            switch_states: Current switch states
            level_id: Level identifier
            result: Reachability analysis result
        """
        cache_key = self._generate_cache_key(ninja_position, switch_states, level_id)
        
        # Create cache entry
        entry = CacheEntry(
            result=result,
            timestamp=time.time(),
            access_count=1,
            ninja_position=ninja_position,
            switch_states=switch_states.copy(),
            level_id=level_id
        )
        
        # Add to cache
        if cache_key in self.cache:
            # Update existing entry
            self.cache.move_to_end(cache_key)
        else:
            # Check size limit
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
        
        self.cache[cache_key] = entry
        
        # Track common scenarios for cache warming
        self.common_scenarios.add(cache_key)
    
    def invalidate_level(self, level_id: str):
        """
        Invalidate all cache entries for a specific level.
        
        Args:
            level_id: Level identifier to invalidate
        """
        keys_to_remove = [
            key for key, entry in self.cache.items() 
            if entry.level_id == level_id
        ]
        
        for key in keys_to_remove:
            del self.cache[key]
            self.invalidation_count += 1
    
    def invalidate_switch_dependent(self, changed_switches: Set[int]):
        """
        Invalidate cache entries that depend on changed switches.
        
        Args:
            changed_switches: Set of switch IDs that have changed state
        """
        keys_to_remove = []
        
        for key, entry in self.cache.items():
            # Check if any changed switch affects this entry
            if any(switch_id in entry.switch_states for switch_id in changed_switches):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
            self.invalidation_count += 1
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        self.invalidation_count = 0
    
    def get_hit_rate(self) -> float:
        """
        Get cache hit rate.
        
        Returns:
            Hit rate as a float between 0.0 and 1.0
        """
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dictionary with cache performance statistics
        """
        total_requests = self.hit_count + self.miss_count
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': self.get_hit_rate(),
            'invalidation_count': self.invalidation_count,
            'total_requests': total_requests,
            'common_scenarios': len(self.common_scenarios),
            'memory_usage_estimate': len(self.cache) * 1024  # Rough estimate in bytes
        }
    
    def warm_cache(self, scenarios: list):
        """
        Pre-populate cache with common scenarios.
        
        Args:
            scenarios: List of (ninja_pos, switch_states, level_id, result) tuples
        """
        for ninja_pos, switch_states, level_id, result in scenarios:
            self.put(ninja_pos, switch_states, level_id, result)
    
    def get_most_accessed_entries(self, limit: int = 10) -> list:
        """
        Get the most frequently accessed cache entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of (cache_key, access_count) tuples
        """
        entries = [
            (key, entry.access_count) 
            for key, entry in self.cache.items()
        ]
        entries.sort(key=lambda x: x[1], reverse=True)
        return entries[:limit]
    
    def cleanup_expired(self):
        """Remove expired cache entries."""
        current_time = time.time()
        keys_to_remove = [
            key for key, entry in self.cache.items()
            if current_time - entry.timestamp > self.ttl_seconds
        ]
        
        for key in keys_to_remove:
            del self.cache[key]
    
    def resize(self, new_max_size: int):
        """
        Resize the cache to a new maximum size.
        
        Args:
            new_max_size: New maximum cache size
        """
        self.max_size = new_max_size
        
        # Remove excess entries if needed
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)