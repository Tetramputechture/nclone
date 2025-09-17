"""
Simplified reachability analyzer using OpenCV flood fill.

This module provides a streamlined ReachabilityAnalyzer that uses OpenCV-based
flood fill algorithms for fast, accurate reachability analysis optimized for
Deep RL training scenarios.
"""

from collections import deque
from dataclasses import dataclass
from typing import Set, Tuple, List, Dict, Optional, Any
import time

from .opencv_flood_fill import OpenCVFloodFill
from .reachability_cache import ReachabilityCache
from .reachability_state import ReachabilityState
from .reachability_types import ReachabilityResult


class ReachabilityAnalyzer:
    """
    Simplified reachability analyzer using OpenCV flood fill.

    This streamlined analyzer uses OpenCV-based flood fill algorithms for fast,
    accurate reachability analysis optimized for Deep RL training scenarios.
    Replaces complex physics calculations with efficient computer vision techniques.
    """

    def __init__(
        self, 
        debug: bool = False,
        enable_caching: bool = True,
        cache_size: int = 1000,
        cache_ttl: float = 300.0
    ):
        """
        Initialize simplified reachability analyzer.

        Args:
            debug: Enable debug output (default: False)
            enable_caching: Enable intelligent caching system (default: True)
            cache_size: Maximum cache size (default: 1000)
            cache_ttl: Cache time-to-live in seconds (default: 300)
        """
        self.debug = debug
        self.enable_caching = enable_caching

        # Initialize OpenCV flood fill engine
        self.opencv_engine = OpenCVFloodFill(debug=debug)
        
        # Initialize caching system
        if enable_caching:
            self.cache = ReachabilityCache(max_size=cache_size, ttl_seconds=cache_ttl)
        else:
            self.cache = None

    def analyze_reachability(
        self,
        level_data,
        ninja_position: Tuple[float, float],
        initial_switch_states: Optional[Dict[int, bool]] = None,
    ) -> ReachabilityState:
        """
        Analyze reachability using OpenCV flood fill algorithm.

        Uses fast OpenCV-based flood fill to determine reachable areas from the
        ninja starting position. Optimized for Deep RL training scenarios.

        Args:
            level_data: Level tile and entity data
            ninja_position: Starting position (x, y) in pixels
            initial_switch_states: Initial state of switches (default: all False)

        Returns:
            ReachabilityState with reachable positions
        """
        if initial_switch_states is None:
            initial_switch_states = {}
        
        # Check cache first
        if self.cache is not None:
            cached_result = self.cache.get(
                ninja_position, 
                initial_switch_states, 
                getattr(level_data, 'level_id', 'unknown')
            )
            if cached_result is not None:
                if self.debug:
                    print("DEBUG: Using cached reachability result")
                return cached_result

        start_time = time.time()
        
        # Use OpenCV flood fill for fast reachability analysis
        reachable_positions = self.opencv_engine.analyze_reachability(
            level_data, ninja_position, initial_switch_states
        )
        
        # Create simplified reachability state
        state = ReachabilityState(
            reachable_positions=reachable_positions,
            switch_states=initial_switch_states.copy(),
            unlocked_areas=set(),
            subgoals=[],
        )

        if self.debug:
            elapsed_ms = (time.time() - start_time) * 1000
            print(f"DEBUG: OpenCV reachability analysis completed in {elapsed_ms:.3f}ms")
            print(f"DEBUG: Found {len(reachable_positions)} reachable positions")

        # Cache the result
        if self.cache is not None:
            self.cache.put(
                ninja_position,
                initial_switch_states,
                getattr(level_data, 'level_id', 'unknown'),
                state
            )

        return state

    def get_cache_hit_rate(self) -> float:
        """
        Get the cache hit rate for performance monitoring.
        
        Returns:
            Cache hit rate as a float between 0.0 and 1.0
        """
        if self.cache is None:
            return 0.0
        return self.cache.get_hit_rate()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dictionary with cache performance statistics
        """
        if self.cache is None:
            return {'caching_disabled': True}
        return self.cache.get_stats()
    
    def clear_cache(self):
        """Clear all cached results."""
        if self.cache is not None:
            self.cache.clear()