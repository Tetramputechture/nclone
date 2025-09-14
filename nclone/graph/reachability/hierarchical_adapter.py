"""
Adapter to integrate hierarchical reachability system with existing interfaces.

This module provides backward compatibility with the existing ReachabilityAnalyzer
interface while using the new hierarchical system under the hood.
"""

from typing import Set, Tuple, List, Dict, Optional, Any
from dataclasses import dataclass

from .hierarchical_reachability import HierarchicalReachabilityAnalyzer
from .hierarchical_constants import ResolutionLevel
from .reachability_state import ReachabilityState
from ..trajectory_calculator import TrajectoryCalculator


class HierarchicalReachabilityAdapter:
    """
    Adapter that provides the old ReachabilityAnalyzer interface using the new hierarchical system.
    
    This allows existing code to use the hierarchical system without changes while
    providing significant performance improvements.
    """
    
    def __init__(
        self, 
        trajectory_calculator: TrajectoryCalculator, 
        debug: bool = False,
        enable_caching: bool = True,
        cache_size: int = 1000,
        cache_ttl: float = 300.0
    ):
        """
        Initialize the hierarchical adapter.
        
        Args:
            trajectory_calculator: Trajectory calculator (kept for compatibility)
            debug: Enable debug output
            enable_caching: Enable caching (handled by hierarchical system)
            cache_size: Cache size (handled by hierarchical system)
            cache_ttl: Cache TTL (handled by hierarchical system)
        """
        self.debug = debug
        self.trajectory_calculator = trajectory_calculator
        
        # Initialize the hierarchical analyzer
        self.hierarchical_analyzer = HierarchicalReachabilityAnalyzer(debug=debug)
        
        # Track compatibility metrics
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
    
    def analyze_reachability(
        self,
        level_data,
        ninja_position: Tuple[float, float],
        initial_switch_states: Optional[Dict[str, bool]] = None
    ) -> ReachabilityState:
        """
        Analyze reachability using hierarchical system and return compatible result.
        
        This method maintains the same interface as the original ReachabilityAnalyzer
        but uses the hierarchical system for dramatically improved performance.
        
        Args:
            level_data: Level data containing tiles and entities
            ninja_position: Ninja's position in pixels
            initial_switch_states: Current switch states
            
        Returns:
            ReachabilityState compatible with existing code
        """
        if self.debug:
            print("DEBUG: Using hierarchical reachability adapter")
        
        # Use hierarchical analyzer
        hierarchical_result = self.hierarchical_analyzer.analyze_reachability(
            level_data, ninja_position, initial_switch_states
        )
        
        # Convert hierarchical result to legacy ReachabilityState format
        legacy_state = self._convert_to_legacy_state(hierarchical_result, initial_switch_states)
        
        if self.debug:
            print(f"DEBUG: Converted hierarchical result to legacy format")
            print(f"DEBUG: Legacy state has {len(legacy_state.reachable_positions)} positions")
        
        return legacy_state
    
    def _convert_to_legacy_state(
        self, 
        hierarchical_result, 
        initial_switch_states: Optional[Dict[str, bool]]
    ) -> ReachabilityState:
        """
        Convert hierarchical result to legacy ReachabilityState format.
        
        Args:
            hierarchical_result: Result from hierarchical analyzer
            initial_switch_states: Initial switch states
            
        Returns:
            ReachabilityState compatible with existing tests
        """
        if initial_switch_states is None:
            initial_switch_states = {}
        
        # Choose best resolution based on coverage
        # Use subcells if they provide good coverage, otherwise fall back to tiles
        expected_subcells_from_tiles = len(hierarchical_result.reachable_tiles) * 16  # 16 subcells per tile
        subcell_coverage_ratio = len(hierarchical_result.reachable_subcells) / max(expected_subcells_from_tiles, 1)
        
        # Use subcells only if they provide at least 50% of expected coverage
        use_subcells = (hierarchical_result.reachable_subcells and 
                       subcell_coverage_ratio >= 0.5)
        
        if self.debug:
            print(f"DEBUG: Subcell coverage: {len(hierarchical_result.reachable_subcells)}/{expected_subcells_from_tiles} = {subcell_coverage_ratio:.2f}")
            print(f"DEBUG: Using {'subcells' if use_subcells else 'tiles'} for legacy conversion")
        
        # Convert from (x, y) format to (sub_row, sub_col) format for legacy compatibility
        if use_subcells:
            # Convert (x, y) subcells to (sub_row, sub_col) format
            reachable_positions = [(y, x) for x, y in hierarchical_result.reachable_subcells]
        else:
            # Convert (x, y) tiles to (sub_row, sub_col) format
            # Each tile contains multiple subcells, so we need to expand
            reachable_positions = []
            subcells_per_tile = ResolutionLevel.TILE.value // ResolutionLevel.SUBCELL.value
            for tile_x, tile_y in hierarchical_result.reachable_tiles:
                # Convert tile to subcells
                start_subcell_x = tile_x * subcells_per_tile
                start_subcell_y = tile_y * subcells_per_tile
                for dx in range(subcells_per_tile):
                    for dy in range(subcells_per_tile):
                        subcell_x = start_subcell_x + dx
                        subcell_y = start_subcell_y + dy
                        # Convert to (sub_row, sub_col) format
                        reachable_positions.append((subcell_y, subcell_x))
        
        # Create legacy state
        legacy_state = ReachabilityState(
            reachable_positions=reachable_positions,
            switch_states=initial_switch_states.copy(),
            unlocked_areas=set(),  # Will be populated if needed
            subgoals=hierarchical_result.subgoals
        )
        
        # Add hierarchical-specific data as attributes for debugging
        legacy_state.hierarchical_regions = hierarchical_result.reachable_regions
        legacy_state.hierarchical_tiles = hierarchical_result.reachable_tiles
        legacy_state.hierarchical_subcells = hierarchical_result.reachable_subcells
        legacy_state.analysis_time_ms = hierarchical_result.analysis_time_ms
        
        return legacy_state
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate from hierarchical analyzer."""
        return self.hierarchical_analyzer.get_cache_hit_rate()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from hierarchical analyzer."""
        return {
            'cache_hit_rate': self.get_cache_hit_rate(),
            'total_queries': self.hierarchical_analyzer.total_queries,
            'cache_hits': self.hierarchical_analyzer.cache_hits
        }