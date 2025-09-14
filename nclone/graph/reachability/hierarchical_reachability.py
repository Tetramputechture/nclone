"""
Hierarchical Reachability Analysis System

This module implements a multi-resolution reachability analysis system optimized for
real-time Deep RL applications. It uses a coarse-to-fine approach:

- Level 1 (96px regions): Strategic planning and switch dependencies
- Level 2 (24px tiles): Standard movement analysis  
- Level 3 (6px sub-cells): High-precision areas only

The system is designed to provide <10ms reachability queries for RL training
while maintaining accuracy for complex physics-based movement.
"""

from dataclasses import dataclass
from typing import Set, Tuple, List, Dict, Optional, Any
from enum import Enum
import time
import numpy as np

from ...constants.physics_constants import TILE_PIXEL_SIZE
from .reachability_state import ReachabilityState
from .hierarchical_constants import ResolutionLevel
from .hierarchical_geometry import HierarchicalGeometryAnalyzer


@dataclass
class HierarchicalNode:
    """Node in the hierarchical reachability graph."""
    position: Tuple[int, int]  # Grid coordinates at this resolution
    resolution: ResolutionLevel
    reachable: bool = False
    children: Set[Tuple[int, int]] = None  # Child nodes at finer resolution
    movement_cost: float = 1.0  # Cost to reach this node
    
    def __post_init__(self):
        if self.children is None:
            self.children = set()


@dataclass 
class HierarchicalReachabilityResult:
    """Result of hierarchical reachability analysis."""
    reachable_regions: Set[Tuple[int, int]]
    reachable_tiles: Set[Tuple[int, int]]
    reachable_subcells: Set[Tuple[int, int]]
    subgoals: List[str]
    analysis_time_ms: float
    cache_hits: int = 0
    total_queries: int = 0
    
    @property
    def total_reachable_positions(self) -> int:
        """Total number of reachable positions across all resolutions."""
        return len(self.reachable_subcells) if self.reachable_subcells else len(self.reachable_tiles)


class HierarchicalReachabilityAnalyzer:
    """
    Multi-resolution reachability analyzer optimized for RL applications.
    
    This analyzer uses a hierarchical approach to dramatically reduce computation
    time while maintaining accuracy for RL subgoal generation and pathfinding.
    """
    
    def __init__(self, debug: bool = False, entity_aware: bool = True):
        """
        Initialize hierarchical reachability analyzer.
        
        Args:
            debug: Enable debug output
            entity_aware: Enable entity-aware reachability analysis
        """
        self.debug = debug
        self.entity_aware = entity_aware
        self.region_cache = {}  # Cache for region-level connectivity
        self.tile_cache = {}    # Cache for tile-level connectivity
        self.movement_templates = {}  # Cache for common movement patterns
        
        # Initialize geometry analyzer with entity awareness
        self.geometry_analyzer = HierarchicalGeometryAnalyzer(debug=debug, entity_aware=entity_aware)
        
        # Performance tracking
        self.cache_hits = 0
        self.total_queries = 0
        
    def analyze_reachability(
        self, 
        level_data,
        ninja_position: Tuple[float, float],
        initial_switch_states: Optional[Dict[str, bool]] = None
    ) -> HierarchicalReachabilityResult:
        """
        Perform hierarchical reachability analysis.
        
        Args:
            level_data: Level data containing tiles and entities
            ninja_position: Ninja's position in pixels
            initial_switch_states: Current switch states
            
        Returns:
            HierarchicalReachabilityResult with multi-resolution reachability data
        """
        start_time = time.time()
        self.total_queries += 1
        
        if initial_switch_states is None:
            initial_switch_states = {}
            
        # Initialize geometry analyzer for this level
        self.geometry_analyzer.initialize_for_level(level_data)
            
        # Convert ninja position to different resolutions
        ninja_region = self._pixel_to_region(ninja_position)
        ninja_tile = self._pixel_to_tile(ninja_position)
        ninja_subcell = self._pixel_to_subcell(ninja_position)
        
        if self.debug:
            print(f"DEBUG: Ninja at pixel {ninja_position}")
            print(f"DEBUG: Region {ninja_region}, Tile {ninja_tile}, SubCell {ninja_subcell}")
        
        # Use multi-state analysis if entity-aware, otherwise use basic analysis
        if self.entity_aware:
            # Multi-state analysis considers switch interactions
            reachable_regions, reachable_tiles, reachable_subcells, final_switch_states = \
                self.geometry_analyzer.analyze_multi_state_reachability(
                    level_data, ninja_region, ninja_position, initial_switch_states
                )
            
            if self.debug:
                print(f"DEBUG: Multi-state analysis completed")
                print(f"DEBUG: Switch states: {final_switch_states}")
        else:
            # Basic hierarchical analysis without entity awareness
            # Phase 1: Region-level analysis (strategic planning)
            reachable_regions = self.geometry_analyzer.analyze_region_reachability(
                level_data, ninja_region, initial_switch_states
            )
            
            # Phase 2: Tile-level analysis (standard movement)
            reachable_tiles = self.geometry_analyzer.analyze_tile_reachability(
                level_data, ninja_tile, reachable_regions, initial_switch_states
            )
            
            # Phase 3: Sub-cell analysis (precision areas only)
            reachable_subcells = self.geometry_analyzer.analyze_subcell_reachability(
                level_data, ninja_subcell, reachable_tiles, initial_switch_states, ninja_position
            )
            
            final_switch_states = initial_switch_states
        
        # Generate RL-optimized subgoals
        subgoals = self._generate_subgoals(
            level_data, reachable_regions, reachable_tiles, initial_switch_states
        )
        
        analysis_time_ms = (time.time() - start_time) * 1000
        
        if self.debug:
            print(f"DEBUG: Analysis completed in {analysis_time_ms:.2f}ms")
            print(f"DEBUG: Found {len(reachable_regions)} regions, {len(reachable_tiles)} tiles, {len(reachable_subcells)} subcells")
        
        return HierarchicalReachabilityResult(
            reachable_regions=reachable_regions,
            reachable_tiles=reachable_tiles,
            reachable_subcells=reachable_subcells,
            subgoals=subgoals,
            analysis_time_ms=analysis_time_ms,
            cache_hits=self.cache_hits,
            total_queries=self.total_queries
        )
    
    def _pixel_to_region(self, pixel_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert pixel position to region coordinates (96px resolution)."""
        x, y = pixel_pos
        return (int(x // ResolutionLevel.REGION.value), int(y // ResolutionLevel.REGION.value))
    
    def _pixel_to_tile(self, pixel_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert pixel position to tile coordinates (24px resolution)."""
        x, y = pixel_pos
        return (int(x // ResolutionLevel.TILE.value), int(y // ResolutionLevel.TILE.value))
    
    def _pixel_to_subcell(self, pixel_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert pixel position to subcell coordinates (6px resolution)."""
        x, y = pixel_pos
        return (int(x // ResolutionLevel.SUBCELL.value), int(y // ResolutionLevel.SUBCELL.value))
    

    
    def _generate_subgoals(
        self,
        level_data,
        reachable_regions: Set[Tuple[int, int]],
        reachable_tiles: Set[Tuple[int, int]],
        switch_states: Dict[str, bool]
    ) -> List[str]:
        """
        Generate RL-optimized subgoals based on reachable areas.
        
        Returns actionable subgoals that the RL agent can pursue.
        """
        subgoals = []
        
        # For now, return basic subgoals
        # TODO: Implement proper subgoal detection based on entities
        if reachable_regions:
            subgoals.append("explore_reachable_areas")
        
        if len(reachable_regions) > 1:
            subgoals.append("navigate_between_regions")
        
        return subgoals
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate for performance monitoring."""
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries