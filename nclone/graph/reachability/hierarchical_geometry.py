"""
Hierarchical geometry analysis for level-aware reachability.

This module provides level geometry analysis at multiple resolutions,
integrating with the existing position validation and collision detection systems.
"""

from typing import Set, Tuple, List, Dict, Optional, Any
import numpy as np
from collections import deque

from .position_validator import PositionValidator
from .collision_checker import CollisionChecker
from .hierarchical_constants import ResolutionLevel
from ..common import SUB_GRID_WIDTH, SUB_GRID_HEIGHT, SUB_CELL_SIZE


class HierarchicalGeometryAnalyzer:
    """
    Analyzes level geometry at multiple resolutions for hierarchical reachability.
    
    This class integrates with existing position validation and collision detection
    while providing multi-resolution analysis for performance optimization.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize hierarchical geometry analyzer.
        
        Args:
            debug: Enable debug output
        """
        self.debug = debug
        self.position_validator = PositionValidator(debug=debug)
        self.collision_checker = CollisionChecker(debug=debug)
        
        # Cache for geometry analysis at different resolutions
        self.region_traversability_cache = {}
        self.tile_traversability_cache = {}
        
    def initialize_for_level(self, level_data):
        """
        Initialize geometry analyzer for a specific level.
        
        Args:
            level_data: Level data containing tiles and entities
        """
        # Initialize underlying components
        self.position_validator.initialize_for_level(level_data.tiles)
        
        # Clear caches for new level
        self.region_traversability_cache.clear()
        self.tile_traversability_cache.clear()
        
        if self.debug:
            print(f"DEBUG: Initialized geometry analyzer for level")
    
    def analyze_region_reachability(
        self,
        level_data,
        start_region: Tuple[int, int],
        switch_states: Dict[str, bool]
    ) -> Set[Tuple[int, int]]:
        """
        Analyze reachability at region level (96px resolution).
        
        This performs a coarse-grained analysis to identify which major
        areas of the level are potentially reachable.
        
        Args:
            level_data: Level data
            start_region: Starting region coordinates
            switch_states: Current switch states
            
        Returns:
            Set of reachable region coordinates
        """
        if self.debug:
            print(f"DEBUG: Analyzing region reachability from {start_region}")
        
        reachable_regions = set()
        queue = deque([start_region])
        visited = set()
        
        # Limit region exploration for performance
        max_regions = 50  # Reasonable limit for RL applications
        processed = 0
        
        while queue and processed < max_regions:
            region = queue.popleft()
            processed += 1
            
            if region in visited:
                continue
            visited.add(region)
            
            # Check if this region contains traversable areas
            is_traversable = self._is_region_traversable(level_data, region)

            
            if is_traversable:
                reachable_regions.add(region)
                
                # Add adjacent regions to queue
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        
                        adjacent_region = (region[0] + dx, region[1] + dy)
                        
                        in_bounds = self._is_region_in_bounds(level_data, adjacent_region)
                        not_visited = adjacent_region not in visited
                        
                        if self.debug:
                            print(f"DEBUG: Adjacent region {adjacent_region}: in_bounds={in_bounds}, not_visited={not_visited}")
                        
                        if not_visited and in_bounds:
                            if self.debug:
                                print(f"DEBUG: Adding adjacent region {adjacent_region} to queue")
                            queue.append(adjacent_region)
        
        if self.debug:
            print(f"DEBUG: Found {len(reachable_regions)} reachable regions")
        
        return reachable_regions
    
    def analyze_tile_reachability(
        self,
        level_data,
        start_tile: Tuple[int, int],
        reachable_regions: Set[Tuple[int, int]],
        switch_states: Dict[str, bool]
    ) -> Set[Tuple[int, int]]:
        """
        Analyze reachability at tile level (24px resolution).
        
        This performs detailed movement analysis within reachable regions.
        
        Args:
            level_data: Level data
            start_tile: Starting tile coordinates
            reachable_regions: Set of reachable regions to analyze
            switch_states: Current switch states
            
        Returns:
            Set of reachable tile coordinates
        """
        if self.debug:
            print(f"DEBUG: Analyzing tile reachability from {start_tile}")
        
        reachable_tiles = set()
        
        # Only analyze tiles within reachable regions (with limits)
        max_tiles_to_check = 500  # Limit for performance
        tiles_checked = 0
        
        for region in reachable_regions:
            if tiles_checked >= max_tiles_to_check:
                break
                
            region_tiles = self._get_tiles_in_region(region)
            
            for tile in region_tiles:
                if tiles_checked >= max_tiles_to_check:
                    break
                tiles_checked += 1
                
                if self._is_tile_traversable(level_data, tile):
                    reachable_tiles.add(tile)
        
        # Perform connectivity analysis within reachable tiles
        connected_tiles = self._analyze_tile_connectivity(
            level_data, start_tile, reachable_tiles, switch_states
        )
        
        if self.debug:
            print(f"DEBUG: Found {len(connected_tiles)} connected tiles")
        
        return connected_tiles
    
    def analyze_subcell_reachability(
        self,
        level_data,
        start_subcell: Tuple[int, int],
        reachable_tiles: Set[Tuple[int, int]],
        switch_states: Dict[str, bool]
    ) -> Set[Tuple[int, int]]:
        """
        Analyze reachability at subcell level (6px resolution).
        
        This provides high-precision analysis for areas that need it.
        
        Args:
            level_data: Level data
            start_subcell: Starting subcell coordinates
            reachable_tiles: Set of reachable tiles to analyze
            switch_states: Current switch states
            
        Returns:
            Set of reachable subcell coordinates
        """
        if self.debug:
            print(f"DEBUG: Analyzing subcell reachability from {start_subcell}")
        
        reachable_subcells = set()
        
        # Only analyze subcells within reachable tiles (with limits)
        max_subcells_to_check = 1000  # Limit for performance
        subcells_checked = 0
        
        for tile in reachable_tiles:
            if subcells_checked >= max_subcells_to_check:
                break
                
            tile_subcells = self._get_subcells_in_tile(tile)
            
            for subcell in tile_subcells:
                if subcells_checked >= max_subcells_to_check:
                    break
                subcells_checked += 1
                
                if self._is_subcell_traversable(level_data, subcell):
                    reachable_subcells.add(subcell)
        
        # Perform detailed connectivity analysis
        connected_subcells = self._analyze_subcell_connectivity(
            level_data, start_subcell, reachable_subcells, switch_states
        )
        
        if self.debug:
            print(f"DEBUG: Found {len(connected_subcells)} connected subcells")
        
        return connected_subcells
    
    def _is_region_traversable(self, level_data, region: Tuple[int, int]) -> bool:
        """
        Check if a region contains any traversable areas.
        
        Args:
            level_data: Level data
            region: Region coordinates
            
        Returns:
            True if region contains traversable areas
        """
        # Cache check
        cache_key = (region, id(level_data))
        if cache_key in self.region_traversability_cache:
            return self.region_traversability_cache[cache_key]
        
        # Sample several points within the region
        region_pixel_x = region[0] * ResolutionLevel.REGION.value
        region_pixel_y = region[1] * ResolutionLevel.REGION.value
        
        # Sample points across the region more comprehensively
        region_size = ResolutionLevel.REGION.value  # 96 pixels
        sample_points = [
            # Corners
            (region_pixel_x + 12, region_pixel_y + 12),
            (region_pixel_x + region_size - 12, region_pixel_y + 12),
            (region_pixel_x + 12, region_pixel_y + region_size - 12),
            (region_pixel_x + region_size - 12, region_pixel_y + region_size - 12),
            # Center and edges
            (region_pixel_x + region_size // 2, region_pixel_y + region_size // 2),
            (region_pixel_x + region_size // 2, region_pixel_y + 12),
            (region_pixel_x + region_size // 2, region_pixel_y + region_size - 12),
            (region_pixel_x + 12, region_pixel_y + region_size // 2),
            (region_pixel_x + region_size - 12, region_pixel_y + region_size // 2)
        ]
        
        traversable = False
        for pixel_x, pixel_y in sample_points:
            if self.position_validator.is_position_traversable_with_radius(
                pixel_x, pixel_y, level_data.tiles, 10.0
            ):
                traversable = True
                break
        
        # Cache result
        self.region_traversability_cache[cache_key] = traversable
        return traversable
    
    def _is_region_in_bounds(self, level_data, region: Tuple[int, int]) -> bool:
        """Check if region is within level bounds."""
        region_pixel_x = region[0] * ResolutionLevel.REGION.value
        region_pixel_y = region[1] * ResolutionLevel.REGION.value
        
        # Calculate level dimensions from tiles array (tiles[row][col])
        # The level_data.width/height seem to be incorrect, so calculate from tiles
        level_width = len(level_data.tiles[0]) * 24  # columns * tile_size
        level_height = len(level_data.tiles) * 24    # rows * tile_size
        

        
        if self.debug:
            print(f"DEBUG: Region {region} at pixel ({region_pixel_x}, {region_pixel_y}), level size ({level_width}, {level_height})")
        
        return (0 <= region_pixel_x < level_width and 
                0 <= region_pixel_y < level_height)
    
    def _get_tiles_in_region(self, region: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get all tile coordinates within a region."""
        tiles = []
        
        # Calculate tile range for this region
        tiles_per_region = ResolutionLevel.REGION.value // ResolutionLevel.TILE.value
        start_tile_x = region[0] * tiles_per_region
        start_tile_y = region[1] * tiles_per_region
        
        for dx in range(tiles_per_region):
            for dy in range(tiles_per_region):
                tiles.append((start_tile_x + dx, start_tile_y + dy))
        
        return tiles
    
    def _is_tile_traversable(self, level_data, tile: Tuple[int, int]) -> bool:
        """Check if a tile is traversable."""
        # Cache check
        cache_key = (tile, id(level_data))
        if cache_key in self.tile_traversability_cache:
            return self.tile_traversability_cache[cache_key]
        
        # Convert tile to pixel coordinates
        pixel_x = tile[0] * ResolutionLevel.TILE.value + ResolutionLevel.TILE.value // 2
        pixel_y = tile[1] * ResolutionLevel.TILE.value + ResolutionLevel.TILE.value // 2
        
        # Check traversability
        traversable = self.position_validator.is_position_traversable_with_radius(
            pixel_x, pixel_y, level_data.tiles, 10.0
        )
        
        # Cache result
        self.tile_traversability_cache[cache_key] = traversable
        return traversable
    
    def _analyze_tile_connectivity(
        self,
        level_data,
        start_tile: Tuple[int, int],
        candidate_tiles: Set[Tuple[int, int]],
        switch_states: Dict[str, bool]
    ) -> Set[Tuple[int, int]]:
        """Analyze connectivity between tiles using BFS."""
        if start_tile not in candidate_tiles:
            # Find closest candidate tile to start from
            if not candidate_tiles:
                return set()
            start_tile = min(candidate_tiles, 
                           key=lambda t: abs(t[0] - start_tile[0]) + abs(t[1] - start_tile[1]))
        
        connected = set()
        queue = deque([start_tile])
        visited = set()
        
        while queue:
            tile = queue.popleft()
            
            if tile in visited or tile not in candidate_tiles:
                continue
            
            visited.add(tile)
            connected.add(tile)
            
            # Check adjacent tiles
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    adjacent_tile = (tile[0] + dx, tile[1] + dy)
                    if adjacent_tile not in visited and adjacent_tile in candidate_tiles:
                        queue.append(adjacent_tile)
        
        return connected
    
    def _get_subcells_in_tile(self, tile: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get all subcell coordinates within a tile."""
        subcells = []
        
        # Calculate subcell range for this tile
        subcells_per_tile = ResolutionLevel.TILE.value // ResolutionLevel.SUBCELL.value
        start_subcell_x = tile[0] * subcells_per_tile
        start_subcell_y = tile[1] * subcells_per_tile
        
        for dx in range(subcells_per_tile):
            for dy in range(subcells_per_tile):
                subcells.append((start_subcell_x + dx, start_subcell_y + dy))
        
        return subcells
    
    def _is_subcell_traversable(self, level_data, subcell: Tuple[int, int]) -> bool:
        """Check if a subcell is traversable."""
        # Convert subcell to pixel coordinates
        pixel_x = subcell[0] * ResolutionLevel.SUBCELL.value + ResolutionLevel.SUBCELL.value // 2
        pixel_y = subcell[1] * ResolutionLevel.SUBCELL.value + ResolutionLevel.SUBCELL.value // 2
        
        # Use position validator
        return self.position_validator.is_position_traversable_with_radius(
            pixel_x, pixel_y, level_data.tiles, 10.0
        )
    
    def _analyze_subcell_connectivity(
        self,
        level_data,
        start_subcell: Tuple[int, int],
        candidate_subcells: Set[Tuple[int, int]],
        switch_states: Dict[str, bool]
    ) -> Set[Tuple[int, int]]:
        """Analyze connectivity between subcells using BFS."""
        if start_subcell not in candidate_subcells:
            # Find closest candidate subcell to start from
            if not candidate_subcells:
                return set()
            start_subcell = min(candidate_subcells, 
                              key=lambda s: abs(s[0] - start_subcell[0]) + abs(s[1] - start_subcell[1]))
        
        connected = set()
        queue = deque([start_subcell])
        visited = set()
        
        # Limit subcell analysis to prevent performance issues
        max_subcells = 5000
        processed = 0
        
        while queue and processed < max_subcells:
            subcell = queue.popleft()
            processed += 1
            
            if subcell in visited or subcell not in candidate_subcells:
                continue
            
            visited.add(subcell)
            connected.add(subcell)
            
            # Check adjacent subcells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    adjacent_subcell = (subcell[0] + dx, subcell[1] + dy)
                    if (adjacent_subcell not in visited and 
                        adjacent_subcell in candidate_subcells):
                        queue.append(adjacent_subcell)
        
        return connected