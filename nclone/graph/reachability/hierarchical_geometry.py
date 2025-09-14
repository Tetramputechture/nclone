"""
Hierarchical geometry analysis for level-aware reachability.

This module provides level geometry analysis at multiple resolutions,
integrating with the existing position validation and collision detection systems.
"""

from typing import Set, Tuple, List, Dict, Optional, Any
import numpy as np
from collections import deque

from .position_validator import PositionValidator
from .entity_aware_validator import EntityAwareValidator
from .collision_checker import CollisionChecker
from .hierarchical_constants import ResolutionLevel
from ..common import SUB_GRID_WIDTH, SUB_GRID_HEIGHT, SUB_CELL_SIZE


class HierarchicalGeometryAnalyzer:
    """
    Analyzes level geometry at multiple resolutions for hierarchical reachability.
    
    This class integrates with existing position validation and collision detection
    while providing multi-resolution analysis for performance optimization.
    """
    
    def __init__(self, debug: bool = False, entity_aware: bool = True):
        """
        Initialize hierarchical geometry analyzer.
        
        Args:
            debug: Enable debug output
            entity_aware: Use entity-aware validation for advanced reachability
        """
        self.debug = debug
        self.entity_aware = entity_aware
        
        if entity_aware:
            self.position_validator = EntityAwareValidator(debug=debug)
        else:
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
        if self.entity_aware:
            # Entity-aware validator needs full level data
            self.position_validator.initialize_for_level(level_data)
        else:
            # Basic validator only needs tiles
            self.position_validator.initialize_for_level(level_data.tiles)
        
        # Clear caches for new level
        self.region_traversability_cache.clear()
        self.tile_traversability_cache.clear()
        
        if self.debug:
            print(f"DEBUG: Initialized geometry analyzer for level (entity_aware={self.entity_aware})")
    
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
        max_regions = 100  # Increased limit to debug long-walk
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
                    if self.debug and len(reachable_tiles) <= 10:
                        print(f"DEBUG: Found traversable tile {tile}")
        
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
        switch_states: Dict[str, bool],
        ninja_pixel_pos: Tuple[float, float] = None
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
                
                is_traversable = self._is_subcell_traversable(level_data, subcell)
                if is_traversable:
                    reachable_subcells.add(subcell)
                

        
        # Ensure starting subcell is included if ninja position is valid
        if start_subcell not in reachable_subcells:
            # Use ninja's actual pixel position if provided, otherwise use subcell center
            if ninja_pixel_pos:
                ninja_pixel_x, ninja_pixel_y = ninja_pixel_pos
            else:
                ninja_pixel_x = start_subcell[0] * ResolutionLevel.SUBCELL.value + ResolutionLevel.SUBCELL.value // 2
                ninja_pixel_y = start_subcell[1] * ResolutionLevel.SUBCELL.value + ResolutionLevel.SUBCELL.value // 2
            
            if self.debug:
                print(f"DEBUG: Ninja subcell {start_subcell} not in candidates, checking pixel ({ninja_pixel_x}, {ninja_pixel_y})")
            
            is_traversable = self.position_validator.is_position_traversable_with_radius(
                ninja_pixel_x, ninja_pixel_y, level_data.tiles, 10.0
            )
            
            if self.debug:
                print(f"DEBUG: Ninja position traversable: {is_traversable}")
            
            if is_traversable:
                reachable_subcells.add(start_subcell)
                if self.debug:
                    print(f"DEBUG: Added ninja starting subcell {start_subcell} to candidates")
        
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
        
        # Check traversability using entity-aware validation if available
        if hasattr(self.position_validator, 'is_position_traversable_with_entities'):
            traversable = self.position_validator.is_position_traversable_with_entities(
                pixel_x, pixel_y, level_data.tiles, 10.0
            )
        else:
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
        if self.debug:
            print(f"DEBUG: Tile connectivity analysis - start: {start_tile}, candidates: {len(candidate_tiles)}")
            if len(candidate_tiles) <= 10:
                print(f"DEBUG: Candidate tiles: {sorted(candidate_tiles)}")
        
        if start_tile not in candidate_tiles:
            # Find closest candidate tile to start from
            if not candidate_tiles:
                return set()
            original_start = start_tile
            start_tile = min(candidate_tiles, 
                           key=lambda t: abs(t[0] - start_tile[0]) + abs(t[1] - start_tile[1]))
            if self.debug:
                print(f"DEBUG: Start tile {original_start} not in candidates, using closest: {start_tile}")
        
        connected = set()
        queue = deque([start_tile])
        visited = set()
        
        while queue:
            tile = queue.popleft()
            
            if tile in visited or tile not in candidate_tiles:
                continue
            
            visited.add(tile)
            connected.add(tile)
            
            # Check adjacent tiles and path connectivity
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    adjacent_tile = (tile[0] + dx, tile[1] + dy)
                    if adjacent_tile not in visited and adjacent_tile in candidate_tiles:
                        # Check if there's actually a traversable path between the tiles
                        if self._can_move_between_tiles(level_data, tile, adjacent_tile):
                            queue.append(adjacent_tile)
        
        return connected
    
    def _can_move_between_tiles(self, level_data, tile1: Tuple[int, int], tile2: Tuple[int, int]) -> bool:
        """Check if ninja can move between two adjacent tiles."""
        # Convert tiles to pixel coordinates (center of tiles)
        x1 = tile1[0] * ResolutionLevel.TILE.value + ResolutionLevel.TILE.value // 2
        y1 = tile1[1] * ResolutionLevel.TILE.value + ResolutionLevel.TILE.value // 2
        x2 = tile2[0] * ResolutionLevel.TILE.value + ResolutionLevel.TILE.value // 2
        y2 = tile2[1] * ResolutionLevel.TILE.value + ResolutionLevel.TILE.value // 2
        
        # Check a few points along the path between tiles
        steps = 3
        for i in range(steps + 1):
            t = i / steps
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            
            # Use entity-aware validation if available
            if hasattr(self.position_validator, 'is_position_traversable_with_entities'):
                if not self.position_validator.is_position_traversable_with_entities(
                    x, y, level_data.tiles, 10.0
                ):
                    return False
            else:
                if not self.position_validator.is_position_traversable_with_radius(
                    x, y, level_data.tiles, 10.0
                ):
                    return False
        
        return True
    
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
        
        # Use entity-aware validation if available, with smaller radius for subcell precision
        if hasattr(self.position_validator, 'is_position_traversable_with_entities'):
            return self.position_validator.is_position_traversable_with_entities(
                pixel_x, pixel_y, level_data.tiles, 6.0
            )
        else:
            return self.position_validator.is_position_traversable_with_radius(
                pixel_x, pixel_y, level_data.tiles, 6.0
            )
    
    def _analyze_subcell_connectivity(
        self,
        level_data,
        start_subcell: Tuple[int, int],
        candidate_subcells: Set[Tuple[int, int]],
        switch_states: Dict[str, bool]
    ) -> Set[Tuple[int, int]]:
        """Analyze connectivity between subcells using BFS."""
        if self.debug:
            print(f"DEBUG: Subcell connectivity - start: {start_subcell}, candidates: {len(candidate_subcells)}")
            
        if start_subcell not in candidate_subcells:
            # Find closest candidate subcell to start from
            if not candidate_subcells:
                if self.debug:
                    print(f"DEBUG: No candidate subcells found!")
                return set()
            original_start = start_subcell
            start_subcell = min(candidate_subcells, 
                              key=lambda s: abs(s[0] - start_subcell[0]) + abs(s[1] - start_subcell[1]))
            if self.debug:
                print(f"DEBUG: Start subcell {original_start} not in candidates, using closest: {start_subcell}")
        
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
            adjacent_found = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    adjacent_subcell = (subcell[0] + dx, subcell[1] + dy)
                    if (adjacent_subcell not in visited and 
                        adjacent_subcell in candidate_subcells):
                        queue.append(adjacent_subcell)
                        adjacent_found += 1
            

        
        return connected
    
    def analyze_multi_state_reachability(
        self,
        level_data,
        start_region: Tuple[int, int],
        ninja_pixel_pos: Tuple[float, float],
        switch_states: Optional[Dict[str, bool]] = None
    ) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], Set[Tuple[int, int]], Dict[int, bool]]:
        """
        Analyze reachability considering multiple switch states and entity interactions.
        
        This method performs a multi-phase analysis:
        1. Initial reachability with current switch states
        2. Find reachable switches and calculate new states
        3. Re-analyze with expanded switch states
        4. Return comprehensive reachability information
        
        Args:
            level_data: Level data
            start_region: Starting region coordinates
            ninja_pixel_pos: Ninja position in pixels
            switch_states: Initial switch states
            
        Returns:
            Tuple of (reachable_regions, reachable_tiles, reachable_subcells, final_switch_states)
        """
        if not self.entity_aware:
            # Fall back to basic analysis if not entity-aware
            regions = self.analyze_region_reachability(level_data, start_region, switch_states)
            tiles = self.analyze_tile_reachability(level_data, self._region_to_tile(start_region), switch_states)
            subcells = self.analyze_subcell_reachability(level_data, self._region_to_subcell(start_region), ninja_pixel_pos, switch_states)
            return regions, tiles, subcells, switch_states or {}
        
        if switch_states is None:
            switch_states = {}
            
        # Phase 1: Initial reachability analysis ignoring door states to find potential switches
        if self.debug:
            print(f"DEBUG: Multi-state analysis Phase 1 - Initial reachability (ignoring doors)")
            
        self.position_validator.update_switch_states(switch_states)
        
        # Temporarily enable ignore_doors mode for initial exploration
        if hasattr(self.position_validator, 'set_ignore_doors_mode'):
            self.position_validator.set_ignore_doors_mode(True)
            # Clear caches since door states are now ignored
            self.region_traversability_cache.clear()
            self.tile_traversability_cache.clear()
        
        initial_regions = self.analyze_region_reachability(level_data, start_region, switch_states)
        initial_tiles = self.analyze_tile_reachability(level_data, self._region_to_tile(start_region), initial_regions, switch_states)
        initial_subcells = self.analyze_subcell_reachability(level_data, self._region_to_subcell(start_region), initial_tiles, switch_states, ninja_pixel_pos)
        
        # Restore normal mode
        if hasattr(self.position_validator, 'set_ignore_doors_mode'):
            self.position_validator.set_ignore_doors_mode(False)
            # Clear caches again since door states are now considered
            self.region_traversability_cache.clear()
            self.tile_traversability_cache.clear()
        
        # Phase 2: Find reachable switches and update states
        if self.debug:
            print(f"DEBUG: Multi-state analysis Phase 2 - Switch discovery")
            
        # Convert tiles to position set for switch reachability check
        reachable_tile_positions = set()
        for tile_x, tile_y in initial_tiles:
            reachable_tile_positions.add((tile_x, tile_y))
        
        # Also add tiles from all reachable regions for broader switch search
        # This allows finding switches in areas that are reachable but not tile-connected
        for region in initial_regions:
            region_tiles = self._get_tiles_in_region(region)
            for tile in region_tiles:
                if self._is_tile_traversable(level_data, tile):
                    reachable_tile_positions.add(tile)
        
        if self.debug:
            print(f"DEBUG: Switch discovery - checking {len(reachable_tile_positions)} tile positions (including region tiles)")
            print(f"DEBUG: Sample positions: {sorted(list(reachable_tile_positions)[:10])}")
        
        # Find newly reachable switches
        if hasattr(self.position_validator, 'find_reachable_switches'):
            achievable_switch_states = self.position_validator.find_reachable_switches(reachable_tile_positions)
            if self.debug:
                print(f"DEBUG: Found {len(achievable_switch_states)} reachable switches: {achievable_switch_states}")
        else:
            achievable_switch_states = {}
            if self.debug:
                print(f"DEBUG: No find_reachable_switches method available")
        
        # Check if any new switches can be activated
        new_switches_found = False
        if self.debug:
            print(f"DEBUG: Checking switch activation - current states: {switch_states}")
        
        for switch_id, can_activate in achievable_switch_states.items():
            current_state = switch_states.get(switch_id, False)
            if self.debug:
                print(f"DEBUG: Switch {switch_id}: can_activate={can_activate}, current_state={current_state}")
            
            if can_activate and not current_state:
                switch_states[switch_id] = True
                new_switches_found = True
                if self.debug:
                    print(f"DEBUG: Activated switch {switch_id}")
        
        if self.debug:
            print(f"DEBUG: New switches found: {new_switches_found}, final states: {switch_states}")
        
        # Phase 3+: Iterative re-analysis until no new switches are found
        current_regions = initial_regions
        current_tiles = initial_tiles
        current_subcells = initial_subcells
        iteration = 0
        max_iterations = 10  # Prevent infinite loops
        
        while new_switches_found and iteration < max_iterations:
            iteration += 1
            if self.debug:
                print(f"DEBUG: Multi-state analysis Phase {2 + iteration} - Re-analysis iteration {iteration}")
                
            self.position_validator.update_switch_states(switch_states)
            
            # Clear caches since switch states changed
            self.region_traversability_cache.clear()
            self.tile_traversability_cache.clear()
            
            # Re-analyze with new switch states
            current_regions = self.analyze_region_reachability(level_data, start_region, switch_states)
            current_tiles = self.analyze_tile_reachability(level_data, self._region_to_tile(start_region), current_regions, switch_states)
            current_subcells = self.analyze_subcell_reachability(level_data, self._region_to_subcell(start_region), current_tiles, switch_states, ninja_pixel_pos)
            
            # Look for MORE switches in the newly accessible areas
            new_reachable_tile_positions = set()
            for tile_x, tile_y in current_tiles:
                new_reachable_tile_positions.add((tile_x, tile_y))
            
            # Also add tiles from all reachable regions
            for region in current_regions:
                region_tiles = self._get_tiles_in_region(region)
                for tile in region_tiles:
                    if self._is_tile_traversable(level_data, tile):
                        new_reachable_tile_positions.add(tile)
            
            if self.debug:
                print(f"DEBUG: Iteration {iteration} - checking {len(new_reachable_tile_positions)} tile positions for more switches")
            
            # Find switches in newly accessible areas
            if hasattr(self.position_validator, 'find_reachable_switches'):
                new_achievable_switch_states = self.position_validator.find_reachable_switches(new_reachable_tile_positions)
                if self.debug:
                    print(f"DEBUG: Iteration {iteration} - found {len(new_achievable_switch_states)} reachable switches: {new_achievable_switch_states}")
            else:
                new_achievable_switch_states = {}
            
            # Check if any NEW switches can be activated
            new_switches_found = False
            for switch_id, can_activate in new_achievable_switch_states.items():
                current_state = switch_states.get(switch_id, False)
                if self.debug:
                    print(f"DEBUG: Iteration {iteration} - Switch {switch_id}: can_activate={can_activate}, current_state={current_state}")
                
                if can_activate and not current_state:
                    switch_states[switch_id] = True
                    new_switches_found = True
                    if self.debug:
                        print(f"DEBUG: Iteration {iteration} - Activated switch {switch_id}")
            
            if self.debug:
                print(f"DEBUG: Iteration {iteration} - New switches found: {new_switches_found}, total states: {switch_states}")
        
        if iteration >= max_iterations:
            if self.debug:
                print(f"DEBUG: Multi-state analysis stopped after {max_iterations} iterations to prevent infinite loop")
        elif self.debug:
            print(f"DEBUG: Multi-state analysis completed after {iteration} iterations")
        
        return current_regions, current_tiles, current_subcells, switch_states
    
    def _region_to_tile(self, region: Tuple[int, int]) -> Tuple[int, int]:
        """Convert region coordinates to tile coordinates."""
        region_x, region_y = region
        # Each region contains 4x4 tiles (96px / 24px = 4)
        tile_x = region_x * 4
        tile_y = region_y * 4
        return (tile_x, tile_y)
    
    def _region_to_subcell(self, region: Tuple[int, int]) -> Tuple[int, int]:
        """Convert region coordinates to subcell coordinates."""
        region_x, region_y = region
        # Each region contains 16x16 subcells (96px / 6px = 16)
        subcell_x = region_x * 16
        subcell_y = region_y * 16
        return (subcell_x, subcell_y)