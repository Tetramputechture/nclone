"""Pre-computed map_data variants for curriculum stages.

This module provides efficient curriculum entity positioning by pre-computing
modified map_data bytes for each stage, eliminating per-reset entity repositioning
overhead.

Design:
- Build cache once during startup with all stage variants
- Load appropriate map_data on reset (entities already at correct positions)
- Zero per-reset overhead vs apply_to_simulator() approach
"""

import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class CurriculumMapCache:
    """Pre-computed map_data variants for each curriculum stage.
    
    Builds modified map_data bytes with exit switch/door entities positioned
    at curriculum-appropriate locations. Enables instant map loading without
    per-reset entity repositioning.
    
    Memory overhead: ~1.3KB per stage (map_data is ~1.3KB)
    Performance: ~0.01ms dict lookup vs ~0.5-1ms apply_to_simulator()
    """
    
    def __init__(
        self, 
        original_map_data: List[int], 
        stage_positions: Dict[int, Dict[str, Tuple[float, float]]]
    ):
        """Initialize curriculum map cache.
        
        Args:
            original_map_data: Original level map data bytes (list of ints)
            stage_positions: Dict mapping stage -> {"switch": (x,y), "exit": (x,y)}
                           Positions in pixel coordinates (world space)
        """
        self._original_map_data = original_map_data
        self._stage_positions = stage_positions
        self._cached_map_data: Dict[int, List[int]] = {}
        
        # Build all stage variants during initialization
        self._build_cache()
        
        num_stages = len(self._cached_map_data)
        total_bytes = sum(len(data) for data in self._cached_map_data.values())
        logger.info(
            f"CurriculumMapCache built: {num_stages} stages, "
            f"{total_bytes / 1024:.1f}KB total "
            f"(~{total_bytes / num_stages / 1024:.2f}KB per stage)"
        )
    
    def _build_cache(self) -> None:
        """Build modified map_data for all curriculum stages."""
        for stage, positions in self._stage_positions.items():
            switch_pos = positions["switch"]
            exit_pos = positions["exit"]
            
            # Create modified map_data with entities at curriculum positions
            modified_map_data = self._modify_entity_positions(
                self._original_map_data.copy(),
                switch_pos,
                exit_pos
            )
            
            self._cached_map_data[stage] = modified_map_data
            
            logger.debug(
                f"  Stage {stage}: switch={switch_pos}, exit={exit_pos}"
            )
    
    def _modify_entity_positions(
        self,
        map_data: List[int],
        switch_pos: Tuple[float, float],
        exit_pos: Tuple[float, float]
    ) -> List[int]:
        """Modify entity positions in map_data bytes.
        
        Map data format (from map_loader.py):
        - Entity data starts at index 1230 (after tiles, header, etc.)
        - Each entity: [type, x, y, orientation, mode] (5 bytes)
        - Exit door (type 3) at base entity index
        - Exit switch (type 4) at index + (5 * exit_door_count)
        - Coordinates in "map_data_units" (pixels / 6)
        
        Args:
            map_data: Original map data to modify (will be modified in-place)
            switch_pos: Switch position in pixels (world space)
            exit_pos: Exit door position in pixels (world space)
        
        Returns:
            Modified map_data with updated entity positions
        """
        # Get exit door count (determines switch offset)
        exit_door_count = map_data[1156]
        
        if exit_door_count == 0:
            logger.warning("No exit doors found in map_data, skipping position modification")
            return map_data
        
        # Find exit door (type 3) and exit switch (type 4) in entity data
        # Entity data starts at index 1230
        index = 1230
        exit_door_index = None
        exit_switch_index = None
        
        entity_count = 0
        while index < len(map_data):
            # Check for complete entity data (5 values needed)
            if index + 4 >= len(map_data):
                break
            
            entity_type = map_data[index]
            
            # Track exit door (type 3)
            if entity_type == 3:
                if exit_door_index is None:
                    exit_door_index = index
                entity_count += 1
            
            # Track exit switch (type 4)
            if entity_type == 4:
                if exit_switch_index is None:
                    exit_switch_index = index
            
            # Handle variable-size entities
            if entity_type in (6, 8):  # Locked door or trap door
                # Try to detect format (9 or 10 bytes)
                if index + 9 < len(map_data):
                    if (
                        map_data[index + 7] != 0
                        and map_data[index + 8] == 0
                        and map_data[index + 9] == 0
                    ):
                        index += 10
                    else:
                        index += 9
                else:
                    index += 9
            else:
                index += 5
        
        if exit_door_index is None:
            logger.warning("Could not find exit door (type 3) in map_data")
            return map_data
        
        if exit_switch_index is None:
            logger.warning("Could not find exit switch (type 4) in map_data")
            return map_data
        
        # Convert pixel positions to map_data_units (divide by 6)
        # Note: map_data stores coordinates as integers in units of 6 pixels
        exit_x_units = int(exit_pos[0] / 6)
        exit_y_units = int(exit_pos[1] / 6)
        switch_x_units = int(switch_pos[0] / 6)
        switch_y_units = int(switch_pos[1] / 6)
        
        # Update exit door position (indices +1 and +2 from entity type)
        map_data[exit_door_index + 1] = exit_x_units
        map_data[exit_door_index + 2] = exit_y_units
        
        # Update exit switch position (indices +1 and +2 from entity type)
        map_data[exit_switch_index + 1] = switch_x_units
        map_data[exit_switch_index + 2] = switch_y_units
        
        logger.debug(
            f"Modified entity positions: "
            f"exit_door[{exit_door_index}]={exit_pos} ({exit_x_units},{exit_y_units} units), "
            f"exit_switch[{exit_switch_index}]={switch_pos} ({switch_x_units},{switch_y_units} units)"
        )
        
        return map_data
    
    def get_map_data_for_stage(self, stage: int) -> List[int]:
        """Get map_data with entities positioned for curriculum stage.
        
        Args:
            stage: Curriculum stage index (0 to num_stages-1)
        
        Returns:
            List of ints representing map_data bytes with modified entity positions
            
        Raises:
            KeyError: If stage not found in cache
        """
        if stage not in self._cached_map_data:
            raise KeyError(
                f"Curriculum stage {stage} not found in map cache. "
                f"Available stages: {sorted(self._cached_map_data.keys())}"
            )
        
        return self._cached_map_data[stage]
    
    def has_stage(self, stage: int) -> bool:
        """Check if cache has data for given stage.
        
        Args:
            stage: Curriculum stage index
        
        Returns:
            True if stage exists in cache
        """
        return stage in self._cached_map_data
    
    @property
    def num_stages(self) -> int:
        """Number of curriculum stages in cache."""
        return len(self._cached_map_data)
    
    @property
    def available_stages(self) -> List[int]:
        """List of available curriculum stage indices."""
        return sorted(self._cached_map_data.keys())
