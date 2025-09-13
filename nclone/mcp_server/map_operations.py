"""
Basic map operations for the N++ MCP server.

This module contains fundamental map creation and manipulation tools including
tile setting, entity placement, and basic map information retrieval.
"""

import logging
from typing import Optional, Literal

from fastmcp import FastMCP
from ..map_generation.map import Map
from ..map_generation.map_generator import generate_map
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT, FULL_MAP_WIDTH, FULL_MAP_HEIGHT

from .constants import (
    ENTITY_TYPE_MAPPING,
    ENTITY_NAME_MAPPING,
    TILE_TYPE_MAPPING,
    TILE_NAME_MAPPING,
)

logger = logging.getLogger(__name__)

# Global variable to hold current map (imported/shared across modules)
current_map: Optional[Map] = None


def register_map_operations(mcp: FastMCP) -> None:
    """Register all map operation tools with the FastMCP server."""

    @mcp.tool()
    async def create_empty_map(seed: Optional[int] = None) -> str:
        """Create a new empty map with basic boundaries.

        Args:
            seed: Optional random seed for reproducible generation

        Returns:
            Map creation status and basic info about the created map
        """
        try:
            map_instance = Map(seed=seed)

            # Note: Boundaries are automatically solid in simulation (1-tile padding)
            # No need to manually set boundary tiles

            # Set ninja spawn to a safe position
            map_instance.set_ninja_spawn(2, 2, orientation=-1)

            # Store the map globally for further editing
            global current_map
            current_map = map_instance

            return f"✓ Created empty map. Map size: {MAP_TILE_WIDTH}x{MAP_TILE_HEIGHT} tiles. Ninja spawns at (2,2) facing left. Boundaries are automatically solid."

        except Exception as e:
            logger.error(f"Error creating empty map: {e}")
            return f"✗ Failed to create empty map: {str(e)}"

    @mcp.tool()
    async def generate_procedural_map(
        level_type: Literal[
            "MAZE", "SIMPLE_HORIZONTAL_NO_BACKTRACK", "MULTI_CHAMBER", "JUMP_REQUIRED"
        ] = "MAZE",
        seed: Optional[int] = None,
    ) -> str:
        """Generate a procedural map using built-in generators.

        Args:
            level_type: Type of level to generate
            seed: Optional random seed for reproducible generation

        Returns:
            Generation status and info about the created map
        """
        try:
            map_instance = generate_map(level_type=level_type, seed=seed)

            # Store the map globally for further editing
            global current_map
            current_map = map_instance

            # Get some basic statistics
            entity_counts = map_instance.entity_counts
            total_entities = len(map_instance.entity_data) // 5

            return (
                f"✓ Generated {level_type} map with seed {seed}. "
                f"Statistics: {entity_counts['exit_door']} exit door(s), "
                f"{entity_counts['gold']} gold, {entity_counts['death_ball']} death ball(s), "
                f"{total_entities} total entities."
            )

        except Exception as e:
            logger.error(f"Error generating procedural map: {e}")
            return f"✗ Failed to generate procedural map: {str(e)}"

    @mcp.tool()
    async def set_tile(x: int, y: int, tile_type: str) -> str:
        """Set a specific tile at the given coordinates.

        Args:
            x: X coordinate (0-43, where 0 and 43 are boundaries)
            y: Y coordinate (0-24, where 0 and 24 are boundaries)
            tile_type: Type of tile (e.g., "empty", "full", "half_top", "slope_tl_br")

        Returns:
            Status of the tile setting operation
        """
        try:
            global current_map
            if current_map is None:
                return "✗ No map loaded. Create or load a map first."

            if tile_type not in TILE_TYPE_MAPPING:
                available_types = ", ".join(TILE_TYPE_MAPPING.keys())
                return f"✗ Invalid tile type '{tile_type}'. Available types: {available_types}"

            if not (0 <= x < FULL_MAP_WIDTH and 0 <= y < FULL_MAP_HEIGHT):
                return f"✗ Coordinates ({x}, {y}) out of bounds. Valid range: x: 0-{FULL_MAP_WIDTH - 1}, y: 0-{FULL_MAP_HEIGHT - 1}"

            tile_id = TILE_TYPE_MAPPING[tile_type]
            current_map.set_tile(x, y, tile_id)

            return f"✓ Set tile at ({x}, {y}) to '{tile_type}' (type {tile_id})"

        except Exception as e:
            logger.error(f"Error setting tile: {e}")
            return f"✗ Failed to set tile: {str(e)}"

    @mcp.tool()
    async def set_tiles_rectangle(
        x1: int, y1: int, x2: int, y2: int, tile_type: str
    ) -> str:
        """Set a rectangular area of tiles to the specified type.

        Args:
            x1: Top-left X coordinate
            y1: Top-left Y coordinate
            x2: Bottom-right X coordinate
            y2: Bottom-right Y coordinate
            tile_type: Type of tile to fill the rectangle with

        Returns:
            Status of the rectangle fill operation
        """
        try:
            global current_map
            if current_map is None:
                return "✗ No map loaded. Create or load a map first."

            if tile_type not in TILE_TYPE_MAPPING:
                available_types = ", ".join(TILE_TYPE_MAPPING.keys())
                return f"✗ Invalid tile type '{tile_type}'. Available types: {available_types}"

            tile_id = TILE_TYPE_MAPPING[tile_type]

            # Ensure coordinates are in bounds and properly ordered
            x1 = max(0, min(x1, FULL_MAP_WIDTH - 1))
            x2 = max(0, min(x2, FULL_MAP_WIDTH - 1))
            y1 = max(0, min(y1, FULL_MAP_HEIGHT - 1))
            y2 = max(0, min(y2, FULL_MAP_HEIGHT - 1))

            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            tiles_set = 0
            for y in range(y1, y2 + 1):
                for x in range(x1, x2 + 1):
                    current_map.set_tile(x, y, tile_id)
                    tiles_set += 1

            return f"✓ Set {tiles_set} tiles in rectangle ({x1},{y1}) to ({x2},{y2}) to '{tile_type}'"

        except Exception as e:
            logger.error(f"Error setting tile rectangle: {e}")
            return f"✗ Failed to set tile rectangle: {str(e)}"

    @mcp.tool()
    async def clear_rectangle(x1: int, y1: int, x2: int, y2: int) -> str:
        """Clear a rectangular area by setting all tiles to empty.

        Args:
            x1: Top-left X coordinate
            y1: Top-left Y coordinate
            x2: Bottom-right X coordinate
            y2: Bottom-right Y coordinate

        Returns:
            Status of the clearing operation
        """
        try:
            global current_map
            if current_map is None:
                return "✗ No map loaded. Create or load a map first."

            current_map.set_empty_rectangle(x1, y1, x2, y2)

            # Calculate cleared area for user feedback
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            cleared_tiles = (x2 - x1 + 1) * (y2 - y1 + 1)

            return f"✓ Cleared {cleared_tiles} tiles in rectangle ({x1},{y1}) to ({x2},{y2})"

        except Exception as e:
            logger.error(f"Error clearing rectangle: {e}")
            return f"✗ Failed to clear rectangle: {str(e)}"

    @mcp.tool()
    async def add_entity(
        entity_type: str,
        x: int,
        y: int,
        orientation: int = 0,
        mode: int = 0,
        switch_x: Optional[int] = None,
        switch_y: Optional[int] = None,
    ) -> str:
        """Add an entity to the map at the specified tile coordinates.

        Args:
            entity_type: Type of entity (e.g., "gold", "exit_door", "toggle_mine")
            x: X coordinate in tiles (1-42 for playable area)
            y: Y coordinate in tiles (1-23 for playable area)
            orientation: Entity orientation (0-7, depends on entity type)
            mode: Entity mode (0-1, depends on entity type)
            switch_x: X coordinate of switch (required for doors)
            switch_y: Y coordinate of switch (required for doors)

        Returns:
            Status of the entity addition
        """
        try:
            global current_map
            if current_map is None:
                return "✗ No map loaded. Create or load a map first."

            if entity_type not in ENTITY_TYPE_MAPPING:
                available_types = ", ".join(ENTITY_TYPE_MAPPING.keys())
                return f"✗ Invalid entity type '{entity_type}'. Available types: {available_types}"

            entity_id = ENTITY_TYPE_MAPPING[entity_type]

            # Validate coordinates are in playable area
            if not (1 <= x <= MAP_TILE_WIDTH and 1 <= y <= MAP_TILE_HEIGHT):
                return f"✗ Coordinates ({x}, {y}) out of playable area. Valid range: x: 1-{MAP_TILE_WIDTH}, y: 1-{MAP_TILE_HEIGHT}"

            # Handle entities that require switch coordinates
            if entity_type in ["exit_door", "locked_door", "trap_door"]:
                if switch_x is None or switch_y is None:
                    return f"✗ Entity type '{entity_type}' requires switch coordinates (switch_x, switch_y)"

                if not (
                    1 <= switch_x <= MAP_TILE_WIDTH and 1 <= switch_y <= MAP_TILE_HEIGHT
                ):
                    return f"✗ Switch coordinates ({switch_x}, {switch_y}) out of playable area"

                current_map.add_entity(
                    entity_id, x, y, orientation, mode, switch_x, switch_y
                )
                return f"✓ Added {entity_type} at ({x}, {y}) with switch at ({switch_x}, {switch_y})"
            else:
                current_map.add_entity(entity_id, x, y, orientation, mode)
                return f"✓ Added {entity_type} at ({x}, {y}) with orientation {orientation}"

        except Exception as e:
            logger.error(f"Error adding entity: {e}")
            return f"✗ Failed to add entity: {str(e)}"

    @mcp.tool()
    async def set_ninja_spawn(x: int, y: int, orientation: Literal[-1, 1] = -1) -> str:
        """Set the ninja spawn point.

        Args:
            x: X coordinate in tiles (1-42 for playable area)
            y: Y coordinate in tiles (1-23 for playable area)
            orientation: Ninja facing direction (-1 for left, 1 for right)

        Returns:
            Status of the spawn point setting
        """
        try:
            global current_map
            if current_map is None:
                return "✗ No map loaded. Create or load a map first."

            if not (1 <= x <= MAP_TILE_WIDTH and 1 <= y <= MAP_TILE_HEIGHT):
                return f"✗ Coordinates ({x}, {y}) out of playable area. Valid range: x: 1-{MAP_TILE_WIDTH}, y: 1-{MAP_TILE_HEIGHT}"

            current_map.set_ninja_spawn(x, y, orientation)
            direction = "left" if orientation == -1 else "right"

            return f"✓ Set ninja spawn at ({x}, {y}) facing {direction}"

        except Exception as e:
            logger.error(f"Error setting ninja spawn: {e}")
            return f"✗ Failed to set ninja spawn: {str(e)}"

    @mcp.tool()
    async def get_map_info() -> str:
        """Get information about the currently loaded map.

        Returns:
            Detailed information about the current map's contents and statistics
        """
        try:
            global current_map
            if current_map is None:
                return "✗ No map loaded. Create or load a map first."

            # Count different tile types
            tile_counts = {}
            for i, tile_type in enumerate(current_map.tile_data):
                if tile_type in tile_counts:
                    tile_counts[tile_type] += 1
                else:
                    tile_counts[tile_type] = 1

            # Format tile statistics
            tile_stats = []
            for tile_id, count in sorted(tile_counts.items()):
                tile_name = TILE_NAME_MAPPING.get(tile_id, f"unknown_{tile_id}")
                tile_stats.append(f"  {tile_name}: {count}")

            # Entity statistics
            entity_counts = current_map.entity_counts
            total_entities = len(current_map.entity_data) // 5

            # Ninja spawn info
            spawn_x, spawn_y = current_map.ninja_spawn_x, current_map.ninja_spawn_y
            spawn_direction = "left" if current_map.ninja_orientation == -1 else "right"

            # Quick validation check
            has_ninja = spawn_x is not None and spawn_y is not None
            has_exit = entity_counts["exit_door"] > 0
            validation_status = "✅ Valid" if (has_ninja and has_exit) else "❌ Invalid"

            validation_issues = []
            if not has_ninja:
                validation_issues.append("Missing ninja spawn")
            if not has_exit:
                validation_issues.append("Missing exit door")

            info = f"""📊 Map Information:

🔍 Validation: {validation_status}"""

            if validation_issues:
                info += f" ({', '.join(validation_issues)})"

            info += f"""

🎯 Ninja Spawn: ({spawn_x}, {spawn_y}) facing {spawn_direction}

📦 Entities:
  Exit doors: {entity_counts["exit_door"]}
  Gold: {entity_counts["gold"]}  
  Death balls: {entity_counts["death_ball"]}
  Total entities: {total_entities}

🧱 Tile Statistics:
{chr(10).join(tile_stats)}

📏 Map Dimensions: {MAP_TILE_WIDTH}x{MAP_TILE_HEIGHT} playable area

💡 Use validate_level() for detailed validation report
"""
            return info

        except Exception as e:
            logger.error(f"Error getting map info: {e}")
            return f"✗ Failed to get map info: {str(e)}"
