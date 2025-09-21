"""
Templates and pattern creation tools for the N++ MCP server.

This module contains pre-designed level structures, pattern generation,
and comprehensive type listings for reference.
"""

import logging
from typing import Literal, List

from fastmcp import FastMCP
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT

from .constants import (
    ENTITY_TYPE_MAPPING,
    TILE_TYPE_MAPPING,
)
from . import map_operations

logger = logging.getLogger(__name__)


def register_template_tools(mcp: FastMCP) -> None:
    """Register all template and pattern creation tools with the FastMCP server."""

    @mcp.tool()
    async def list_available_types() -> str:
        """List all available entity types and tile types for reference.

        Returns:
            Comprehensive list of available types for map creation
        """
        try:
            entity_list = "\n".join(
                [
                    f"  {name}: {id_}"
                    for name, id_ in sorted(ENTITY_TYPE_MAPPING.items())
                ]
            )

            # Group tile types by category for better readability
            basic_tiles = {k: v for k, v in TILE_TYPE_MAPPING.items() if v <= 5}
            slope_45 = {k: v for k, v in TILE_TYPE_MAPPING.items() if 6 <= v <= 9}
            quarter_moons = {
                k: v for k, v in TILE_TYPE_MAPPING.items() if 10 <= v <= 13
            }
            quarter_pipes = {
                k: v for k, v in TILE_TYPE_MAPPING.items() if 14 <= v <= 17
            }
            mild_slopes = {k: v for k, v in TILE_TYPE_MAPPING.items() if 18 <= v <= 25}
            steep_slopes = {k: v for k, v in TILE_TYPE_MAPPING.items() if 26 <= v <= 33}

            tile_categories = f"""  Basic Tiles:
{chr(10).join([f"    {name}: {id_}" for name, id_ in sorted(basic_tiles.items()) if "slope_" not in name and "quarter_" not in name])}
            
  45-Degree Slopes:
{chr(10).join([f"    {name}: {id_}" for name, id_ in sorted(slope_45.items()) if not name.startswith(("slope_tl", "slope_tr", "slope_bl", "slope_br"))])}
        
  Quarter Moons (Convex Corners):
{chr(10).join([f"    {name}: {id_}" for name, id_ in sorted(quarter_moons.items()) if not name.startswith("quarter_circle")])}
        
  Quarter Pipes (Concave Corners):
{chr(10).join([f"    {name}: {id_}" for name, id_ in sorted(quarter_pipes.items())])}
        
  Mild/Shallow Slopes:
{chr(10).join([f"    {name}: {id_}" for name, id_ in sorted(mild_slopes.items())])}
        
  Steep Slopes:
{chr(10).join([f"    {name}: {id_}" for name, id_ in sorted(steep_slopes.items())])}"""

            return f"""📋 Available Types:

🎮 Entity Types:
{entity_list}

🧱 Tile Types:
{tile_categories}

💡 Advanced Commands Available:
- create_slope(): Create slopes with varying steepness and length
- create_corridor(): Connect areas with corridors  
- create_platform(): Build platforms with optional supports
- create_ramp(): Create ramps between height levels
- create_tunnel(): Carve tunnels through terrain
- create_chamber(): Build enclosed rooms
- validate_level(): Check if level meets N++ playability requirements

🎯 Level Requirements:
- ✅ REQUIRED: Ninja spawn point (use set_ninja_spawn())  
- ✅ REQUIRED: At least one exit door with switch (use add_entity("exit_door"))
- ⚠️  RECOMMENDED: Reasonable amount of navigable space (5-80%)
- ℹ️  INFO: Boundaries are automatically solid (1-tile padding added by simulation)

🎯 Tips:
- Use proper nclone terminology: quarter_moon, quarter_pipe, slope_mild_, slope_steep_
- Advanced commands handle complex structures automatically  
- Coordinates are in tiles: x: 1-{MAP_TILE_WIDTH}, y: 1-{MAP_TILE_HEIGHT} for playable area
- Exit doors, locked doors, and trap doors require switch coordinates
- Use validate_level() to check completeness before export/save
- Use legacy names for backward compatibility if needed
"""

        except Exception as e:
            logger.error(f"Error listing types: {e}")
            return f"✗ Failed to list types: {str(e)}"

    @mcp.tool()
    async def create_template(
        template_name: Literal[
            "simple_room",
            "jump_challenge",
            "parkour_section",
            "maze_segment",
            "vertical_shaft",
        ] = "simple_room",
        x: int = 5,
        y: int = 5,
        scale: int = 1,
    ) -> str:
        """Create pre-designed level structures using templates.

        Args:
            template_name: Name of the template to create
            x: X coordinate for placement
            y: Y coordinate for placement
            scale: Scale multiplier for template size (1-3)

        Returns:
            Status of template creation
        """
        try:
            current_map = map_operations.current_map
            if current_map is None:
                return "✗ No map loaded. Create or load a map first."

            if scale < 1 or scale > 3:
                return "✗ Scale must be between 1 and 3"

            tiles_set = 0

            if template_name == "simple_room":
                # 7x5 room with entrance
                width, height = 7 * scale, 5 * scale

                # Create walls
                for i in range(width + 2):
                    for j in range(height + 2):
                        tx, ty = x + i, y + j
                        if 1 <= tx < MAP_TILE_WIDTH and 1 <= ty < MAP_TILE_HEIGHT:
                            if i == 0 or i == width + 1 or j == 0 or j == height + 1:
                                if not (
                                    i == width // 2 and j == height + 1
                                ):  # Leave entrance
                                    current_map.set_tile(
                                        tx, ty, TILE_TYPE_MAPPING["full"]
                                    )
                            else:
                                current_map.set_tile(tx, ty, TILE_TYPE_MAPPING["empty"])
                            tiles_set += 1

            elif template_name == "jump_challenge":
                # Series of platforms requiring jumps
                platform_width = 3 * scale
                gap = 4 * scale

                for i in range(3):  # 3 platforms
                    px = x + i * (platform_width + gap)
                    py = y - i * 2  # Ascending height

                    for pw in range(platform_width):
                        for ph in range(2):
                            tx, ty = px + pw, py + ph
                            if 1 <= tx < MAP_TILE_WIDTH and 1 <= ty < MAP_TILE_HEIGHT:
                                current_map.set_tile(tx, ty, TILE_TYPE_MAPPING["full"])
                                tiles_set += 1

            elif template_name == "parkour_section":
                # Complex parkour with slopes and quarter pipes
                section_width = 12 * scale

                # Base platform
                for i in range(section_width):
                    tx, ty = x + i, y + 3
                    if 1 <= tx < MAP_TILE_WIDTH and 1 <= ty < MAP_TILE_HEIGHT:
                        current_map.set_tile(tx, ty, TILE_TYPE_MAPPING["full"])
                        tiles_set += 1

                # Add slopes and curves
                slope_positions = [2 * scale, 6 * scale, 10 * scale]
                for pos in slope_positions:
                    if 1 <= x + pos < MAP_TILE_WIDTH and 1 <= y + 2 < MAP_TILE_HEIGHT:
                        current_map.set_tile(
                            x + pos, y + 2, TILE_TYPE_MAPPING["slope_45_bl_tr"]
                        )
                        current_map.set_tile(
                            x + pos + 1, y + 1, TILE_TYPE_MAPPING["quarter_pipe_br"]
                        )
                        tiles_set += 2

            elif template_name == "maze_segment":
                # Small maze section with walls and passages
                maze_size = 8 * scale

                # Create maze walls in a pattern
                for i in range(maze_size):
                    for j in range(maze_size):
                        tx, ty = x + i, y + j
                        if 1 <= tx < MAP_TILE_WIDTH and 1 <= ty < MAP_TILE_HEIGHT:
                            # Simple maze pattern
                            if (
                                (i % 3 == 0 and j % 2 == 1)
                                or (j % 3 == 0 and i % 2 == 1)
                                or (
                                    i == 0
                                    or j == 0
                                    or i == maze_size - 1
                                    or j == maze_size - 1
                                )
                            ):
                                if not (i == 1 and j == maze_size - 1):  # Entrance
                                    current_map.set_tile(
                                        tx, ty, TILE_TYPE_MAPPING["full"]
                                    )
                            else:
                                current_map.set_tile(tx, ty, TILE_TYPE_MAPPING["empty"])
                            tiles_set += 1

            elif template_name == "vertical_shaft":
                # Vertical climbing shaft with platforms
                shaft_height = 15 * scale
                shaft_width = 4 * scale

                # Create shaft walls
                for h in range(shaft_height):
                    for w in range(shaft_width + 2):
                        tx, ty = x + w, y + h
                        if 1 <= tx < MAP_TILE_WIDTH and 1 <= ty < MAP_TILE_HEIGHT:
                            if w == 0 or w == shaft_width + 1:
                                current_map.set_tile(
                                    tx, ty, TILE_TYPE_MAPPING["full"]
                                )  # Walls
                            else:
                                current_map.set_tile(
                                    tx, ty, TILE_TYPE_MAPPING["empty"]
                                )  # Interior
                            tiles_set += 1

                # Add climbing platforms
                for p in range(0, shaft_height, 4):
                    platform_x = x + 1 + (p // 4) % shaft_width
                    platform_y = y + p
                    if (
                        1 <= platform_x < MAP_TILE_WIDTH
                        and 1 <= platform_y < MAP_TILE_HEIGHT
                    ):
                        current_map.set_tile(
                            platform_x, platform_y, TILE_TYPE_MAPPING["half_top"]
                        )
                        tiles_set += 1

            return f"✓ Created '{template_name}' template at ({x},{y}) scale {scale}x using {tiles_set} tiles"

        except Exception as e:
            logger.error(f"Error creating template: {e}")
            return f"✗ Failed to create template: {str(e)}"

    @mcp.tool()
    async def create_pattern_line(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        pattern: Literal["alternating", "checkerboard", "gradient"] = "alternating",
        tile_types: List[str] = None,
    ) -> str:
        """Create patterned lines between two points.

        Args:
            x1: Starting X coordinate
            y1: Starting Y coordinate
            x2: Ending X coordinate
            y2: Ending Y coordinate
            pattern: Pattern type to apply
            tile_types: List of tile types to use in pattern (defaults to ["full", "empty"])

        Returns:
            Status of pattern creation
        """
        try:
            current_map = map_operations.current_map
            if current_map is None:
                return "✗ No map loaded. Create or load a map first."

            if tile_types is None:
                tile_types = ["full", "empty"]

            # Validate tile types
            for tile_type in tile_types:
                if tile_type not in TILE_TYPE_MAPPING:
                    return f"✗ Invalid tile type '{tile_type}'"

            tiles_set = 0

            # Calculate line using Bresenham's algorithm
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy

            current_x, current_y = x1, y1
            position = 0

            while True:
                if 1 <= current_x < MAP_TILE_WIDTH and 1 <= current_y < MAP_TILE_HEIGHT:
                    if pattern == "alternating":
                        tile_type = tile_types[position % len(tile_types)]
                    elif pattern == "checkerboard":
                        tile_type = tile_types[
                            (current_x + current_y) % len(tile_types)
                        ]
                    elif pattern == "gradient":
                        # Gradient based on distance along line
                        total_distance = max(dx, dy)
                        progress = position / max(1, total_distance)
                        tile_index = int(progress * (len(tile_types) - 1))
                        tile_type = tile_types[min(tile_index, len(tile_types) - 1)]

                    current_map.set_tile(
                        current_x, current_y, TILE_TYPE_MAPPING[tile_type]
                    )
                    tiles_set += 1

                if current_x == x2 and current_y == y2:
                    break

                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    current_x += sx
                if e2 < dx:
                    err += dx
                    current_y += sy
                position += 1

            pattern_info = f" using {len(tile_types)} tile types"
            return f"✓ Created {pattern} pattern line from ({x1},{y1}) to ({x2},{y2}){pattern_info}, {tiles_set} tiles set"

        except Exception as e:
            logger.error(f"Error creating pattern line: {e}")
            return f"✗ Failed to create pattern line: {str(e)}"
