"""
Advanced building tools for the N++ MCP server.

This module contains complex map construction tools including slopes,
corridors, platforms, ramps, tunnels, and chambers.
"""

import logging
from typing import Literal

from fastmcp import FastMCP
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT

from .constants import TILE_TYPE_MAPPING
from . import map_operations

logger = logging.getLogger(__name__)


def register_building_tools(mcp: FastMCP) -> None:
    """Register all building tool functions with the FastMCP server."""

    @mcp.tool()
    async def add_random_entities_outside_playspace(
        playspace_x1: int, playspace_y1: int, playspace_x2: int, playspace_y2: int
    ) -> str:
        """Add random decorative entities outside the main playspace area.

        Args:
            playspace_x1: Left bound of main playspace
            playspace_y1: Top bound of main playspace
            playspace_x2: Right bound of main playspace
            playspace_y2: Bottom bound of main playspace

        Returns:
            Status of the random entity addition
        """
        try:
            current_map = map_operations.current_map
            if current_map is None:
                return "✗ No map loaded. Create or load a map first."

            entities_before = len(current_map.entity_data) // 5
            current_map.add_random_entities_outside_playspace(
                playspace_x1, playspace_y1, playspace_x2, playspace_y2
            )
            entities_after = len(current_map.entity_data) // 5
            entities_added = entities_after - entities_before

            return f"✓ Added {entities_added} random entities outside playspace ({playspace_x1},{playspace_y1}) to ({playspace_x2},{playspace_y2})"

        except Exception as e:
            logger.error(f"Error adding random entities: {e}")
            return f"✗ Failed to add random entities: {str(e)}"

    @mcp.tool()
    async def create_slope(
        x: int,
        y: int,
        direction: Literal["up", "down"],
        steepness: Literal["45", "mild", "steep"] = "45",
        length: int = 1,
        raised: bool = False,
    ) -> str:
        """Create a slope of specified type, direction, and length.

        Args:
            x: Starting X coordinate
            y: Starting Y coordinate
            direction: Slope direction ("up" rises to the right, "down" drops to the right)
            steepness: Slope steepness ("45" for 45-degree, "mild" for gentle, "steep" for sharp)
            length: Number of tiles to extend the slope (1-10)
            raised: Whether to use raised platform variants (only for mild/steep slopes)

        Returns:
            Status of slope creation
        """
        try:
            current_map = map_operations.current_map
            if current_map is None:
                return "✗ No map loaded. Create or load a map first."

            if length < 1 or length > 10:
                return "✗ Length must be between 1 and 10 tiles"

            if not (1 <= x < MAP_TILE_WIDTH - length and 1 <= y < MAP_TILE_HEIGHT):
                return f"✗ Slope would extend outside playable area. Ensure x: 1-{MAP_TILE_WIDTH - length}, y: 1-{MAP_TILE_HEIGHT - 1}"

            tiles_set = 0

            for i in range(length):
                tile_x = x + i

                if steepness == "45":
                    # 45-degree slopes - alternate pattern for longer slopes
                    if direction == "up":
                        if i % 2 == 0:
                            tile_type = "slope_45_bl_tr" if i == 0 else "slope_45_tl_br"
                        else:
                            tile_type = "slope_45_tl_br"
                    else:  # down
                        if i % 2 == 0:
                            tile_type = "slope_45_tl_br" if i == 0 else "slope_45_bl_tr"
                        else:
                            tile_type = "slope_45_bl_tr"

                elif steepness == "mild":
                    # Mild/shallow slopes
                    if raised:
                        if direction == "up":
                            tile_type = (
                                "slope_mild_raised_left"
                                if i == 0
                                else "slope_mild_raised_right"
                            )
                        else:  # down
                            tile_type = (
                                "slope_mild_raised_drop_right"
                                if i == 0
                                else "slope_mild_raised_drop_left"
                            )
                    else:
                        if direction == "up":
                            tile_type = (
                                "slope_mild_up_left"
                                if i == 0
                                else "slope_mild_up_right"
                            )
                        else:  # down
                            tile_type = (
                                "slope_mild_down_right"
                                if i == 0
                                else "slope_mild_down_left"
                            )

                elif steepness == "steep":
                    # Steep slopes
                    if raised:
                        if direction == "up":
                            tile_type = (
                                "slope_steep_raised_left"
                                if i == 0
                                else "slope_steep_raised_right"
                            )
                        else:  # down
                            tile_type = (
                                "slope_steep_raised_drop_right"
                                if i == 0
                                else "slope_steep_raised_drop_left"
                            )
                    else:
                        if direction == "up":
                            tile_type = (
                                "slope_steep_up_left"
                                if i == 0
                                else "slope_steep_up_right"
                            )
                        else:  # down
                            tile_type = (
                                "slope_steep_down_right"
                                if i == 0
                                else "slope_steep_down_left"
                            )

                tile_id = TILE_TYPE_MAPPING[tile_type]
                current_map.set_tile(tile_x, y, tile_id)
                tiles_set += 1

            raised_str = " raised" if raised else ""
            return f"✓ Created {steepness}{raised_str} slope going {direction} at ({x},{y}) with {tiles_set} tiles"

        except Exception as e:
            logger.error(f"Error creating slope: {e}")
            return f"✗ Failed to create slope: {str(e)}"

    @mcp.tool()
    async def create_corridor(
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        width: int = 1,
        height: int = 3,
        wall_type: str = "full",
        floor_type: str = "full",
    ) -> str:
        """Create a corridor connecting two points.

        Args:
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            end_x: Ending X coordinate
            end_y: Ending Y coordinate
            width: Internal width of corridor (1-5)
            height: Internal height of corridor (2-8)
            wall_type: Tile type for walls
            floor_type: Tile type for floor

        Returns:
            Status of corridor creation
        """
        try:
            current_map = map_operations.current_map
            if current_map is None:
                return "✗ No map loaded. Create or load a map first."

            if width < 1 or width > 5:
                return "✗ Width must be between 1 and 5"

            if height < 2 or height > 8:
                return "✗ Height must be between 2 and 8"

            if (
                wall_type not in TILE_TYPE_MAPPING
                or floor_type not in TILE_TYPE_MAPPING
            ):
                return "✗ Invalid wall_type or floor_type"

            wall_id = TILE_TYPE_MAPPING[wall_type]
            floor_id = TILE_TYPE_MAPPING[floor_type]

            # Ensure coordinates are ordered
            if start_x > end_x:
                start_x, end_x = end_x, start_x
            if start_y > end_y:
                start_y, end_y = end_y, start_y

            tiles_set = 0

            # Create L-shaped corridor (horizontal first, then vertical)
            # Horizontal segment
            for x in range(start_x, end_x + width + 2):  # +2 for wall thickness
                for y in range(start_y, start_y + height + 2):
                    if (
                        y == start_y
                        or y == start_y + height + 1
                        or x == start_x
                        or x == start_x + width + 1
                    ):
                        # Wall tiles
                        current_map.set_tile(x, y, wall_id)
                    else:
                        # Clear interior
                        current_map.set_tile(x, y, 0)  # Empty
                        if y == start_y + height:  # Floor
                            current_map.set_tile(x, y, floor_id)
                    tiles_set += 1

            # Vertical segment (if needed)
            if end_y > start_y:
                for y in range(start_y, end_y + 2):
                    for x in range(end_x, end_x + width + 2):
                        if (
                            x == end_x
                            or x == end_x + width + 1
                            or y == start_y
                            or y == end_y + 1
                        ):
                            # Wall tiles
                            current_map.set_tile(x, y, wall_id)
                        else:
                            # Clear interior
                            current_map.set_tile(x, y, 0)  # Empty
                            if y == end_y:  # Floor
                                current_map.set_tile(x, y, floor_id)
                        tiles_set += 1

            return f"✓ Created corridor from ({start_x},{start_y}) to ({end_x},{end_y}) with {tiles_set} tiles"

        except Exception as e:
            logger.error(f"Error creating corridor: {e}")
            return f"✗ Failed to create corridor: {str(e)}"

    @mcp.tool()
    async def create_platform(
        x: int,
        y: int,
        width: int,
        height: int = 1,
        platform_type: str = "full",
        supports: bool = True,
    ) -> str:
        """Create a platform structure.

        Args:
            x: Left X coordinate
            y: Top Y coordinate
            width: Platform width (1-15)
            height: Platform thickness (1-3)
            platform_type: Tile type for platform surface
            supports: Whether to add support pillars underneath

        Returns:
            Status of platform creation
        """
        try:
            current_map = map_operations.current_map
            if current_map is None:
                return "✗ No map loaded. Create or load a map first."

            if width < 1 or width > 15:
                return "✗ Width must be between 1 and 15"

            if height < 1 or height > 3:
                return "✗ Height must be between 1 and 3"

            if platform_type not in TILE_TYPE_MAPPING:
                return f"✗ Invalid platform_type '{platform_type}'"

            platform_id = TILE_TYPE_MAPPING[platform_type]
            tiles_set = 0

            # Create platform
            for px in range(x, x + width):
                for py in range(y, y + height):
                    if 1 <= px < MAP_TILE_WIDTH and 1 <= py < MAP_TILE_HEIGHT:
                        current_map.set_tile(px, py, platform_id)
                        tiles_set += 1

            # Add support pillars if requested
            if supports and y + height < MAP_TILE_HEIGHT - 1:
                support_spacing = max(2, width // 4)  # Support every 2-4 tiles
                for sx in range(x + 1, x + width - 1, support_spacing):
                    # Create pillar down to ground or existing structure
                    for sy in range(y + height, MAP_TILE_HEIGHT - 1):
                        if current_map.tile_data[sx + sy * MAP_TILE_WIDTH] != 0:
                            break  # Hit existing structure
                        current_map.set_tile(sx, sy, platform_id)
                        tiles_set += 1

            support_str = " with supports" if supports else ""
            return f"✓ Created platform at ({x},{y}) size {width}x{height}{support_str} using {tiles_set} tiles"

        except Exception as e:
            logger.error(f"Error creating platform: {e}")
            return f"✗ Failed to create platform: {str(e)}"

    @mcp.tool()
    async def create_ramp(
        x: int,
        y: int,
        width: int,
        height: int,
        direction: Literal[
            "up_right", "up_left", "down_right", "down_left"
        ] = "up_right",
        steepness: Literal["mild", "steep"] = "mild",
    ) -> str:
        """Create a ramp structure connecting different heights.

        Args:
            x: Starting X coordinate
            y: Starting Y coordinate (bottom of ramp)
            width: Horizontal length of ramp (2-10)
            height: Vertical height to traverse (1-5)
            direction: Direction of ramp
            steepness: Ramp steepness (mild or steep)

        Returns:
            Status of ramp creation
        """
        try:
            current_map = map_operations.current_map
            if current_map is None:
                return "✗ No map loaded. Create or load a map first."

            if width < 2 or width > 10:
                return "✗ Width must be between 2 and 10"

            if height < 1 or height > 5:
                return "✗ Height must be between 1 and 5"

            tiles_set = 0

            # Calculate step size
            step_height = height / width

            for i in range(width):
                current_height = int(i * step_height)
                ramp_x = x + (
                    i if direction in ["up_right", "down_right"] else width - 1 - i
                )
                ramp_y = y - current_height

                if not (1 <= ramp_x < MAP_TILE_WIDTH and 1 <= ramp_y < MAP_TILE_HEIGHT):
                    continue

                # Choose appropriate slope tile based on position and steepness
                if i == 0:  # Starting tile
                    if steepness == "mild":
                        tile_type = (
                            "slope_mild_up_left"
                            if "up" in direction
                            else "slope_mild_down_left"
                        )
                    else:
                        tile_type = (
                            "slope_steep_up_left"
                            if "up" in direction
                            else "slope_steep_down_left"
                        )
                elif i == width - 1:  # Ending tile
                    if steepness == "mild":
                        tile_type = (
                            "slope_mild_up_right"
                            if "up" in direction
                            else "slope_mild_down_right"
                        )
                    else:
                        tile_type = (
                            "slope_steep_up_right"
                            if "up" in direction
                            else "slope_steep_down_right"
                        )
                else:  # Middle tiles
                    if steepness == "mild":
                        tile_type = (
                            "slope_mild_raised_right"
                            if "up" in direction
                            else "slope_mild_raised_drop_right"
                        )
                    else:
                        tile_type = (
                            "slope_steep_raised_right"
                            if "up" in direction
                            else "slope_steep_raised_drop_right"
                        )

                tile_id = TILE_TYPE_MAPPING[tile_type]
                current_map.set_tile(ramp_x, ramp_y, tile_id)
                tiles_set += 1

                # Fill in support structure below ramp
                for support_y in range(ramp_y + 1, y + 1):
                    if 1 <= support_y < MAP_TILE_HEIGHT:
                        current_map.set_tile(
                            ramp_x, support_y, TILE_TYPE_MAPPING["full"]
                        )
                        tiles_set += 1

            return f"✓ Created {steepness} ramp going {direction} at ({x},{y}) size {width}x{height} using {tiles_set} tiles"

        except Exception as e:
            logger.error(f"Error creating ramp: {e}")
            return f"✗ Failed to create ramp: {str(e)}"

    @mcp.tool()
    async def create_tunnel(
        x: int,
        y: int,
        length: int,
        direction: Literal["horizontal", "vertical"] = "horizontal",
        width: int = 3,
        add_supports: bool = True,
    ) -> str:
        """Create a tunnel through solid terrain.

        Args:
            x: Starting X coordinate
            y: Starting Y coordinate
            length: Length of tunnel (2-20)
            direction: Tunnel orientation
            width: Internal width/height of tunnel (2-6)
            add_supports: Whether to add structural supports

        Returns:
            Status of tunnel creation
        """
        try:
            current_map = map_operations.current_map
            if current_map is None:
                return "✗ No map loaded. Create or load a map first."

            if length < 2 or length > 20:
                return "✗ Length must be between 2 and 20"

            if width < 2 or width > 6:
                return "✗ Width must be between 2 and 6"

            tiles_set = 0

            if direction == "horizontal":
                # Create horizontal tunnel
                for tx in range(x, x + length):
                    for ty in range(y, y + width):
                        if 1 <= tx < MAP_TILE_WIDTH and 1 <= ty < MAP_TILE_HEIGHT:
                            if ty == y or ty == y + width - 1:
                                # Ceiling and floor - use half tiles for more natural look
                                tile_type = "half_bottom" if ty == y else "half_top"
                            else:
                                # Clear interior space
                                tile_type = "empty"
                            current_map.set_tile(tx, ty, TILE_TYPE_MAPPING[tile_type])
                            tiles_set += 1

                # Add support pillars if requested
                if add_supports and length > 6:
                    support_spacing = length // 3
                    for sx in range(x + support_spacing, x + length, support_spacing):
                        if 1 <= sx < MAP_TILE_WIDTH:
                            # Central support pillar
                            mid_y = y + width // 2
                            current_map.set_tile(sx, mid_y, TILE_TYPE_MAPPING["full"])
                            tiles_set += 1

            else:  # vertical
                # Create vertical tunnel
                for tx in range(x, x + width):
                    for ty in range(y, y + length):
                        if 1 <= tx < MAP_TILE_WIDTH and 1 <= ty < MAP_TILE_HEIGHT:
                            if tx == x or tx == x + width - 1:
                                # Left and right walls - use half tiles
                                tile_type = "half_right" if tx == x else "half_left"
                            else:
                                # Clear interior space
                                tile_type = "empty"
                            current_map.set_tile(tx, ty, TILE_TYPE_MAPPING[tile_type])
                            tiles_set += 1

                # Add support pillars if requested
                if add_supports and length > 6:
                    support_spacing = length // 3
                    for sy in range(y + support_spacing, y + length, support_spacing):
                        if 1 <= sy < MAP_TILE_HEIGHT:
                            # Central support pillar
                            mid_x = x + width // 2
                            current_map.set_tile(mid_x, sy, TILE_TYPE_MAPPING["full"])
                            tiles_set += 1

            support_str = " with supports" if add_supports else ""
            return f"✓ Created {direction} tunnel at ({x},{y}) length {length} width {width}{support_str} using {tiles_set} tiles"

        except Exception as e:
            logger.error(f"Error creating tunnel: {e}")
            return f"✗ Failed to create tunnel: {str(e)}"

    @mcp.tool()
    async def create_chamber(
        x: int,
        y: int,
        width: int,
        height: int,
        wall_thickness: int = 1,
        add_corners: bool = True,
    ) -> str:
        """Create a rectangular chamber with walls.

        Args:
            x: Left X coordinate
            y: Top Y coordinate
            width: Internal width (3-15)
            height: Internal height (3-10)
            wall_thickness: Thickness of walls (1-2)
            add_corners: Whether to add decorative corner pieces

        Returns:
            Status of chamber creation
        """
        try:
            current_map = map_operations.current_map
            if current_map is None:
                return "✗ No map loaded. Create or load a map first."

            if width < 3 or width > 15:
                return "✗ Width must be between 3 and 15"

            if height < 3 or height > 10:
                return "✗ Height must be between 3 and 10"

            if wall_thickness < 1 or wall_thickness > 2:
                return "✗ Wall thickness must be 1 or 2"

            tiles_set = 0
            total_width = width + 2 * wall_thickness
            total_height = height + 2 * wall_thickness

            # Create chamber structure
            for cx in range(x, x + total_width):
                for cy in range(y, y + total_height):
                    if not (1 <= cx < MAP_TILE_WIDTH and 1 <= cy < MAP_TILE_HEIGHT):
                        continue

                    # Determine if this is a wall or interior tile
                    is_wall_x = (
                        cx < x + wall_thickness or cx >= x + width + wall_thickness
                    )
                    is_wall_y = (
                        cy < y + wall_thickness or cy >= y + height + wall_thickness
                    )

                    if is_wall_x or is_wall_y:
                        # Wall tile
                        if add_corners and is_wall_x and is_wall_y:
                            # Corner - use quarter moon tiles
                            if cx < x + wall_thickness and cy < y + wall_thickness:
                                tile_type = "quarter_moon_br"  # Top-left corner
                            elif (
                                cx >= x + width + wall_thickness
                                and cy < y + wall_thickness
                            ):
                                tile_type = "quarter_moon_bl"  # Top-right corner
                            elif (
                                cx < x + wall_thickness
                                and cy >= y + height + wall_thickness
                            ):
                                tile_type = "quarter_moon_tr"  # Bottom-left corner
                            else:
                                tile_type = "quarter_moon_tl"  # Bottom-right corner
                        else:
                            tile_type = "full"  # Regular wall
                    else:
                        # Interior - empty space
                        tile_type = "empty"

                    current_map.set_tile(cx, cy, TILE_TYPE_MAPPING[tile_type])
                    tiles_set += 1

            corner_str = " with corner details" if add_corners else ""
            return f"✓ Created chamber at ({x},{y}) size {width}x{height}{corner_str} using {tiles_set} tiles"

        except Exception as e:
            logger.error(f"Error creating chamber: {e}")
            return f"✗ Failed to create chamber: {str(e)}"
