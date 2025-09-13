"""
Map analysis and validation tools for the N++ MCP server.

This module provides connectivity analysis, level validation,
and playability checks for N++ levels.
"""

import logging

from fastmcp import FastMCP
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT

# TILE_NAME_MAPPING not currently used in this module
from . import map_operations

logger = logging.getLogger(__name__)


def register_analysis_tools(mcp: FastMCP) -> None:
    """Register all analysis and validation tools with the FastMCP server."""

    @mcp.tool()
    async def analyze_map_connectivity() -> str:
        """Analyze the current map's connectivity and provide navigation insights.

        Returns:
            Analysis of map connectivity, isolated areas, and navigation paths
        """
        try:
            current_map = map_operations.current_map
            if current_map is None:
                return "✗ No map loaded. Create or load a map first."

            # Find empty spaces (navigable areas)
            empty_spaces = []
            for y in range(1, MAP_TILE_HEIGHT):
                for x in range(1, MAP_TILE_WIDTH):
                    tile_type = current_map.tile_data[x + y * MAP_TILE_WIDTH]
                    if tile_type == 0:  # Empty tile
                        empty_spaces.append((x, y))

            if not empty_spaces:
                return "📊 Map Analysis: No navigable empty spaces found. Map is completely solid."

            # Simple connectivity analysis using flood fill
            visited = set()
            connected_regions = []

            def flood_fill(start_x, start_y):
                """Flood fill to find connected regions."""
                if (start_x, start_y) in visited:
                    return []

                region = []
                stack = [(start_x, start_y)]

                while stack:
                    x, y = stack.pop()
                    if (x, y) in visited:
                        continue
                    if (x, y) not in empty_spaces:
                        continue

                    visited.add((x, y))
                    region.append((x, y))

                    # Check 4-connected neighbors
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 1 <= nx < MAP_TILE_WIDTH and 1 <= ny < MAP_TILE_HEIGHT:
                            if (nx, ny) not in visited:
                                stack.append((nx, ny))

                return region

            for x, y in empty_spaces:
                if (x, y) not in visited:
                    region = flood_fill(x, y)
                    if region:
                        connected_regions.append(region)

            # Count different tile types
            tile_counts = {}
            for tile_type in current_map.tile_data:
                tile_counts[tile_type] = tile_counts.get(tile_type, 0) + 1

            # Analyze slopes and special tiles
            slope_tiles = sum(1 for t in current_map.tile_data if 6 <= t <= 33)
            quarter_features = sum(1 for t in current_map.tile_data if 10 <= t <= 17)

            # Find ninja spawn connectivity
            ninja_x = (
                current_map.ninja_spawn_x // 6
            )  # Convert from screen to grid coords
            ninja_y = current_map.ninja_spawn_y // 6
            spawn_region = None
            for i, region in enumerate(connected_regions):
                if (ninja_x, ninja_y) in region or any(
                    abs(ninja_x - x) <= 1 and abs(ninja_y - y) <= 1 for x, y in region
                ):
                    spawn_region = i
                    break

            analysis = f"""📊 Map Connectivity Analysis:

🌍 Navigable Areas:
  Total empty spaces: {len(empty_spaces)} tiles
  Connected regions: {len(connected_regions)}
  Largest region: {max(len(r) for r in connected_regions) if connected_regions else 0} tiles
  Average region size: {len(empty_spaces) // len(connected_regions) if connected_regions else 0} tiles

🎯 Ninja Spawn Analysis:
  Spawn position: ({ninja_x}, {ninja_y})
  Spawn connectivity: {"Connected to main area" if spawn_region is not None else "Isolated or needs analysis"}
  
🧱 Terrain Features:
  Total tiles: {len(current_map.tile_data)}
  Solid tiles: {tile_counts.get(1, 0)}
  Empty tiles: {tile_counts.get(0, 0)}
  Slope tiles: {slope_tiles}
  Quarter features (moons/pipes): {quarter_features}
  
{"⚠️  Warning: Multiple disconnected regions found! Consider adding corridors." if len(connected_regions) > 1 else "✅ Good connectivity - single navigable area."}

💡 Suggestions:
{("- Connect isolated regions with corridors or ramps" + chr(10)) if len(connected_regions) > 1 else ""}- Add more slope variations for interesting traversal
- Consider adding platforms or quarter-pipe features
- Use create_corridor() to link separated areas"""

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing map connectivity: {e}")
            return f"✗ Failed to analyze map connectivity: {str(e)}"

    @mcp.tool()
    async def validate_level() -> str:
        """Validate that the current level meets N++ requirements for playability.

        Returns:
            Detailed validation report with any issues found
        """
        try:
            current_map = map_operations.current_map
            if current_map is None:
                return "✗ No map loaded. Create or load a map first."

            issues = []
            warnings = []

            # Check for ninja spawn point
            has_ninja = True
            if current_map.ninja_spawn_x is None or current_map.ninja_spawn_y is None:
                has_ninja = False
                issues.append("❌ No ninja spawn point defined")
            else:
                # Validate ninja spawn position is within playable area
                ninja_grid_x = (
                    current_map.ninja_spawn_x // 6
                )  # Convert screen to grid coords
                ninja_grid_y = current_map.ninja_spawn_y // 6
                if not (
                    1 <= ninja_grid_x < MAP_TILE_WIDTH
                    and 1 <= ninja_grid_y < MAP_TILE_HEIGHT
                ):
                    issues.append(
                        f"❌ Ninja spawn at ({ninja_grid_x}, {ninja_grid_y}) is outside playable area"
                    )
                else:
                    # Check if ninja spawns in a solid tile
                    tile_index = ninja_grid_x + ninja_grid_y * MAP_TILE_WIDTH
                    if (
                        tile_index < len(current_map.tile_data)
                        and current_map.tile_data[tile_index] == 1
                    ):
                        warnings.append(
                            f"⚠️  Ninja spawns inside solid tile at ({ninja_grid_x}, {ninja_grid_y})"
                        )

            # Check for exit doors
            exit_door_count = current_map.entity_counts.get("exit_door", 0)
            if exit_door_count == 0:
                issues.append("❌ No exit doors found - level cannot be completed")
            elif exit_door_count > 1:
                warnings.append(
                    f"⚠️  Multiple exit doors ({exit_door_count}) found - unusual but allowed"
                )

            # Check entity data integrity for exit doors
            exit_doors_found = 0
            switches_found = 0
            for i in range(0, len(current_map.entity_data), 5):
                if i + 4 < len(current_map.entity_data):
                    entity_type = current_map.entity_data[i]
                    if entity_type == 3:  # Exit door
                        exit_doors_found += 1
                    elif entity_type == 4:  # Switch
                        switches_found += 1

            if exit_doors_found != current_map.entity_counts.get("exit_door", 0):
                issues.append(
                    f"❌ Exit door count mismatch: expected {current_map.entity_counts.get('exit_door', 0)}, found {exit_doors_found}"
                )

            # Each exit door should have a corresponding switch
            if exit_doors_found > 0 and switches_found == 0:
                issues.append(
                    "❌ Exit doors found but no switches - doors cannot be activated"
                )
            elif exit_doors_found != switches_found and exit_doors_found > 0:
                warnings.append(
                    f"⚠️  Exit door/switch count mismatch: {exit_doors_found} doors, {switches_found} switches"
                )

            # Check for reasonable playable area
            empty_tiles = sum(1 for tile in current_map.tile_data if tile == 0)
            total_tiles = len(current_map.tile_data)
            empty_percentage = (empty_tiles / total_tiles) * 100

            if empty_tiles == 0:
                issues.append("❌ No empty tiles found - ninja cannot move")
            elif empty_percentage < 5:
                warnings.append(
                    f"⚠️  Very little empty space ({empty_percentage:.1f}%) - may be unplayable"
                )
            elif empty_percentage > 80:
                warnings.append(
                    f"⚠️  Mostly empty space ({empty_percentage:.1f}%) - may be too easy"
                )

            # Note: Boundaries are automatically solid due to simulation padding, so no need to check

            # Generate validation report
            status = "✅ VALID" if len(issues) == 0 else "❌ INVALID"

            report = f"""🔍 Level Validation Report: {status}

📊 Requirements Check:
  Ninja spawn point: {"✅ Present" if has_ninja else "❌ Missing"}
  Exit door(s): {"✅ Present" if exit_door_count > 0 else "❌ Missing"} ({exit_door_count} found)
  
📈 Level Statistics:
  Total tiles: {total_tiles}
  Empty tiles: {empty_tiles} ({empty_percentage:.1f}%)
  Total entities: {len(current_map.entity_data) // 5}
  Gold pieces: {current_map.entity_counts.get("gold", 0)}
  Death balls: {current_map.entity_counts.get("death_ball", 0)}"""

            if issues:
                report += f"\n\n❌ Critical Issues ({len(issues)}):"
                for issue in issues:
                    report += f"\n  {issue}"

            if warnings:
                report += f"\n\n⚠️  Warnings ({len(warnings)}):"
                for warning in warnings:
                    report += f"\n  {warning}"

            if len(issues) == 0:
                report += "\n\n✅ Level is valid and ready for gameplay!"
            else:
                report += (
                    "\n\n💡 Fix the critical issues above to make this level playable."
                )

            return report

        except Exception as e:
            logger.error(f"Error validating level: {e}")
            return f"✗ Failed to validate level: {str(e)}"
