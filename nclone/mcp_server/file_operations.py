"""
File import/export operations for the N++ MCP server.

This module handles saving and loading maps in various formats,
including binary files and JSON data export.
"""

import json
import logging
from pathlib import Path

from fastmcp import FastMCP
from ..map_generation.map import Map
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from . import map_operations

logger = logging.getLogger(__name__)


def register_file_operations(mcp: FastMCP) -> None:
    """Register all file operation tools with the FastMCP server."""

    @mcp.tool()
    async def export_map_data() -> str:
        """Export the current map as raw binary data that can be used by the simulator.

        Returns:
            JSON string containing the map data array
        """
        try:
            current_map = map_operations.current_map
            if current_map is None:
                return "✗ No map loaded. Create or load a map first."

            # Quick validation check before export
            has_ninja = (
                current_map.ninja_spawn_x is not None
                and current_map.ninja_spawn_y is not None
            )
            has_exit = current_map.entity_counts.get("exit_door", 0) > 0
            is_valid = has_ninja and has_exit

            map_data = current_map.map_data()

            # Create export info
            export_info = {
                "map_data": map_data,
                "metadata": {
                    "size": f"{MAP_TILE_WIDTH}x{MAP_TILE_HEIGHT}",
                    "total_data_points": len(map_data),
                    "ninja_spawn": [
                        current_map.ninja_spawn_x,
                        current_map.ninja_spawn_y,
                    ],
                    "ninja_orientation": current_map.ninja_orientation,
                    "entity_counts": current_map.entity_counts,
                    "total_entities": len(current_map.entity_data) // 5,
                    "is_valid_level": is_valid,
                },
            }

            result = json.dumps(export_info, indent=2)

            # Add validation warning if needed
            if not is_valid:
                validation_issues = []
                if not has_ninja:
                    validation_issues.append("missing ninja spawn")
                if not has_exit:
                    validation_issues.append("missing exit door")

                warning = f"\n\n⚠️  WARNING: This level is incomplete ({', '.join(validation_issues)}) and may not be playable. Use validate_level() for details."
                result += warning

            return result

        except Exception as e:
            logger.error(f"Error exporting map data: {e}")
            return f"✗ Failed to export map data: {str(e)}"

    @mcp.tool()
    async def save_map_to_file(filepath: str) -> str:
        """Save the current map to a binary file compatible with the nclone simulator.

        Args:
            filepath: Path where to save the map file

        Returns:
            Status of the save operation
        """
        try:
            current_map = map_operations.current_map
            if current_map is None:
                return "✗ No map loaded. Create or load a map first."

            # Quick validation check before saving
            has_ninja = (
                current_map.ninja_spawn_x is not None
                and current_map.ninja_spawn_y is not None
            )
            has_exit = current_map.entity_counts.get("exit_door", 0) > 0
            is_valid = has_ninja and has_exit

            map_data = current_map.map_data()

            # Convert to bytes and write to file
            file_path = Path(filepath)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "wb") as f:
                for value in map_data:
                    f.write(value.to_bytes(4, byteorder="little", signed=True))

            result = f"✓ Saved map to {filepath} ({len(map_data)} data points, {file_path.stat().st_size} bytes)"

            # Add validation warning if needed
            if not is_valid:
                validation_issues = []
                if not has_ninja:
                    validation_issues.append("missing ninja spawn")
                if not has_exit:
                    validation_issues.append("missing exit door")

                result += f"\n\n⚠️  WARNING: Saved level is incomplete ({', '.join(validation_issues)}) and may not be playable. Use validate_level() for details."

            return result

        except Exception as e:
            logger.error(f"Error saving map to file: {e}")
            return f"✗ Failed to save map: {str(e)}"

    @mcp.tool()
    async def load_map_from_file(filepath: str) -> str:
        """Load a map from a binary file.

        Args:
            filepath: Path to the map file to load

        Returns:
            Status of the load operation and basic map info
        """
        try:
            file_path = Path(filepath)
            if not file_path.exists():
                return f"✗ File not found: {filepath}"

            # Read binary file
            with open(file_path, "rb") as f:
                data = f.read()

            # Convert bytes to integers
            if len(data) % 4 != 0:
                return f"✗ Invalid file format. File size {len(data)} is not divisible by 4"

            map_data = []
            for i in range(0, len(data), 4):
                value = int.from_bytes(data[i : i + 4], byteorder="little", signed=True)
                map_data.append(value)

            # Create map from data
            map_operations.current_map = Map.from_map_data(map_data)
            current_map = map_operations.current_map

            # Get basic info
            entity_counts = current_map.entity_counts
            total_entities = len(current_map.entity_data) // 5

            return (
                f"✓ Loaded map from {filepath} ({len(map_data)} data points). "
                f"Statistics: {entity_counts['exit_door']} exit door(s), "
                f"{entity_counts['gold']} gold, {total_entities} total entities."
            )

        except Exception as e:
            logger.error(f"Error loading map from file: {e}")
            return f"✗ Failed to load map: {str(e)}"
