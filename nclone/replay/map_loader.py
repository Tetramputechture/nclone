"""
Map loader for N++ replay files.

This module provides functionality to load actual map data from N++ map definition files
when available, with fallback to empty maps for attract replay files.

Note: npp_attract files contain complete map data and use the perfect decoder.
This loader provides enhanced map data from official sources when available.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import re

logger = logging.getLogger(__name__)


class MapLoader:
    """Loads N++ map data from definition files."""

    def __init__(self, map_path: Optional[Path] = None):
        """
        Initialize the map loader.

        Args:
            map_path: Path to N++ map definition file or directory containing map files
                     (format: $name#mapdata#)
        """
        # Convert string path to Path object if needed
        if isinstance(map_path, str):
            map_path = Path(map_path)

        self.map_path = map_path
        self.level_cache: Dict[
            str, Tuple[str, int, str, str]
        ] = {}  # name -> (file, line_num, name, map_data)
        self.level_id_cache: Dict[
            int, Tuple[str, int, str, str]
        ] = {}  # level_id -> (file, line_num, name, map_data)
        self.loaded_files: List[str] = []

        if map_path and map_path.exists():
            if map_path.is_file():
                self._load_map_file(map_path)

    def _load_map_file(self, map_file: Path) -> None:
        """Load and parse a single map definition file."""
        if not map_file.exists():
            logger.warning(f"Map file not found: {map_file}")
            return

        try:
            # Check if this is a binary map file (like those created by npp_attract_decoder)
            if self._is_binary_map_file(map_file):
                self._load_binary_map_file(map_file)
                return

            # Load text-based map definition file
            with open(map_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            logger.info(f"Loading map definitions from {map_file} ({len(lines)} lines)")
            file_name = map_file.name
            levels_loaded = 0

            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith("$") and "#" in line:
                    # Parse format: $name#mapdata#
                    parts = line.split("#", 2)
                    if len(parts) >= 2:
                        level_name = parts[0][1:]  # Remove $ prefix
                        map_data = parts[1] if len(parts) > 1 else ""

                        # Store in cache with file information
                        cache_key = level_name.lower()
                        self.level_cache[cache_key] = (
                            file_name,
                            i + 1,
                            level_name,
                            map_data,
                        )
                        levels_loaded += 1

                        # Try to extract level ID from name if it contains numbers
                        level_id = self._extract_level_id_from_name(level_name)
                        if level_id is not None:
                            self.level_id_cache[level_id] = (
                                file_name,
                                i + 1,
                                level_name,
                                map_data,
                            )

            logger.info(f"Loaded {levels_loaded} level definitions from {file_name}")
            self.loaded_files.append(file_name)

        except Exception as e:
            logger.error(f"Error loading map file {map_file}: {e}")

    def _is_binary_map_file(self, map_file: Path) -> bool:
        """Check if a file is a binary map file (like those from npp_attract_decoder)."""
        try:
            # Check file size - nclone binary maps are typically 1335 bytes
            file_size = map_file.stat().st_size
            if file_size == 1335:
                return True
            
            # Check if file contains binary data by trying to read first few bytes
            with open(map_file, "rb") as f:
                header = f.read(10)
                # If it contains non-printable bytes, it's likely binary
                if any(b < 32 and b not in [9, 10, 13] for b in header):
                    return True
                    
        except Exception:
            pass
        return False

    def _load_binary_map_file(self, map_file: Path) -> None:
        """Load a binary map file created by npp_attract_decoder."""
        try:
            with open(map_file, "rb") as f:
                binary_data = f.read()
            
            logger.info(f"Loading binary map from {map_file} ({len(binary_data)} bytes)")
            
            # For binary maps, we don't have a level name, so use the filename
            level_name = map_file.stem
            
            # Convert binary data to a format that can be used by the environment
            # For now, we'll store the raw binary data and let the environment handle it
            cache_key = level_name.lower()
            self.level_cache[cache_key] = (
                map_file.name,
                1,
                level_name,
                binary_data,  # Store binary data directly
            )
            
            # Also store with a generic key for binary maps
            self.level_cache["binary_map"] = (
                map_file.name,
                1,
                level_name,
                binary_data,
            )
            
            logger.info(f"Loaded binary map: {level_name}")
            self.loaded_files.append(map_file.name)
            
        except Exception as e:
            logger.error(f"Error loading binary map file {map_file}: {e}")

    def _extract_level_id_from_name(self, level_name: str) -> Optional[int]:
        """
        Try to extract a level ID from the level name.

        This is speculative - looks for patterns like numbers in the name.
        """
        # Look for numbers in the level name
        numbers = re.findall(r"\d+", level_name)
        if numbers:
            try:
                # Use the first number found as potential level ID
                return int(numbers[0])
            except ValueError:
                pass
        return None

    def get_statistics(self) -> Dict[str, any]:
        """
        Get loader statistics.

        Returns:
            Dictionary with loader statistics
        """
        return {
            "total_levels": len(self.level_cache),
            "levels_with_ids": len(self.level_id_cache),
            "loaded_files": self.loaded_files.copy(),
            "files_count": len(self.loaded_files),
        }

    def _parse_npp_tile_data(self, map_data: str) -> List[int]:
        """
        Parse N++ map data string to extract tile information.

        N++ uses character-based encoding for tiles. This function converts
        the string representation to the integer tile values expected by nclone.

        Args:
            map_data: N++ map data string

        Returns:
            List of tile values (0-255) for 42x23 grid
        """
        # Initialize 42x23 tile grid (966 tiles total)
        tiles = [0] * (42 * 23)

        # Basic tile parsing - npp_attract files contain complete map data
        for i, char in enumerate(map_data):
            if i >= len(tiles):
                break
            if char.isdigit():
                tiles[i] = int(char)
            else:
                tiles[i] = 0
        logger.debug(f"Parsed {len(tiles)} tiles from map data")
        return tiles


def create_map_loader(map_file_path: Optional[str] = None) -> MapLoader:
    """
    Create a map loader instance.

    Args:
        map_file_path: Optional path to N++ map definition file

    Returns:
        MapLoader instance
    """
    path = Path(map_file_path) if map_file_path else None
    return MapLoader(path)
