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
        self.level_cache: Dict[str, Tuple[str, int, str, str]] = {}  # name -> (file, line_num, name, map_data)
        self.level_id_cache: Dict[int, Tuple[str, int, str, str]] = {}  # level_id -> (file, line_num, name, map_data)
        self.loaded_files: List[str] = []
        
        if map_path and map_path.exists():
            if map_path.is_file():
                self._load_map_file(map_path)
            elif map_path.is_dir():
                self._load_map_directory(map_path)
    
    def _load_map_directory(self, directory: Path) -> None:
        """Load and parse all map definition files in a directory."""
        map_files = list(directory.glob("*"))
        map_files = [f for f in map_files if f.is_file()]
        if not map_files:
            logger.warning(f"No map files found in directory: {directory}")
            return
            
        logger.info(f"Loading map definitions from {len(map_files)} files in {directory}")
        
        for map_file in map_files:
            self._load_map_file(map_file)
    
    def _load_map_file(self, map_file: Path) -> None:
        """Load and parse a single map definition file."""
        if not map_file.exists():
            logger.warning(f"Map file not found: {map_file}")
            return
            
        try:
            with open(map_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            logger.info(f"Loading map definitions from {map_file} ({len(lines)} lines)")
            file_name = map_file.name
            levels_loaded = 0
            
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('$') and '#' in line:
                    # Parse format: $name#mapdata#
                    parts = line.split('#', 2)
                    if len(parts) >= 2:
                        level_name = parts[0][1:]  # Remove $ prefix
                        map_data = parts[1] if len(parts) > 1 else ""
                        
                        # Store in cache with file information
                        cache_key = level_name.lower()
                        self.level_cache[cache_key] = (file_name, i + 1, level_name, map_data)
                        levels_loaded += 1
                        
                        # Try to extract level ID from name if it contains numbers
                        level_id = self._extract_level_id_from_name(level_name)
                        if level_id is not None:
                            self.level_id_cache[level_id] = (file_name, i + 1, level_name, map_data)
                        
            logger.info(f"Loaded {levels_loaded} level definitions from {file_name}")
            self.loaded_files.append(file_name)
            
        except Exception as e:
            logger.error(f"Error loading map file {map_file}: {e}")
    
    def _extract_level_id_from_name(self, level_name: str) -> Optional[int]:
        """
        Try to extract a level ID from the level name.
        
        This is speculative - looks for patterns like numbers in the name.
        """
        # Look for numbers in the level name
        numbers = re.findall(r'\d+', level_name)
        if numbers:
            try:
                # Use the first number found as potential level ID
                return int(numbers[0])
            except ValueError:
                pass
        return None
    
    def find_map_by_name(self, level_name: str, level_id: Optional[int] = None) -> Optional[Tuple[str, str, str]]:
        """
        Find map data by level name with exact matching only.
        
        Note: Pattern matching removed as npp_attract files contain complete data.
        
        Args:
            level_name: Name of the level to find
            level_id: Optional level ID for additional matching
            
        Returns:
            Tuple of (level_name, map_data, source_file) if found, None otherwise
        """
        if not self.level_cache:
            return None
            
        logger.debug(f"Searching for level: '{level_name}' (ID: {level_id})")
        
        # Try exact match
        level_key = level_name.lower()
        if level_key in self.level_cache:
            file_name, _, name, map_data = self.level_cache[level_key]
            logger.info(f"Found exact match: '{level_name}' -> '{name}' in {file_name}")
            return (name, map_data, file_name)
        
        # Try level ID match if available
        if level_id is not None and level_id in self.level_id_cache:
            file_name, _, name, map_data = self.level_id_cache[level_id]
            logger.info(f"Found ID match: {level_id} -> '{name}' in {file_name}")
            return (name, map_data, file_name)
        
        logger.debug(f"No exact match found for level: '{level_name}' (ID: {level_id})")
        return None
    

    
    def find_map_by_id(self, level_id: int) -> Optional[Tuple[str, str, str]]:
        """
        Find map data by level ID.
        
        Args:
            level_id: ID of the level to find
            
        Returns:
            Tuple of (level_name, map_data, source_file) if found, None otherwise
        """
        if level_id in self.level_id_cache:
            file_name, _, name, map_data = self.level_id_cache[level_id]
            logger.info(f"Found level by ID {level_id}: '{name}' in {file_name}")
            return (name, map_data, file_name)
        
        return None
    
    def get_available_levels(self) -> List[Tuple[str, str, int]]:
        """
        Get list of available levels.
        
        Returns:
            List of (level_name, source_file, line_number) tuples
        """
        return [(name, file_name, line_num) for name, (file_name, line_num, _, _) in self.level_cache.items()]
    
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
            "files_count": len(self.loaded_files)
        }
    
    def convert_map_data_to_nclone_format(self, map_data: str) -> bytes:
        """
        Convert N++ map data format to nclone binary format.
        
        N++ uses a string-based format where each character represents map elements.
        nclone expects a 1245-byte binary format with header + tile data + entity data.
        
        Args:
            map_data: Raw map data string from N++ definition file
            
        Returns:
            Binary map data in nclone format
        """
        try:
            logger.info(f"Converting N++ map data ({len(map_data)} chars) to nclone format")
            
            # Create the binary map structure (1245 bytes total)
            binary_map = bytearray(1245)
            
            # Set header (first 184 bytes) - based on simple-walk format
            binary_map[0:4] = [6, 0, 0, 0]      # Map type
            binary_map[4:8] = [221, 4, 0, 0]    # Size/checksum
            binary_map[8:12] = [255, 255, 255, 255]  # Unknown field
            binary_map[12:16] = [4, 0, 0, 0]    # Entity count
            binary_map[16:20] = [37, 0, 0, 0]   # Additional header data
            
            # Parse N++ map data to extract tile information
            # N++ format appears to be a string representation of the map
            tile_data = self._parse_npp_tile_data(map_data)
            
            # Fill tile data section (bytes 184-1150, 42x23 grid = 966 tiles)
            tile_start = 184
            for i, tile_value in enumerate(tile_data):
                if tile_start + i < 1150:  # Don't exceed tile data section
                    binary_map[tile_start + i] = tile_value
            
            # Parse and add entity data (bytes 1150+)
            entity_data = self._parse_npp_entity_data(map_data)
            entity_start = 1150
            for i, entity_byte in enumerate(entity_data):
                if entity_start + i < 1245:  # Don't exceed total size
                    binary_map[entity_start + i] = entity_byte
            
            logger.info(f"Successfully converted N++ map to nclone binary format ({len(binary_map)} bytes)")
            return bytes(binary_map)
            
        except Exception as e:
            logger.error(f"Failed to convert N++ map data: {e}")
            logger.warning("Falling back to empty map generation")
            return self._generate_empty_map_data()
    
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
    
    def _parse_npp_entity_data(self, map_data: str) -> List[int]:
        """
        Parse N++ map data to extract entity information.
        
        Note: npp_attract files contain complete entity data that is handled
        by the perfect npp_attract decoder. This method provides fallback
        basic entity data for other use cases.
        
        Args:
            map_data: N++ map data string
            
        Returns:
            List of entity data bytes (basic fallback)
        """
        # Basic fallback entity data - npp_attract files use perfect decoder
        entity_data = []
        
        # Add basic ninja spawn and exit
        entity_data.extend([
            0, 0, 0, 0,  # Ninja spawn entity
            100, 0, 0, 0,  # X position
            100, 0, 0, 0,  # Y position
            3, 0, 0, 0,  # Exit door entity type
            200, 0, 0, 0,  # X position
            100, 0, 0, 0,  # Y position
        ])
        
        # Pad remaining space
        remaining_space = 95 - len(entity_data)
        entity_data.extend([0] * max(0, remaining_space))
        
        return entity_data[:95]
    
    def _generate_empty_map_data(self) -> bytes:
        """Generate empty map data for fallback."""
        # This matches the implementation in binary_replay_parser.py
        empty_map = bytearray(1245)  # Standard empty map size
        
        # Set basic map structure (minimal viable map)
        empty_map[0:4] = b'\x00\x00\x00\x00'  # Map header
        empty_map[4:8] = b'\x00\x00\x00\x00'  # Dimensions or flags
        
        # Fill rest with zeros (empty space)
        return bytes(empty_map)


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