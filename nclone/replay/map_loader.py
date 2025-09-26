"""
Map loader for N++ replay files.

This module provides functionality to load actual map data from N++ map definition files
when available, with fallback to empty maps for attract replay files.

Supports both single map files and directories containing multiple .txt map definition files.
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
            map_path: Path to N++ map definition file or directory containing .txt files
                     (format: $name#mapdata#)
        """
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
        """Load and parse all .txt map definition files in a directory."""
        txt_files = list(directory.glob("*.txt"))
        if not txt_files:
            logger.warning(f"No .txt files found in directory: {directory}")
            return
            
        logger.info(f"Loading map definitions from {len(txt_files)} files in {directory}")
        
        for txt_file in txt_files:
            self._load_map_file(txt_file)
    
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
        Find map data by level name with enhanced correlation logic.
        
        Args:
            level_name: Name of the level to find
            level_id: Optional level ID for additional correlation
            
        Returns:
            Tuple of (level_name, map_data, source_file) if found, None otherwise
        """
        if not self.level_cache:
            return None
            
        logger.debug(f"Searching for level: '{level_name}' (ID: {level_id})")
        
        # Strategy 1: Try exact match first
        level_key = level_name.lower()
        if level_key in self.level_cache:
            file_name, _, name, map_data = self.level_cache[level_key]
            logger.info(f"Found exact match: '{level_name}' -> '{name}' in {file_name}")
            return (name, map_data, file_name)
        
        # Strategy 2: Try fuzzy matching with common N++ level name patterns
        best_match = self._find_fuzzy_match(level_name, level_id)
        if best_match:
            return best_match
        
        # Strategy 3: If we have a level ID, try to find levels with similar IDs
        if level_id is not None:
            id_match = self._find_by_similar_id(level_id)
            if id_match:
                return id_match
        
        logger.debug(f"No match found for level: '{level_name}' (ID: {level_id})")
        return None
    
    def _find_fuzzy_match(self, level_name: str, level_id: Optional[int] = None) -> Optional[Tuple[str, str, str]]:
        """Find level using fuzzy matching strategies with scoring."""
        level_key = level_name.lower()
        cleaned_name = self._clean_level_name(level_name)
        
        # Collect all potential matches with scores
        candidates = []
        
        for cached_name, (file_name, line_num, name, map_data) in self.level_cache.items():
            score = self._calculate_match_score(level_name, cleaned_name, cached_name, name, file_name)
            if score > 0:
                candidates.append((score, name, map_data, file_name, line_num))
        
        if candidates:
            # Sort by score (highest first) and return the best match
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_name, best_map_data, best_file, best_line = candidates[0]
            
            logger.info(f"Found fuzzy match: '{level_name}' -> '{best_name}' in {best_file} (line {best_line}, score: {best_score:.3f})")
            return (best_name, best_map_data, best_file)
        
        return None
    
    def _calculate_match_score(self, original_name: str, cleaned_name: str, cached_name: str, full_name: str, file_name: str) -> float:
        """Calculate a match score between level names."""
        score = 0.0
        
        # Bonus for intro files (SI.txt, CI.txt, RI.txt) since attract files are intro levels
        if file_name.startswith(('SI.txt', 'CI.txt', 'RI.txt')):
            score += 0.3
        
        # Check for exact word matches
        original_words = set(original_name.lower().split())
        cached_words = set(full_name.lower().split())
        
        if original_words == cached_words:
            score += 1.0  # Perfect match
        elif original_words.issubset(cached_words) or cached_words.issubset(original_words):
            score += 0.8  # Subset match
        else:
            # Calculate word overlap
            common_words = original_words.intersection(cached_words)
            if common_words:
                word_score = len(common_words) / max(len(original_words), len(cached_words))
                score += word_score * 0.6
        
        # Check for substring matches
        if cleaned_name in cached_name or cached_name in cleaned_name:
            score += 0.4
        
        # Use similarity score for character-level matching
        similarity = self._similarity_score(cached_name, cleaned_name)
        score += similarity * 0.3
        
        # Special handling for common attract level patterns
        if self._is_attract_level_pattern_match(original_name, full_name):
            score += 0.5
        
        return score
    
    def _is_attract_level_pattern_match(self, attract_name: str, level_name: str) -> bool:
        """Check for specific attract level patterns."""
        attract_lower = attract_name.lower()
        level_lower = level_name.lower()
        
        # Handle common truncations in attract files
        patterns = [
            ('he basics', 'the basics'),
            ('alljumptroduction', 'walljumptroduction'),
            ('ntro to accepting', 'intro to accepting'),
            ('all-to-wall', 'wall-to-wall'),
            ('ump mechanics', 'jump mechanics'),
            ('all jumping', 'wall jumping'),
        ]
        
        for attract_pattern, level_pattern in patterns:
            if attract_pattern in attract_lower and level_pattern in level_lower:
                return True
        
        return False
    
    def _find_by_similar_id(self, level_id: int) -> Optional[Tuple[str, str, str]]:
        """Find level by similar ID (within a range)."""
        # Look for levels with IDs within ±50 of the target ID
        id_range = 50
        for candidate_id in range(level_id - id_range, level_id + id_range + 1):
            if candidate_id in self.level_id_cache:
                file_name, line_num, name, map_data = self.level_id_cache[candidate_id]
                logger.info(f"Found similar ID match: {level_id} -> {candidate_id} ('{name}') in {file_name}")
                return (name, map_data, file_name)
        
        return None
    
    def _clean_level_name(self, level_name: str) -> str:
        """Clean level name for better matching."""
        # Remove common artifacts and normalize
        cleaned = level_name.lower()
        
        # Remove common N++ prefixes/suffixes
        prefixes_to_remove = ['intro to ', 'the ', 'a ', 'an ']
        suffixes_to_remove = [' intro', ' tutorial', ' basics']
        
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break
                
        for suffix in suffixes_to_remove:
            if cleaned.endswith(suffix):
                cleaned = cleaned[:-len(suffix)]
                break
        
        # Remove punctuation and extra spaces
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _similarity_score(self, str1: str, str2: str) -> float:
        """Calculate similarity score between two strings."""
        # Simple Jaccard similarity based on character n-grams
        def get_ngrams(s: str, n: int = 2) -> set:
            return set(s[i:i+n] for i in range(len(s) - n + 1))
        
        ngrams1 = get_ngrams(str1)
        ngrams2 = get_ngrams(str2)
        
        if not ngrams1 and not ngrams2:
            return 1.0
        if not ngrams1 or not ngrams2:
            return 0.0
            
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
    
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
        
        try:
            # N++ map data appears to be encoded as characters
            # Common patterns observed: '0' = empty, '1' = wall, '6' = special
            
            # Process the map data string
            data_index = 0
            for tile_index in range(len(tiles)):
                if data_index < len(map_data):
                    char = map_data[data_index]
                    
                    # Convert character to tile value
                    if char.isdigit():
                        tile_value = int(char)
                        # Map N++ tile values to nclone tile values
                        if tile_value == 0:
                            tiles[tile_index] = 0  # Empty space
                        elif tile_value == 1:
                            tiles[tile_index] = 1  # Wall/solid tile
                        elif tile_value == 6:
                            tiles[tile_index] = 6  # Special tile (spawn/exit)
                        else:
                            tiles[tile_index] = min(tile_value, 255)  # Other tile types
                    else:
                        # Non-digit characters might represent special elements
                        tiles[tile_index] = 0  # Default to empty
                    
                    data_index += 1
                else:
                    # Fill remaining tiles with empty space
                    tiles[tile_index] = 0
            
            logger.debug(f"Parsed {len(tiles)} tiles from N++ map data")
            return tiles
            
        except Exception as e:
            logger.error(f"Error parsing N++ tile data: {e}")
            # Return empty tile grid on error
            return [0] * (42 * 23)
    
    def _parse_npp_entity_data(self, map_data: str) -> List[int]:
        """
        Parse N++ map data to extract entity information.
        
        This is a simplified implementation that creates basic entity data.
        A full implementation would need to parse N++ entity encoding.
        
        Args:
            map_data: N++ map data string
            
        Returns:
            List of entity data bytes
        """
        # For now, create minimal entity data
        # This would need to be enhanced to parse actual N++ entity format
        entity_data = []
        
        # Add basic ninja spawn point (required for simulation)
        # Entity format: [type, x, y, additional_data...]
        entity_data.extend([
            0, 0, 0, 0,  # Ninja spawn entity
            100, 0, 0, 0,  # X position (low bytes first)
            100, 0, 0, 0,  # Y position (low bytes first)
        ])
        
        # Add exit door (required for level completion)
        entity_data.extend([
            3, 0, 0, 0,  # Exit door entity type
            200, 0, 0, 0,  # X position
            100, 0, 0, 0,  # Y position
        ])
        
        # Pad to fill remaining space
        remaining_space = 95 - len(entity_data)  # 95 bytes available for entities
        entity_data.extend([0] * max(0, remaining_space))
        
        return entity_data[:95]  # Ensure we don't exceed available space
    
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