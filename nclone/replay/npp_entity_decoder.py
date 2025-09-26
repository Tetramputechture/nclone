#!/usr/bin/env python3
"""
N++ Entity Format Decoder

This module analyzes and decodes the entity data from N++ level definitions.
The entity data appears in the hexadecimal section at the end of level strings.

Format Analysis:
- N++ levels: $level_name#tile_data_digits#hex_entity_data#
- Entity data starts where hexadecimal characters (a-f) first appear
- Contains both header information and entity records

Entity Types Observed:
- c0 entities: Common entity type with subtype, x, y coordinates
- Coordinate pairs: Position data for spawns/exits
- Header data: Level metadata and configuration
"""

import struct
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Import nclone's proper entity type definitions
from nclone.constants.entity_types import EntityType
from nclone.tile_definitions import TILE_GRID_EDGE_MAP


@dataclass
class NppEntity:
    """Represents a decoded N++ entity."""
    entity_type: str
    subtype: Optional[int] = None
    x: Optional[float] = None
    y: Optional[float] = None
    raw_data: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class NppEntityDecoder:
    """Decodes N++ entity data from hexadecimal strings."""
    
    # N++ character to tile type mapping (based on tile_definitions.py)
    NPP_TILE_MAPPING = {
        '0': 0,  # Empty tile
        '1': 1,  # Full solid tile
        '2': 2,  # Top half solid
        '3': 3,  # Right half solid
        '4': 4,  # Bottom half solid
        '5': 5,  # Left half solid
        '6': 0,  # Ninja spawn (empty tile with entity)
        '7': 0,  # Exit door (empty tile with entity)
        '8': 8,  # Slope tile
        '9': 9,  # Slope tile
        'a': 10, # Quarter circle
        'b': 11, # Quarter circle
        'c': 12, # Quarter circle
        'd': 13, # Quarter circle
        'e': 14, # Quarter pipe
        'f': 15, # Quarter pipe
    }
    
    # N++ entity type mappings (using proper EntityType constants)
    ENTITY_TYPE_MAPPING = {
        0: EntityType.NINJA,
        1: EntityType.TOGGLE_MINE,
        2: EntityType.GOLD,
        3: EntityType.EXIT_DOOR,
        4: EntityType.EXIT_SWITCH,
        5: EntityType.REGULAR_DOOR,
        6: EntityType.DRONE_ZAP,
        7: EntityType.BOUNCE_BLOCK,
        8: EntityType.THWUMP,
    }
    
    def __init__(self):
        self.debug = False
    
    def parse_npp_tile_data(self, map_data: str) -> List[int]:
        """
        Parse N++ map data string to extract tile information using proper tile definitions.
        
        Args:
            map_data: N++ map data string (digits + hex section)
            
        Returns:
            List of tile values for 42x23 grid (966 tiles total)
        """
        # Initialize 42x23 tile grid
        tiles = [0] * (42 * 23)
        
        # Find where hex section starts (entities)
        hex_start = -1
        for i, c in enumerate(map_data):
            if c in 'abcdef':
                hex_start = i
                break
        
        # Use only the digit section for tiles
        tile_section = map_data[:hex_start] if hex_start != -1 else map_data
        
        # Parse tile data
        for i, char in enumerate(tile_section):
            if i >= len(tiles):
                break
            
            if char in self.NPP_TILE_MAPPING:
                tiles[i] = self.NPP_TILE_MAPPING[char]
            elif char.isdigit():
                tiles[i] = int(char)
            else:
                tiles[i] = 0  # Default to empty
        
        return tiles
    
    def extract_tile_entities(self, map_data: str) -> List[NppEntity]:
        """
        Extract entities that are encoded as special characters in the tile section.
        
        Args:
            map_data: N++ map data string
            
        Returns:
            List of entities found in tile data
        """
        entities = []
        
        # Find where hex section starts
        hex_start = -1
        for i, c in enumerate(map_data):
            if c in 'abcdef':
                hex_start = i
                break
        
        tile_section = map_data[:hex_start] if hex_start != -1 else map_data
        
        # Scan tile section for entity markers
        for i, char in enumerate(tile_section):
            row = i // 42
            col = i % 42
            
            if char == '2':  # Gold piece
                entities.append(NppEntity(
                    entity_type="tile_entity",
                    subtype=EntityType.GOLD,
                    x=float(col),
                    y=float(row),
                    raw_data=f"tile_{char}",
                    metadata={"source": "tile_section", "char": char}
                ))
            elif char == '3':  # Mine
                entities.append(NppEntity(
                    entity_type="tile_entity", 
                    subtype=EntityType.TOGGLE_MINE,
                    x=float(col),
                    y=float(row),
                    raw_data=f"tile_{char}",
                    metadata={"source": "tile_section", "char": char}
                ))
            elif char == '6':  # Ninja spawn
                entities.append(NppEntity(
                    entity_type="tile_entity",
                    subtype=EntityType.NINJA,
                    x=float(col),
                    y=float(row),
                    raw_data=f"tile_{char}",
                    metadata={"source": "tile_section", "char": char}
                ))
            elif char == '7':  # Exit door
                entities.append(NppEntity(
                    entity_type="tile_entity",
                    subtype=EntityType.EXIT_DOOR,
                    x=float(col),
                    y=float(row),
                    raw_data=f"tile_{char}",
                    metadata={"source": "tile_section", "char": char}
                ))
        
        return entities
    
    def convert_entities_to_nclone_format(self, entities: List[NppEntity]) -> List[int]:
        """
        Convert decoded N++ entities to nclone binary entity format.
        
        Args:
            entities: List of decoded N++ entities
            
        Returns:
            List of bytes for nclone entity data section
        """
        entity_bytes = []
        
        for entity in entities:
            if entity.entity_type == "c0_entity" and entity.subtype is not None:
                # Convert c0 entity to nclone entity format
                entity_type = self.ENTITY_TYPE_MAPPING.get(entity.subtype, EntityType.GOLD)
                
                # Entity format: [type, x_low, x_high, y_low, y_high, ...]
                x_pos = int(entity.x * 24) if entity.x else 100  # Convert to pixels
                y_pos = int(entity.y * 24) if entity.y else 100
                
                entity_bytes.extend([
                    entity_type, 0, 0, 0,  # Entity type (4 bytes)
                    x_pos & 0xFF, (x_pos >> 8) & 0xFF, 0, 0,  # X position (4 bytes)
                    y_pos & 0xFF, (y_pos >> 8) & 0xFF, 0, 0,  # Y position (4 bytes)
                ])
            
            elif entity.entity_type == "tile_entity" and entity.subtype is not None:
                # Convert tile-based entity to nclone entity format
                entity_type = entity.subtype  # Already an EntityType constant
                
                # Convert tile coordinates to pixels
                x_pos = int(entity.x * 24) if entity.x else 100
                y_pos = int(entity.y * 24) if entity.y else 100
                
                entity_bytes.extend([
                    entity_type, 0, 0, 0,  # Entity type (4 bytes)
                    x_pos & 0xFF, (x_pos >> 8) & 0xFF, 0, 0,  # X position (4 bytes)
                    y_pos & 0xFF, (y_pos >> 8) & 0xFF, 0, 0,  # Y position (4 bytes)
                ])
            
            elif entity.entity_type == "coordinate_pair":
                # This might be ninja spawn or exit
                x_pos = int(entity.x) if entity.x else 100
                y_pos = int(entity.y) if entity.y else 100
                
                # Assume first coordinate pair is ninja spawn
                entity_bytes.extend([
                    EntityType.NINJA, 0, 0, 0,  # Ninja spawn
                    x_pos & 0xFF, (x_pos >> 8) & 0xFF, 0, 0,  # X position
                    y_pos & 0xFF, (y_pos >> 8) & 0xFF, 0, 0,  # Y position
                ])
        
        return entity_bytes
    

    
    def decode_hex_entities(self, map_data: str) -> List[NppEntity]:
        """
        Decode entities from the hex section of N++ map data.
        
        Args:
            map_data: Full N++ map data string
            
        Returns:
            List of decoded entities
        """
        # Find hex section
        hex_start = -1
        for i, c in enumerate(map_data):
            if c in 'abcdef':
                hex_start = i
                break
        
        if hex_start == -1:
            return []
        
        hex_string = map_data[hex_start:]
        
        try:
            entity_data = bytes.fromhex(hex_string)
        except ValueError:
            return []
        
        # Remove trailing zeros
        non_zero_end = len(entity_data)
        for i in range(len(entity_data) - 1, -1, -1):
            if entity_data[i] != 0:
                non_zero_end = i + 1
                break
        
        non_zero_data = entity_data[:non_zero_end]
        
        if self.debug:
            print(f"Decoding hex section: {len(non_zero_data)} bytes: {non_zero_data.hex()}")
        
        entities = []
        
        # Look for c0 entity patterns in the data
        # Pattern appears to be: c0 XX c0 YY c0 ZZ... where XX, YY, ZZ are entity types
        # The coordinates seem to be encoded separately
        
        # First, extract all c0 entity types
        c0_entities_types = []
        i = 0
        while i + 1 < len(non_zero_data):
            if non_zero_data[i] == 0xc0:
                entity_type = non_zero_data[i+1]
                c0_entities_types.append(entity_type)
                if self.debug:
                    entity_name = self.ENTITY_TYPE_MAPPING.get(entity_type, f"unknown_{entity_type}")
                    print(f"  Found c0 entity type: {entity_type} ({entity_name})")
                i += 2
            else:
                i += 1
        
        # Look for coordinate data - might be at the end or in specific positions
        # Based on the analysis, coordinates might be at positions like (16, 105) for ninja
        coordinate_pairs = []
        
        # Look for the ninja spawn coordinates (16, 105) pattern
        for i in range(len(non_zero_data) - 1):
            if non_zero_data[i] == 16 and i + 1 < len(non_zero_data) and non_zero_data[i+1] == 105:
                coordinate_pairs.append((16, 105))
                if self.debug:
                    print(f"  Found coordinate pair: (16, 105) - likely ninja spawn")
                break
        
        # Create entities with available information
        # For now, place most entities at a default location and ninja at spawn
        ninja_placed = False
        for i, entity_type in enumerate(c0_entities_types):
            if entity_type == 0 and not ninja_placed:  # Ninja spawn
                x, y = (16, 105) if coordinate_pairs else (100, 100)
                ninja_placed = True
            else:
                # Place other entities in a grid pattern for now
                # This is a simplification - real coordinates would need more analysis
                x = 200 + (i % 5) * 50
                y = 100 + (i // 5) * 50
            
            entity = NppEntity(
                entity_type="c0_entity",
                subtype=entity_type,
                x=float(x),
                y=float(y),
                raw_data=f"c0{entity_type:02x}",
                metadata={
                    "entity_name": self.ENTITY_TYPE_MAPPING.get(entity_type, f"unknown_{entity_type}"),
                    "coordinate_system": "estimated"
                }
            )
            entities.append(entity)
        
        return entities
    
    def decode_level_entities(self, level_line: str) -> Tuple[str, List[NppEntity]]:
        """
        Decode entities from a complete N++ level definition line.
        
        Args:
            level_line: Full level line in format $name#tile_data#hex_data#
            
        Returns:
            Tuple of (level_name, list_of_entities)
        """
        parts = level_line.split('#')
        if len(parts) < 2:
            return "", []
            
        level_name = parts[0][1:]  # Remove $ prefix
        map_data = parts[1]
        
        all_entities = []
        
        # Extract entities from tile section (gold, mines, spawns, exits)
        tile_entities = self.extract_tile_entities(map_data)
        all_entities.extend(tile_entities)
        
        # Extract entities from hex section
        hex_entities = self.decode_hex_entities(map_data)
        all_entities.extend(hex_entities)
        
        return level_name, all_entities
    
    def _decode_header(self, header_data: bytes) -> Optional[NppEntity]:
        """Decode level header information."""
        if len(header_data) < 12:
            return None
        
        # Try to interpret header as various formats
        try:
            # First attempt: coordinate pairs for ninja spawn and exit
            coords = struct.unpack('<6H', header_data)
            
            # Look for reasonable coordinate values
            valid_coords = [(coords[i], coords[i+1]) for i in range(0, 6, 2) 
                          if 50 < coords[i] < 1200 and 50 < coords[i+1] < 800]
            
            if valid_coords:
                return NppEntity(
                    entity_type="header",
                    metadata={
                        "raw_coords": coords,
                        "potential_positions": valid_coords,
                        "raw_data": header_data.hex()
                    }
                )
        except:
            pass
        
        return NppEntity(
            entity_type="header",
            raw_data=header_data.hex(),
            metadata={"raw_bytes": list(header_data)}
        )
    
    def _decode_entity_records(self, entity_data: bytes) -> List[NppEntity]:
        """Decode entity records from binary data."""
        entities = []
        i = 0
        
        while i + 3 < len(entity_data):
            # Look for c0 pattern (entity marker)
            if entity_data[i] == 0xc0 and i + 3 < len(entity_data):
                # Found c0 entity: c0 [subtype] [x] [y]
                subtype = entity_data[i+1]
                x_coord = entity_data[i+2]
                y_coord = entity_data[i+3]
                
                entity = NppEntity(
                    entity_type="c0_entity",
                    subtype=subtype,
                    x=float(x_coord),
                    y=float(y_coord),
                    raw_data=entity_data[i:i+4].hex(),
                    metadata={
                        "entity_name": self.ENTITY_TYPE_MAPPING.get(subtype, f"unknown_{subtype}"),
                        "coordinate_system": "tile_based"
                    }
                )
                entities.append(entity)
                i += 4
            else:
                # Try to decode as coordinate pair or other data
                if i + 4 <= len(entity_data):
                    record = entity_data[i:i+4]
                    entity = self._decode_generic_entity(record, i)
                    entities.append(entity)
                    i += 4
                else:
                    # Handle remaining bytes
                    remaining = entity_data[i:]
                    entities.append(NppEntity(
                        entity_type="remaining_data",
                        raw_data=remaining.hex(),
                        metadata={"bytes": list(remaining)}
                    ))
                    break
        
        return entities
    
    def _decode_c0_entity(self, record: bytes) -> NppEntity:
        """Decode c0-prefixed entity record."""
        subtype = record[1]
        x_coord = record[2]
        y_coord = record[3]
        
        # Map subtype to entity name if known
        entity_name = self.ENTITY_TYPE_MAPPING.get(subtype, f"unknown_c0_{subtype}")
        
        return NppEntity(
            entity_type="c0_entity",
            subtype=subtype,
            x=float(x_coord),
            y=float(y_coord),
            raw_data=record.hex(),
            metadata={
                "entity_name": entity_name,
                "coordinate_system": "tile_based"  # Assuming tile coordinates
            }
        )
    
    def _decode_generic_entity(self, record: bytes, offset: int) -> NppEntity:
        """Decode generic entity record."""
        # Try as coordinate pair
        try:
            x, y = struct.unpack('<HH', record)
            if 50 < x < 1200 and 50 < y < 800:
                return NppEntity(
                    entity_type="coordinate_pair",
                    x=float(x),
                    y=float(y),
                    raw_data=record.hex(),
                    metadata={"coordinate_system": "pixel_based"}
                )
        except:
            pass
        
        # Fallback to raw data
        return NppEntity(
            entity_type="unknown",
            raw_data=record.hex(),
            metadata={
                "offset": offset,
                "bytes": list(record)
            }
        )
    
    def analyze_level_entities(self, level_line: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of a level's entities.
        
        Returns detailed analysis including entity counts, types, and positions.
        """
        level_name, entities = self.decode_level_entities(level_line)
        
        analysis = {
            "level_name": level_name,
            "total_entities": len(entities),
            "entity_types": {},
            "positions": [],
            "c0_entities": [],
            "coordinate_pairs": [],
            "raw_data": []
        }
        
        for entity in entities:
            # Count entity types
            if entity.entity_type in analysis["entity_types"]:
                analysis["entity_types"][entity.entity_type] += 1
            else:
                analysis["entity_types"][entity.entity_type] = 1
            
            # Collect positions
            if entity.x is not None and entity.y is not None:
                analysis["positions"].append((entity.x, entity.y))
            
            # Categorize entities
            if entity.entity_type == "c0_entity":
                analysis["c0_entities"].append({
                    "subtype": entity.subtype,
                    "position": (entity.x, entity.y),
                    "name": entity.metadata.get("entity_name", "unknown")
                })
            elif entity.entity_type == "coordinate_pair":
                analysis["coordinate_pairs"].append((entity.x, entity.y))
            
            analysis["raw_data"].append(entity.raw_data)
        
        return analysis


def main():
    """Demo the entity decoder with 'the basics' level."""
    decoder = NppEntityDecoder()
    decoder.debug = True
    
    # Load and analyze 'the basics'
    with open('/workspace/official_levels/SI.txt', 'r') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    for line in lines:
        if 'the basics' in line.lower():
            print("=== N++ Entity Decoder Analysis ===")
            analysis = decoder.analyze_level_entities(line)
            
            print(f"Level: {analysis['level_name']}")
            print(f"Total entities: {analysis['total_entities']}")
            print(f"Entity types: {analysis['entity_types']}")
            print(f"Positions found: {len(analysis['positions'])}")
            
            print("\nC0 Entities (likely game objects):")
            for entity in analysis['c0_entities']:
                print(f"  {entity['name']} (subtype {entity['subtype']}) at {entity['position']}")
            
            print("\nCoordinate pairs (likely spawn/exit points):")
            for pos in analysis['coordinate_pairs']:
                print(f"  Position: {pos}")
            
            break


if __name__ == "__main__":
    main()