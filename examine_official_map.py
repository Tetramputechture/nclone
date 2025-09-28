#!/usr/bin/env python3
"""
Examine the structure of official map files to understand the correct format.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from pathlib import Path
import struct

def examine_map_structure(map_path):
    """Examine the byte structure of an official map file."""
    with open(map_path, 'rb') as f:
        data = f.read()
    
    print(f"=== EXAMINING {map_path.name} ===")
    print(f"Total size: {len(data)} bytes")
    print()
    
    # Show first 200 bytes in hex
    print("First 200 bytes (hex):")
    for i in range(0, min(200, len(data)), 16):
        hex_part = ' '.join(f'{b:02x}' for b in data[i:i+16])
        ascii_part = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in data[i:i+16])
        print(f"{i:04x}: {hex_part:<48} {ascii_part}")
    print()
    
    # Check tiles section (184-1149)
    print("=== TILES SECTION (184-1149) ===")
    tiles = data[184:1150]
    print(f"Tiles section size: {len(tiles)} bytes")
    tile_counts = {}
    for tile in tiles:
        tile_counts[tile] = tile_counts.get(tile, 0) + 1
    print(f"Tile type distribution: {dict(sorted(tile_counts.items()))}")
    print()
    
    # Check entity section area (around 1230+)
    print("=== ENTITY SECTION AREA (1200-1300) ===")
    entity_area = data[1200:1300] if len(data) >= 1300 else data[1200:]
    print(f"Entity area size: {len(entity_area)} bytes")
    
    # Show bytes around entity area
    for i in range(0, len(entity_area), 16):
        start_pos = 1200 + i
        hex_part = ' '.join(f'{b:02x}' for b in entity_area[i:i+16])
        ascii_part = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in entity_area[i:i+16])
        print(f"{start_pos:04x}: {hex_part:<48} {ascii_part}")
    print()
    
    # Try to find entity patterns
    print("=== LOOKING FOR ENTITY PATTERNS ===")
    
    # Look for common entity type values (0, 1, 2, 3, 4, etc.)
    for pos in range(1200, min(len(data) - 12, 1300), 4):
        try:
            val = struct.unpack('<I', data[pos:pos+4])[0]
            if 0 <= val <= 10:  # Reasonable entity type range
                x = struct.unpack('<I', data[pos+4:pos+8])[0] if pos+8 <= len(data) else 0
                y = struct.unpack('<I', data[pos+8:pos+12])[0] if pos+12 <= len(data) else 0
                if 0 <= x <= 1000 and 0 <= y <= 1000:  # Reasonable coordinate range
                    print(f"  Pos {pos}: type={val}, x={x}, y={y}")
        except:
            continue
    
    # Look for ninja spawn pattern (type 0)
    print("\n=== LOOKING FOR NINJA SPAWN (type 0) ===")
    for pos in range(1200, min(len(data) - 12, 1300), 4):
        try:
            val = struct.unpack('<I', data[pos:pos+4])[0]
            if val == 0:
                x = struct.unpack('<I', data[pos+4:pos+8])[0] if pos+8 <= len(data) else 0
                y = struct.unpack('<I', data[pos+8:pos+12])[0] if pos+12 <= len(data) else 0
                print(f"  Pos {pos}: NINJA SPAWN at ({x}, {y})")
        except:
            continue
    
    return data

def examine_multiple_maps():
    """Examine multiple official maps to understand the pattern."""
    maps_dir = Path("nclone/maps/official")
    
    if not maps_dir.exists():
        print(f"❌ Official maps directory not found: {maps_dir}")
        return
    
    map_files = list(maps_dir.glob("*"))[:3]  # Examine first 3 maps
    
    for map_file in map_files:
        if map_file.is_file():
            examine_map_structure(map_file)
            print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    examine_multiple_maps()