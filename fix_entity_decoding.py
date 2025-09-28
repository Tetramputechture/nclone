#!/usr/bin/env python3
"""
Fix entity decoding to match official map format exactly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from pathlib import Path
import struct

def decode_official_correctly(map_path):
    """Decode official map with correct format understanding."""
    with open(map_path, 'rb') as f:
        data = f.read()
    
    print(f"=== CORRECT DECODING OF {map_path.name} ===")
    
    # Ninja spawn at position 1231 (8-bit values)
    spawn_x = data[1231]
    spawn_y = data[1232]
    ninja_spawn = (spawn_x, spawn_y)
    print(f"Ninja spawn: ({spawn_x}, {spawn_y})")
    
    # Entities start at position 1235 (after 00 00 padding)
    entities = []
    pos = 1235
    entity_count = 0
    
    print(f"\nDecoding entities starting at position {pos}:")
    while pos + 4 < len(data) and entity_count < 25:  # Safety limit
        entity_type = data[pos]
        if entity_type == 0:
            break
            
        # Read coordinates as 16-bit little endian
        x_raw = struct.unpack('<H', data[pos+1:pos+3])[0]
        y_raw = struct.unpack('<H', data[pos+3:pos+5])[0]
        
        entities.append({
            'type': entity_type,
            'x_raw': x_raw,
            'y_raw': y_raw,
            'pos': pos
        })
        
        print(f"  Entity {entity_count}: type={entity_type}, raw_pos=({x_raw}, {y_raw}) at byte {pos}")
        pos += 5
        entity_count += 1
    
    return entities, ninja_spawn

def analyze_coordinate_system():
    """Analyze the coordinate system used in official vs attract files."""
    from nclone.replay.npp_attract_decoder import NppAttractDecoder
    
    # Load official data
    official_file = Path("nclone/maps/official/000 the basics")
    official_entities, official_spawn = decode_official_correctly(official_file)
    
    # Load attract data  
    attract_file = Path("nclone/example_replays/npp_attract/0")
    decoder = NppAttractDecoder()
    attract_data = decoder.decode_npp_attract_file(str(attract_file))
    
    print(f"\n=== COORDINATE SYSTEM ANALYSIS ===")
    print(f"Official spawn: {official_spawn}")
    print(f"Attract spawn:  {attract_data['ninja_spawn']}")
    print(f"Spawn match: {official_spawn == attract_data['ninja_spawn']}")
    
    print(f"\nEntity coordinate analysis:")
    print(f"Official entities: {len(official_entities)}")
    print(f"Attract entities:  {len(attract_data['entities'])}")
    
    # The key insight: official maps store coordinates in a different format
    # Let me check if the raw coordinates need to be converted to tile coordinates
    
    # N++ uses 24x24 pixel tiles typically
    # But let me check what conversion makes sense
    
    print(f"\nTesting coordinate conversions:")
    for i in range(min(5, len(official_entities), len(attract_data['entities']))):
        off_ent = official_entities[i]
        att_ent = attract_data['entities'][i]
        
        print(f"Entity {i}:")
        print(f"  Official: type={off_ent['type']}, raw=({off_ent['x_raw']}, {off_ent['y_raw']})")
        print(f"  Attract:  type={att_ent.get('type')}, pos=({att_ent.get('x')}, {att_ent.get('y')})")
        
        # Try different conversions
        if off_ent['x_raw'] > 0 and att_ent.get('x', 0) > 0:
            # Test if it's a simple division
            for divisor in [24, 32, 48, 64, 96]:
                converted_x = off_ent['x_raw'] // divisor
                if abs(converted_x - att_ent.get('x', 0)) <= 2:  # Allow small difference
                    print(f"    Possible X conversion: {off_ent['x_raw']} / {divisor} = {converted_x} (attract: {att_ent.get('x')})")
        
        if off_ent['y_raw'] > 0 and att_ent.get('y', 0) > 0:
            for divisor in [24, 32, 48, 64, 96]:
                converted_y = off_ent['y_raw'] // divisor
                if abs(converted_y - att_ent.get('y', 0)) <= 2:
                    print(f"    Possible Y conversion: {off_ent['y_raw']} / {divisor} = {converted_y} (attract: {att_ent.get('y')})")

def check_y_coordinates():
    """Check if Y coordinates are stored differently."""
    
    map_path = Path("nclone/maps/official/000 the basics")
    with open(map_path, 'rb') as f:
        data = f.read()
    
    print("=== Y COORDINATE INVESTIGATION ===")
    
    # The Y coordinates in my previous analysis were all 0, which seems wrong
    # Let me check if they're stored in a different location or format
    
    # Look at the raw bytes again, focusing on potential Y coordinate storage
    print("Raw entity bytes with potential Y coordinates:")
    pos = 1235
    for i in range(5):  # First 5 entities
        if pos + 5 <= len(data):
            entity_type = data[pos]
            x_raw = struct.unpack('<H', data[pos+1:pos+3])[0]
            y_raw = struct.unpack('<H', data[pos+3:pos+5])[0]
            
            # Show the actual bytes
            bytes_str = ' '.join(f'{b:02x}' for b in data[pos:pos+5])
            print(f"  Entity {i} bytes: {bytes_str}")
            print(f"    Type: {entity_type}, X: {x_raw}, Y: {y_raw}")
            
            # Maybe Y is stored as a single byte?
            y_byte = data[pos+3]
            print(f"    Y as single byte: {y_byte}")
            
            pos += 5

if __name__ == "__main__":
    decode_official_correctly(Path("nclone/maps/official/000 the basics"))
    print("\n" + "="*60 + "\n")
    analyze_coordinate_system()
    print("\n" + "="*60 + "\n")
    check_y_coordinates()