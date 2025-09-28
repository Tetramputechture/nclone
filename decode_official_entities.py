#!/usr/bin/env python3
"""
Properly decode entities from official map files.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from pathlib import Path
import struct

def decode_official_entities(map_path):
    """Properly decode entities from official map file."""
    with open(map_path, 'rb') as f:
        data = f.read()
    
    print(f"=== DECODING {map_path.name} ===")
    print(f"Total size: {len(data)} bytes")
    
    # Look at the hex data around 1231 (0x04cf)
    print("\nHex data around entity section (1220-1280):")
    for i in range(1220, min(1280, len(data)), 16):
        hex_part = ' '.join(f'{b:02x}' for b in data[i:i+16])
        print(f"{i:04x}: {hex_part}")
    
    # The pattern I see is:
    # 04cf: 42 1a 00 00 01 10 12 00 00 01 50 10 00 00 01 ac
    # This looks like: 42 1a 00 00 (ninja spawn x,y) followed by entities
    
    # Try decoding ninja spawn at position 1231
    if len(data) >= 1235:
        spawn_x = struct.unpack('<H', data[1231:1233])[0]  # 16-bit little endian
        spawn_y = struct.unpack('<H', data[1233:1235])[0]  # 16-bit little endian
        print(f"\nNinja spawn (16-bit): ({spawn_x}, {spawn_y})")
    
    # Try entities starting at 1235
    entities = []
    pos = 1235
    entity_count = 0
    
    print(f"\nDecoding entities starting at position {pos}:")
    while pos + 4 < len(data) and entity_count < 20:  # Safety limit
        # Try reading entity type as single byte
        entity_type = data[pos]
        if entity_type == 0:
            break
            
        # Try reading coordinates as 16-bit values
        if pos + 4 < len(data):
            x = struct.unpack('<H', data[pos+1:pos+3])[0]
            y = struct.unpack('<H', data[pos+3:pos+5])[0]
            
            entities.append({
                'type': entity_type,
                'x': x,
                'y': y,
                'pos': pos
            })
            
            print(f"  Entity {entity_count}: type={entity_type}, pos=({x}, {y}) at byte {pos}")
            pos += 5  # Move to next entity (1 byte type + 2 bytes x + 2 bytes y)
            entity_count += 1
        else:
            break
    
    print(f"\nFound {len(entities)} entities")
    return entities, (spawn_x, spawn_y) if len(data) >= 1235 else (0, 0)

def compare_with_attract():
    """Compare official map entities with attract file entities."""
    from nclone.replay.npp_attract_decoder import NppAttractDecoder
    
    # Load official data
    official_file = Path("nclone/maps/official/000 the basics")
    official_entities, official_spawn = decode_official_entities(official_file)
    
    # Load attract data
    attract_file = Path("nclone/example_replays/npp_attract/0")
    decoder = NppAttractDecoder()
    attract_data = decoder.decode_npp_attract_file(str(attract_file))
    
    print(f"\n=== COMPARISON ===")
    print(f"Official: {len(official_entities)} entities, spawn {official_spawn}")
    print(f"Attract:  {len(attract_data['entities'])} entities, spawn {attract_data['ninja_spawn']}")
    
    print(f"\nOfficial entities:")
    for i, entity in enumerate(official_entities):
        print(f"  {i}: type={entity['type']}, pos=({entity['x']}, {entity['y']})")
    
    print(f"\nAttract entities:")
    for i, entity in enumerate(attract_data['entities']):
        print(f"  {i}: type={entity.get('type', 'unknown')}, pos=({entity.get('x', 0)}, {entity.get('y', 0)})")

if __name__ == "__main__":
    compare_with_attract()