#!/usr/bin/env python3
"""
Analyze current accuracy of npp_attract decoder vs official maps.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from pathlib import Path
from nclone.replay.npp_attract_decoder import NppAttractDecoder
import struct

def load_official_map(map_path):
    """Load official map data and extract entities and spawn."""
    with open(map_path, 'rb') as f:
        data = f.read()
    
    print(f"Official map size: {len(data)} bytes")
    
    # Extract tiles (184-1149, 966 bytes)
    tiles = list(data[184:1150])
    print(f"Official tiles: {len(tiles)} values")
    
    # Extract entities (starting at 1230, 5 bytes each)
    entities = []
    pos = 1230
    while pos + 4 < len(data):
        entity_type = struct.unpack('<I', data[pos:pos+4])[0]
        if entity_type == 0:
            break
        x = struct.unpack('<I', data[pos+4:pos+8])[0] if pos+8 <= len(data) else 0
        y = struct.unpack('<I', data[pos+8:pos+12])[0] if pos+12 <= len(data) else 0
        entities.append({
            'type': entity_type,
            'x': x,
            'y': y,
            'pos': pos
        })
        pos += 12  # Each entity is 12 bytes (type + x + y)
    
    # Extract ninja spawn (1231-1232 for x, 1235-1236 for y)
    if len(data) >= 1240:
        spawn_x = struct.unpack('<I', data[1231:1235])[0]
        spawn_y = struct.unpack('<I', data[1235:1239])[0]
        ninja_spawn = (spawn_x, spawn_y)
    else:
        ninja_spawn = (0, 0)
    
    return {
        'tiles': tiles,
        'entities': entities,
        'ninja_spawn': ninja_spawn
    }

def analyze_attract_vs_official():
    """Compare attract decoder output to official map data."""
    
    # Test files
    attract_file = Path("nclone/example_replays/npp_attract/0")
    official_file = Path("nclone/maps/official/000 the basics")
    
    if not attract_file.exists():
        print(f"❌ Attract file not found: {attract_file}")
        return
    
    if not official_file.exists():
        print(f"❌ Official file not found: {official_file}")
        return
    
    print("=== ACCURACY ANALYSIS ===")
    print(f"Attract file: {attract_file}")
    print(f"Official file: {official_file}")
    print()
    
    # Load official data
    print("Loading official map data...")
    official_data = load_official_map(official_file)
    
    print(f"Official: {len(official_data['tiles'])} tiles, {len(official_data['entities'])} entities, spawn {official_data['ninja_spawn']}")
    
    # Load attract data
    print("\nLoading attract file data...")
    decoder = NppAttractDecoder()
    attract_data = decoder.decode_npp_attract_file(str(attract_file))
    
    print(f"Attract: {len(attract_data['tiles'])} tiles, {len(attract_data['entities'])} entities, spawn {attract_data['ninja_spawn']}")
    
    # Compare tiles
    print("\n=== TILE COMPARISON ===")
    tile_matches = sum(1 for i, (a, b) in enumerate(zip(attract_data['tiles'], official_data['tiles'])) if a == b)
    tile_accuracy = tile_matches / len(official_data['tiles']) * 100
    print(f"Tile accuracy: {tile_matches}/{len(official_data['tiles'])} = {tile_accuracy:.1f}%")
    
    if tile_accuracy < 100:
        print("First 10 tile mismatches:")
        for i, (a, b) in enumerate(zip(attract_data['tiles'], official_data['tiles'])):
            if a != b:
                print(f"  Tile {i}: attract={a}, official={b}")
                if len([x for x in zip(attract_data['tiles'], official_data['tiles']) if x[0] != x[1]]) >= 10:
                    break
    
    # Compare entities
    print("\n=== ENTITY COMPARISON ===")
    print(f"Entity count: attract={len(attract_data['entities'])}, official={len(official_data['entities'])}")
    
    print("\nOfficial entities:")
    for i, entity in enumerate(official_data['entities']):
        print(f"  {i}: type={entity['type']}, pos=({entity['x']}, {entity['y']})")
    
    print("\nAttract entities:")
    for i, entity in enumerate(attract_data['entities']):
        print(f"  {i}: type={entity.get('type', 'unknown')}, pos=({entity.get('x', 0)}, {entity.get('y', 0)})")
    
    # Compare spawn
    print("\n=== SPAWN COMPARISON ===")
    spawn_match = attract_data['ninja_spawn'] == official_data['ninja_spawn']
    print(f"Spawn match: {spawn_match}")
    print(f"  Attract spawn: {attract_data['ninja_spawn']}")
    print(f"  Official spawn: {official_data['ninja_spawn']}")
    
    # Summary
    print("\n=== ACCURACY SUMMARY ===")
    print(f"✅ Tiles: {tile_accuracy:.1f}% ({tile_matches}/{len(official_data['tiles'])})")
    print(f"{'✅' if len(attract_data['entities']) == len(official_data['entities']) else '❌'} Entity count: {len(attract_data['entities'])}/{len(official_data['entities'])}")
    print(f"{'✅' if spawn_match else '❌'} Spawn position: {spawn_match}")
    
    return {
        'tile_accuracy': tile_accuracy,
        'entity_count_match': len(attract_data['entities']) == len(official_data['entities']),
        'spawn_match': spawn_match,
        'attract_data': attract_data,
        'official_data': official_data
    }

if __name__ == "__main__":
    analyze_attract_vs_official()