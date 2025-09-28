#!/usr/bin/env python3
"""
Understand the exact entity format by comparing with attract data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from pathlib import Path
import struct

def analyze_entity_format():
    """Analyze entity format by reverse engineering from attract data."""
    from nclone.replay.npp_attract_decoder import NppAttractDecoder
    
    # Get attract data first (this is our ground truth)
    attract_file = Path("nclone/example_replays/npp_attract/0")
    decoder = NppAttractDecoder()
    attract_data = decoder.decode_npp_attract_file(str(attract_file))
    
    print("=== ATTRACT DATA (GROUND TRUTH) ===")
    print(f"Spawn: {attract_data['ninja_spawn']}")
    print(f"Entities ({len(attract_data['entities'])}):")
    for i, entity in enumerate(attract_data['entities']):
        print(f"  {i}: type={entity.get('type')}, pos=({entity.get('x')}, {entity.get('y')})")
    
    # Now examine official map
    official_file = Path("nclone/maps/official/000 the basics")
    with open(official_file, 'rb') as f:
        data = f.read()
    
    print(f"\n=== OFFICIAL MAP ANALYSIS ===")
    
    # We know spawn is correct at (66, 26)
    spawn_x = data[1231]
    spawn_y = data[1232]
    print(f"Spawn: ({spawn_x}, {spawn_y}) ✅")
    
    # Now let's figure out the entity format
    # The attract data shows entities with both X and Y coordinates
    # But official map Y coordinates are all 0
    
    # Maybe the format is different than I thought?
    # Let me try different interpretations of the entity bytes
    
    print(f"\nEntity format analysis:")
    pos = 1235
    
    for i in range(min(5, len(attract_data['entities']))):
        attract_entity = attract_data['entities'][i]
        
        if pos + 5 <= len(data):
            # Current interpretation
            entity_type = data[pos]
            x_raw = struct.unpack('<H', data[pos+1:pos+3])[0]
            y_raw = struct.unpack('<H', data[pos+3:pos+5])[0]
            
            print(f"\nEntity {i}:")
            print(f"  Attract: type={attract_entity.get('type')}, pos=({attract_entity.get('x')}, {attract_entity.get('y')})")
            print(f"  Official raw: type={entity_type}, x={x_raw}, y={y_raw}")
            
            # Try different coordinate interpretations
            # Maybe it's not 16-bit little endian for coordinates?
            
            # Try single bytes for coordinates
            x_byte1 = data[pos+1]
            x_byte2 = data[pos+2]
            y_byte1 = data[pos+3]
            y_byte2 = data[pos+4]
            
            print(f"  Bytes: {data[pos]:02x} {x_byte1:02x} {x_byte2:02x} {y_byte1:02x} {y_byte2:02x}")
            print(f"  Single bytes: x1={x_byte1}, x2={x_byte2}, y1={y_byte1}, y2={y_byte2}")
            
            # Try big endian
            x_big = struct.unpack('>H', data[pos+1:pos+3])[0]
            y_big = struct.unpack('>H', data[pos+3:pos+5])[0]
            print(f"  Big endian: x={x_big}, y={y_big}")
            
            # Check if any of these match the attract coordinates
            attract_x = attract_entity.get('x', 0)
            attract_y = attract_entity.get('y', 0)
            
            # Test coordinate conversions
            if x_raw > 0 and attract_x > 0:
                x_factor = x_raw / attract_x
                print(f"  X conversion factor: {x_factor:.2f}")
                
                # Test common factors
                for factor in [24, 32, 48, 64, 96]:
                    converted = x_raw // factor
                    if abs(converted - attract_x) <= 1:
                        print(f"    ✅ X: {x_raw} / {factor} = {converted} (attract: {attract_x})")
            
            # For Y, since official is 0, maybe Y is encoded differently
            # Let me check if Y might be in the high byte of X or something
            if x_raw > 0:
                x_high_byte = (x_raw >> 8) & 0xFF
                x_low_byte = x_raw & 0xFF
                print(f"  X split: high={x_high_byte}, low={x_low_byte}")
                
                # Check if high byte matches Y
                if x_high_byte == attract_y:
                    print(f"    ✅ Y might be in X high byte: {x_high_byte} (attract Y: {attract_y})")
                if x_low_byte == attract_x:
                    print(f"    ✅ X might be in X low byte: {x_low_byte} (attract X: {attract_x})")
            
            pos += 5

if __name__ == "__main__":
    analyze_entity_format()