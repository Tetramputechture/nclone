#!/usr/bin/env python3
"""
Examine the exact entity structure in official maps.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from pathlib import Path
import struct

def examine_entity_bytes():
    """Examine the exact byte structure of entities."""
    
    map_path = Path("nclone/maps/official/000 the basics")
    with open(map_path, 'rb') as f:
        data = f.read()
    
    print("=== DETAILED ENTITY BYTE ANALYSIS ===")
    
    # Look at the exact bytes around the entity section
    print("Raw hex data from 1230-1340:")
    for i in range(1230, min(1340, len(data)), 16):
        hex_part = ' '.join(f'{b:02x}' for b in data[i:i+16])
        ascii_part = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in data[i:i+16])
        print(f"{i:04x}: {hex_part:<48} {ascii_part}")
    
    print(f"\nLet's analyze the pattern starting at 1231:")
    print(f"Bytes 1231-1235: {' '.join(f'{b:02x}' for b in data[1231:1235])}")
    
    # The pattern I see is: 42 1a 00 00 01 10 12 00 00 01 50 10 00 00 01 ac 0e 00 00 02
    # This could be:
    # 42 1a = spawn x (6722)
    # 00 00 = spawn y (0) - but this seems wrong
    # 01 = entity type 1
    # 10 12 = entity x (4624)  
    # 00 00 = entity y (0) - also seems wrong
    
    # Let me try a different interpretation
    print(f"\nTrying different interpretations:")
    
    # Maybe the spawn is at a different location?
    # Let me look for the pattern that matches attract spawn (66, 26)
    
    # Convert attract spawn to different formats to find pattern
    attract_spawn = (66, 26)
    print(f"Looking for spawn pattern matching attract ({attract_spawn[0]}, {attract_spawn[1]}):")
    
    # Try different byte positions and formats
    for pos in range(1200, 1250):
        if pos + 4 <= len(data):
            # Try 16-bit little endian
            val1 = struct.unpack('<H', data[pos:pos+2])[0]
            val2 = struct.unpack('<H', data[pos+2:pos+4])[0]
            if val1 == attract_spawn[0] and val2 == attract_spawn[1]:
                print(f"  Found spawn at pos {pos}: ({val1}, {val2})")
            
            # Try 8-bit values
            if data[pos] == attract_spawn[0] and data[pos+1] == attract_spawn[1]:
                print(f"  Found spawn (8-bit) at pos {pos}: ({data[pos]}, {data[pos+1]})")
    
    # Let me also check if the entity coordinates make more sense in a different format
    print(f"\nAnalyzing entity coordinate patterns:")
    
    # Look at the attract entities again
    attract_entities = [
        (1, 16, 18), (1, 80, 16), (1, 172, 14),
        (2, 40, 12), (2, 48, 12), (2, 56, 12), (2, 64, 12), (2, 72, 12), (2, 80, 12),
        (2, 24, 12), (2, 32, 12), (2, 88, 12), (2, 96, 12), (2, 104, 12), (2, 112, 12),
        (2, 120, 12), (2, 128, 12), (2, 136, 12), (3, 150, 14), (4, 162, 14)
    ]
    
    # Try to find these patterns in the official data
    pos = 1235  # Start after potential spawn
    for i, (exp_type, exp_x, exp_y) in enumerate(attract_entities[:5]):  # Check first 5
        if pos + 5 <= len(data):
            actual_type = data[pos]
            actual_x = struct.unpack('<H', data[pos+1:pos+3])[0]
            actual_y = struct.unpack('<H', data[pos+3:pos+5])[0]
            
            print(f"  Entity {i}: expected ({exp_type}, {exp_x}, {exp_y}), actual ({actual_type}, {actual_x}, {actual_y})")
            
            # Check if coordinates need conversion
            if actual_x > 0 and exp_x > 0:
                x_ratio = actual_x / exp_x
                print(f"    X ratio: {x_ratio:.2f}")
            
            pos += 5

if __name__ == "__main__":
    examine_entity_bytes()