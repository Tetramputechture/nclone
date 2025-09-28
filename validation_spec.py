#!/usr/bin/env python3
"""
Comprehensive validation specification for npp_attract decoder accuracy.

This module provides 100% accuracy validation by comparing attract file decoder
output against official map data with exact entity and spawn position matching.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from pathlib import Path
import struct
from typing import Dict, List, Tuple, Any
from nclone.replay.npp_attract_decoder import NppAttractDecoder

class ValidationSpec:
    """Comprehensive validation specification for npp_attract decoder."""
    
    def __init__(self):
        self.decoder = NppAttractDecoder()
    
    def load_official_map_data(self, map_path: Path) -> Dict[str, Any]:
        """Load official map data with correct entity format."""
        with open(map_path, 'rb') as f:
            data = f.read()
        
        # Ninja spawn at position 1231-1232 (8-bit values)
        spawn_x = data[1231]
        spawn_y = data[1232]
        ninja_spawn = (spawn_x, spawn_y)
        
        # Tiles at position 184-1149 (966 bytes)
        tiles = list(data[184:1150])
        
        # Entities start at position 1235 (after spawn + padding)
        entities = []
        pos = 1235
        entity_count = 0
        
        while pos + 4 < len(data) and entity_count < 25:  # Safety limit
            entity_type = data[pos]
            if entity_type == 0:
                break
            
            # Coordinates are single bytes (not 16-bit!)
            x = data[pos + 1]  # Single byte X coordinate
            y = data[pos + 2]  # Single byte Y coordinate
            # pos+3 and pos+4 are padding (00 00)
            
            entities.append({
                'type': entity_type,
                'x': x,
                'y': y,
                'pos': pos
            })
            
            pos += 5  # Move to next entity (1 byte type + 1 byte x + 1 byte y + 2 bytes padding)
            entity_count += 1
        
        return {
            'tiles': tiles,
            'entities': entities,
            'ninja_spawn': ninja_spawn,
            'total_size': len(data)
        }
    
    def validate_attract_vs_official(self, attract_file: Path, official_file: Path) -> Dict[str, Any]:
        """Validate attract decoder output against official map data."""
        
        if not attract_file.exists():
            return {'error': f'Attract file not found: {attract_file}'}
        
        if not official_file.exists():
            return {'error': f'Official file not found: {official_file}'}
        
        # Load official data
        official_data = self.load_official_map_data(official_file)
        
        # Load attract data
        attract_data = self.decoder.decode_npp_attract_file(str(attract_file))
        
        # Validate tiles
        tile_matches = 0
        tile_mismatches = []
        
        for i, (attract_tile, official_tile) in enumerate(zip(attract_data['tiles'], official_data['tiles'])):
            if attract_tile == official_tile:
                tile_matches += 1
            else:
                tile_mismatches.append({
                    'position': i,
                    'attract': attract_tile,
                    'official': official_tile
                })
        
        tile_accuracy = tile_matches / len(official_data['tiles']) * 100
        
        # Validate spawn
        spawn_match = attract_data['ninja_spawn'] == official_data['ninja_spawn']
        
        # Validate entities
        entity_count_match = len(attract_data['entities']) == len(official_data['entities'])
        entity_matches = 0
        entity_mismatches = []
        
        for i in range(min(len(attract_data['entities']), len(official_data['entities']))):
            attract_entity = attract_data['entities'][i]
            official_entity = official_data['entities'][i]
            
            type_match = attract_entity.get('type') == official_entity['type']
            x_match = attract_entity.get('x') == official_entity['x']
            y_match = attract_entity.get('y') == official_entity['y']
            
            if type_match and x_match and y_match:
                entity_matches += 1
            else:
                entity_mismatches.append({
                    'index': i,
                    'attract': {
                        'type': attract_entity.get('type'),
                        'x': attract_entity.get('x'),
                        'y': attract_entity.get('y')
                    },
                    'official': {
                        'type': official_entity['type'],
                        'x': official_entity['x'],
                        'y': official_entity['y']
                    },
                    'type_match': type_match,
                    'x_match': x_match,
                    'y_match': y_match
                })
        
        entity_accuracy = entity_matches / len(official_data['entities']) * 100 if official_data['entities'] else 100
        
        # Overall accuracy
        overall_accuracy = (
            tile_accuracy == 100 and 
            spawn_match and 
            entity_count_match and 
            entity_accuracy == 100
        )
        
        return {
            'attract_file': str(attract_file),
            'official_file': str(official_file),
            'tile_accuracy': tile_accuracy,
            'tile_matches': tile_matches,
            'tile_total': len(official_data['tiles']),
            'tile_mismatches': tile_mismatches[:10],  # First 10 mismatches
            'spawn_match': spawn_match,
            'attract_spawn': attract_data['ninja_spawn'],
            'official_spawn': official_data['ninja_spawn'],
            'entity_count_match': entity_count_match,
            'attract_entity_count': len(attract_data['entities']),
            'official_entity_count': len(official_data['entities']),
            'entity_accuracy': entity_accuracy,
            'entity_matches': entity_matches,
            'entity_mismatches': entity_mismatches,
            'overall_accuracy': overall_accuracy,
            'summary': {
                'tiles': f"✅ {tile_accuracy:.1f}%" if tile_accuracy == 100 else f"❌ {tile_accuracy:.1f}%",
                'spawn': "✅ Match" if spawn_match else "❌ Mismatch",
                'entity_count': "✅ Match" if entity_count_match else "❌ Mismatch",
                'entity_positions': f"✅ {entity_accuracy:.1f}%" if entity_accuracy == 100 else f"❌ {entity_accuracy:.1f}%",
                'overall': "✅ 100% ACCURATE" if overall_accuracy else "❌ NEEDS FIXING"
            }
        }
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation across multiple test files."""
        
        test_cases = [
            {
                'attract': Path("nclone/example_replays/npp_attract/0"),
                'official': Path("nclone/maps/official/000 the basics")
            },
            # Add more test cases as needed
        ]
        
        results = []
        overall_success = True
        
        for test_case in test_cases:
            result = self.validate_attract_vs_official(test_case['attract'], test_case['official'])
            results.append(result)
            
            if not result.get('overall_accuracy', False):
                overall_success = False
        
        return {
            'test_cases': len(test_cases),
            'results': results,
            'overall_success': overall_success,
            'summary': "✅ ALL TESTS PASS - 100% ACCURACY" if overall_success else "❌ SOME TESTS FAIL"
        }
    
    def print_validation_report(self, result: Dict[str, Any]) -> None:
        """Print a detailed validation report."""
        
        print("=" * 80)
        print("NPP_ATTRACT DECODER VALIDATION REPORT")
        print("=" * 80)
        
        if 'error' in result:
            print(f"❌ ERROR: {result['error']}")
            return
        
        print(f"Attract File: {result['attract_file']}")
        print(f"Official File: {result['official_file']}")
        print()
        
        print("ACCURACY RESULTS:")
        for component, status in result['summary'].items():
            print(f"  {component.upper()}: {status}")
        print()
        
        # Detailed results
        print("DETAILED RESULTS:")
        print(f"  Tiles: {result['tile_matches']}/{result['tile_total']} = {result['tile_accuracy']:.1f}%")
        print(f"  Spawn: {result['attract_spawn']} vs {result['official_spawn']} = {'✅' if result['spawn_match'] else '❌'}")
        print(f"  Entity Count: {result['attract_entity_count']} vs {result['official_entity_count']} = {'✅' if result['entity_count_match'] else '❌'}")
        print(f"  Entity Positions: {result['entity_matches']}/{result['official_entity_count']} = {result['entity_accuracy']:.1f}%")
        
        # Show mismatches
        if result['tile_mismatches']:
            print(f"\nTILE MISMATCHES (first 10):")
            for mismatch in result['tile_mismatches']:
                print(f"  Tile {mismatch['position']}: attract={mismatch['attract']}, official={mismatch['official']}")
        
        if result['entity_mismatches']:
            print(f"\nENTITY MISMATCHES:")
            for mismatch in result['entity_mismatches']:
                print(f"  Entity {mismatch['index']}:")
                print(f"    Attract:  type={mismatch['attract']['type']}, pos=({mismatch['attract']['x']}, {mismatch['attract']['y']})")
                print(f"    Official: type={mismatch['official']['type']}, pos=({mismatch['official']['x']}, {mismatch['official']['y']})")
                print(f"    Matches: type={mismatch['type_match']}, x={mismatch['x_match']}, y={mismatch['y_match']}")
        
        print("\n" + "=" * 80)

def main():
    """Run validation specification."""
    validator = ValidationSpec()
    
    # Single test validation
    attract_file = Path("nclone/example_replays/npp_attract/0")
    official_file = Path("nclone/maps/official/000 the basics")
    
    print("Running single test validation...")
    result = validator.validate_attract_vs_official(attract_file, official_file)
    validator.print_validation_report(result)
    
    # Comprehensive validation
    print("\nRunning comprehensive validation...")
    comprehensive_result = validator.run_comprehensive_validation()
    print(f"\nCOMPREHENSIVE VALIDATION: {comprehensive_result['summary']}")

if __name__ == "__main__":
    main()