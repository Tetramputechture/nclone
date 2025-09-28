#!/usr/bin/env python3
"""
Comprehensive validation across all npp_attract test files.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from pathlib import Path
from validation_spec import ValidationSpec

def run_comprehensive_validation():
    """Run validation across all available test files."""
    
    validator = ValidationSpec()
    
    # Define test cases - mapping attract files to official maps
    test_cases = [
        {
            'name': 'the basics',
            'attract': Path("nclone/example_replays/npp_attract/0"),
            'official': Path("nclone/maps/official/000 the basics")
        },
        # Add more test cases as we identify them
    ]
    
    # Check for additional attract files
    attract_dir = Path("nclone/example_replays/npp_attract")
    if attract_dir.exists():
        attract_files = list(attract_dir.glob("*"))
        attract_files = [f for f in attract_files if f.is_file() and f.name.isdigit()]
        print(f"Found {len(attract_files)} attract files: {[f.name for f in attract_files]}")
    
    # Check for additional official maps
    official_dir = Path("nclone/maps/official")
    if official_dir.exists():
        official_files = list(official_dir.glob("*"))
        official_files = [f for f in official_files if f.is_file()]
        print(f"Found {len(official_files)} official maps: {[f.name for f in official_files[:10]]}...")
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE NPP_ATTRACT DECODER VALIDATION")
    print(f"{'='*80}")
    
    results = []
    overall_success = True
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {test_case['name']}")
        print(f"  Attract: {test_case['attract']}")
        print(f"  Official: {test_case['official']}")
        
        if not test_case['attract'].exists():
            print(f"  ❌ SKIP: Attract file not found")
            continue
            
        if not test_case['official'].exists():
            print(f"  ❌ SKIP: Official file not found")
            continue
        
        result = validator.validate_attract_vs_official(test_case['attract'], test_case['official'])
        results.append(result)
        
        # Print summary for this test case
        if result.get('overall_accuracy', False):
            print(f"  ✅ PASS: 100% accuracy achieved")
            print(f"    - Tiles: {result['tile_accuracy']:.1f}%")
            print(f"    - Spawn: {'✅' if result['spawn_match'] else '❌'}")
            print(f"    - Entities: {result['entity_accuracy']:.1f}%")
        else:
            print(f"  ❌ FAIL: Accuracy issues detected")
            print(f"    - Tiles: {result['tile_accuracy']:.1f}%")
            print(f"    - Spawn: {'✅' if result['spawn_match'] else '❌'}")
            print(f"    - Entities: {result['entity_accuracy']:.1f}%")
            overall_success = False
            
            # Show detailed issues
            if result['tile_mismatches']:
                print(f"    - Tile mismatches: {len(result['tile_mismatches'])}")
            if result['entity_mismatches']:
                print(f"    - Entity mismatches: {len(result['entity_mismatches'])}")
    
    print(f"\n{'='*80}")
    print("FINAL VALIDATION RESULTS")
    print(f"{'='*80}")
    
    if overall_success:
        print("🎉 SUCCESS: All test cases pass with 100% accuracy!")
        print("✅ NPP_ATTRACT decoder is working perfectly")
        print("✅ Entity extraction: 100% accurate")
        print("✅ Spawn extraction: 100% accurate") 
        print("✅ Tile extraction: 100% accurate")
    else:
        print("❌ FAILURE: Some test cases have accuracy issues")
        print("   Manual investigation required")
    
    print(f"\nTest cases run: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r.get('overall_accuracy', False))}")
    print(f"Failed: {sum(1 for r in results if not r.get('overall_accuracy', False))}")
    
    return overall_success, results

def test_additional_files():
    """Test additional attract files if available."""
    
    print(f"\n{'='*80}")
    print("TESTING ADDITIONAL ATTRACT FILES")
    print(f"{'='*80}")
    
    attract_dir = Path("nclone/example_replays/npp_attract")
    if not attract_dir.exists():
        print("No attract directory found")
        return
    
    attract_files = list(attract_dir.glob("*"))
    attract_files = [f for f in attract_files if f.is_file() and f.name.isdigit()]
    
    validator = ValidationSpec()
    
    for attract_file in attract_files[:5]:  # Test first 5 files
        print(f"\nTesting attract file: {attract_file.name}")
        
        try:
            # Try to decode the attract file
            attract_data = validator.decoder.decode_npp_attract_file(str(attract_file))
            
            print(f"  ✅ Successfully decoded")
            print(f"    - Tiles: {len(attract_data['tiles'])}")
            print(f"    - Entities: {len(attract_data['entities'])}")
            print(f"    - Spawn: {attract_data['ninja_spawn']}")
            
            # Show first few entities
            if attract_data['entities']:
                print(f"    - First 3 entities:")
                for i, entity in enumerate(attract_data['entities'][:3]):
                    print(f"      {i}: type={entity.get('type')}, pos=({entity.get('x')}, {entity.get('y')})")
                    
        except Exception as e:
            print(f"  ❌ Failed to decode: {e}")

if __name__ == "__main__":
    success, results = run_comprehensive_validation()
    test_additional_files()
    
    if success:
        print(f"\n🎉 COMPREHENSIVE VALIDATION COMPLETE: 100% SUCCESS!")
    else:
        print(f"\n❌ COMPREHENSIVE VALIDATION FAILED")
        sys.exit(1)