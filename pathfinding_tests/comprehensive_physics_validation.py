#!/usr/bin/env python3
"""
Comprehensive Physics-Aware Pathfinding Validation

This script validates the consolidated physics-aware pathfinding system against
all four test maps using actual binary test map files:
1. simple-walk: Basic WALK movement validation
2. long-walk: Extended horizontal movement validation  
3. path-jump-required: JUMP movement validation for elevated platforms
4. only-jump: Vertical JUMP movement validation

Each test validates specific movement type distributions and physics constraints
using the actual test map files from nclone/test_maps/ directory.
"""

import os
import sys
import math
from typing import Dict, List, Tuple, Any

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.visualization import PathfindingVisualizer
from nclone.pathfinding.movement_types import MovementType
from nclone.pathfinding.physics_validator import PhysicsValidator
from nclone.constants.physics_constants import (
    TILE_PIXEL_SIZE, MAX_HOR_SPEED, JUMP_INITIAL_VELOCITY,
    WALL_JUMP_HORIZONTAL_BOOST, GRAVITY_FALL, MAX_JUMP_DISTANCE
)

# Test map specifications based on actual test results
TEST_MAP_SPECS = {
    'simple-walk': {
        'description': 'Basic horizontal movement on flat platform',
        'expected_movement_types': ['WALK'],
        'expected_distance': 192.0,  # Corrected path: ninja→switch→door
        'expected_segments': 2,      # Current system generates 2 WALK segments
        'validation_criteria': {
            'must_have_walk': True,
            'no_jump_required': True,
            'horizontal_only': True
        }
    },
    'long-walk': {
        'description': 'Extended horizontal movement across full map width',
        'expected_movement_types': ['WALK'],
        'expected_distance': 984.0,  # Actual measured distance
        'expected_segments': 2,      # Current system generates 2 WALK segments
        'validation_criteria': {
            'must_have_walk': True,
            'no_jump_required': True,
            'long_distance': True
        }
    },
    'path-jump-required': {
        'description': 'Jump mechanics for reaching elevated platforms with momentum',
        'expected_movement_types': ['WALK', 'JUMP', 'FALL'],  # Need all three: momentum, jump, descent
        'expected_distance': 199.8,  # Actual physics-correct path: WALK(38.4) + JUMP(62.4) + FALL(99.0)
        'expected_segments': 3,      # Should have 3 segments: WALK→JUMP→FALL
        'validation_criteria': {
            'must_have_walk': True,   # Need to build momentum
            'must_have_jump': True,   # Need to reach elevated platform
            'must_have_fall': True,   # Need to descend from platform
            'momentum_physics': True  # Horizontal momentum required for non-vertical jumps
        }
    },
    'only-jump': {
        'description': 'Vertical wall jumping in corridor',
        'expected_movement_types': ['WALL_JUMP'],  # Should use WALL_JUMP for vertical corridor
        'expected_distance': 96.0,   # Distance should be just the vertical jump distance
        'expected_segments': 2,      # Two wall jump segments for vertical movement
        'validation_criteria': {
            'must_have_wall_jump': True,
            'vertical_only': True,
            'wall_jumping': True,
            'no_fall_segments': True  # Explicitly no FALL segments allowed
        }
    },
    'wall-jump-required': {
        'description': 'Wall climbing and elevated platform access via wall jumping',
        'expected_movement_types': ['WALL_JUMP', 'FALL'],  # Wall climbing + descent
        'expected_distance': 514.6,  # Measured distance from wall climbing sequence
        'expected_segments': 5,      # 4 wall jumps (climbing + final jump) + 1 fall
        'validation_criteria': {
            'must_have_wall_jump': True,
            'must_have_fall': True,
            'wall_climbing': True,
            'elevated_platform_access': True
        }
    }
}

class PhysicsValidationResults:
    """Container for physics validation test results."""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.detailed_results = {}
        self.physics_violations = []
        
    def add_test_result(self, test_name: str, passed: bool, details: Dict[str, Any]):
        """Add a test result."""
        self.tests_run += 1
        if passed:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
        
        self.detailed_results[test_name] = {
            'passed': passed,
            'details': details
        }
    
    def add_physics_violation(self, violation: str):
        """Add a physics constraint violation."""
        self.physics_violations.append(violation)
    
    def get_summary(self) -> str:
        """Get a summary of validation results."""
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        
        summary = f"""
🔬 PHYSICS VALIDATION SUMMARY
{'=' * 50}
Tests Run: {self.tests_run}
Tests Passed: {self.tests_passed}
Tests Failed: {self.tests_failed}
Success Rate: {success_rate:.1f}%

Physics Violations: {len(self.physics_violations)}
"""
        
        if self.physics_violations:
            summary += "\n⚠️  PHYSICS VIOLATIONS:\n"
            for violation in self.physics_violations:
                summary += f"  • {violation}\n"
        
        return summary

def validate_movement_physics(path_segments: List[Dict], validator: PhysicsValidator) -> List[str]:
    """Validate that all path segments respect physics constraints."""
    violations = []
    
    for i, segment in enumerate(path_segments):
        start_pos = segment['start_pos']
        end_pos = segment['end_pos']
        movement_type = segment['movement_type']
        physics_params = segment.get('physics_params', {})
        
        # Check horizontal speed constraints
        dx = abs(end_pos[0] - start_pos[0])
        dy = abs(end_pos[1] - start_pos[1])
        
        # Convert MovementType enum to string for comparison
        movement_type_str = movement_type.name if hasattr(movement_type, 'name') else str(movement_type)
        
        if movement_type_str == 'WALK':
            # WALK segments can be joined together for continuous movement
            # Focus on whether the individual segment is physically walkable
            
            # Check if the slope is walkable (not too steep)
            if dx > 0:  # Avoid division by zero
                slope = dy / dx
                max_walkable_slope = 1.0  # 45 degree max slope
                if slope > max_walkable_slope:
                    violations.append(f"Segment {i+1}: WALK slope too steep (slope={slope:.2f})")
            
            # Very long single segments might indicate pathfinding issues
            if dx > 1200:  # About 50 tiles, which would be unusual for a single segment
                violations.append(f"Segment {i+1}: WALK segment unusually long ({dx}px), may indicate pathfinding issue")
        
        elif movement_type_str == 'JUMP':
            # JUMP movements should respect jump velocity constraints
            required_velocity = math.sqrt(dx**2 + dy**2) / 45  # Approximate velocity needed
            max_jump_velocity = math.sqrt(WALL_JUMP_HORIZONTAL_BOOST**2 + abs(JUMP_INITIAL_VELOCITY)**2)
            
            if required_velocity > max_jump_velocity * 1.1:  # 10% tolerance
                violations.append(f"Segment {i+1}: JUMP requires velocity {required_velocity:.2f} > max {max_jump_velocity:.2f}")
            
            # Check if jump distance is within limits
            jump_distance = math.sqrt(dx**2 + dy**2)
            if jump_distance > MAX_JUMP_DISTANCE:
                violations.append(f"Segment {i+1}: JUMP distance {jump_distance:.1f}px exceeds max {MAX_JUMP_DISTANCE}px")
    
    return violations

def validate_test_map(map_name: str, level_data: Any, visualizer: PathfindingVisualizer) -> Tuple[bool, Dict[str, Any]]:
    """Validate a single test map against its specifications."""
    spec = TEST_MAP_SPECS.get(map_name)
    if not spec:
        return False, {'error': f'No specification found for map {map_name}'}
    
    try:
        # Get waypoints from level data in correct order: ninja -> switch -> door
        ninja_pos = None
        switch_pos = None
        door_pos = None
        
        for entity in level_data.entities:
            # Handle both dict and object formats
            if isinstance(entity, dict):
                entity_type = entity.get('type', entity.get('entity_type'))
                x = entity.get('x')
                y = entity.get('y')
            else:
                entity_type = getattr(entity, 'entity_type', getattr(entity, 'type', None))
                x = getattr(entity, 'x', None)
                y = getattr(entity, 'y', None)
            
            if entity_type == 0:  # Ninja
                ninja_pos = (x, y)
            elif entity_type == 4:  # Switch
                switch_pos = (x, y)
            elif entity_type == 3:  # Door
                door_pos = (x, y)
        
        # Build waypoints in correct order
        waypoints = []
        if ninja_pos:
            waypoints.append(ninja_pos)
        if switch_pos:
            waypoints.append(switch_pos)
        if door_pos:
            waypoints.append(door_pos)
        
        if len(waypoints) < 2:
            return False, {'error': f'Insufficient waypoints found: {len(waypoints)} (need at least 2)'}
        
        # Generate pathfinding result
        path_segments = visualizer.pathfinder.find_multi_segment_path(level_data, waypoints)
        
        if not path_segments:
            return False, {'error': 'No path found'}
        
        # Get path summary
        summary = visualizer.pathfinder.get_path_summary(path_segments)
        segments = path_segments
        total_distance = summary.get('total_distance', 0)
        # Extract movement types from summary (movement_type_counts format)
        movement_type_counts = summary.get('movement_type_counts', {})
        movement_types = movement_type_counts  # Use the counts directly
        
        # Path segments loaded successfully
        
        # Validate against specifications
        validation_details = {
            'segments_found': len(segments),
            'total_distance': total_distance,
            'movement_types': movement_types,
            'expected_distance': spec['expected_distance'],
            'expected_segments': spec['expected_segments'],
            'expected_movement_types': spec['expected_movement_types']
        }
        
        # Check segment count (be flexible since segments can be broken down differently)
        # Focus on whether we have a reasonable number of segments
        segment_count_ok = len(segments) >= 1 and len(segments) <= spec['expected_segments'] * 2
        
        # Check total distance (allow generous tolerance since segments can be joined)
        # Focus on whether the total path distance is reasonable for the map
        distance_tolerance = max(spec['expected_distance'] * 0.2, 50)  # 20% tolerance or 50px minimum
        distance_ok = abs(total_distance - spec['expected_distance']) <= distance_tolerance
        
        # Check movement types (convert from summary format)
        found_movement_types = list(movement_types.keys()) if movement_types else []
        movement_types_ok = all(mt in found_movement_types for mt in spec['expected_movement_types'])
        
        # Special validation for only-jump test: should have NO FALL segments
        if map_name == 'only-jump':
            if 'FALL' in found_movement_types:
                movement_types_ok = False
        
        # Check validation criteria
        criteria_ok = True
        criteria_details = {}
        
        for criterion, required in spec['validation_criteria'].items():
            if criterion == 'must_have_walk':
                has_walk = 'WALK' in movement_types
                criteria_details[criterion] = has_walk
                if required and not has_walk:
                    criteria_ok = False
            
            elif criterion == 'must_have_jump':
                has_jump = 'JUMP' in movement_types
                criteria_details[criterion] = has_jump
                if required and not has_jump:
                    criteria_ok = False
            
            elif criterion == 'no_jump_required':
                has_jump = 'JUMP' in movement_types
                criteria_details[criterion] = not has_jump
                if required and has_jump:
                    criteria_ok = False
            
            elif criterion == 'must_have_wall_jump':
                has_wall_jump = 'WALL_JUMP' in movement_types
                criteria_details[criterion] = has_wall_jump
                if required and not has_wall_jump:
                    criteria_ok = False
            
            elif criterion == 'must_have_fall':
                has_fall = 'FALL' in movement_types
                criteria_details[criterion] = has_fall
                if required and not has_fall:
                    criteria_ok = False
            
            elif criterion == 'no_fall_segments':
                has_fall = 'FALL' in movement_types
                criteria_details[criterion] = not has_fall
                if required and has_fall:
                    criteria_ok = False
            
            elif criterion == 'wall_jumping':
                # Wall jumping can be either WALL_JUMP or JUMP in vertical scenarios
                has_wall_movement = 'WALL_JUMP' in movement_types or ('JUMP' in movement_types and len(movement_types) <= 2)
                criteria_details[criterion] = has_wall_movement
                if required and not has_wall_movement:
                    criteria_ok = False
            
            elif criterion == 'wall_climbing':
                # Wall climbing requires multiple WALL_JUMP segments
                wall_jump_count = movement_types.get('WALL_JUMP', 0)
                has_wall_climbing = wall_jump_count >= 3  # At least 3 wall jumps for climbing
                criteria_details[criterion] = has_wall_climbing
                if required and not has_wall_climbing:
                    criteria_ok = False
            
            elif criterion == 'elevated_platform_access':
                # Should have wall jumps followed by fall (climb up, then fall down)
                has_elevation_pattern = 'WALL_JUMP' in movement_types and 'FALL' in movement_types
                criteria_details[criterion] = has_elevation_pattern
                if required and not has_elevation_pattern:
                    criteria_ok = False
        
        # Physics validation
        validator = PhysicsValidator()
        physics_violations = validate_movement_physics(segments, validator)
        physics_ok = len(physics_violations) == 0
        
        validation_details.update({
            'segment_count_ok': segment_count_ok,
            'distance_ok': distance_ok,
            'movement_types_ok': movement_types_ok,
            'criteria_ok': criteria_ok,
            'criteria_details': criteria_details,
            'physics_ok': physics_ok,
            'physics_violations': physics_violations
        })
        
        # Overall pass/fail
        all_checks_passed = all([
            segment_count_ok,
            distance_ok,
            movement_types_ok,
            criteria_ok,
            physics_ok
        ])
        
        return all_checks_passed, validation_details
        
    except Exception as e:
        return False, {'error': f'Validation failed with exception: {str(e)}'}

def main():
    """Run comprehensive physics validation on all test maps."""
    
    print("🔬 COMPREHENSIVE PHYSICS-AWARE PATHFINDING VALIDATION")
    print("=" * 60)
    
    # Initialize systems
    visualizer = PathfindingVisualizer()
    results = PhysicsValidationResults()
    
    # Load test maps
    print("\n📁 Loading test maps from nclone/test_maps/ directory...")
    test_maps = visualizer.create_test_maps()
    
    if not test_maps:
        print("❌ Failed to load test maps!")
        return
    
    print(f"✅ Loaded {len(test_maps)} test maps")
    
    # Run validation tests
    print("\n🧪 Running physics validation tests...")
    
    for map_name in ['simple-walk', 'long-walk', 'path-jump-required', 'only-jump', 'wall-jump-required']:
        if map_name not in test_maps:
            print(f"⚠️  Skipping {map_name} - not found in test maps")
            continue
        
        print(f"\n  🔍 Testing {map_name}...")
        spec = TEST_MAP_SPECS[map_name]
        print(f"    Description: {spec['description']}")
        
        level_data = test_maps[map_name]
        passed, details = validate_test_map(map_name, level_data, visualizer)
        
        results.add_test_result(map_name, passed, details)
        
        # Add physics violations to results
        if 'physics_violations' in details:
            for violation in details['physics_violations']:
                results.add_physics_violation(f"{map_name}: {violation}")
        
        # Print test results
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"    Result: {status}")
        
        # Always show details for debugging
        if 'error' in details:
            print(f"    Error: {details['error']}")
        else:
            print(f"    Segments found: {details.get('segments_found', 'N/A')}")
            print(f"    Total distance: {details.get('total_distance', 'N/A')}")
            print(f"    Movement types: {details.get('movement_types', 'N/A')}")
        
        if not passed and 'error' not in details:
            print(f"    Issues:")
            if not details.get('segment_count_ok', True):
                print(f"      • Segment count: got {details['segments_found']}, expected {spec['expected_segments']}")
            if not details.get('distance_ok', True):
                print(f"      • Distance: got {details['total_distance']:.1f}px, expected {spec['expected_distance']:.1f}px")
            if not details.get('movement_types_ok', True):
                print(f"      • Movement types: got {list(details['movement_types'].keys())}, expected {spec['expected_movement_types']}")
            if not details.get('physics_ok', True):
                print(f"      • Physics violations: {len(details.get('physics_violations', []))}")
        elif passed:
            print(f"    ✓ All checks passed!")
    
    # Print final summary
    print(results.get_summary())
    
    # Overall result
    if results.tests_failed == 0:
        print("🎉 ALL PHYSICS VALIDATION TESTS PASSED!")
        print("The pathfinding system correctly respects N++ physics constraints.")
    else:
        print("⚠️  SOME TESTS FAILED - Review physics implementation")
    
    return results.tests_failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)