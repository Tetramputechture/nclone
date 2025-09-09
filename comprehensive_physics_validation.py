#!/usr/bin/env python3
"""
Comprehensive Physics-Aware Pathfinding Validation

This script validates the enhanced physics-aware pathfinding system against
all four test maps as specified in the work document:
1. simple-walk: Single WALK segment validation
2. long-walk: Extended horizontal movement validation  
3. path-jump-required: Multi-segment WALK/JUMP/FALL validation
4. only-jump: Wall jumping mechanics validation

Each test validates specific movement type distributions and physics constraints.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
import math
from typing import Dict, List, Tuple, Any

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine, PathfindingAlgorithm
from nclone.constants.entity_types import EntityType
from nclone.constants.physics_constants import (
    TILE_PIXEL_SIZE, NINJA_RADIUS, MAP_PADDING, 
    MAX_JUMP_DISTANCE, MAX_FALL_DISTANCE, MAX_HOR_SPEED
)
from nclone.graph.common import EdgeType
from nclone.graph.movement_classifier import MovementType
from nclone.graph.segment_consolidator import SegmentConsolidator

# Test map specifications from work document
TEST_MAP_SPECS = {
    'simple-walk': {
        'description': 'Single horizontal platform - validate basic WALK movement',
        'expected_movement_types': ['WALK'],
        'expected_distance_range': (180, 200),  # ~192 pixels ± tolerance
        'max_segments': 3,  # Allow some consolidation flexibility
        'validation_criteria': {
            'must_have_walk': True,
            'no_jump_required': True,
            'no_fall_required': True,
            'single_platform': True
        }
    },
    'long-walk': {
        'description': 'Extended horizontal movement across full map width',
        'expected_movement_types': ['WALK'],
        'expected_distance_range': (970, 1000),  # ~984 pixels ± tolerance
        'max_segments': 5,  # Allow some consolidation flexibility
        'validation_criteria': {
            'must_have_walk': True,
            'no_jump_required': True,
            'no_fall_required': True,
            'full_map_traversal': True
        }
    },
    'path-jump-required': {
        'description': 'Elevated platform requiring jump mechanics',
        'expected_movement_types': ['WALK', 'JUMP', 'FALL'],
        'expected_distance_range': (200, 300),  # Variable based on path
        'max_segments': 10,  # Complex multi-segment path
        'validation_criteria': {
            'must_have_walk': True,
            'must_have_jump': True,
            'must_have_fall': True,
            'elevated_target': True
        }
    },
    'only-jump': {
        'description': 'Vertical corridor requiring wall jumping',
        'expected_movement_types': ['JUMP', 'WALL_JUMP'],
        'expected_distance_range': (100, 200),  # Vertical movement
        'max_segments': 8,  # Multiple wall jumps
        'validation_criteria': {
            'no_walk_required': True,
            'must_have_jump': True,
            'vertical_movement': True,
            'wall_jumping_preferred': True
        }
    }
}

# Movement type colors for visualization
MOVEMENT_COLORS = {
    EdgeType.WALK: '#0080FF',      # Blue - walking
    EdgeType.JUMP: '#FF8000',      # Orange - jumping  
    EdgeType.FALL: '#00FFFF',      # Cyan - falling
    EdgeType.WALL_SLIDE: '#FF00FF', # Magenta - wall sliding
    EdgeType.ONE_WAY: '#FFFF00',   # Yellow - one way platform
    EdgeType.FUNCTIONAL: '#FF0000'  # Red - functional
}

class PhysicsValidationResult:
    """Results of physics validation for a test map."""
    
    def __init__(self, map_name: str):
        self.map_name = map_name
        self.success = False
        self.path_found = False
        self.movement_types = {}
        self.total_distance = 0.0
        self.segment_count = 0
        self.physics_violations = []
        self.validation_errors = []
        self.consolidated_segments = []
        
    def add_violation(self, violation: str):
        self.physics_violations.append(violation)
        
    def add_error(self, error: str):
        self.validation_errors.append(error)
        
    def is_valid(self) -> bool:
        return self.success and len(self.physics_violations) == 0 and len(self.validation_errors) == 0


def load_test_map(map_name: str) -> Tuple[Any, Tuple[float, float], List[Dict], Any]:
    """Load a test map and return environment data."""
    print(f"📁 Loading test map: {map_name}")
    
    try:
        # Construct path to test map
        map_path = os.path.join('nclone', 'test_maps', map_name)
        
        # Create environment with custom map path
        env = BasicLevelNoGold(render_mode="rgb_array", custom_map_path=map_path)
        env.reset()
        
        ninja_position = env.nplay_headless.ninja_position()
        level_data = env.level_data
        entities = level_data.entities
        
        print(f"✅ Map loaded: {level_data.width}x{level_data.height} tiles")
        print(f"✅ Ninja position: {ninja_position}")
        print(f"✅ Found {len(entities)} entities")
        
        return env, ninja_position, entities, level_data
        
    except Exception as e:
        print(f"❌ Failed to load test map {map_name}: {e}")
        return None, None, None, None


def find_target_entity(entities: List[Dict], map_name: str) -> Tuple[float, float]:
    """Find the target entity for the test map."""
    
    # For most test maps, target the exit switch
    target_type = EntityType.EXIT_SWITCH
    
    # Special case for maps with locked doors
    if 'door' in map_name.lower():
        target_type = EntityType.LOCKED_DOOR
    
    targets = []
    for entity in entities:
        if entity.get("type") == target_type:
            targets.append((entity.get("x", 0), entity.get("y", 0)))
    
    if not targets:
        # Fallback to any switch type
        for entity in entities:
            entity_type = entity.get("type", 0)
            if entity_type in [EntityType.EXIT_SWITCH, EntityType.LOCKED_DOOR]:
                targets.append((entity.get("x", 0), entity.get("y", 0)))
    
    if not targets:
        raise ValueError(f"No target entity found in {map_name}")
    
    # Return the first target (or leftmost for consistency)
    target = min(targets, key=lambda pos: pos[0])
    print(f"🎯 Target entity at: {target}")
    return target


def validate_physics_constraints(
    path_nodes: List[Tuple[float, float]], 
    movement_types: List[EdgeType],
    map_name: str
) -> List[str]:
    """Validate that path respects N++ physics constraints."""
    violations = []
    
    for i in range(len(path_nodes) - 1):
        start_pos = path_nodes[i]
        end_pos = path_nodes[i + 1]
        movement_type = movement_types[i]
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        # Check movement type constraints
        if movement_type == EdgeType.JUMP:
            if distance > MAX_JUMP_DISTANCE:
                violations.append(
                    f"Jump segment {i} exceeds max jump distance: {distance:.1f}px > {MAX_JUMP_DISTANCE}px"
                )
            
            # Jumps should generally be upward or horizontal
            if dy > distance * 0.5:  # More than 50% downward
                violations.append(
                    f"Jump segment {i} is primarily downward: dy={dy:.1f}px"
                )
        
        elif movement_type == EdgeType.FALL:
            if distance > MAX_FALL_DISTANCE:
                violations.append(
                    f"Fall segment {i} exceeds max fall distance: {distance:.1f}px > {MAX_FALL_DISTANCE}px"
                )
            
            # Falls should be downward
            if dy <= 0:
                violations.append(
                    f"Fall segment {i} is not downward: dy={dy:.1f}px"
                )
        
        elif movement_type == EdgeType.WALK:
            # Walking should be mostly horizontal
            if abs(dy) > TILE_PIXEL_SIZE:  # More than one tile height
                violations.append(
                    f"Walk segment {i} has excessive vertical movement: dy={dy:.1f}px"
                )
    
    return violations


def validate_test_map_criteria(
    result: PhysicsValidationResult,
    map_name: str,
    movement_types: List[EdgeType],
    path_nodes: List[Tuple[float, float]]
) -> None:
    """Validate test map specific criteria."""
    
    spec = TEST_MAP_SPECS.get(map_name, {})
    criteria = spec.get('validation_criteria', {})
    
    # Count movement types
    type_counts = {}
    for movement_type in movement_types:
        type_name = movement_type.name
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
    
    result.movement_types = type_counts
    
    # Validate specific criteria
    if criteria.get('must_have_walk') and type_counts.get('WALK', 0) == 0:
        result.add_error(f"{map_name}: Must have WALK segments but found none")
    
    if criteria.get('no_jump_required') and type_counts.get('JUMP', 0) > 0:
        result.add_error(f"{map_name}: Should not require JUMP but found {type_counts.get('JUMP', 0)} segments")
    
    if criteria.get('must_have_jump') and type_counts.get('JUMP', 0) == 0:
        result.add_error(f"{map_name}: Must have JUMP segments but found none")
    
    if criteria.get('must_have_fall') and type_counts.get('FALL', 0) == 0:
        result.add_error(f"{map_name}: Must have FALL segments but found none")
    
    if criteria.get('no_walk_required') and type_counts.get('WALK', 0) > 0:
        result.add_error(f"{map_name}: Should not require WALK but found {type_counts.get('WALK', 0)} segments")
    
    # Validate distance range
    expected_range = spec.get('expected_distance_range', (0, float('inf')))
    if not (expected_range[0] <= result.total_distance <= expected_range[1]):
        result.add_error(
            f"{map_name}: Distance {result.total_distance:.1f}px outside expected range {expected_range}"
        )
    
    # Validate segment count
    max_segments = spec.get('max_segments', float('inf'))
    if result.segment_count > max_segments:
        result.add_error(
            f"{map_name}: Too many segments {result.segment_count} > {max_segments} (may indicate micro-movements)"
        )


def test_single_map(map_name: str) -> PhysicsValidationResult:
    """Test pathfinding on a single map."""
    
    print(f"\n{'='*80}")
    print(f"🧪 TESTING MAP: {map_name.upper()}")
    print(f"{'='*80}")
    
    result = PhysicsValidationResult(map_name)
    
    # Load test map
    env, ninja_position, entities, level_data = load_test_map(map_name)
    if not env:
        result.add_error(f"Failed to load map {map_name}")
        return result
    
    try:
        # Find target entity
        target_pos = find_target_entity(entities, map_name)
        
        # Build graph
        print("\n🔧 Building hierarchical graph...")
        builder = HierarchicalGraphBuilder()
        hierarchical_graph = builder.build_graph(level_data, ninja_position)
        
        if not hierarchical_graph or not hierarchical_graph.sub_cell_graph:
            result.add_error("Failed to build graph")
            return result
        
        graph = hierarchical_graph.sub_cell_graph
        print(f"✅ Graph built: {graph.num_nodes} nodes, {graph.num_edges} edges")
        
        # Find ninja and target nodes
        ninja_node = None
        target_node = None
        
        # Find ninja node (closest to ninja position)
        min_ninja_dist = float('inf')
        for i in range(graph.num_nodes):
            node_x = graph.node_features[i, 0]
            node_y = graph.node_features[i, 1]
            dist = math.sqrt((node_x - ninja_position[0])**2 + (node_y - ninja_position[1])**2)
            if dist < min_ninja_dist:
                min_ninja_dist = dist
                ninja_node = i
        
        # Find target node (closest to target position)
        min_target_dist = float('inf')
        for i in range(graph.num_nodes):
            node_x = graph.node_features[i, 0]
            node_y = graph.node_features[i, 1]
            dist = math.sqrt((node_x - target_pos[0])**2 + (node_y - target_pos[1])**2)
            if dist < min_target_dist:
                min_target_dist = dist
                target_node = i
        
        if ninja_node is None or target_node is None:
            result.add_error("Could not find ninja or target nodes")
            return result
        
        print(f"✅ Ninja node: {ninja_node} at ({graph.node_features[ninja_node, 0]:.1f}, {graph.node_features[ninja_node, 1]:.1f})")
        print(f"✅ Target node: {target_node} at ({graph.node_features[target_node, 0]:.1f}, {graph.node_features[target_node, 1]:.1f})")
        
        # Find path
        print("\n🚀 Finding physics-accurate path...")
        pathfinding_engine = PathfindingEngine(level_data, entities)
        path_result = pathfinding_engine.find_shortest_path(
            graph, ninja_node, target_node, PathfindingAlgorithm.DIJKSTRA
        )
        
        if not path_result.success:
            result.add_error("No path found")
            return result
        
        result.path_found = True
        path_nodes = []
        for node_id in path_result.path:
            x = graph.node_features[node_id, 0]
            y = graph.node_features[node_id, 1]
            path_nodes.append((x, y))
        
        movement_types = path_result.edge_types
        result.segment_count = len(movement_types)
        
        # Calculate total distance
        total_distance = 0
        for i in range(len(path_nodes) - 1):
            dx = path_nodes[i+1][0] - path_nodes[i][0]
            dy = path_nodes[i+1][1] - path_nodes[i][1]
            total_distance += math.sqrt(dx**2 + dy**2)
        
        result.total_distance = total_distance
        
        print(f"✅ Path found: {len(path_nodes)} nodes, {total_distance:.1f}px")
        print(f"🎯 Movement types: {dict(zip(*np.unique(movement_types, return_counts=True)))}")
        
        # Validate physics constraints
        print("\n🔍 Validating physics constraints...")
        violations = validate_physics_constraints(path_nodes, movement_types, map_name)
        for violation in violations:
            result.add_violation(violation)
        
        # Validate test map criteria
        print("🔍 Validating test map criteria...")
        validate_test_map_criteria(result, map_name, movement_types, path_nodes)
        
        # Test segment consolidation
        print("🔍 Testing segment consolidation...")
        consolidator = SegmentConsolidator()
        try:
            # Convert EdgeType to MovementType for consolidation
            movement_type_mapping = {
                EdgeType.WALK: MovementType.WALK,
                EdgeType.JUMP: MovementType.JUMP,
                EdgeType.FALL: MovementType.FALL,
                EdgeType.WALL_SLIDE: MovementType.WALL_SLIDE,
            }
            
            consolidated_movement_types = []
            physics_params = []
            
            for edge_type in movement_types:
                movement_type = movement_type_mapping.get(edge_type, MovementType.WALK)
                consolidated_movement_types.append(movement_type)
                # Create dummy physics params
                physics_params.append({
                    'time_estimate': 1.0,
                    'energy_cost': 1.0,
                    'difficulty': 0.1
                })
            
            consolidated_segments = consolidator.consolidate_path(
                path_nodes, consolidated_movement_types, physics_params
            )
            
            result.consolidated_segments = consolidated_segments
            print(f"✅ Consolidated to {len(consolidated_segments)} segments")
            
            for i, segment in enumerate(consolidated_segments):
                print(f"   Segment {i+1}: {segment.movement_type.name} - {segment.total_distance:.1f}px")
        
        except Exception as e:
            print(f"⚠️  Segment consolidation failed: {e}")
        
        # Determine overall success
        result.success = (len(result.physics_violations) == 0 and 
                         len(result.validation_errors) == 0 and
                         result.path_found)
        
        if result.success:
            print(f"✅ {map_name}: PASSED all validation criteria")
        else:
            print(f"❌ {map_name}: FAILED validation")
            for violation in result.physics_violations:
                print(f"   Physics violation: {violation}")
            for error in result.validation_errors:
                print(f"   Validation error: {error}")
    
    except Exception as e:
        result.add_error(f"Test execution failed: {e}")
        print(f"❌ Test failed with exception: {e}")
    
    return result


def generate_summary_report(results: Dict[str, PhysicsValidationResult]) -> None:
    """Generate a comprehensive summary report."""
    
    print(f"\n{'='*80}")
    print("📊 COMPREHENSIVE PHYSICS VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result.success)
    
    print(f"📈 Overall Results: {passed_tests}/{total_tests} tests passed")
    print(f"📈 Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    print(f"\n📋 Individual Test Results:")
    print("-" * 80)
    
    for map_name, result in results.items():
        status = "✅ PASS" if result.success else "❌ FAIL"
        spec = TEST_MAP_SPECS.get(map_name, {})
        description = spec.get('description', 'No description')
        
        print(f"{status} {map_name:15} | {description}")
        
        if result.path_found:
            print(f"     Distance: {result.total_distance:6.1f}px | Segments: {result.segment_count:2d} | Types: {result.movement_types}")
        
        if result.physics_violations:
            print(f"     Physics violations: {len(result.physics_violations)}")
            for violation in result.physics_violations[:2]:  # Show first 2
                print(f"       - {violation}")
        
        if result.validation_errors:
            print(f"     Validation errors: {len(result.validation_errors)}")
            for error in result.validation_errors[:2]:  # Show first 2
                print(f"       - {error}")
        
        print()
    
    # Movement type analysis
    print("🎯 Movement Type Analysis:")
    print("-" * 40)
    
    all_movement_types = set()
    for result in results.values():
        all_movement_types.update(result.movement_types.keys())
    
    for movement_type in sorted(all_movement_types):
        maps_with_type = []
        total_count = 0
        
        for map_name, result in results.items():
            count = result.movement_types.get(movement_type, 0)
            if count > 0:
                maps_with_type.append(f"{map_name}({count})")
                total_count += count
        
        if maps_with_type:
            print(f"{movement_type:12} | Total: {total_count:3d} | Maps: {', '.join(maps_with_type)}")
    
    # Physics constraint validation
    print(f"\n⚖️  Physics Constraint Validation:")
    print("-" * 40)
    
    total_violations = sum(len(result.physics_violations) for result in results.values())
    if total_violations == 0:
        print("✅ All paths respect N++ physics constraints")
    else:
        print(f"❌ Found {total_violations} physics constraint violations")
        
        for map_name, result in results.items():
            if result.physics_violations:
                print(f"   {map_name}: {len(result.physics_violations)} violations")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    print("-" * 40)
    
    if passed_tests == total_tests:
        print("🎉 Excellent! All tests passed. The physics-aware pathfinding system is working correctly.")
        print("   Consider testing with more complex scenarios or edge cases.")
    else:
        failed_maps = [name for name, result in results.items() if not result.success]
        print(f"🔧 Focus on fixing issues in: {', '.join(failed_maps)}")
        
        # Specific recommendations based on failures
        for map_name in failed_maps:
            result = results[map_name]
            if 'simple-walk' in map_name and result.movement_types.get('JUMP', 0) > 0:
                print(f"   - {map_name}: Should use only WALK movements for simple horizontal navigation")
            elif 'only-jump' in map_name and result.movement_types.get('WALK', 0) > 0:
                print(f"   - {map_name}: Should use JUMP/WALL_JUMP movements for vertical navigation")


def main():
    """Run comprehensive physics validation on all test maps."""
    
    print("🚀 COMPREHENSIVE PHYSICS-AWARE PATHFINDING VALIDATION")
    print("=" * 80)
    print("Testing enhanced physics system against all validation test maps")
    print("Validating movement classification, segment consolidation, and physics constraints")
    print()
    
    # Test all maps
    test_maps = ['simple-walk', 'long-walk', 'path-jump-required', 'only-jump']
    results = {}
    
    for map_name in test_maps:
        try:
            result = test_single_map(map_name)
            results[map_name] = result
        except Exception as e:
            print(f"❌ Critical error testing {map_name}: {e}")
            result = PhysicsValidationResult(map_name)
            result.add_error(f"Critical test failure: {e}")
            results[map_name] = result
    
    # Generate summary report
    generate_summary_report(results)
    
    # Return overall success
    overall_success = all(result.success for result in results.values())
    
    print(f"\n{'='*80}")
    if overall_success:
        print("🎉 COMPREHENSIVE VALIDATION: SUCCESS")
        print("All test maps passed validation criteria!")
    else:
        print("❌ COMPREHENSIVE VALIDATION: FAILED")
        print("Some test maps failed validation criteria.")
    print(f"{'='*80}")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)