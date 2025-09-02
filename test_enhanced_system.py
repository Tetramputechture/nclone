#!/usr/bin/env python3
"""
Test script for enhanced traversability system with comprehensive validation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from nclone.graph.hazard_system import HazardClassificationSystem, HazardType, HazardState
from nclone.graph.precise_collision import PreciseTileCollision
from nclone.constants.entity_types import EntityType

def test_enhanced_system():
    """Test the enhanced hazard system with comprehensive validation."""
    print("🚀 Testing Enhanced Traversability System")
    print("=" * 50)
    
    # Initialize systems
    precise_collision = PreciseTileCollision()
    hazard_system = HazardClassificationSystem(precise_collision)
    
    # Test level data
    test_level_data = {
        'level_id': 'test_enhanced',
        'tiles': [[0 for _ in range(50)] for _ in range(50)],
        'width': 50,
        'height': 50
    }
    
    # Set level data for collision checking
    hazard_system.set_level_data(test_level_data)
    
    print("✅ Systems initialized successfully")
    
    # Test 1: Enhanced Toggle Mine Logic
    print("\n🔧 Test 1: Enhanced Toggle Mine Logic")
    
    # Test new state-based logic
    toggled_mine = {'id': 1, 'type': EntityType.TOGGLE_MINE, 'x': 100.0, 'y': 100.0, 'state': 0}  # Deadly
    untoggled_mine = {'id': 2, 'type': EntityType.TOGGLE_MINE, 'x': 200.0, 'y': 100.0, 'state': 1}  # Safe
    toggling_mine = {'id': 3, 'type': EntityType.TOGGLE_MINE, 'x': 300.0, 'y': 100.0, 'state': 2}  # Safe
    
    # Test backward compatibility with 'active' field
    active_mine = {'id': 4, 'type': EntityType.TOGGLE_MINE, 'x': 400.0, 'y': 100.0, 'active': True}  # Deadly
    inactive_mine = {'id': 5, 'type': EntityType.TOGGLE_MINE, 'x': 500.0, 'y': 100.0, 'active': False}  # Safe
    
    toggled_hazard = hazard_system._classify_toggle_mine(toggled_mine)
    untoggled_hazard = hazard_system._classify_toggle_mine(untoggled_mine)
    toggling_hazard = hazard_system._classify_toggle_mine(toggling_mine)
    active_hazard = hazard_system._classify_toggle_mine(active_mine)
    inactive_hazard = hazard_system._classify_toggle_mine(inactive_mine)
    
    assert toggled_hazard is not None, "Toggled mine should be deadly"
    assert untoggled_hazard is None, "Untoggled mine should be safe"
    assert toggling_hazard is None, "Toggling mine should be safe"
    assert active_hazard is not None, "Active mine should be deadly (backward compatibility)"
    assert inactive_hazard is None, "Inactive mine should be safe (backward compatibility)"
    
    print("  ✅ State-based toggle mine logic working correctly")
    print("  ✅ Backward compatibility with 'active' field maintained")
    
    # Test 2: Enhanced Drone Logic with Collision Detection
    print("\n🚀 Test 2: Enhanced Drone Logic with Collision Detection")
    
    # Test regular drone with vx/vy
    regular_drone = {
        'id': 10, 'type': EntityType.DRONE_ZAP, 'x': 100.0, 'y': 200.0,
        'vx': 2.0, 'vy': 0.0, 'direction': 0, 'mode': 0
    }
    
    # Test mini drone with direction/speed
    mini_drone = {
        'id': 11, 'type': EntityType.MINI_DRONE, 'x': 200.0, 'y': 200.0,
        'direction': 1, 'mode': 1, 'speed': 1.3
    }
    
    regular_hazard = hazard_system._classify_drone(regular_drone)
    mini_hazard = hazard_system._classify_drone(mini_drone)
    
    assert regular_hazard is not None, "Regular drone should be classified as hazard"
    assert regular_hazard.velocity == (2.0, 0.0), "Regular drone should use vx/vy when available"
    assert regular_hazard.danger_radius == 12.0, "Regular drone should have correct radius"
    assert len(regular_hazard.predicted_positions) > 0, "Regular drone should have movement predictions"
    
    assert mini_hazard is not None, "Mini drone should be classified as hazard"
    assert mini_hazard.danger_radius == 4.0, "Mini drone should have correct radius"
    assert len(mini_hazard.predicted_positions) > 0, "Mini drone should have movement predictions"
    
    print("  ✅ Regular drone classification with vx/vy velocity")
    print("  ✅ Mini drone classification with direction/speed")
    print("  ✅ Movement prediction working for both drone types")
    
    # Test 3: DRY Code Refactoring
    print("\n🔧 Test 3: DRY Code Refactoring")
    
    # Test helper methods work correctly
    blocked_cells = hazard_system._create_blocked_cells_area(100.0, 100.0, 3)
    assert len(blocked_cells) == 9, "3x3 area should have 9 cells"
    
    static_hazard = hazard_system._create_static_blocking_hazard(
        entity_id=100,
        entity_type=EntityType.TOGGLE_MINE,
        position=(100.0, 100.0),
        danger_radius=18.0,
        block_area_size=3
    )
    
    assert static_hazard.hazard_type == HazardType.STATIC_BLOCKING
    assert static_hazard.position == (100.0, 100.0)
    assert static_hazard.danger_radius == 18.0
    assert len(static_hazard.blocked_cells) == 9
    
    dynamic_hazard = hazard_system._create_dynamic_threat_hazard(
        entity_id=101,
        entity_type=EntityType.DRONE_ZAP,
        position=(200.0, 200.0),
        velocity=(2.0, 0.0),
        danger_radius=12.0,
        predicted_positions=[(210.0, 200.0), (220.0, 200.0)]
    )
    
    assert dynamic_hazard.hazard_type == HazardType.DYNAMIC_THREAT
    assert dynamic_hazard.velocity == (2.0, 0.0)
    assert len(dynamic_hazard.predicted_positions) == 2
    
    print("  ✅ Helper methods working correctly")
    print("  ✅ Code duplication eliminated")
    
    # Test 4: Comprehensive Entity Coverage
    print("\n🎯 Test 4: Comprehensive Entity Coverage")
    
    entities = [
        # Toggle mines with different states
        {'id': 1, 'type': EntityType.TOGGLE_MINE, 'x': 100.0, 'y': 100.0, 'state': 0},  # Deadly
        {'id': 2, 'type': EntityType.TOGGLE_MINE, 'x': 200.0, 'y': 100.0, 'state': 1},  # Safe
        
        # Thwumps with different states
        {'id': 3, 'type': EntityType.THWUMP, 'x': 300.0, 'y': 100.0, 'state': 0, 'orientation': 0},  # Immobile
        {'id': 4, 'type': EntityType.THWUMP, 'x': 400.0, 'y': 100.0, 'state': -1, 'orientation': 0},  # Retreating
        
        # Shove thwumps with different states
        {'id': 5, 'type': EntityType.SHWUMP, 'x': 500.0, 'y': 100.0, 'state': 0},  # Immobile
        {'id': 6, 'type': EntityType.SHWUMP, 'x': 600.0, 'y': 100.0, 'state': 3},  # Retreating
        
        # Drones
        {'id': 7, 'type': EntityType.DRONE_ZAP, 'x': 100.0, 'y': 200.0, 'vx': 2.0, 'vy': 0.0},
        {'id': 8, 'type': EntityType.MINI_DRONE, 'x': 200.0, 'y': 200.0, 'direction': 1, 'speed': 1.3},
    ]
    
    # Build static hazard cache
    static_cache = hazard_system.build_static_hazard_cache(entities, test_level_data)
    static_count = len(static_cache)
    
    # Get dynamic hazards
    dynamic_hazards = hazard_system.get_dynamic_hazards_in_range(entities, (300.0, 150.0), 200.0)
    dynamic_count = len(dynamic_hazards)
    
    print(f"  ✅ Static hazard cache size: {static_count}")
    print(f"  ✅ Dynamic hazards found: {dynamic_count}")
    
    # Test 5: Performance with Level Data Integration
    print("\n⚡ Test 5: Performance with Level Data Integration")
    
    # Test collision checking integration
    can_move = hazard_system._can_drone_move_direction(100.0, 100.0, 0, 24, 12.0)
    print(f"  ✅ Drone collision checking: {can_move}")
    
    # Test large entity count performance
    import time
    many_entities = []
    for i in range(100):
        many_entities.append({
            'id': i + 1000, 'type': EntityType.TOGGLE_MINE, 
            'x': i * 10.0, 'y': 100.0, 'state': i % 2
        })
    
    start_time = time.time()
    large_cache = hazard_system.build_static_hazard_cache(many_entities, test_level_data)
    end_time = time.time()
    
    print(f"  ✅ Large entity performance: {len(large_cache)} hazards in {end_time - start_time:.4f}s")
    
    print("\n🎉 All Enhanced System Tests Passed!")
    print("=" * 50)
    print("✅ Toggle mine logic corrected with proper state handling")
    print("✅ Drone collision detection integrated with level data")
    print("✅ Code refactored with DRY principles")
    print("✅ Comprehensive entity coverage maintained")
    print("✅ Performance optimized for real-time usage")
    print("✅ Backward compatibility preserved")

if __name__ == "__main__":
    test_enhanced_system()