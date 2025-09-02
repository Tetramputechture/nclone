#!/usr/bin/env python3
"""
Test script to validate toggle mine radii implementation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from nclone.graph.hazard_system import HazardClassificationSystem
from nclone.constants.entity_types import EntityType

def test_toggle_mine_radii():
    """Test that toggle mine radii match the entity class definition."""
    print("🔧 Testing Toggle Mine Radii Implementation")
    print("=" * 50)
    
    hazard_system = HazardClassificationSystem()
    
    # Expected radii from entity_toggle_mine.py
    expected_radii = {0: 4.0, 1: 3.5, 2: 4.5}  # 0:toggled, 1:untoggled, 2:toggling
    
    print("Expected radii from entity_toggle_mine.py:")
    print(f"  State 0 (toggled/deadly): {expected_radii[0]} pixels")
    print(f"  State 1 (untoggled/safe): {expected_radii[1]} pixels")
    print(f"  State 2 (toggling/safe): {expected_radii[2]} pixels")
    
    # Test each state
    for state in [0, 1, 2]:
        mine = {
            'id': state + 1, 
            'type': EntityType.TOGGLE_MINE, 
            'x': 100.0, 
            'y': 100.0, 
            'state': state
        }
        
        hazard_info = hazard_system._classify_toggle_mine(mine)
        
        if state == 0:  # Toggled state - should be deadly
            assert hazard_info is not None, f"State {state} should create hazard"
            actual_radius = hazard_info.danger_radius
            expected_radius = expected_radii[state]
            assert actual_radius == expected_radius, f"State {state}: expected {expected_radius}, got {actual_radius}"
            print(f"  ✅ State {state} (toggled/deadly): {actual_radius} pixels - CORRECT")
        else:  # States 1 and 2 - should be safe
            assert hazard_info is None, f"State {state} should not create hazard (safe)"
            print(f"  ✅ State {state} ({'untoggled' if state == 1 else 'toggling'}/safe): No hazard - CORRECT")
    
    # Test backward compatibility with 'active' field
    print("\nTesting backward compatibility with 'active' field:")
    
    # Active mine (should use toggled radius)
    active_mine = {
        'id': 10, 
        'type': EntityType.TOGGLE_MINE, 
        'x': 100.0, 
        'y': 100.0, 
        'active': True
    }
    
    active_hazard = hazard_system._classify_toggle_mine(active_mine)
    assert active_hazard is not None, "Active mine should create hazard"
    assert active_hazard.danger_radius == 4.0, f"Active mine should use toggled radius (4.0), got {active_hazard.danger_radius}"
    print(f"  ✅ Active mine: {active_hazard.danger_radius} pixels (toggled radius) - CORRECT")
    
    # Inactive mine (should be safe)
    inactive_mine = {
        'id': 11, 
        'type': EntityType.TOGGLE_MINE, 
        'x': 100.0, 
        'y': 100.0, 
        'active': False
    }
    
    inactive_hazard = hazard_system._classify_toggle_mine(inactive_mine)
    assert inactive_hazard is None, "Inactive mine should not create hazard"
    print(f"  ✅ Inactive mine: No hazard - CORRECT")
    
    # Test TOGGLE_MINE_TOGGLED entity type
    print("\nTesting TOGGLE_MINE_TOGGLED entity type:")
    
    toggled_entity = {
        'id': 20,
        'type': EntityType.TOGGLE_MINE_TOGGLED,
        'x': 100.0,
        'y': 100.0
    }
    
    toggled_hazard = hazard_system._classify_toggle_mine_toggled(toggled_entity)
    assert toggled_hazard is not None, "TOGGLE_MINE_TOGGLED should create hazard"
    assert toggled_hazard.danger_radius == 4.0, f"TOGGLE_MINE_TOGGLED should use toggled radius (4.0), got {toggled_hazard.danger_radius}"
    print(f"  ✅ TOGGLE_MINE_TOGGLED: {toggled_hazard.danger_radius} pixels - CORRECT")
    
    # Verify the constants are correctly defined
    print("\nVerifying hazard system constants:")
    assert hasattr(hazard_system, 'TOGGLE_MINE_RADII'), "TOGGLE_MINE_RADII constant should exist"
    assert hazard_system.TOGGLE_MINE_RADII == expected_radii, "TOGGLE_MINE_RADII should match entity definition"
    print(f"  ✅ TOGGLE_MINE_RADII constant: {hazard_system.TOGGLE_MINE_RADII} - CORRECT")
    
    print("\n🎉 All Toggle Mine Radii Tests Passed!")
    print("=" * 50)
    print("✅ State-based radii correctly implemented")
    print("✅ Backward compatibility maintained")
    print("✅ TOGGLE_MINE_TOGGLED entity type handled")
    print("✅ Constants match entity class definition")

if __name__ == "__main__":
    test_toggle_mine_radii()