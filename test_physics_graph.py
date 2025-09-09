#!/usr/bin/env python3
"""
Test physics-aware graph construction and pathfinding.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.graph.pathfinding import PathfindingEngine, PathfindingAlgorithm
from nclone.graph.movement_classifier import MovementType
from nclone.graph.level_data import LevelData

def test_physics_aware_pathfinding():
    """Test that pathfinding uses physics-aware costs and classifications."""
    print("=== Testing Physics-Aware Movement Classification ===")
    
    # Create pathfinding engine to test movement classification
    pathfinder = PathfindingEngine()
    
    # Test movement classification on various scenarios
    test_scenarios = [
        ("Simple Walk", (0, 100), (100, 100)),
        ("Upward Jump", (50, 150), (100, 100)),
        ("Downward Fall", (50, 100), (100, 150)),
        ("Vertical Ascent", (50, 150), (50, 100)),
        ("Long Horizontal", (0, 100), (200, 100)),
    ]
    
    for name, start_pos, end_pos in test_scenarios:
        print(f"\n{name}: {start_pos} -> {end_pos}")
        
        # Get movement classification
        movement_type, physics_params = pathfinder.movement_classifier.classify_movement(
            start_pos, end_pos, None, None
        )
        
        print(f"  Movement Type: {MovementType(movement_type).name}")
        print(f"  Distance: {physics_params.get('distance', 0):.1f} pixels")
        print(f"  Energy Cost: {physics_params.get('energy_cost', 0):.2f}")
        print(f"  Risk Factor: {physics_params.get('risk_factor', 0):.2f}")
        
        # Calculate what the pathfinding cost would be
        base_cost = physics_params.get('distance', 0)
        energy_cost = physics_params.get('energy_cost', 0)
        risk_factor = physics_params.get('risk_factor', 0)
        total_cost = base_cost + energy_cost * 10 + risk_factor * 5
        
        print(f"  Total Pathfinding Cost: {total_cost:.2f}")
        
        # Compare with simple distance
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        simple_distance = (dx**2 + dy**2)**0.5
        print(f"  Simple Distance: {simple_distance:.1f}")
        print(f"  Cost/Distance Ratio: {total_cost / simple_distance:.2f}x")

def test_cost_comparison():
    """Compare physics-aware costs vs simple distance costs."""
    print("\n=== Testing Cost Calculation Comparison ===")
    
    pathfinder = PathfindingEngine()
    
    # Test different movement scenarios
    test_cases = [
        ("Horizontal Walk", (0, 100), (100, 100)),
        ("Upward Jump", (0, 100), (50, 50)),
        ("Downward Fall", (0, 50), (50, 100)),
        ("Large Vertical", (0, 100), (0, 0)),
    ]
    
    for name, start_pos, end_pos in test_cases:
        print(f"\n{name}: {start_pos} -> {end_pos}")
        
        # Get movement classification
        movement_type, physics_params = pathfinder.movement_classifier.classify_movement(
            start_pos, end_pos, None, None
        )
        
        # Calculate simple distance
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        simple_distance = (dx**2 + dy**2)**0.5
        
        print(f"  Movement Type: {MovementType(movement_type).name}")
        print(f"  Simple Distance: {simple_distance:.1f}")
        print(f"  Physics Distance: {physics_params.get('distance', 0):.1f}")
        print(f"  Energy Cost: {physics_params.get('energy_cost', 0):.2f}")
        print(f"  Risk Factor: {physics_params.get('risk_factor', 0):.2f}")
        
        # Calculate what the pathfinding cost would be
        base_cost = physics_params.get('distance', 0)
        energy_cost = physics_params.get('energy_cost', 0)
        risk_factor = physics_params.get('risk_factor', 0)
        total_cost = base_cost + energy_cost * 10 + risk_factor * 5
        
        print(f"  Total Pathfinding Cost: {total_cost:.2f}")
        print(f"  Cost vs Distance Ratio: {total_cost / simple_distance:.2f}x")

if __name__ == "__main__":
    test_physics_aware_pathfinding()
    test_cost_comparison()