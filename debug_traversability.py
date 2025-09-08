#!/usr/bin/env python3
"""
Debug traversability of different areas in the level
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.graph_construction import GraphConstructor
from nclone.graph.common import SUB_CELL_SIZE

def debug_traversability():
    print("=" * 60)
    print("TRAVERSABILITY DEBUG")
    print("=" * 60)
    
    # Create environment
    env = BasicLevelNoGold(render_mode="rgb_array", enable_frame_stack=False, enable_debug_overlay=False, eval_mode=False, seed=42)
    env.reset()
    ninja_position = env.nplay_headless.ninja_position()
    level_data = env.level_data
    
    print(f"Ninja position: {ninja_position}")
    print(f"Level size: {level_data.width}x{level_data.height} tiles")
    
    # Create graph constructor to access traversability logic
    from nclone.graph.feature_extraction import FeatureExtractor
    from nclone.graph.edge_building import EdgeBuilder
    
    feature_extractor = FeatureExtractor()
    edge_builder = EdgeBuilder(feature_extractor)
    constructor = GraphConstructor(feature_extractor, edge_builder)
    
    # Test traversability in a grid pattern
    test_positions = []
    traversable_positions = []
    non_traversable_positions = []
    
    # Test every 12 pixels (2 sub-cells) for performance
    step = 12
    for y in range(0, level_data.height * 24, step):
        for x in range(0, level_data.width * 24, step):
            test_positions.append((x, y))
            
            # Check if position is traversable
            is_traversable = constructor._is_position_traversable(x, y, level_data.tiles)
            
            if is_traversable:
                traversable_positions.append((x, y))
            else:
                non_traversable_positions.append((x, y))
    
    print(f"Tested {len(test_positions)} positions")
    print(f"Traversable: {len(traversable_positions)}")
    print(f"Non-traversable: {len(non_traversable_positions)}")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot non-traversable positions in red
    if non_traversable_positions:
        non_trav_x = [pos[0] for pos in non_traversable_positions]
        non_trav_y = [pos[1] for pos in non_traversable_positions]
        plt.scatter(non_trav_x, non_trav_y, c='red', alpha=0.6, s=2, label=f'Non-traversable ({len(non_traversable_positions)})')
    
    # Plot traversable positions in green
    if traversable_positions:
        trav_x = [pos[0] for pos in traversable_positions]
        trav_y = [pos[1] for pos in traversable_positions]
        plt.scatter(trav_x, trav_y, c='green', alpha=0.6, s=2, label=f'Traversable ({len(traversable_positions)})')
    
    # Highlight ninja position
    plt.scatter([ninja_position[0]], [ninja_position[1]], c='blue', s=100, marker='*', label='Ninja', edgecolors='white', linewidth=2)
    
    # Highlight leftmost switch
    leftmost_switch = None
    leftmost_x = float('inf')
    
    for entity in level_data.entities:
        if entity.get("type") == 6:  # LOCKED_DOOR
            switch_x = entity.get("x", 0)
            if switch_x < leftmost_x:
                leftmost_x = switch_x
                leftmost_switch = entity
    
    if leftmost_switch:
        switch_x = leftmost_switch.get("x", 0)
        switch_y = leftmost_switch.get("y", 0)
        plt.scatter([switch_x], [switch_y], c='purple', s=100, marker='s', label='Leftmost Switch', edgecolors='white', linewidth=2)
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Level Traversability Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()  # Invert Y axis to match game coordinates
    
    plt.tight_layout()
    plt.savefig('/workspace/nclone/traversability_debug.png', dpi=150, bbox_inches='tight')
    print("📊 Saved traversability visualization to traversability_debug.png")
    
    # Check specific path between ninja and switch
    if leftmost_switch:
        print(f"\n🔍 Checking path from ninja to leftmost switch:")
        print(f"Ninja: ({ninja_position[0]}, {ninja_position[1]})")
        print(f"Switch: ({switch_x}, {switch_y})")
        
        # Sample points along the direct path
        num_samples = 20
        for i in range(num_samples + 1):
            t = i / num_samples
            sample_x = ninja_position[0] + t * (switch_x - ninja_position[0])
            sample_y = ninja_position[1] + t * (switch_y - ninja_position[1])
            
            is_traversable = constructor._is_position_traversable(sample_x, sample_y, level_data.tiles)
            status = "✅" if is_traversable else "❌"
            print(f"  Point {i:2d}: ({sample_x:6.1f}, {sample_y:6.1f}) {status}")

if __name__ == "__main__":
    debug_traversability()
    print("\n🎉 TRAVERSABILITY DEBUG COMPLETE!")