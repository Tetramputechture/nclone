#!/usr/bin/env python3
"""
Simple visualization test to verify graph construction and basic rendering.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.constants.entity_types import EntityType
from nclone.constants.physics_constants import TILE_PIXEL_SIZE

def simple_test():
    print("=" * 60)
    print("SIMPLE VISUALIZATION TEST")
    print("=" * 60)
    
    # Verify doortest map
    doortest_path = "/workspace/nclone/nclone/test_maps/doortest"
    if not os.path.exists(doortest_path):
        print(f"❌ doortest map not found at {doortest_path}")
        return False
    print(f"✅ doortest map found")
    
    # Create environment
    try:
        env = BasicLevelNoGold(render_mode="rgb_array", enable_frame_stack=False, enable_debug_overlay=False, eval_mode=False, seed=42)
        env.reset()
        ninja_position = env.nplay_headless.ninja_position()
        level_data = env.level_data
        print(f"✅ Environment created: {level_data.width}x{level_data.height} tiles")
        print(f"✅ Ninja position: {ninja_position}")
        print(f"✅ Entities: {len(level_data.entities)}")
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return False
    
    # Build graph
    try:
        builder = HierarchicalGraphBuilder()
        hierarchical_graph = builder.build_graph(level_data, ninja_position)
        graph = hierarchical_graph.sub_cell_graph
        print(f"✅ Graph built: {graph.num_nodes} nodes, {graph.num_edges} edges")
    except Exception as e:
        print(f"❌ Graph building failed: {e}")
        return False
    
    # Create simple visualization
    try:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Map dimensions
        map_width = level_data.width * TILE_PIXEL_SIZE
        map_height = level_data.height * TILE_PIXEL_SIZE
        
        ax.set_xlim(0, map_width)
        ax.set_ylim(0, map_height)
        ax.invert_yaxis()  # Y-axis: max at top, 0 at bottom
        
        # Draw tiles
        for tile_y in range(level_data.height):
            for tile_x in range(level_data.width):
                tile_value = level_data.tiles[tile_y, tile_x]
                if tile_value == 1:  # Solid tiles
                    pixel_x = tile_x * TILE_PIXEL_SIZE
                    pixel_y = tile_y * TILE_PIXEL_SIZE
                    rect = patches.Rectangle((pixel_x, pixel_y), TILE_PIXEL_SIZE, TILE_PIXEL_SIZE, 
                                           facecolor='gray', edgecolor='black', linewidth=0.5)
                    ax.add_patch(rect)
        
        # Draw entities
        entity_colors = {
            EntityType.NINJA: 'black',
            EntityType.EXIT_SWITCH: 'yellow', 
            EntityType.EXIT_DOOR: 'gold',
            EntityType.LOCKED_DOOR: 'blue',
            EntityType.TRAP_DOOR: 'magenta',
            EntityType.ONE_WAY: 'green'
        }
        
        for entity in level_data.entities:
            entity_type = entity.get("type", 0)
            entity_x = entity.get("x", 0)
            entity_y = entity.get("y", 0)
            
            color = entity_colors.get(entity_type, 'gray')
            if entity_type == EntityType.NINJA:
                circle = patches.Circle((entity_x, entity_y), 8, facecolor=color, edgecolor='white', linewidth=2)
            else:
                circle = patches.Circle((entity_x, entity_y), 6, facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(circle)
        
        # Draw some graph nodes (sample)
        node_sample = min(100, graph.num_nodes)  # Sample first 100 nodes
        for i in range(node_sample):
            if graph.node_mask[i] == 1:  # Valid node
                node_x = graph.node_features[i, 0]
                node_y = graph.node_features[i, 1]
                ax.plot(node_x, node_y, 'r.', markersize=2, alpha=0.5)
        
        ax.set_title(f'Simple Doortest Visualization\n'
                    f'{level_data.width}x{level_data.height} tiles, {len(level_data.entities)} entities, '
                    f'{graph.num_nodes} nodes, {graph.num_edges} edges\n'
                    f'Y-axis corrected: max at top, 0 at bottom', fontsize=12)
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = "/workspace/simple_visualization_test.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved simple visualization to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_test()
    if success:
        print("\n🎉 SIMPLE TEST SUCCESSFUL!")
    else:
        print("\n❌ SIMPLE TEST FAILED!")
    
    sys.exit(0 if success else 1)