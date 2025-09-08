#!/usr/bin/env python3
"""
Debug with visual tile graphics and traversability overlay
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.graph_construction import GraphConstructor
from nclone.graph.feature_extraction import FeatureExtractor
from nclone.graph.edge_building import EdgeBuilder

def debug_visual_overlay():
    print("=" * 60)
    print("VISUAL TILE GRAPHICS + TRAVERSABILITY OVERLAY")
    print("=" * 60)
    
    # Create environment
    env = BasicLevelNoGold(render_mode="rgb_array", enable_frame_stack=False, enable_debug_overlay=False, eval_mode=False, seed=42)
    env.reset()
    ninja_position = env.nplay_headless.ninja_position()
    level_data = env.level_data
    
    print(f"Ninja position: {ninja_position}")
    print(f"Level size: {level_data.width}x{level_data.height} tiles")
    
    # Create graph constructor for traversability testing
    feature_extractor = FeatureExtractor()
    edge_builder = EdgeBuilder(feature_extractor)
    constructor = GraphConstructor(feature_extractor, edge_builder)
    
    # Get the actual rendered level image from the environment
    print("Rendering level image...")
    level_image = env.render()  # This should give us the actual tile graphics
    
    # Test traversability on a fine grid
    print("Testing traversability on fine grid...")
    traversable_positions = []
    non_traversable_positions = []
    
    # Test every 3 pixels for higher resolution
    step = 3
    for y in range(0, level_data.height * 24, step):
        for x in range(0, level_data.width * 24, step):
            is_traversable = constructor._is_position_traversable(x, y, level_data.tiles)
            
            if is_traversable:
                traversable_positions.append((x, y))
            else:
                non_traversable_positions.append((x, y))
    
    print(f"Found {len(traversable_positions)} traversable positions")
    print(f"Found {len(non_traversable_positions)} non-traversable positions")
    
    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left plot: Actual level rendering with traversability overlay
    ax1.imshow(level_image, origin='upper', extent=[0, level_data.width * 24, level_data.height * 24, 0])
    
    # Overlay traversability points
    if traversable_positions:
        trav_x = [pos[0] for pos in traversable_positions]
        trav_y = [pos[1] for pos in traversable_positions]
        ax1.scatter(trav_x, trav_y, c='lime', alpha=0.6, s=1, label=f'Traversable ({len(traversable_positions)})')
    
    if non_traversable_positions:
        # Sample non-traversable positions to avoid overcrowding
        sample_size = min(1000, len(non_traversable_positions))
        sampled_indices = np.random.choice(len(non_traversable_positions), sample_size, replace=False)
        sampled_non_trav = [non_traversable_positions[i] for i in sampled_indices]
        
        non_trav_x = [pos[0] for pos in sampled_non_trav]
        non_trav_y = [pos[1] for pos in sampled_non_trav]
        ax1.scatter(non_trav_x, non_trav_y, c='red', alpha=0.3, s=0.5, label=f'Non-traversable (sampled)')
    
    # Mark ninja and entities
    ax1.scatter([ninja_position[0]], [ninja_position[1]], c='blue', s=100, marker='*', 
               label='Ninja', edgecolors='white', linewidth=2, zorder=10)
    
    # Mark leftmost switch
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
        ax1.scatter([switch_x], [switch_y], c='purple', s=100, marker='s', 
                   label='Leftmost Switch', edgecolors='white', linewidth=2, zorder=10)
        
        # Draw direct path line
        ax1.plot([ninja_position[0], switch_x], [ninja_position[1], switch_y], 
                'yellow', linewidth=2, alpha=0.7, linestyle='--', label='Direct Path')
    
    ax1.set_xlabel('X Position (pixels)')
    ax1.set_ylabel('Y Position (pixels)')
    ax1.set_title('Level Rendering + Traversability Overlay')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Tile map with traversability density
    tiles = level_data.tiles
    
    # Create traversability density map
    density_map = np.zeros((level_data.height, level_data.width))
    
    for x, y in traversable_positions:
        tile_x = min(int(x // 24), level_data.width - 1)
        tile_y = min(int(y // 24), level_data.height - 1)
        density_map[tile_y, tile_x] += 1
    
    # Normalize density map
    max_density = np.max(density_map) if np.max(density_map) > 0 else 1
    density_map = density_map / max_density
    
    # Show tile map with density overlay
    im = ax2.imshow(tiles, cmap='tab20', interpolation='nearest', aspect='auto', alpha=0.7)
    ax2.imshow(density_map, cmap='Greens', interpolation='nearest', aspect='auto', alpha=0.8)
    
    # Mark ninja and switch positions in tile coordinates
    ninja_tile_x = int(ninja_position[0] // 24)
    ninja_tile_y = int(ninja_position[1] // 24)
    ax2.scatter([ninja_tile_x], [ninja_tile_y], c='blue', s=100, marker='*', 
               label='Ninja', edgecolors='white', linewidth=2)
    
    if leftmost_switch:
        switch_tile_x = int(switch_x // 24)
        switch_tile_y = int(switch_y // 24)
        ax2.scatter([switch_tile_x], [switch_tile_y], c='purple', s=100, marker='s', 
                   label='Leftmost Switch', edgecolors='white', linewidth=2)
    
    ax2.set_xlabel('Tile X')
    ax2.set_ylabel('Tile Y')
    ax2.set_title('Tile Map + Traversability Density (Green = More Traversable)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/nclone/visual_overlay_debug.png', dpi=150, bbox_inches='tight')
    print("📊 Saved visual overlay to visual_overlay_debug.png")
    
    # Analyze connectivity gaps
    print(f"\n🔍 CONNECTIVITY ANALYSIS:")
    
    # Find clusters of traversable positions
    print(f"Analyzing traversable position clusters...")
    
    # Group traversable positions by proximity
    clusters = []
    used_positions = set()
    
    for pos in traversable_positions:
        if pos in used_positions:
            continue
            
        # Start a new cluster
        cluster = [pos]
        queue = [pos]
        used_positions.add(pos)
        
        while queue:
            current = queue.pop(0)
            cx, cy = current
            
            # Check nearby positions (within 12 pixels)
            for other_pos in traversable_positions:
                if other_pos in used_positions:
                    continue
                    
                ox, oy = other_pos
                dist = ((cx - ox)**2 + (cy - oy)**2)**0.5
                
                if dist <= 12:  # Within 2 sub-cells
                    cluster.append(other_pos)
                    queue.append(other_pos)
                    used_positions.add(other_pos)
        
        clusters.append(cluster)
    
    # Sort clusters by size
    clusters.sort(key=len, reverse=True)
    
    print(f"Found {len(clusters)} traversable clusters:")
    for i, cluster in enumerate(clusters[:10]):  # Show top 10 clusters
        if len(cluster) > 1:
            cluster_x = [pos[0] for pos in cluster]
            cluster_y = [pos[1] for pos in cluster]
            min_x, max_x = min(cluster_x), max(cluster_x)
            min_y, max_y = min(cluster_y), max(cluster_y)
            
            print(f"  Cluster {i+1}: {len(cluster):3d} positions, bounds x=[{min_x:3.0f}, {max_x:3.0f}], y=[{min_y:3.0f}, {max_y:3.0f}]")
    
    # Check which cluster contains ninja and switch
    ninja_cluster = None
    switch_cluster = None
    
    for i, cluster in enumerate(clusters):
        cluster_x = [pos[0] for pos in cluster]
        cluster_y = [pos[1] for pos in cluster]
        
        # Check if ninja is in this cluster (within 15 pixels)
        for cx, cy in cluster:
            if ((ninja_position[0] - cx)**2 + (ninja_position[1] - cy)**2)**0.5 <= 15:
                ninja_cluster = i
                break
        
        # Check if switch is in this cluster
        if leftmost_switch:
            for cx, cy in cluster:
                if ((switch_x - cx)**2 + (switch_y - cy)**2)**0.5 <= 15:
                    switch_cluster = i
                    break
    
    print(f"\nNinja is in cluster: {ninja_cluster + 1 if ninja_cluster is not None else 'None'}")
    print(f"Switch is in cluster: {switch_cluster + 1 if switch_cluster is not None else 'None'}")
    
    if ninja_cluster != switch_cluster:
        print("❌ NINJA AND SWITCH ARE IN DIFFERENT CLUSTERS - This explains the connectivity issue!")
    else:
        print("✅ Ninja and switch are in the same cluster")

if __name__ == "__main__":
    debug_visual_overlay()
    print("\n🎉 VISUAL OVERLAY DEBUG COMPLETE!")