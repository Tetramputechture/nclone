#!/usr/bin/env python3
"""
Debug graph connectivity between ninja and target areas
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.constants.entity_types import EntityType

def find_connected_component(graph, start_node):
    """Find all nodes in the same connected component as start_node"""
    visited = set()
    queue = deque([start_node])
    visited.add(start_node)
    
    while queue:
        current = queue.popleft()
        
        # Check all edges from this node
        for edge_idx in range(graph.num_edges):
            if graph.edge_mask[edge_idx] == 1:
                src = graph.edge_index[0, edge_idx]
                dst = graph.edge_index[1, edge_idx]
                
                if src == current and dst not in visited:
                    visited.add(dst)
                    queue.append(dst)
                elif dst == current and src not in visited:
                    visited.add(src)
                    queue.append(src)
    
    return visited

def debug_connectivity():
    print("=" * 60)
    print("GRAPH CONNECTIVITY DEBUG")
    print("=" * 60)
    
    # Create environment
    env = BasicLevelNoGold(render_mode="rgb_array", enable_frame_stack=False, enable_debug_overlay=False, eval_mode=False, seed=42)
    env.reset()
    ninja_position = env.nplay_headless.ninja_position()
    level_data = env.level_data
    
    print(f"Ninja position: {ninja_position}")
    print(f"Level size: {level_data.width}x{level_data.height} tiles")
    
    # Build graph
    builder = HierarchicalGraphBuilder()
    hierarchical_graph = builder.build_graph(level_data, ninja_position)
    graph = hierarchical_graph.sub_cell_graph
    
    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Find ninja node
    ninja_candidates = []
    for node_idx in range(graph.num_nodes):
        if graph.node_mask[node_idx] == 1:  # Valid node
            node_x = graph.node_features[node_idx, 0]
            node_y = graph.node_features[node_idx, 1]
            dist = ((node_x - ninja_position[0])**2 + (node_y - ninja_position[1])**2)**0.5
            if dist < 10:  # Close enough
                ninja_candidates.append((node_idx, node_x, node_y, dist))
    
    if not ninja_candidates:
        print("❌ Could not find ninja node")
        return False
    
    ninja_candidates.sort(key=lambda x: x[3])  # Sort by distance
    ninja_node = ninja_candidates[0][0]
    ninja_x, ninja_y = ninja_candidates[0][1], ninja_candidates[0][2]
    
    print(f"✅ Found ninja node: {ninja_node} at ({ninja_x:.1f}, {ninja_y:.1f})")
    
    # Find leftmost switch
    leftmost_switch = None
    leftmost_x = float('inf')
    
    for entity in level_data.entities:
        if entity.get("type") == EntityType.LOCKED_DOOR:
            switch_x = entity.get("x", 0)
            if switch_x < leftmost_x:
                leftmost_x = switch_x
                leftmost_switch = entity
    
    if leftmost_switch is None:
        print("❌ Could not find leftmost switch")
        return False
    
    switch_x = leftmost_switch.get("x", 0)
    switch_y = leftmost_switch.get("y", 0)
    print(f"✅ Found leftmost switch at ({switch_x}, {switch_y})")
    
    # Find switch node
    switch_node = None
    for node_idx in range(graph.num_nodes):
        if graph.node_mask[node_idx] == 1:  # Valid node
            node_x = graph.node_features[node_idx, 0]
            node_y = graph.node_features[node_idx, 1]
            dist = ((node_x - switch_x)**2 + (node_y - switch_y)**2)**0.5
            if dist < 10:  # Close enough
                switch_node = node_idx
                break
    
    if switch_node is None:
        print("❌ Could not find switch node")
        return False
    
    print(f"✅ Found switch node: {switch_node}")
    
    # Find connected components
    ninja_component = find_connected_component(graph, ninja_node)
    switch_component = find_connected_component(graph, switch_node)
    
    print(f"Ninja connected component: {len(ninja_component)} nodes")
    print(f"Switch connected component: {len(switch_component)} nodes")
    
    if ninja_node in switch_component:
        print("✅ Ninja and switch are in the same connected component!")
        return True
    else:
        print("❌ Ninja and switch are in DIFFERENT connected components!")
        
        # Analyze the components
        ninja_positions = []
        switch_positions = []
        
        for node_idx in ninja_component:
            if graph.node_mask[node_idx] == 1:
                x = graph.node_features[node_idx, 0]
                y = graph.node_features[node_idx, 1]
                ninja_positions.append((x, y))
        
        for node_idx in switch_component:
            if graph.node_mask[node_idx] == 1:
                x = graph.node_features[node_idx, 0]
                y = graph.node_features[node_idx, 1]
                switch_positions.append((x, y))
        
        # Find bounds of each component
        if ninja_positions:
            ninja_min_x = min(pos[0] for pos in ninja_positions)
            ninja_max_x = max(pos[0] for pos in ninja_positions)
            ninja_min_y = min(pos[1] for pos in ninja_positions)
            ninja_max_y = max(pos[1] for pos in ninja_positions)
            print(f"Ninja component bounds: x=[{ninja_min_x:.0f}, {ninja_max_x:.0f}], y=[{ninja_min_y:.0f}, {ninja_max_y:.0f}]")
        
        if switch_positions:
            switch_min_x = min(pos[0] for pos in switch_positions)
            switch_max_x = max(pos[0] for pos in switch_positions)
            switch_min_y = min(pos[1] for pos in switch_positions)
            switch_max_y = max(pos[1] for pos in switch_positions)
            print(f"Switch component bounds: x=[{switch_min_x:.0f}, {switch_max_x:.0f}], y=[{switch_min_y:.0f}, {switch_max_y:.0f}]")
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot ninja component in blue
        if ninja_positions:
            ninja_x_coords = [pos[0] for pos in ninja_positions]
            ninja_y_coords = [pos[1] for pos in ninja_positions]
            plt.scatter(ninja_x_coords, ninja_y_coords, c='blue', alpha=0.6, s=1, label=f'Ninja Component ({len(ninja_component)} nodes)')
        
        # Plot switch component in red
        if switch_positions:
            switch_x_coords = [pos[0] for pos in switch_positions]
            switch_y_coords = [pos[1] for pos in switch_positions]
            plt.scatter(switch_x_coords, switch_y_coords, c='red', alpha=0.6, s=1, label=f'Switch Component ({len(switch_component)} nodes)')
        
        # Highlight ninja and switch
        plt.scatter([ninja_x], [ninja_y], c='darkblue', s=100, marker='*', label='Ninja', edgecolors='white', linewidth=2)
        plt.scatter([switch_x], [switch_y], c='darkred', s=100, marker='s', label='Leftmost Switch', edgecolors='white', linewidth=2)
        
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Graph Connectivity Analysis - Disconnected Components')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # Invert Y axis to match game coordinates
        
        plt.tight_layout()
        plt.savefig('/workspace/nclone/connectivity_debug.png', dpi=150, bbox_inches='tight')
        print("📊 Saved connectivity visualization to connectivity_debug.png")
        
        return False

if __name__ == "__main__":
    success = debug_connectivity()
    if success:
        print("\n🎉 CONNECTIVITY DEBUG SUCCESSFUL!")
    else:
        print("\n❌ CONNECTIVITY DEBUG FAILED!")
    
    sys.exit(0 if success else 1)