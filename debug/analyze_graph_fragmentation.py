#!/usr/bin/env python3
"""
Analyze graph fragmentation to understand why long-distance pathfinding fails.
"""

import os
import sys
import numpy as np

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine


def analyze_graph_fragmentation():
    """Analyze why the graph is highly fragmented."""
    print("=" * 80)
    print("ANALYZING GRAPH FRAGMENTATION FOR LONG-DISTANCE PATHFINDING")
    print("=" * 80)
    
    # Create environment
    env = BasicLevelNoGold(
        render_mode="rgb_array",
        enable_frame_stack=False,
        enable_debug_overlay=False,
        eval_mode=False,
        seed=42
    )
    
    # Reset to load the map
    env.reset()
    
    # Get level data and ninja position
    level_data = env.level_data
    ninja_pos = env.nplay_headless.ninja_position()
    
    print(f"Map size: {level_data.width}x{level_data.height} tiles")
    print(f"Ninja position: {ninja_pos}")
    
    # Analyze tile distribution
    solid_tiles = 0
    empty_tiles = 0
    other_tiles = 0
    
    for y in range(level_data.height):
        for x in range(level_data.width):
            tile_value = level_data.get_tile(y, x)
            if tile_value == 0:
                empty_tiles += 1
            elif tile_value == 1:
                solid_tiles += 1
            else:
                other_tiles += 1
    
    total_tiles = level_data.width * level_data.height
    print(f"\nTile distribution:")
    print(f"  Empty tiles (0): {empty_tiles} ({empty_tiles/total_tiles*100:.1f}%)")
    print(f"  Solid tiles (1): {solid_tiles} ({solid_tiles/total_tiles*100:.1f}%)")
    print(f"  Other tiles: {other_tiles} ({other_tiles/total_tiles*100:.1f}%)")
    
    # Build graph
    print(f"\nBuilding graph...")
    graph_builder = HierarchicalGraphBuilder()
    hierarchical_data = graph_builder.build_graph(level_data, ninja_pos)
    graph_data = hierarchical_data.sub_cell_graph
    pathfinding_engine = PathfindingEngine()
    
    print(f"Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # Find all connected components
    print(f"\nAnalyzing connected components...")
    
    all_visited = set()
    components = []
    
    for node_idx in range(graph_data.num_nodes):
        if graph_data.node_mask[node_idx] == 0 or node_idx in all_visited:
            continue
        
        # Find this component
        visited = set()
        stack = [node_idx]
        component = []
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            
            visited.add(current)
            component.append(current)
            
            # Find neighbors
            for edge_idx in range(graph_data.num_edges):
                if graph_data.edge_mask[edge_idx] == 0:
                    continue
                
                src = int(graph_data.edge_index[0, edge_idx])
                dst = int(graph_data.edge_index[1, edge_idx])
                
                if src == current and dst not in visited:
                    stack.append(dst)
                elif dst == current and src not in visited:
                    stack.append(src)
        
        components.append(component)
        all_visited.update(component)
    
    # Sort components by size
    components.sort(key=len, reverse=True)
    
    print(f"Found {len(components)} connected components")
    print(f"Top 10 largest components:")
    
    ninja_node = pathfinding_engine._find_node_at_position(graph_data, ninja_pos)
    
    for i, component in enumerate(components[:10]):
        contains_ninja = ninja_node in component
        ninja_status = "🥷 NINJA" if contains_ninja else ""
        
        # Analyze component positions
        positions = []
        empty_nodes = 0
        solid_nodes = 0
        
        for node_idx in component:
            node_pos = pathfinding_engine._get_node_position(graph_data, node_idx)
            positions.append(node_pos)
            
            # Check tile type
            tile_x = int(node_pos[0] // 24)
            tile_y = int(node_pos[1] // 24)
            
            if 0 <= tile_x < level_data.width and 0 <= tile_y < level_data.height:
                tile_value = level_data.get_tile(tile_y, tile_x)
                if tile_value == 0:
                    empty_nodes += 1
                elif tile_value == 1:
                    solid_nodes += 1
        
        # Calculate bounding box
        if positions:
            min_x = min(pos[0] for pos in positions)
            max_x = max(pos[0] for pos in positions)
            min_y = min(pos[1] for pos in positions)
            max_y = max(pos[1] for pos in positions)
            
            width = max_x - min_x
            height = max_y - min_y
            
            print(f"  Component {i+1}: {len(component)} nodes {ninja_status}")
            print(f"    Area: ({min_x:.0f},{min_y:.0f}) to ({max_x:.0f},{max_y:.0f}) - {width:.0f}x{height:.0f}px")
            print(f"    Nodes: {empty_nodes} empty, {solid_nodes} solid")
    
    # Analyze empty tile connectivity specifically
    print(f"\n" + "=" * 60)
    print("ANALYZING EMPTY TILE CONNECTIVITY")
    print("=" * 60)
    
    # Find nodes in empty tiles
    empty_tile_nodes = []
    
    for node_idx in range(graph_data.num_nodes):
        if graph_data.node_mask[node_idx] == 0:
            continue
        
        node_pos = pathfinding_engine._get_node_position(graph_data, node_idx)
        tile_x = int(node_pos[0] // 24)
        tile_y = int(node_pos[1] // 24)
        
        if (0 <= tile_x < level_data.width and 0 <= tile_y < level_data.height and
            level_data.get_tile(tile_y, tile_x) == 0):
            empty_tile_nodes.append((node_idx, node_pos, tile_x, tile_y))
    
    print(f"Found {len(empty_tile_nodes)} nodes in empty tiles")
    
    # Group empty tile nodes by tile
    empty_tiles_with_nodes = {}
    for node_idx, node_pos, tile_x, tile_y in empty_tile_nodes:
        tile_key = (tile_x, tile_y)
        if tile_key not in empty_tiles_with_nodes:
            empty_tiles_with_nodes[tile_key] = []
        empty_tiles_with_nodes[tile_key].append((node_idx, node_pos))
    
    print(f"Empty tiles with nodes: {len(empty_tiles_with_nodes)}")
    
    # Check connectivity between empty tiles
    print(f"\nChecking connectivity between empty tiles...")
    
    connected_empty_tiles = set()
    isolated_empty_tiles = []
    
    for tile_key, nodes_in_tile in empty_tiles_with_nodes.items():
        tile_x, tile_y = tile_key
        
        # Check if any node in this tile connects to nodes in other empty tiles
        has_external_connection = False
        
        for node_idx, node_pos in nodes_in_tile:
            # Find neighbors of this node
            for edge_idx in range(graph_data.num_edges):
                if graph_data.edge_mask[edge_idx] == 0:
                    continue
                
                src = int(graph_data.edge_index[0, edge_idx])
                dst = int(graph_data.edge_index[1, edge_idx])
                
                other_node = None
                if src == node_idx:
                    other_node = dst
                elif dst == node_idx:
                    other_node = src
                
                if other_node is not None:
                    # Check if other node is in a different empty tile
                    other_pos = pathfinding_engine._get_node_position(graph_data, other_node)
                    other_tile_x = int(other_pos[0] // 24)
                    other_tile_y = int(other_pos[1] // 24)
                    
                    if (0 <= other_tile_x < level_data.width and 
                        0 <= other_tile_y < level_data.height and
                        level_data.get_tile(other_tile_y, other_tile_x) == 0 and
                        (other_tile_x, other_tile_y) != tile_key):
                        has_external_connection = True
                        break
            
            if has_external_connection:
                break
        
        if has_external_connection:
            connected_empty_tiles.add(tile_key)
        else:
            isolated_empty_tiles.append((tile_key, len(nodes_in_tile)))
    
    print(f"Connected empty tiles: {len(connected_empty_tiles)}")
    print(f"Isolated empty tiles: {len(isolated_empty_tiles)}")
    
    if isolated_empty_tiles:
        print(f"\nFirst 10 isolated empty tiles:")
        for i, ((tile_x, tile_y), node_count) in enumerate(isolated_empty_tiles[:10], 1):
            pixel_x = tile_x * 24 + 12
            pixel_y = tile_y * 24 + 12
            print(f"  {i}. Tile ({tile_x}, {tile_y}) at pixel ({pixel_x}, {pixel_y}) - {node_count} nodes")
    
    # Identify potential connectivity issues
    print(f"\n" + "=" * 60)
    print("CONNECTIVITY BOTTLENECK ANALYSIS")
    print("=" * 60)
    
    fragmentation_ratio = len(components) / graph_data.num_nodes
    print(f"Fragmentation ratio: {fragmentation_ratio:.4f} (lower is better)")
    
    largest_component_ratio = len(components[0]) / graph_data.num_nodes if components else 0
    print(f"Largest component ratio: {largest_component_ratio:.4f} (higher is better)")
    
    empty_tile_isolation_ratio = len(isolated_empty_tiles) / len(empty_tiles_with_nodes) if empty_tiles_with_nodes else 0
    print(f"Empty tile isolation ratio: {empty_tile_isolation_ratio:.4f} (lower is better)")
    
    print(f"\nDiagnosis:")
    if fragmentation_ratio > 0.5:
        print("❌ SEVERE FRAGMENTATION: Graph is extremely fragmented")
    elif fragmentation_ratio > 0.1:
        print("⚠️  HIGH FRAGMENTATION: Graph connectivity needs improvement")
    else:
        print("✅ GOOD CONNECTIVITY: Graph fragmentation is acceptable")
    
    if largest_component_ratio < 0.1:
        print("❌ POOR CONNECTIVITY: Largest component is too small")
    elif largest_component_ratio < 0.5:
        print("⚠️  LIMITED CONNECTIVITY: Largest component could be bigger")
    else:
        print("✅ GOOD CONNECTIVITY: Largest component covers significant portion")
    
    if empty_tile_isolation_ratio > 0.5:
        print("❌ EMPTY TILE ISOLATION: Many empty tiles are disconnected")
        print("   💡 Suggestion: Collision detection may be too restrictive")
    elif empty_tile_isolation_ratio > 0.2:
        print("⚠️  SOME EMPTY TILE ISOLATION: Some empty tiles are disconnected")
    else:
        print("✅ GOOD EMPTY TILE CONNECTIVITY: Most empty tiles are connected")
    
    return {
        'total_components': len(components),
        'largest_component_size': len(components[0]) if components else 0,
        'ninja_component_size': len([c for c in components if ninja_node in c][0]) if any(ninja_node in c for c in components) else 0,
        'fragmentation_ratio': fragmentation_ratio,
        'empty_tile_isolation_ratio': empty_tile_isolation_ratio,
        'isolated_empty_tiles': len(isolated_empty_tiles),
        'connected_empty_tiles': len(connected_empty_tiles)
    }


if __name__ == '__main__':
    results = analyze_graph_fragmentation()
    print(f"\nAnalysis completed. Results: {results}")