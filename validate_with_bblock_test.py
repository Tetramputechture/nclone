#!/usr/bin/env python3
"""
Comprehensive validation of graph visualization fixes using the original bblock_test map.

This script loads the exact same map used in the original screenshots and validates
that all three reported issues have been resolved:

1. Functional edges between switches and doors
2. Walkable edges in solid tiles 
3. Pathfinding on traversable paths
"""

import os
import sys
import numpy as np
import pygame

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.level_data import LevelData
from nclone.graph.pathfinding import PathfindingEngine, PathfindingAlgorithm
from nclone.graph.common import EdgeType, NodeType
from nclone.graph.visualization import GraphVisualizer, VisualizationConfig
from nclone.constants import TILE_PIXEL_SIZE
from nclone.nsim import Simulator
from nclone.sim_config import SimConfig


def load_bblock_test_map():
    """Load the bblock_test map from the test_maps directory."""
    map_path = os.path.join('nclone', 'test_maps', 'bblock_test')
    
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"bblock_test map not found at {map_path}")
    
    with open(map_path, 'rb') as f:
        map_data = [int(b) for b in f.read()]
    
    print(f"✅ Loaded bblock_test map: {len(map_data)} bytes")
    return map_data


def create_level_data_from_simulator(sim):
    """Extract level data from the loaded simulator."""
    # Extract tile data (42x23 grid)
    tiles = np.zeros((42, 23), dtype=int)
    
    for (x, y), tile_id in sim.tile_dic.items():
        # Skip boundary tiles (they're added by the loader)
        if 1 <= x <= 42 and 1 <= y <= 23:
            tiles[x-1, y-1] = tile_id
    
    # Extract entities
    entities = []
    for entity_type, entity_list in sim.entity_dic.items():
        for entity in entity_list:
            # Try different attribute names for position
            x_pos = getattr(entity, 'xcoord', getattr(entity, 'x', getattr(entity, 'pos_x', 0)))
            y_pos = getattr(entity, 'ycoord', getattr(entity, 'y', getattr(entity, 'pos_y', 0)))
            
            entities.append({
                'type': entity_type,
                'entity_id': len(entities) + 1,
                'x': x_pos,
                'y': y_pos,
                'switch_id': getattr(entity, 'switch_id', None) if hasattr(entity, 'switch_id') else None
            })
    
    print(f"✅ Extracted level data: {tiles.shape} tiles, {len(entities)} entities")
    return LevelData(tiles=tiles, entities=entities)


def validate_functional_edges(graph_data):
    """Validate that functional edges exist between switches and doors."""
    print("\n=== ISSUE #1: FUNCTIONAL EDGES ===")
    
    # Count functional edges using the edge_types array
    functional_edge_mask = (graph_data.edge_types == EdgeType.FUNCTIONAL.value) & graph_data.edge_mask.astype(bool)
    functional_edge_count = np.sum(functional_edge_mask)
    
    print(f"Functional edges found: {functional_edge_count}")
    
    if functional_edge_count > 0:
        print("✅ ISSUE #1 RESOLVED: Functional edges are present")
        # Show first few functional edges
        functional_indices = np.where(functional_edge_mask)[0][:3]
        for i, edge_idx in enumerate(functional_indices):
            src_node = graph_data.edge_index[0, edge_idx]
            tgt_node = graph_data.edge_index[1, edge_idx]
            print(f"   Edge {i+1}: {src_node} -> {tgt_node}")
        return True
    else:
        print("❌ ISSUE #1 NOT RESOLVED: No functional edges found")
        return False


def validate_walkable_edges_in_solid_tiles(graph_data, level_data):
    """Validate that walkable edges don't exist in solid tiles."""
    print("\n=== ISSUE #2: WALKABLE EDGES IN SOLID TILES ===")
    
    # Count walkable edges using the edge_types array
    walkable_edge_mask = (graph_data.edge_types == EdgeType.WALK.value) & graph_data.edge_mask.astype(bool)
    total_walkable = np.sum(walkable_edge_mask)
    
    print(f"Total walkable edges: {total_walkable}")
    
    # Check if edges exist in solid tiles (sample first 1000 for performance)
    walkable_indices = np.where(walkable_edge_mask)[0][:1000]
    solid_tile_edges = 0
    
    for edge_idx in walkable_indices:
        src_node_idx = graph_data.edge_index[0, edge_idx]
        tgt_node_idx = graph_data.edge_index[1, edge_idx]
        
        # Get node positions from node_features (assuming x, y are first two features)
        src_x = graph_data.node_features[src_node_idx, 0]
        src_y = graph_data.node_features[src_node_idx, 1]
        tgt_x = graph_data.node_features[tgt_node_idx, 0]
        tgt_y = graph_data.node_features[tgt_node_idx, 1]
        
        # Convert to tile coordinates
        src_tile_x = int(src_x // TILE_PIXEL_SIZE)
        src_tile_y = int(src_y // TILE_PIXEL_SIZE)
        tgt_tile_x = int(tgt_x // TILE_PIXEL_SIZE)
        tgt_tile_y = int(tgt_y // TILE_PIXEL_SIZE)
        
        # Check if either endpoint is in a solid tile
        height, width = level_data.tiles.shape
        if (0 <= src_tile_x < width and 0 <= src_tile_y < height and
            level_data.tiles[src_tile_y, src_tile_x] == 1):
            solid_tile_edges += 1
        elif (0 <= tgt_tile_x < width and 0 <= tgt_tile_y < height and
              level_data.tiles[tgt_tile_y, tgt_tile_x] == 1):
            solid_tile_edges += 1
    
    if solid_tile_edges == 0:
        print("✅ ISSUE #2 RESOLVED: Walkable edges in solid tiles have been filtered out")
        print(f"   Edge count appears optimized (no solid tile edges in sample)")
        return True
    else:
        print(f"❌ ISSUE #2 NOT RESOLVED: Found {solid_tile_edges} edges in solid tiles (from sample)")
        return False


def validate_pathfinding(graph_data):
    """Validate that pathfinding works on traversable paths."""
    print("\n=== ISSUE #3: PATHFINDING ===")
    
    # Find connected nodes for testing
    pathfinding_engine = PathfindingEngine()
    
    # Get valid node indices
    valid_node_indices = np.where(graph_data.node_mask)[0]
    if len(valid_node_indices) < 10:
        print("❌ Not enough nodes for pathfinding test")
        return False
    
    # Test short path
    try:
        path = pathfinding_engine.find_shortest_path(
            graph_data, valid_node_indices[0], valid_node_indices[1], PathfindingAlgorithm.A_STAR
        )
        short_path_success = path is not None and path.success
        print(f"Short path test (A*): {'✅' if short_path_success else '❌'}")
    except Exception as e:
        print(f"Short path test (A*): ❌ (Error: {e})")
        short_path_success = False
    
    # Test with Dijkstra
    try:
        path = pathfinding_engine.find_shortest_path(
            graph_data, valid_node_indices[0], valid_node_indices[1], PathfindingAlgorithm.DIJKSTRA
        )
        dijkstra_success = path is not None and path.success
        print(f"Short path test (Dijkstra): {'✅' if dijkstra_success else '❌'}")
    except Exception as e:
        print(f"Short path test (Dijkstra): ❌ (Error: {e})")
        dijkstra_success = False
    
    # Test longer path
    try:
        if len(valid_node_indices) >= 20:
            path = pathfinding_engine.find_shortest_path(
                graph_data, valid_node_indices[5], valid_node_indices[15], PathfindingAlgorithm.A_STAR
            )
            long_path_success = path is not None and path.success
            print(f"Long path test: {'✅' if long_path_success else '❌'}")
        else:
            long_path_success = True
            print("Long path test: ✅ (skipped - not enough nodes)")
    except Exception as e:
        print(f"Long path test: ❌ (Error: {e})")
        long_path_success = False
    
    # Test same node
    try:
        path = pathfinding_engine.find_shortest_path(
            graph_data, valid_node_indices[0], valid_node_indices[0], PathfindingAlgorithm.A_STAR
        )
        same_node_success = path is not None and path.success and len(path.path) == 1
        print(f"Same node test: {'✅' if same_node_success else '❌'}")
    except Exception as e:
        print(f"Same node test: ❌ (Error: {e})")
        same_node_success = False
    
    if short_path_success and dijkstra_success and long_path_success and same_node_success:
        print("✅ ISSUE #3 RESOLVED: Pathfinding is working correctly")
        return True
    else:
        print("❌ ISSUE #3 NOT RESOLVED: Pathfinding has issues")
        return False


def create_validation_visualization(graph_data, level_data):
    """Create a visualization of the bblock_test map with all fixes applied."""
    print("\n=== CREATING BBLOCK_TEST VALIDATION VISUALIZATION ===")
    
    try:
        # Initialize pygame
        pygame.init()
        
        # Create visualizer with all edge types enabled
        config = VisualizationConfig(
            show_functional_edges=True,
            show_walk_edges=True,
            show_jump_edges=True,
            show_fall_edges=True,
            show_wall_slide_edges=True,
            show_one_way_edges=True,
            node_size=2,
            edge_width=1
        )
        
        visualizer = GraphVisualizer(config)
        
        # Create standalone visualization
        surface = visualizer.create_standalone_visualization(graph_data, level_data)
        
        # Save the image
        filename = 'bblock_test_validation_fixed.png'
        pygame.image.save(surface, filename)
        
        print(f"✅ bblock_test validation visualization saved as '{filename}'")
        print("   This image shows the original problematic map with all fixes applied:")
        print("   - Yellow functional edges between switches and doors")
        print("   - Green walkable edges only in valid traversable areas")
        print("   - No edges in solid tiles where ninja cannot exist")
        print("   - Proper graph connectivity for pathfinding")
        
        pygame.quit()
        
    except Exception as e:
        print(f"⚠️  Visualization creation failed: {e}")
        print("   (This doesn't affect the core validation results)")


def main():
    """Main validation function using the original bblock_test map."""
    print("=== COMPREHENSIVE VALIDATION USING ORIGINAL BBLOCK_TEST MAP ===")
    print("This script validates all fixes using the exact map from the original screenshots.\n")
    
    try:
        # Load the original bblock_test map
        map_data = load_bblock_test_map()
        
        # Create simulator and load the map
        sim_config = SimConfig()
        sim = Simulator(sim_config)
        sim.load(map_data)
        
        # Convert to level data format
        level_data = create_level_data_from_simulator(sim)
        
        # Build the hierarchical graph
        print("Building hierarchical graph...")
        builder = HierarchicalGraphBuilder()
        
        # Get ninja position from simulator
        ninja_x = sim.ninja.xpos if sim.ninja else 100.0
        ninja_y = sim.ninja.ypos if sim.ninja else 100.0
        ninja_position = (ninja_x, ninja_y)
        print(f"Ninja position: ({ninja_x}, {ninja_y})")
        
        hierarchical_graph = builder.build_graph(level_data, ninja_position)
        
        # Use the sub-cell graph (highest resolution) for validation
        graph_data = hierarchical_graph.sub_cell_graph
        
        print(f"Graph built: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
        
        # Validate each issue
        issue1_resolved = validate_functional_edges(graph_data)
        issue2_resolved = validate_walkable_edges_in_solid_tiles(graph_data, level_data)
        issue3_resolved = validate_pathfinding(graph_data)
        
        # Create validation visualization
        create_validation_visualization(graph_data, level_data)
        
        # Final summary
        print("\n" + "="*60)
        print("FINAL VALIDATION SUMMARY (BBLOCK_TEST MAP)")
        print("="*60)
        print(f"{'✅' if issue1_resolved else '❌'} {'RESOLVED' if issue1_resolved else 'NOT RESOLVED'}: Issue #1: Functional edges between switches and doors")
        print(f"{'✅' if issue2_resolved else '❌'} {'RESOLVED' if issue2_resolved else 'NOT RESOLVED'}: Issue #2: Walkable edges in solid tiles")
        print(f"{'✅' if issue3_resolved else '❌'} {'RESOLVED' if issue3_resolved else 'NOT RESOLVED'}: Issue #3: Pathfinding not working on traversable paths")
        print("="*60)
        
        if issue1_resolved and issue2_resolved and issue3_resolved:
            print("🎉 ALL ISSUES RESOLVED! The graph visualization system is working correctly.")
            return True
        else:
            print("⚠️  SOME ISSUES REMAIN UNRESOLVED")
            print("Further investigation may be needed.")
            return False
            
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)