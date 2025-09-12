#!/usr/bin/env python3
"""
Direct test of doortest map using Simulator to validate graph visualization fixes.

This script directly loads the doortest map and tests all three issues:
1. Functional edges between switches and doors
2. Walkable edges in solid tiles (ninja radius clearance)  
3. Pathfinding on traversable paths
"""

import os
import sys
import numpy as np

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nsim import Simulator
from nclone.sim_config import SimConfig
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.level_data import LevelData
from nclone.graph.navigation import PathfindingEngine
from nclone.graph.common import EdgeType
from nclone.constants import TILE_PIXEL_SIZE


def load_doortest_map():
    """Load the doortest map directly using Simulator."""
    map_path = os.path.join('nclone', 'test_maps', 'doortest')
    
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Doortest map not found at {map_path}")
    
    with open(map_path, 'rb') as f:
        map_data = bytearray(f.read())  # Use bytearray for mutability
    
    print(f"✅ Loaded doortest map: {len(map_data)} bytes")
    
    # Create simulator and load map
    sim = Simulator(SimConfig())
    sim.load(map_data)
    
    return sim


def extract_level_data_from_sim(sim):
    """Extract level data from simulator (similar to BaseEnvironment logic)."""
    # Get tile data
    tile_data = sim.get_tile_data()
    
    # Determine map dimensions by finding max coordinates
    max_x = max_y = 0
    for (x, y), tile_id in tile_data.items():
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    
    # Create tiles array (simulator includes 1-tile border)
    width = max_x - 1  # Remove border
    height = max_y - 1  # Remove border
    tiles = np.zeros((height, width), dtype=np.int32)
    
    # Fill tiles array (map inner area, excluding border)
    for (x, y), tile_id in tile_data.items():
        inner_x = x - 1
        inner_y = y - 1
        if 0 <= inner_x < width and 0 <= inner_y < height:
            tiles[inner_y, inner_x] = int(tile_id)
    
    # Extract entities
    entities = []
    
    # Get switches
    switches = sim.get_switches()
    for switch in switches:
        entities.append({
            'type': 'switch',
            'x': switch.x,
            'y': switch.y,
            'id': switch.id,
            'door_id': getattr(switch, 'door_id', None)
        })
    
    # Get doors
    doors = sim.get_doors()
    for door in doors:
        entities.append({
            'type': 'door',
            'x': door.x,
            'y': door.y,
            'id': door.id,
            'switch_id': getattr(door, 'switch_id', None)
        })
    
    # Get ninja position
    ninja = sim.ninja
    entities.append({
        'type': 'ninja',
        'x': ninja.x,
        'y': ninja.y,
        'id': 'ninja'
    })
    
    # Get exit door
    exit_door = sim.get_exit_door()
    if exit_door:
        entities.append({
            'type': 'exit_door',
            'x': exit_door.x,
            'y': exit_door.y,
            'id': 'exit_door'
        })
    
    level_data = LevelData(
        tiles=tiles,
        entities=entities,
        level_id="doortest"
    )
    
    print(f"✅ Extracted level data: ({width}, {height}) tiles, {len(entities)} entities")
    
    return level_data


def test_doortest_issues():
    """Test all three graph visualization issues with doortest map."""
    print("=" * 80)
    print("TESTING DOORTEST MAP GRAPH VISUALIZATION (DIRECT)")
    print("=" * 80)
    
    # Load doortest map
    sim = load_doortest_map()
    
    # Extract level data
    level_data = extract_level_data_from_sim(sim)
    
    # Get ninja position
    ninja_pos = (sim.ninja.x, sim.ninja.y)
    print(f"✅ Ninja position: {ninja_pos}")
    
    # Print entity information
    print(f"\nEntities in doortest map:")
    for entity in level_data.entities:
        print(f"  {entity['type']}: ({entity['x']}, {entity['y']}) id={entity['id']}")
    
    # Build graph
    print(f"\nBuilding hierarchical graph...")
    graph_builder = HierarchicalGraphBuilder()
    graph_data = graph_builder.build_hierarchical_graph(level_data, ninja_pos)
    
    print(f"✅ Built graph with {graph_data.sub_cell_graph.num_nodes} nodes and {graph_data.sub_cell_graph.num_edges} edges")
    
    # Test Issue #1: Functional edges
    print("\n" + "=" * 60)
    print("=== ISSUE #1: FUNCTIONAL EDGES ===")
    print("=" * 60)
    
    edge_types = graph_data.sub_cell_graph.edge_types
    functional_edges = np.sum(edge_types == EdgeType.FUNCTIONAL.value)
    
    print(f"Functional edges found: {functional_edges}")
    
    if functional_edges > 0:
        print("✅ ISSUE #1 RESOLVED: Functional edges are present")
        
        # Show functional edge details
        edge_index = graph_data.sub_cell_graph.edge_index
        node_features = graph_data.sub_cell_graph.node_features
        
        print("Sample functional edges:")
        functional_count = 0
        for i in range(graph_data.sub_cell_graph.num_edges):
            if edge_types[i] == EdgeType.FUNCTIONAL.value:
                node1_id = edge_index[0, i]
                node2_id = edge_index[1, i]
                
                node1_pos = (float(node_features[node1_id, 0]), float(node_features[node1_id, 1]))
                node2_pos = (float(node_features[node2_id, 0]), float(node_features[node2_id, 1]))
                
                print(f"  Functional edge {functional_count + 1}: {node1_pos} -> {node2_pos}")
                functional_count += 1
                if functional_count >= 5:  # Show first 5
                    break
    else:
        print("❌ ISSUE #1 NOT RESOLVED: No functional edges found")
    
    # Test Issue #2: Walkable edges in solid tiles
    print("\n" + "=" * 60)
    print("=== ISSUE #2: WALKABLE EDGES IN SOLID TILES ===")
    print("=" * 60)
    
    edge_index = graph_data.sub_cell_graph.edge_index
    edge_types = graph_data.sub_cell_graph.edge_types
    node_features = graph_data.sub_cell_graph.node_features
    
    walkable_edges_in_solid = 0
    solid_tile_violations = []
    
    # Count solid tiles first
    solid_tiles = 0
    for y in range(level_data.height):
        for x in range(level_data.width):
            if level_data.get_tile(y, x) == 1:
                solid_tiles += 1
    
    print(f"Total solid tiles in map: {solid_tiles}")
    
    for i in range(graph_data.sub_cell_graph.num_edges):
        if edge_types[i] == EdgeType.WALK.value:
            node1_id = edge_index[0, i]
            node2_id = edge_index[1, i]
            
            # Get node positions
            node1_pos = (float(node_features[node1_id, 0]), float(node_features[node1_id, 1]))
            node2_pos = (float(node_features[node2_id, 0]), float(node_features[node2_id, 1]))
            
            # Check if either node is in a solid tile
            for pos in [node1_pos, node2_pos]:
                tile_x = int(pos[0] // TILE_PIXEL_SIZE)
                tile_y = int(pos[1] // TILE_PIXEL_SIZE)
                
                if (0 <= tile_y < level_data.height and 0 <= tile_x < level_data.width and
                    level_data.get_tile(tile_y, tile_x) == 1):
                    walkable_edges_in_solid += 1
                    solid_tile_violations.append({
                        'edge_id': i,
                        'node1_pos': node1_pos,
                        'node2_pos': node2_pos,
                        'tile_pos': (tile_x, tile_y),
                        'tile_value': level_data.get_tile(tile_y, tile_x)
                    })
                    break
    
    print(f"Walkable edges in solid tiles: {walkable_edges_in_solid}")
    
    if walkable_edges_in_solid == 0:
        print("✅ ISSUE #2 RESOLVED: Walkable edges in solid tiles have been filtered out")
    else:
        print("❌ ISSUE #2 NOT RESOLVED: Found walkable edges in solid tiles")
        print("First few violations:")
        for i, violation in enumerate(solid_tile_violations[:5]):
            print(f"  Violation {i+1}: Edge {violation['edge_id']} at tile {violation['tile_pos']} (value={violation['tile_value']})")
            print(f"    Node positions: {violation['node1_pos']} -> {violation['node2_pos']}")
    
    # Test Issue #3: Pathfinding
    print("\n" + "=" * 60)
    print("=== ISSUE #3: PATHFINDING ===")
    print("=" * 60)
    
    navigation_engine = PathfindingEngine()
    
    # Test navigation between nearby positions
    start_pos = ninja_pos
    # Choose positions that should be reachable
    test_positions = [
        (ninja_pos[0] + 50, ninja_pos[1]),      # 50 pixels right
        (ninja_pos[0] - 50, ninja_pos[1]),      # 50 pixels left  
        (ninja_pos[0], ninja_pos[1] + 50),      # 50 pixels down
        (ninja_pos[0], ninja_pos[1] - 50),      # 50 pixels up
        (ninja_pos[0] + 25, ninja_pos[1] + 25), # diagonal
    ]
    
    successful_paths = 0
    
    for i, end_pos in enumerate(test_positions):
        print(f"\nTest {i+1}: Pathfinding from {start_pos} to {end_pos}")
        
        # Find nodes at these positions
        start_node = navigation_engine._find_node_at_position(graph_data.sub_cell_graph, start_pos)
        end_node = navigation_engine._find_node_at_position(graph_data.sub_cell_graph, end_pos)
        
        if start_node is not None and end_node is not None:
            path_result = navigation_engine.find_shortest_path(graph_data.sub_cell_graph, start_node, end_node)
            
            if path_result.success:
                successful_paths += 1
                print(f"  ✅ Path found: {len(path_result.path)} nodes, cost {path_result.total_cost:.2f}")
            else:
                print(f"  ❌ No path found")
        else:
            print(f"  ❌ Could not find nodes (start: {start_node}, end: {end_node})")
    
    print(f"\nPathfinding success rate: {successful_paths}/{len(test_positions)} ({100*successful_paths/len(test_positions):.1f}%)")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL VALIDATION SUMMARY (DOORTEST MAP)")
    print("=" * 80)
    
    issue1_resolved = functional_edges > 0
    issue2_resolved = walkable_edges_in_solid == 0
    issue3_resolved = successful_paths > 0
    
    print(f"{'✅' if issue1_resolved else '❌'} RESOLVED: Issue #1: Functional edges between switches and doors")
    print(f"{'✅' if issue2_resolved else '❌'} RESOLVED: Issue #2: Walkable edges in solid tiles")
    print(f"{'✅' if issue3_resolved else '❌'} RESOLVED: Issue #3: Pathfinding not working on traversable paths")
    
    if issue1_resolved and issue2_resolved and issue3_resolved:
        print("\n🎉 ALL ISSUES RESOLVED! The graph visualization system is working correctly.")
    else:
        print(f"\n⚠️  {sum([issue1_resolved, issue2_resolved, issue3_resolved])}/3 issues resolved. More work needed.")
    
    print("=" * 80)
    
    return {
        'functional_edges': functional_edges,
        'walkable_edges_in_solid': walkable_edges_in_solid,
        'navigation_success_rate': successful_paths / len(test_positions),
        'all_resolved': issue1_resolved and issue2_resolved and issue3_resolved,
        'solid_tiles': solid_tiles
    }


if __name__ == '__main__':
    try:
        results = test_doortest_issues()
        
        print(f"\nTest Results Summary:")
        print(f"- Functional edges: {results['functional_edges']}")
        print(f"- Walkable edges in solid tiles: {results['walkable_edges_in_solid']}")
        print(f"- Pathfinding success rate: {results['navigation_success_rate']:.1%}")
        print(f"- Solid tiles in map: {results['solid_tiles']}")
        
        # Exit with appropriate code
        if results['all_resolved']:
            print("✅ All tests passed!")
            sys.exit(0)
        else:
            print("❌ Some tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)