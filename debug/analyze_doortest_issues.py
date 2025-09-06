#!/usr/bin/env python3
"""
Analyze doortest map issues using the actual BasicLevelNoGold environment.

This script uses the real environment and graph system to diagnose the three issues:
1. Missing functional edges between switches and doors
2. Walkable edges in solid tiles 
3. Pathfinding not working on traversable paths
"""

import os
import sys
import numpy as np

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.pathfinding import PathfindingEngine
from nclone.graph.common import EdgeType
from nclone.constants import TILE_PIXEL_SIZE


def analyze_doortest_map():
    """Analyze the doortest map using the actual BasicLevelNoGold environment."""
    print("=" * 80)
    print("ANALYZING DOORTEST MAP WITH ACTUAL ENVIRONMENT")
    print("=" * 80)
    
    # Create the actual environment (it loads doortest by default)
    print("Creating BasicLevelNoGold environment...")
    env = BasicLevelNoGold(
        render_mode="rgb_array",
        enable_frame_stack=False,
        enable_debug_overlay=False,
        eval_mode=False,
        seed=42
    )
    
    # Reset to load the map
    print("Resetting environment to load doortest map...")
    env.reset()
    
    # Get level data using the environment's method
    level_data = env.level_data
    print(f"✅ Loaded level data: ({level_data.width}, {level_data.height}) tiles, {len(level_data.entities)} entities")
    
    # Get ninja position from the environment
    ninja_pos = env.nplay_headless.ninja_position()
    print(f"✅ Ninja position: {ninja_pos}")
    
    # Print detailed entity information
    print(f"\nEntities in doortest map:")
    for i, entity in enumerate(level_data.entities):
        print(f"  {i+1}. {entity}")
    
    # Analyze tile distribution
    solid_tiles = 0
    empty_tiles = 0
    for y in range(level_data.height):
        for x in range(level_data.width):
            tile_value = level_data.get_tile(y, x)
            if tile_value == 1:
                solid_tiles += 1
            elif tile_value == 0:
                empty_tiles += 1
    
    print(f"\nTile analysis:")
    print(f"  Total tiles: {level_data.width * level_data.height}")
    print(f"  Solid tiles (value=1): {solid_tiles}")
    print(f"  Empty tiles (value=0): {empty_tiles}")
    print(f"  Other tiles: {level_data.width * level_data.height - solid_tiles - empty_tiles}")
    
    # Build graph using the hierarchical builder
    print(f"\nBuilding graph with HierarchicalGraphBuilder...")
    graph_builder = HierarchicalGraphBuilder()
    
    try:
        hierarchical_data = graph_builder.build_graph(level_data, ninja_pos)
        graph_data = hierarchical_data.sub_cell_graph
        
        print(f"✅ Graph built successfully:")
        print(f"  Nodes: {graph_data.num_nodes}")
        print(f"  Edges: {graph_data.num_edges}")
        print(f"  Node features shape: {graph_data.node_features.shape}")
        print(f"  Edge types shape: {graph_data.edge_types.shape}")
        
    except Exception as e:
        print(f"❌ Failed to build graph: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Analyze Issue #1: Functional edges
    print("\n" + "=" * 60)
    print("=== ISSUE #1: FUNCTIONAL EDGES ANALYSIS ===")
    print("=" * 60)
    
    edge_types = graph_data.edge_types
    edge_type_counts = {}
    
    for edge_type in EdgeType:
        count = np.sum(edge_types == edge_type.value)
        edge_type_counts[edge_type.name] = count
        print(f"  {edge_type.name}: {count} edges")
    
    functional_edges = edge_type_counts.get('FUNCTIONAL', 0)
    
    if functional_edges > 0:
        print(f"\n✅ ISSUE #1 STATUS: {functional_edges} functional edges found")
        
        # Show details of functional edges
        edge_index = graph_data.edge_index
        node_features = graph_data.node_features
        
        print("\nFunctional edge details:")
        functional_count = 0
        for i in range(graph_data.num_edges):
            if edge_types[i] == EdgeType.FUNCTIONAL.value:
                node1_id = edge_index[0, i]
                node2_id = edge_index[1, i]
                
                node1_pos = (float(node_features[node1_id, 0]), float(node_features[node1_id, 1]))
                node2_pos = (float(node_features[node2_id, 0]), float(node_features[node2_id, 1]))
                
                distance = np.sqrt((node1_pos[0] - node2_pos[0])**2 + (node1_pos[1] - node2_pos[1])**2)
                
                print(f"  Edge {functional_count + 1}: {node1_pos} -> {node2_pos} (distance: {distance:.1f})")
                functional_count += 1
                if functional_count >= 10:  # Show first 10
                    break
        
        if functional_count < functional_edges:
            print(f"  ... and {functional_edges - functional_count} more functional edges")
    else:
        print(f"\n❌ ISSUE #1 STATUS: No functional edges found")
        
        # Check if we have switches and doors in entities
        switches = [e for e in level_data.entities if e.get('type') == 'switch']
        doors = [e for e in level_data.entities if e.get('type') == 'door']
        
        print(f"  Switches in level data: {len(switches)}")
        print(f"  Doors in level data: {len(doors)}")
        
        for switch in switches:
            print(f"    Switch: {switch}")
        for door in doors:
            print(f"    Door: {door}")
    
    # Analyze Issue #2: Walkable edges in solid tiles
    print("\n" + "=" * 60)
    print("=== ISSUE #2: WALKABLE EDGES IN SOLID TILES ANALYSIS ===")
    print("=" * 60)
    
    walkable_edges = edge_type_counts.get('WALK', 0)
    print(f"Total walkable edges: {walkable_edges}")
    
    if walkable_edges > 0:
        edge_index = graph_data.edge_index
        node_features = graph_data.node_features
        
        walkable_edges_in_solid = 0
        violations = []
        
        for i in range(graph_data.num_edges):
            if edge_types[i] == EdgeType.WALK.value:
                node1_id = edge_index[0, i]
                node2_id = edge_index[1, i]
                
                # Get node positions
                node1_pos = (float(node_features[node1_id, 0]), float(node_features[node1_id, 1]))
                node2_pos = (float(node_features[node2_id, 0]), float(node_features[node2_id, 1]))
                
                # Check if either node is in a solid tile
                # Skip ninja edges - ninja is allowed to have edges even in solid tiles
                ninja_pos = env.nplay_headless.ninja_position()
                is_ninja_edge = False
                
                for pos in [node1_pos, node2_pos]:
                    # Check if this position is the ninja position (within 1 pixel)
                    if abs(pos[0] - ninja_pos[0]) < 1.0 and abs(pos[1] - ninja_pos[1]) < 1.0:
                        is_ninja_edge = True
                        break
                
                if is_ninja_edge:
                    continue  # Skip ninja edges
                
                for pos in [node1_pos, node2_pos]:
                    tile_x = int(pos[0] // TILE_PIXEL_SIZE)
                    tile_y = int(pos[1] // TILE_PIXEL_SIZE)
                    
                    if (0 <= tile_y < level_data.height and 0 <= tile_x < level_data.width):
                        tile_value = level_data.get_tile(tile_y, tile_x)
                        if tile_value == 1:  # Solid tile
                            walkable_edges_in_solid += 1
                            violations.append({
                                'edge_id': i,
                                'node_pos': pos,
                                'tile_pos': (tile_x, tile_y),
                                'tile_value': tile_value
                            })
                            break
        
        print(f"Walkable edges in solid tiles: {walkable_edges_in_solid}")
        
        if walkable_edges_in_solid == 0:
            print("✅ ISSUE #2 STATUS: No walkable edges in solid tiles")
        else:
            print(f"❌ ISSUE #2 STATUS: {walkable_edges_in_solid} walkable edges found in solid tiles")
            
            print("\nFirst 10 violations:")
            for i, violation in enumerate(violations[:10]):
                print(f"  Violation {i+1}: Edge {violation['edge_id']}")
                print(f"    Node at {violation['node_pos']} in solid tile {violation['tile_pos']} (value={violation['tile_value']})")
    else:
        print("No walkable edges to analyze")
    
    # Analyze Issue #3: Pathfinding
    print("\n" + "=" * 60)
    print("=== ISSUE #3: PATHFINDING ANALYSIS ===")
    print("=" * 60)
    
    pathfinding_engine = PathfindingEngine()
    
    # Test pathfinding from ninja position to nearby locations
    start_pos = ninja_pos
    print(f"Testing pathfinding from ninja position: {start_pos}")
    
    # Find empty tiles to test pathfinding to
    test_positions = []
    
    # Look for empty tiles in the level
    for tile_y in range(level_data.height):
        for tile_x in range(level_data.width):
            if level_data.get_tile(tile_y, tile_x) == 0:  # Empty tile
                # Convert tile coordinates to pixel coordinates (center of tile)
                pixel_x = tile_x * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
                pixel_y = tile_y * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
                
                # Calculate distance from ninja
                distance = ((pixel_x - start_pos[0])**2 + (pixel_y - start_pos[1])**2)**0.5
                
                # Add tiles at various distances
                if 50 < distance < 200:  # Not too close, not too far
                    test_positions.append((pixel_x, pixel_y))
                    
                    if len(test_positions) >= 8:  # Limit to 8 tests
                        break
        if len(test_positions) >= 8:
            break
    
    # If no empty tiles found, use entity positions
    if not test_positions:
        for entity in level_data.entities:
            entity_x = entity.get('x', 0)
            entity_y = entity.get('y', 0)
            distance = ((entity_x - start_pos[0])**2 + (entity_y - start_pos[1])**2)**0.5
            
            if distance > 50:  # Not too close to ninja
                test_positions.append((entity_x, entity_y))
                
                if len(test_positions) >= 8:
                    break
    
    successful_paths = 0
    total_tests = len(test_positions)
    
    for i, end_pos in enumerate(test_positions):
        print(f"\nTest {i+1}: Pathfinding to {end_pos}")
        
        # Find nodes at positions
        start_node = pathfinding_engine._find_node_at_position(graph_data, start_pos)
        end_node = pathfinding_engine._find_node_at_position(graph_data, end_pos)
        
        print(f"  Start node: {start_node}, End node: {end_node}")
        
        if start_node is not None and end_node is not None:
            try:
                path_result = pathfinding_engine.find_shortest_path(graph_data, start_node, end_node)
                
                if path_result.success:
                    successful_paths += 1
                    print(f"  ✅ Path found: {len(path_result.path)} nodes, cost {path_result.total_cost:.2f}")
                else:
                    print(f"  ❌ No path found")
            except Exception as e:
                print(f"  ❌ Pathfinding error: {e}")
        else:
            print(f"  ❌ Could not find nodes at positions")
    
    success_rate = successful_paths / total_tests
    print(f"\nPathfinding success rate: {successful_paths}/{total_tests} ({success_rate:.1%})")
    
    if success_rate > 0:
        print("✅ ISSUE #3 STATUS: Pathfinding is working")
    else:
        print("❌ ISSUE #3 STATUS: Pathfinding is not working")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL ANALYSIS SUMMARY")
    print("=" * 80)
    
    issue1_ok = functional_edges > 0
    issue2_ok = walkable_edges_in_solid == 0 if walkable_edges > 0 else True
    issue3_ok = success_rate > 0
    
    print(f"Issue #1 (Functional edges): {'✅ RESOLVED' if issue1_ok else '❌ NOT RESOLVED'}")
    print(f"Issue #2 (Walkable edges in solid): {'✅ RESOLVED' if issue2_ok else '❌ NOT RESOLVED'}")
    print(f"Issue #3 (Pathfinding): {'✅ RESOLVED' if issue3_ok else '❌ NOT RESOLVED'}")
    
    issues_resolved = sum([issue1_ok, issue2_ok, issue3_ok])
    print(f"\nOverall status: {issues_resolved}/3 issues resolved")
    
    if issues_resolved == 3:
        print("🎉 ALL ISSUES RESOLVED!")
    else:
        print("⚠️ Some issues still need attention")
    
    return {
        'functional_edges': functional_edges,
        'walkable_edges_in_solid': walkable_edges_in_solid if walkable_edges > 0 else 0,
        'pathfinding_success_rate': success_rate,
        'edge_type_counts': edge_type_counts,
        'issues_resolved': issues_resolved
    }


if __name__ == '__main__':
    try:
        results = analyze_doortest_map()
        
        print(f"\n" + "=" * 40)
        print("DETAILED RESULTS")
        print("=" * 40)
        print(f"Functional edges: {results['functional_edges']}")
        print(f"Walkable edges in solid tiles: {results['walkable_edges_in_solid']}")
        print(f"Pathfinding success rate: {results['pathfinding_success_rate']:.1%}")
        print(f"Issues resolved: {results['issues_resolved']}/3")
        
        print(f"\nEdge type distribution:")
        for edge_type, count in results['edge_type_counts'].items():
            print(f"  {edge_type}: {count}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)