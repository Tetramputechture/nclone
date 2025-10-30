#!/usr/bin/env python3
"""
Quick test script for path-aware reward shaping system.

Usage:
    python test_path_aware.py
"""

import sys
import os

# Add nclone to path
sys.path.insert(0, os.path.dirname(__file__))

def test_tile_connectivity_loader():
    """Test that the precomputed tile connectivity data can be loaded."""
    print("\n" + "=" * 60)
    print("TEST 1: Tile Connectivity Loader")
    print("=" * 60)
    
    try:
        from nclone.graph.reachability.tile_connectivity_loader import TileConnectivityLoader
        
        loader = TileConnectivityLoader()
        print(f"✅ Connectivity table loaded successfully")
        print(f"   - Shape: {loader.table_shape}")
        print(f"   - Size: {loader.table_size_kb:.2f} KB")
        print(f"   - Total combinations: {loader.table_shape[0] * loader.table_shape[1] * loader.table_shape[2]}")
        
        # Test a few lookups
        print("\n   Testing lookups:")
        for tile_a in [0, 5, 10]:
            for tile_b in [0, 5, 10]:
                for direction in ['N', 'E']:  # North and East
                    traversable = loader.is_traversable(tile_a, tile_b, direction)
                    print(f"     Tile {tile_a} -> {direction} -> Tile {tile_b}: {traversable}")
                    if tile_a >= 3:  # Only test a few
                        break
                if tile_b >= 3:
                    break
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fast_graph_builder():
    """Test building adjacency graph from tile data."""
    print("\n" + "=" * 60)
    print("TEST 2: Fast Graph Builder")
    print("=" * 60)
    
    try:
        from nclone.graph.reachability.fast_graph_builder import FastGraphBuilder
        import numpy as np
        
        # Create a simple test map (42x23 grid of empty tiles)
        map_data = np.zeros((42, 23), dtype=np.int32)
        # Add some walls (tile 1)
        map_data[10:15, 10] = 1  # vertical wall
        level_data = {"tiles": map_data}
        
        builder = FastGraphBuilder()
        
        import time
        start = time.perf_counter()
        graph_data = builder.build_graph(level_data)
        build_time = (time.perf_counter() - start) * 1000
        
        print(f"✅ Graph built successfully")
        print(f"   - Build time: {build_time:.3f} ms")
        print(f"   - Nodes: {len(graph_data['adjacency'])}")
        print(f"   - Edges: {sum(len(neighbors) for neighbors in graph_data['adjacency'].values())}")
        
        # Test cache hit
        start = time.perf_counter()
        graph_data2 = builder.build_graph(level_data)
        cache_time = (time.perf_counter() - start) * 1000
        
        print(f"\n   Cache test:")
        print(f"   - Cache hit time: {cache_time:.3f} ms")
        print(f"   - Speedup: {build_time / cache_time:.1f}x")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_path_distance_calculator():
    """Test pathfinding on a simple graph."""
    print("\n" + "=" * 60)
    print("TEST 3: Path Distance Calculator")
    print("=" * 60)
    
    try:
        from nclone.graph.reachability.path_distance_calculator import PathDistanceCalculator
        
        # Create simple test graph (straight corridor)
        adjacency = {}
        for i in range(10):
            x = i * 24
            y = 100
            adjacency[(x, y)] = []
            if i > 0:
                adjacency[(x, y)].append((((i-1)*24, y), 24.0))  # left neighbor: ((x,y), cost)
            if i < 9:
                adjacency[(x, y)].append((((i+1)*24, y), 24.0))  # right neighbor: ((x,y), cost)
        
        calculator = PathDistanceCalculator()
        
        # Test straight path
        start = (0, 100)
        goal = (9*24, 100)
        
        import time
        start_time = time.perf_counter()
        distance = calculator.calculate_distance(start, goal, adjacency)
        calc_time = (time.perf_counter() - start_time) * 1000
        
        expected = 9 * 24  # 9 moves of 24 pixels each
        
        print(f"✅ Pathfinding test passed")
        print(f"   - Start: {start}")
        print(f"   - Goal: {goal}")
        print(f"   - Distance: {distance:.1f} px")
        print(f"   - Expected: {expected:.1f} px")
        print(f"   - Calculation time: {calc_time:.3f} ms")
        print(f"   - Match: {'✓' if abs(distance - expected) < 1 else '✗'}")
        
        return abs(distance - expected) < 1
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_entity_mask():
    """Test entity masking with doors and mines."""
    print("\n" + "=" * 60)
    print("TEST 4: Entity Mask")
    print("=" * 60)
    
    try:
        from nclone.graph.reachability.entity_mask import EntityMask
        from nclone.constants.entity_types import EntityType
        
        # Create mock entities as dicts
        entities = [
            {'type': EntityType.LOCKED_DOOR, 'x': 100, 'y': 100, 'id': 1},
            {'type': EntityType.TOGGLE_MINE, 'x': 200, 'y': 200, 'mine_type': 1},  # Type 1 - starts safe
            {'type': EntityType.TOGGLE_MINE, 'x': 300, 'y': 300, 'mine_type': 21},  # Type 21 - starts deadly
        ]
        
        level_data = {"entities": entities}
        entity_mask = EntityMask(level_data)
        
        print(f"✅ Entity mask created")
        print(f"   - Doors: {len(entity_mask.doors)}")
        print(f"   - Mines: {len(entity_mask.mines)}")
        
        # Test blocking
        blocked_positions = entity_mask.get_blocked_positions()
        blocked_edges = entity_mask.get_blocked_edges()
        
        print(f"\n   Blocking status:")
        print(f"   - Blocked positions: {len(blocked_positions)}")
        print(f"   - Blocked edges: {len(blocked_edges)}")
        
        # Test door opening
        print(f"\n   Testing door state changes:")
        blocked_before = len(entity_mask.get_blocked_positions())
        entity_mask.update_switch_state('1', True)  # Activate switch 1
        blocked_after = len(entity_mask.get_blocked_positions())
        print(f"   - Blocked positions before: {blocked_before}")
        print(f"   - Blocked positions after: {blocked_after}")
        print(f"   - Door opened: {blocked_after < blocked_before}")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PATH-AWARE REWARD SHAPING SYSTEM TESTS")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Tile Connectivity Loader", test_tile_connectivity_loader()))
    results.append(("Fast Graph Builder", test_fast_graph_builder()))
    results.append(("Path Distance Calculator", test_path_distance_calculator()))
    results.append(("Entity Mask", test_entity_mask()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
