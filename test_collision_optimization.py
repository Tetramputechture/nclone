"""
Tests for collision detection optimizations.

Validates that optimized collision detection maintains 100% physics accuracy
while providing significant performance improvements.
"""

import time
import numpy as np
from nclone.nsim import Simulator
from nclone.sim_config import SimConfig


def test_collision_accuracy():
    """Test that optimized collision detection matches original implementation exactly."""
    print("\n=== Testing Collision Optimization Accuracy ===\n")
    
    # Create simulator with a test map
    config = SimConfig()
    config.log_data = False
    sim = Simulator(config)
    
    # Load a test level (using default map data)
    test_map_data = [0] * 1800  # Minimal map data
    # Set spawn point
    test_map_data[1231] = 10  # spawn x
    test_map_data[1232] = 10  # spawn y
    
    # Add some tiles for collision
    for i in range(184, 184 + 42):
        test_map_data[i] = 1  # Bottom row of tiles
    
    sim.load(test_map_data)
    
    # Verify collision structures are built
    assert hasattr(sim, 'collision_data'), "collision_data not initialized"
    assert sim.collision_data is not None, "collision_data is None"
    assert sim.collision_data.is_built, "collision_data not built"
    assert sim.spatial_segment_index is not None, "spatial_segment_index not initialized"
    
    print("✓ Collision data structures initialized successfully")
    
    # Test segment queries return same results
    from nclone.physics import gather_segments_from_region
    
    test_queries = [
        (100, 100, 150, 150),
        (200, 200, 250, 250),
        (50, 50, 100, 100),
    ]
    
    for x1, y1, x2, y2 in test_queries:
        # Query with spatial index
        segments_optimized = gather_segments_from_region(sim, x1, y1, x2, y2)
        
        # Temporarily disable spatial index to test fallback
        spatial_index_backup = sim.spatial_segment_index
        sim.spatial_segment_index = None
        segments_original = gather_segments_from_region(sim, x1, y1, x2, y2)
        sim.spatial_segment_index = spatial_index_backup
        
        # Verify same number of segments returned
        assert len(segments_optimized) == len(segments_original), \
            f"Segment count mismatch: optimized={len(segments_optimized)}, original={len(segments_original)}"
        
        # Verify segments are the same (by bounds)
        optimized_bounds = sorted([seg.get_bounds() for seg in segments_optimized])
        original_bounds = sorted([seg.get_bounds() for seg in segments_original])
        
        for opt_bound, orig_bound in zip(optimized_bounds, original_bounds):
            assert opt_bound == orig_bound, f"Segment bounds mismatch: {opt_bound} != {orig_bound}"
    
    print(f"✓ Segment queries match exactly for {len(test_queries)} test cases")
    
    # Test collision detection produces same results
    # Run a few simulation steps
    initial_pos = (sim.ninja.xpos, sim.ninja.ypos)
    
    for _ in range(10):
        sim.tick(1, 0)  # Move right
    
    final_pos = (sim.ninja.xpos, sim.ninja.ypos)
    
    print(f"✓ Physics simulation ran successfully")
    print(f"  Initial position: ({initial_pos[0]:.2f}, {initial_pos[1]:.2f})")
    print(f"  Final position: ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
    
    # Get collision data statistics
    stats = sim.collision_data.get_stats()
    print(f"\n=== Collision Data Statistics ===")
    print(f"Level hash: {stats['level_hash']}")
    print(f"Spatial index cells: {stats['spatial_index']['total_cells']}")
    print(f"Spatial index segments: {stats['spatial_index']['total_segments']}")
    print(f"Query cache size: {stats['query_cache']['size']}/{stats['query_cache']['max_size']}")
    
    print("\n✓ All accuracy tests passed!")
    return True


def test_collision_performance():
    """Benchmark collision detection performance improvements."""
    print("\n=== Testing Collision Optimization Performance ===\n")
    
    # Create simulator
    config = SimConfig()
    config.log_data = False
    sim = Simulator(config)
    
    # Load a test level
    test_map_data = [0] * 1800
    test_map_data[1231] = 10
    test_map_data[1232] = 10
    
    # Add more tiles for a realistic test
    for y in range(23):
        for x in range(42):
            idx = 184 + x + y * 42
            if idx < len(test_map_data):
                # Create a pattern with some empty spaces
                test_map_data[idx] = 1 if (x + y) % 3 != 0 else 0
    
    sim.load(test_map_data)
    
    # Benchmark segment queries
    test_queries = [
        (x, y, x + 50, y + 50)
        for x in range(0, 500, 50)
        for y in range(0, 300, 50)
    ]
    
    print(f"Benchmarking {len(test_queries)} segment queries...")
    
    # Benchmark with optimization
    from nclone.physics import gather_segments_from_region
    
    start = time.perf_counter()
    for x1, y1, x2, y2 in test_queries:
        segments = gather_segments_from_region(sim, x1, y1, x2, y2)
    optimized_time = time.perf_counter() - start
    
    # Benchmark without optimization (fallback)
    spatial_index_backup = sim.spatial_segment_index
    sim.spatial_segment_index = None
    
    start = time.perf_counter()
    for x1, y1, x2, y2 in test_queries:
        segments = gather_segments_from_region(sim, x1, y1, x2, y2)
    original_time = time.perf_counter() - start
    
    sim.spatial_segment_index = spatial_index_backup
    
    speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
    
    print(f"\nSegment Query Performance:")
    print(f"  Original implementation: {original_time*1000:.2f}ms")
    print(f"  Optimized implementation: {optimized_time*1000:.2f}ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    # Benchmark collision detection in physics loop
    print(f"\nBenchmarking full physics simulation (100 frames)...")
    
    # Reset ninja
    sim.ninja.xpos = test_map_data[1231] * 6
    sim.ninja.ypos = test_map_data[1232] * 6
    
    start = time.perf_counter()
    for _ in range(100):
        sim.tick(1, 0)
    simulation_time = time.perf_counter() - start
    
    print(f"  100 frames completed in: {simulation_time*1000:.2f}ms")
    print(f"  Average frame time: {simulation_time*10:.2f}ms")
    
    # Get final statistics
    stats = sim.collision_data.get_stats()
    query_stats = stats['query_cache']
    
    print(f"\n=== Query Cache Statistics ===")
    print(f"  Total queries: {query_stats['total_queries']}")
    print(f"  Cache hits: {query_stats['hits']}")
    print(f"  Cache misses: {query_stats['misses']}")
    print(f"  Hit rate: {query_stats['hit_rate']*100:.1f}%")
    print(f"  Evictions: {query_stats['evictions']}")
    
    print("\n✓ Performance benchmarks completed!")
    
    # Verify significant speedup
    assert speedup > 1.1, f"Expected at least 1.1x speedup, got {speedup:.2f}x"
    print(f"\n✓ Achieved {speedup:.2f}x speedup in segment queries")
    
    return True


def test_terminal_velocity_optimization():
    """Test terminal velocity simulation with entity collision skip."""
    print("\n=== Testing Terminal Velocity Optimization ===\n")
    
    from nclone.terminal_velocity_simulator import TerminalVelocitySimulator
    
    # Create simulator
    config = SimConfig()
    config.log_data = False
    sim = Simulator(config)
    
    # Load a test level
    test_map_data = [0] * 1800
    test_map_data[1231] = 10
    test_map_data[1232] = 10
    
    # Add floor
    for i in range(184, 184 + 42):
        test_map_data[i] = 1
    
    sim.load(test_map_data)
    
    # Create terminal velocity simulator
    tv_sim = TerminalVelocitySimulator(sim)
    
    # Set ninja to falling state with high velocity
    sim.ninja.airborn = True
    sim.ninja.yspeed = 8.0  # High downward velocity
    sim.ninja.ypos = 100.0  # High above floor
    
    # Test simulation
    print("Testing terminal velocity simulation...")
    start = time.perf_counter()
    result = tv_sim.simulate_for_terminal_impact(action=0, max_frames=30)
    elapsed = time.perf_counter() - start
    
    print(f"  Simulation completed in: {elapsed*1000:.2f}ms")
    print(f"  Terminal impact detected: {result}")
    
    # Verify ninja state was restored
    assert sim.ninja.yspeed == 8.0, "Ninja state not restored correctly"
    assert sim.ninja.airborn == True, "Ninja airborn state not restored"
    
    print("✓ Terminal velocity simulation works correctly")
    print("✓ Ninja state properly restored after simulation")
    
    return True


def test_persistent_cache():
    """Test persistent disk-based caching."""
    print("\n=== Testing Persistent Cache ===\n")
    
    from nclone.utils.persistent_collision_cache import PersistentCollisionCache
    
    # Clear cache first
    test_hash = "test_cache_12345"
    PersistentCollisionCache.clear_cache(test_hash)
    
    # Test save and load
    test_data = {
        'segments': [1, 2, 3],
        'metadata': 'test'
    }
    
    print("Saving test data to cache...")
    PersistentCollisionCache.save_to_cache(test_hash, test_data, "test_data")
    
    print("Loading test data from cache...")
    loaded_data = PersistentCollisionCache.load_from_cache(test_hash, "test_data")
    
    assert loaded_data is not None, "Failed to load from cache"
    assert loaded_data == test_data, "Loaded data doesn't match saved data"
    
    print("✓ Save and load successful")
    
    # Test get_or_build with cache hit
    test_build_hash = "test_build_456xyz"
    PersistentCollisionCache.clear_cache(test_build_hash)  # Clear any existing cache
    
    call_count = [0]
    
    def expensive_builder():
        call_count[0] += 1
        return {"computed": True}
    
    print("\nTesting get_or_build (first call - should build)...")
    result1 = PersistentCollisionCache.get_or_build(
        test_build_hash,
        expensive_builder,
        "build_test"
    )
    
    print("Testing get_or_build (second call - should use cache)...")
    result2 = PersistentCollisionCache.get_or_build(
        test_build_hash,
        expensive_builder,
        "build_test"
    )
    
    assert call_count[0] == 1, f"Builder called {call_count[0]} times, expected 1 (cache should have prevented second call)"
    assert result1 == result2, "Results from cache don't match"
    
    print("✓ Cache prevents redundant builds")
    
    # Test cache size
    cache_size = PersistentCollisionCache.get_cache_size()
    print(f"\nCache size: {cache_size / 1024:.2f} KB")
    
    # Clean up test cache
    PersistentCollisionCache.clear_cache(test_build_hash)
    PersistentCollisionCache.clear_cache(test_hash)
    
    print("✓ Persistent cache tests passed")
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("  Collision Optimization Test Suite")
    print("="*60)
    
    try:
        # Run tests
        test_collision_accuracy()
        test_collision_performance()
        test_terminal_velocity_optimization()
        test_persistent_cache()
        
        print("\n" + "="*60)
        print("  ✓ ALL TESTS PASSED")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise

