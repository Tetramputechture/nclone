#!/usr/bin/env python3
"""
Performance profiling script for the current reachability system.
This helps identify bottlenecks before implementing optimizations.
Uses real level data from test maps for accurate profiling.
"""

import cProfile
import pstats
import time
import sys
import os
from pathlib import Path
import numpy as np

# Add the nclone package to the path
sys.path.insert(0, str(Path(__file__).parent))

from nclone.graph.reachability.reachability_analyzer import ReachabilityAnalyzer
from nclone.graph.trajectory_calculator import TrajectoryCalculator
from nclone.graph.level_data import LevelData
from nclone.constants.entity_types import EntityType
from nclone.gym_environment.npp_environment import NppEnvironment


def load_real_level_data(map_path):
    """Load real level data from test maps using NppEnvironment."""
    try:
        # Create environment and load the specific map
        env = NppEnvironment(render_mode="rgb_array", custom_map_path=map_path)
        env.reset()
        
        # Extract level data
        level_data = env.level_data
        ninja_pos = env.nplay_headless.ninja_position()
        
        # Clean up
        env.close()
        
        return level_data, ninja_pos
        
    except Exception as e:
        print(f"Error loading {map_path}: {e}")
        return None, None


def profile_reachability_analysis():
    """Profile the current reachability analysis system using real test maps."""
    print("Profiling current reachability analysis system with real test maps...")
    
    # Initialize analyzer
    trajectory_calculator = TrajectoryCalculator()
    analyzer = ReachabilityAnalyzer(trajectory_calculator, debug=False)
    
    # Test with real test maps of different complexities
    test_maps = [
        ('nclone/test_maps/simple-walk', 'simple'),
        ('nclone/test_maps/complex-path-switch-required', 'complex'),
        ('nclone/test_maps/map-unreachable-areas', 'frontier'),
        ('nclone/test_maps/bounce-block-reachable', 'entity'),
        ('nclone/test_maps/wall-jump-required', 'physics')
    ]
    
    results = {}
    
    for map_path, map_type in test_maps:
        print(f"\nTesting {map_type} map: {map_path}")
        
        try:
            # Load real level data
            level_data, ninja_pos = load_real_level_data(map_path)
            
            if level_data is None:
                print(f"  Skipping {map_path} - failed to load")
                continue
            
            print(f"  Level size: {level_data.tiles.shape}")
            print(f"  Entities: {len(level_data.entities)}")
            print(f"  Ninja position: {ninja_pos}")
            
            # Profile single analysis
            profiler = cProfile.Profile()
            profiler.enable()
            
            start_time = time.time()
            result = analyzer.analyze_reachability(level_data, ninja_pos, {})
            end_time = time.time()
            
            profiler.disable()
            
            # Store results
            analysis_time = (end_time - start_time) * 1000  # Convert to ms
            results[map_type] = {
                'map_path': map_path,
                'time_ms': analysis_time,
                'reachable_positions': len(result.reachable_positions),
                'subgoals': len(result.subgoals),
                'level_size': level_data.tiles.shape,
                'entity_count': len(level_data.entities)
            }
            
            print(f"  Analysis time: {analysis_time:.2f}ms")
            print(f"  Reachable positions: {len(result.reachable_positions)}")
            print(f"  Subgoals found: {len(result.subgoals)}")
            
            # Performance assessment
            if analysis_time > 100:
                print(f"  ⚠️  Exceeds 100ms limit for large levels")
            elif analysis_time > 10:
                print(f"  ⚠️  Exceeds 10ms limit for typical levels")
            else:
                print(f"  ✓ Within performance limits")
            
            # Save profiling stats for complex map
            if map_type == 'complex':
                stats = pstats.Stats(profiler)
                stats.sort_stats('cumulative')
                print(f"\nTop 10 time consumers for {map_type} map:")
                stats.print_stats(10)
            
        except Exception as e:
            print(f"  Error testing {map_path}: {e}")
            import traceback
            traceback.print_exc()
            results[map_type] = {'error': str(e)}
    
    return results


def test_repeated_queries():
    """Test performance with repeated queries (simulating RL training)."""
    print("\n" + "="*60)
    print("Testing repeated queries (RL training simulation)")
    print("="*60)
    
    trajectory_calculator = TrajectoryCalculator()
    analyzer = ReachabilityAnalyzer(trajectory_calculator, debug=False)
    
    try:
        # Load a medium complexity map
        level_data, base_ninja_pos = load_real_level_data('nclone/test_maps/complex-path-switch-required')
        
        if level_data is None:
            print("Failed to load test map for repeated queries")
            return {'error': 'Failed to load test map'}
        
        # Generate test positions around the ninja starting position
        test_positions = [
            (base_ninja_pos[0] + i*10, base_ninja_pos[1] + j*10) 
            for i in range(-5, 6) 
            for j in range(-5, 6)
        ]
        
        print(f"Testing {len(test_positions)} repeated queries...")
        print(f"Base ninja position: {base_ninja_pos}")
        
        # Time repeated queries
        start_time = time.time()
        for pos in test_positions:
            result = analyzer.analyze_reachability(level_data, pos, {})
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = (total_time / len(test_positions)) * 1000  # Convert to ms
        
        print(f"Total time: {total_time:.3f}s")
        print(f"Average time per query: {avg_time:.2f}ms")
        print(f"Queries per second: {len(test_positions) / total_time:.1f}")
        
        # Check if we meet performance requirements
        if avg_time < 10:
            print("✓ PASS: Average query time meets <10ms requirement")
        else:
            print("✗ FAIL: Average query time exceeds 10ms requirement")
            
        return {
            'total_time': total_time,
            'avg_time_ms': avg_time,
            'queries_per_second': len(test_positions) / total_time,
            'meets_requirement': avg_time < 10
        }
        
    except Exception as e:
        print(f"Error in repeated queries test: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def main():
    """Main profiling function."""
    print("Reachability System Performance Profiling")
    print("="*50)
    
    # Profile basic analysis
    basic_results = profile_reachability_analysis()
    
    # Test repeated queries
    repeated_results = test_repeated_queries()
    
    # Summary
    print("\n" + "="*60)
    print("PROFILING SUMMARY")
    print("="*60)
    
    print("\nBasic Analysis Results:")
    for map_path, result in basic_results.items():
        if 'error' not in result:
            print(f"  {map_path}: {result['time_ms']:.2f}ms")
            if result['time_ms'] > 100:
                print(f"    ⚠️  Exceeds 100ms limit for large levels")
            elif result['time_ms'] > 10:
                print(f"    ⚠️  Exceeds 10ms limit for typical levels")
            else:
                print(f"    ✓ Within performance limits")
    
    if 'error' not in repeated_results:
        print(f"\nRepeated Queries: {repeated_results['avg_time_ms']:.2f}ms average")
        if repeated_results['meets_requirement']:
            print("  ✓ Meets RL training requirements")
        else:
            print("  ✗ Does not meet RL training requirements")
    
    print("\nNext steps:")
    print("1. Implement caching system to improve repeated query performance")
    print("2. Add incremental updates to reduce computation")
    print("3. Optimize physics calculations for common cases")
    print("4. Add parallel processing for BFS exploration")


if __name__ == "__main__":
    main()