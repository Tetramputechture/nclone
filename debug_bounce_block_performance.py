#!/usr/bin/env python3
"""
Debug script to investigate the performance issue with bounce-block-reachable map.
"""

import cProfile
import pstats
import time
import sys
from pathlib import Path

# Add the nclone package to the path
sys.path.insert(0, str(Path(__file__).parent))

from nclone.graph.reachability.reachability_analyzer import ReachabilityAnalyzer
from nclone.graph.trajectory_calculator import TrajectoryCalculator
from nclone.gym_environment.npp_environment import NppEnvironment


def debug_bounce_block_performance():
    """Debug the performance issue with bounce-block-reachable map."""
    print("Debugging bounce-block-reachable performance issue...")
    
    # Initialize analyzer with debug enabled
    trajectory_calculator = TrajectoryCalculator()
    analyzer = ReachabilityAnalyzer(trajectory_calculator, debug=True)
    
    try:
        # Load the problematic map
        map_path = 'nclone/test_maps/bounce-block-reachable'
        env = NppEnvironment(render_mode="rgb_array", custom_map_path=map_path)
        env.reset()
        
        level_data = env.level_data
        ninja_pos = env.nplay_headless.ninja_position()
        
        print(f"Level size: {level_data.tiles.shape}")
        print(f"Entities: {len(level_data.entities)}")
        print(f"Ninja position: {ninja_pos}")
        
        # Print entity details
        print("\nEntity details:")
        for i, entity in enumerate(level_data.entities):
            print(f"  Entity {i}: {entity}")
        
        # Profile with timeout
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.time()
        
        # Set a timeout to prevent infinite execution
        timeout = 30  # 30 seconds max
        
        try:
            result = analyzer.analyze_reachability(level_data, ninja_pos, {})
            end_time = time.time()
            
            analysis_time = (end_time - start_time) * 1000
            print(f"\nAnalysis completed in {analysis_time:.2f}ms")
            print(f"Reachable positions: {len(result.reachable_positions)}")
            print(f"Subgoals found: {len(result.subgoals)}")
            
        except KeyboardInterrupt:
            print("\nAnalysis interrupted by user")
            end_time = time.time()
            analysis_time = (end_time - start_time) * 1000
            print(f"Partial analysis time: {analysis_time:.2f}ms")
        
        profiler.disable()
        
        # Show detailed profiling stats
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        print(f"\nTop 20 time consumers:")
        stats.print_stats(20)
        
        # Show functions that took the most time
        print(f"\nFunctions sorted by total time:")
        stats.sort_stats('tottime')
        stats.print_stats(10)
        
        env.close()
        
    except Exception as e:
        print(f"Error during debugging: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_bounce_block_performance()