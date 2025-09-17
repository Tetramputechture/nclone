#!/usr/bin/env python3
"""
Generate OpenCV flood fill visualizations for all test maps.
Saves visualization images to nclone/reachability_viz/ directory.
"""

import sys
import os
import json
import shutil
from pathlib import Path

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def generate_map_visualizations():
    """Generate reachability visualizations for all test maps."""
    print("OpenCV Reachability Visualization Generator")
    print("=" * 50)
    
    # Setup paths
    test_maps_dir = Path("nclone/test_maps")
    viz_dir = Path("nclone/reachability_viz")
    
    # Clear and recreate visualization directory
    if viz_dir.exists():
        shutil.rmtree(viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Load maps.json to get map metadata
    maps_json_path = test_maps_dir / "maps.json"
    if maps_json_path.exists():
        with open(maps_json_path, 'r') as f:
            maps_metadata = json.load(f)
    else:
        maps_metadata = {}
    
    # Get all map files (excluding maps.json)
    map_files = [f for f in test_maps_dir.iterdir() 
                 if f.is_file() and f.name != "maps.json"]
    map_files.sort()  # Process in alphabetical order
    
    print(f"Found {len(map_files)} test maps to process")
    print()
    
    successful_maps = []
    failed_maps = []
    
    for map_file in map_files:
        map_name = map_file.name
        print(f"Processing map: {map_name}")
        print("-" * 30)
        
        try:
            # Generate visualization for this map
            metadata = {}
            if isinstance(maps_metadata, dict):
                metadata = maps_metadata.get(map_name, {})
            result = generate_single_map_visualization(map_name, viz_dir, metadata)
            
            if result:
                successful_maps.append(map_name)
                print(f"✓ Successfully generated visualization for {map_name}")
            else:
                failed_maps.append(map_name)
                print(f"✗ Failed to generate visualization for {map_name}")
                
        except Exception as e:
            failed_maps.append(map_name)
            print(f"✗ Error processing {map_name}: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Summary
    print("VISUALIZATION GENERATION SUMMARY")
    print("=" * 50)
    print(f"Total maps processed: {len(map_files)}")
    print(f"Successful: {len(successful_maps)}")
    print(f"Failed: {len(failed_maps)}")
    
    if successful_maps:
        print(f"\n✓ Successful maps:")
        for map_name in successful_maps:
            print(f"  - {map_name}")
    
    if failed_maps:
        print(f"\n✗ Failed maps:")
        for map_name in failed_maps:
            print(f"  - {map_name}")
    
    print(f"\nVisualization images saved to: {viz_dir.absolute()}")
    
    return len(successful_maps), len(failed_maps)

def generate_single_map_visualization(map_name: str, viz_dir: Path, metadata: dict):
    """Generate visualization for a single map."""
    from nclone.gym_environment.npp_environment import NppEnvironment
    from nclone.graph.reachability.opencv_flood_fill import OpenCVFloodFill
    import shutil
    
    try:
        # Initialize environment with the specific map (like test_environment.py)
        map_path = f"nclone/test_maps/{map_name}"
        env = NppEnvironment(
            render_mode="rgb_array",
            enable_frame_stack=False,
            enable_debug_overlay=False,  # Disable for cleaner visualization
            eval_mode=False,
            seed=42,
            custom_map_path=map_path,
        )
        env.reset()
        
        # Get ninja position and level data (like test_environment.py)
        ninja_pos = (
            env.nplay_headless.ninja_position()
            if hasattr(env, "nplay_headless")
            else (100, 100)
        )
        level_data = getattr(env, "level_data", None)
        
        # Get switch states from environment
        switch_states = {}
        if hasattr(env.nplay_headless, 'get_switch_states'):
            switch_states = env.nplay_headless.get_switch_states()
        
        # Get entities from environment
        entities = []
        if hasattr(env.nplay_headless, 'get_entities'):
            entities = env.nplay_headless.get_entities()
        elif hasattr(env, 'entities'):
            entities = env.entities
        
        print(f"  Map loaded: {level_data.shape}")
        print(f"  Ninja position: {ninja_pos}")
        print(f"  Entities: {len(entities) if entities else 0}")
        
        # Create OpenCV analyzer with debug enabled and optimized scale
        opencv_analyzer = OpenCVFloodFill(debug=True, render_scale=0.25)
        
        # Perform reachability analysis
        result = opencv_analyzer.quick_check(ninja_pos, level_data, switch_states, entities)
        
        print(f"  Analysis complete:")
        print(f"    Positions found: {len(result.reachable_positions)}")
        print(f"    Computation time: {result.computation_time_ms:.2f}ms")
        print(f"    Confidence: {result.confidence}")
        
        # Move debug images from /tmp to our visualization directory
        debug_source = Path("/tmp/opencv_flood_fill_debug")
        if debug_source.exists():
            map_viz_dir = viz_dir / map_name
            map_viz_dir.mkdir(exist_ok=True)
            
            # Copy all debug images
            for debug_file in debug_source.glob("*.png"):
                dest_file = map_viz_dir / debug_file.name
                shutil.copy2(debug_file, dest_file)
                print(f"    Saved: {dest_file.name}")
            
            # Create a summary file with analysis results
            summary_file = map_viz_dir / "analysis_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"OpenCV Flood Fill Analysis - {map_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Map: {map_name}\n")
                f.write(f"Level dimensions: {level_data.shape}\n")
                f.write(f"Ninja position: {ninja_pos}\n")
                f.write(f"Entities: {len(entities) if entities else 0}\n\n")
                f.write("Analysis Results:\n")
                f.write(f"  Reachable positions: {len(result.reachable_positions)}\n")
                f.write(f"  Computation time: {result.computation_time_ms:.2f}ms\n")
                f.write(f"  Confidence: {result.confidence}\n")
                f.write(f"  Method: {result.method}\n")
                f.write(f"  Tier used: {result.tier_used}\n")
                f.write(f"  Render scale: 0.25x\n\n")
                
                if metadata:
                    f.write("Map Metadata:\n")
                    for key, value in metadata.items():
                        f.write(f"  {key}: {value}\n")
            
            print(f"    Saved: analysis_summary.txt")
            
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    """Main function."""
    try:
        successful, failed = generate_map_visualizations()
        
        if failed == 0:
            print("\n🎉 All visualizations generated successfully!")
            sys.exit(0)
        else:
            print(f"\n⚠️  {failed} maps failed to generate visualizations")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nVisualization generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()