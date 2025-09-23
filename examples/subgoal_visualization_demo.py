#!/usr/bin/env python3
"""
Subgoal Visualization System Demo

This script demonstrates how to use the subgoal visualization system
for analyzing subgoal planning and reachability in N++ levels.

Usage examples:
    # Basic visualization
    python examples/subgoal_visualization_demo.py

    # Export visualization
    python examples/subgoal_visualization_demo.py --export

    # Different modes
    python examples/subgoal_visualization_demo.py --mode reachability
"""

import argparse
import sys
import os

# Add the parent directory to the path so we can import nclone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description="Subgoal Visualization Demo")
    parser.add_argument(
        "--mode", 
        choices=["basic", "detailed", "reachability"], 
        default="detailed",
        help="Visualization mode"
    )
    parser.add_argument(
        "--export", 
        action="store_true",
        help="Export visualization to image file"
    )
    parser.add_argument(
        "--custom-map", 
        type=str,
        help="Path to custom map file"
    )
    
    args = parser.parse_args()
    
    print("🎮 Subgoal Visualization System Demo")
    print("=" * 50)
    
    # Build command for test_environment
    cmd_parts = ["python", "-m", "nclone.test_environment"]
    
    # Add subgoal visualization
    cmd_parts.extend(["--visualize-subgoals", "--subgoal-mode", args.mode])
    
    # Add reachability for better visualization
    if args.mode == "reachability":
        cmd_parts.append("--visualize-reachability")
    
    # Add custom map if specified
    if args.custom_map:
        cmd_parts.extend(["--custom-map-path", args.custom_map])
    
    # Add export if requested
    if args.export:
        export_filename = f"subgoal_demo_{args.mode}.png"
        cmd_parts.extend(["--export-subgoals", export_filename])
        print(f"📸 Will export visualization to: {export_filename}")
    
    print(f"🔧 Mode: {args.mode}")
    print(f"🗺️  Custom map: {args.custom_map or 'Default level'}")
    
    if not args.export:
        print("\n🎯 Interactive Controls:")
        print("  S - Toggle subgoal visualization")
        print("  M - Cycle through modes")
        print("  P - Update subgoal plan")
        print("  O - Export screenshot")
        print("  ESC - Exit")
    
    print("\n🚀 Starting visualization...")
    print("Command:", " ".join(cmd_parts))
    print("-" * 50)
    
    # Execute the command
    import subprocess
    try:
        result = subprocess.run(cmd_parts, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if result.returncode == 0:
            print("✅ Demo completed successfully!")
        else:
            print(f"❌ Demo exited with code {result.returncode}")
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Error running demo: {e}")

if __name__ == "__main__":
    main()