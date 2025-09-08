#!/usr/bin/env python3
"""
Debug script to check ninja position methods.
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold

def main():
    """Debug ninja position."""
    print("=" * 60)
    print("🔍 DEBUGGING NINJA POSITION METHODS")
    print("=" * 60)
    
    # Create environment
    env = BasicLevelNoGold(render_mode="rgb_array")
    env.reset()
    
    # Check different ways to get ninja position
    print(f"🥷 env.nplay_headless.ninja_position(): {env.nplay_headless.ninja_position()}")
    
    # Check if there are other methods
    ninja = env.nplay_headless.ninja
    print(f"🥷 ninja.position: {ninja.position}")
    print(f"🥷 ninja.x, ninja.y: ({ninja.x}, {ninja.y})")

if __name__ == "__main__":
    main()