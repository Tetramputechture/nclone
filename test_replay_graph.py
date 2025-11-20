#!/usr/bin/env python3
"""
Quick test to diagnose graph building issues in replay execution.
Run this to see if the graph is being built correctly with filter_by_reachability=False.
"""

import logging
import sys
from pathlib import Path

# Setup logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Import replay executor
from nclone.replay.replay_executor import ReplayExecutor
from nclone.replay.types import CompactReplay

def test_replay_graph_building():
    """Test graph building with a sample replay."""
    print("=" * 60)
    print("Testing Replay Graph Building")
    print("=" * 60)
    
    # Find a sample replay file
    replay_dir = Path("~/datasets/train").expanduser()
    if not replay_dir.exists():
        print(f"Dataset directory not found: {replay_dir}")
        return
    
    replay_files = list(replay_dir.glob("*.replay"))
    if not replay_files:
        print(f"No replay files found in {replay_dir}")
        return
    
    # Load first replay
    replay_file = replay_files[0]
    print(f"\nLoading replay: {replay_file.name}")
    
    with open(replay_file, "rb") as f:
        replay_data = f.read()
    
    replay = CompactReplay.from_binary(replay_data)
    print(f"Replay loaded: episode_id={replay.episode_id}, success={replay.success}")
    print(f"Input sequence length: {len(replay.input_sequence)}")
    
    # Create executor
    print("\nCreating ReplayExecutor...")
    executor = ReplayExecutor()
    
    # Execute replay (this should trigger graph building)
    print("\nExecuting replay...")
    print("Watch for [SURFACE_AREA_COMPUTE] and [FLOOD_FILL] debug messages\n")
    
    try:
        observations = executor.execute_replay(replay.map_data, replay.input_sequence)
        print(f"\n✓ Success! Generated {len(observations)} observations")
    except Exception as e:
        print(f"\n✗ Failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        executor.close()
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_replay_graph_building()


