#!/usr/bin/env python3
"""Test replay playback to verify it reaches termination."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nclone.replay.gameplay_recorder import CompactReplay
from nclone.replay.replay_executor import decode_input_to_controls
from nclone.nplay_headless import NPlayHeadless


def test_replay_termination(replay_path: str):
    """Test if replay reaches termination state."""

    # Load replay
    with open(replay_path, "rb") as f:
        replay_data = f.read()

    replay = CompactReplay.from_binary(replay_data, episode_id=Path(replay_path).stem)

    print(f"Testing replay: {replay.episode_id}")
    print(f"Input sequence length: {len(replay.input_sequence)}")
    print(f"Map data size: {len(replay.map_data)} bytes")

    # Create environment
    nplay = NPlayHeadless(
        render_mode="rgb_array",
        enable_animation=False,
        enable_logging=False,
        enable_debug_overlay=False,
        seed=42,
    )

    # Load map
    nplay.load_map_from_map_data(list(replay.map_data))

    # Check initial state
    initial_pos = nplay.ninja_position()
    print(f"\nInitial ninja position: {initial_pos}")
    print(f"Initial has_won state: {nplay.ninja_has_won()}")
    print(f"Initial ninja dead: {nplay.ninja_has_died()}")

    # Execute replay
    terminated = False
    truncated = False

    for frame_idx, input_byte in enumerate(replay.input_sequence):
        horizontal, jump = decode_input_to_controls(input_byte)
        nplay.tick(horizontal, jump)

        # Check termination conditions
        has_won = nplay.ninja_has_won()
        has_died = nplay.ninja_has_died()

        if has_won:
            print(
                f"\n✅ Ninja won at frame {frame_idx + 1}/{len(replay.input_sequence)}"
            )
            terminated = True
            break

        if has_died:
            print(
                f"\n❌ Ninja died at frame {frame_idx + 1}/{len(replay.input_sequence)}"
            )
            terminated = True
            break

    # Final state
    final_pos = nplay.ninja_position()
    print(f"\nFinal ninja position: {final_pos}")
    print(f"Final has_won state: {nplay.ninja_has_won()}")
    print(f"Final ninja dead: {nplay.ninja_has_died()}")
    print(f"Frames processed: {len(replay.input_sequence)}")

    if terminated:
        print(f"\n✅ Replay reached termination state")
    else:
        print(f"\n⚠️  Replay did NOT reach termination state")
        print(f"   This indicates the recording was incomplete or stopped early")

    return terminated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test replay playback termination")
    parser.add_argument("replay_file", help="Path to .replay file")
    args = parser.parse_args()

    terminated = test_replay_termination(args.replay_file)
    sys.exit(0 if terminated else 1)
