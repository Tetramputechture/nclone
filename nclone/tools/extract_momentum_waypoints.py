#!/usr/bin/env python3
"""Extract momentum waypoints from expert demonstrations.

This script analyzes expert demonstration replays to identify momentum-building
waypoints where the optimal strategy involves temporarily moving away from the
goal to build velocity for momentum-dependent maneuvers.

Usage:
    python extract_momentum_waypoints.py --replay-dir path/to/replays --output-dir momentum_waypoints_cache

The extracted waypoints are saved to cache files that the environment automatically
loads during training, enabling momentum-aware PBRS reward shaping.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add nclone to path if running as script
if __name__ == "__main__":
    nclone_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(nclone_root))

from nclone.analysis.momentum_waypoint_extractor import MomentumWaypointExtractor
from nclone.replay.replay_executor import ReplayExecutor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_waypoints_from_replay_file(
    replay_path: str, extractor: MomentumWaypointExtractor
) -> tuple:
    """Extract waypoints from a single replay file.

    Args:
        replay_path: Path to .npp replay file
        extractor: MomentumWaypointExtractor instance

    Returns:
        Tuple of (level_id, waypoints) or (None, []) if extraction failed
    """
    try:
        # Execute replay to get trajectory
        executor = ReplayExecutor(
            replay_path=replay_path,
            enable_rendering=False,
            enable_logging=False,
        )

        # Run replay to completion
        positions = []
        velocities = []
        actions = []
        goal_position = None
        switch_position = None
        switch_activation_frame = None
        level_id = None

        frame_idx = 0
        while not executor.is_complete():
            # Get current state
            ninja_pos = executor.nplay_headless.ninja_position()
            ninja_vel = executor.nplay_headless.ninja_velocity()
            switch_activated = executor.nplay_headless.exit_switch_activated()

            positions.append(ninja_pos)
            velocities.append(ninja_vel)

            # Track switch activation
            if switch_activated and switch_activation_frame is None:
                switch_activation_frame = frame_idx

            # Get action and step
            action = executor.get_next_action()
            if action is None:
                break

            actions.append(action)
            executor.step()
            frame_idx += 1

        # Get goal positions from final state
        if executor.nplay_headless:
            goal_position = executor.nplay_headless.exit_door_position()
            switch_position = executor.nplay_headless.exit_switch_position()
            level_id = Path(replay_path).stem  # Use filename as level_id

        if not positions or goal_position is None:
            logger.warning(f"Failed to extract trajectory from {replay_path}")
            return None, []

        # Extract waypoints
        waypoints = extractor.extract_from_episode(
            positions=positions,
            velocities=velocities,
            actions=actions,
            goal_position=goal_position,
            switch_position=switch_position,
            switch_activation_frame=switch_activation_frame,
        )

        logger.info(
            f"Extracted {len(waypoints)} waypoints from {replay_path} ({len(positions)} frames)"
        )
        return level_id, waypoints

    except Exception as e:
        logger.error(f"Failed to process replay {replay_path}: {e}")
        return None, []


def main():
    parser = argparse.ArgumentParser(
        description="Extract momentum waypoints from expert demonstrations"
    )
    parser.add_argument(
        "--replay-dir",
        type=str,
        required=True,
        help="Directory containing .npp replay files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="momentum_waypoints_cache",
        help="Output directory for waypoint cache files",
    )
    parser.add_argument(
        "--min-speed",
        type=float,
        default=1.5,
        help="Minimum speed for momentum detection (pixels/frame)",
    )
    parser.add_argument(
        "--speed-increase",
        type=float,
        default=0.8,
        help="Required speed increase for momentum-building detection",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize extractor
    extractor = MomentumWaypointExtractor(
        min_speed=args.min_speed,
        speed_increase_threshold=args.speed_increase,
    )

    # Find all replay files
    replay_dir = Path(args.replay_dir)
    if not replay_dir.exists():
        logger.error(f"Replay directory not found: {replay_dir}")
        return 1

    replay_files = list(replay_dir.glob("*.npp"))
    if not replay_files:
        logger.error(f"No .npp replay files found in {replay_dir}")
        return 1

    logger.info(f"Found {len(replay_files)} replay files to process")

    # Extract waypoints from all replays
    waypoints_by_level = {}
    for replay_file in replay_files:
        logger.info(f"Processing {replay_file.name}...")
        level_id, waypoints = extract_waypoints_from_replay_file(
            str(replay_file), extractor
        )

        if level_id and waypoints:
            if level_id not in waypoints_by_level:
                waypoints_by_level[level_id] = []
            waypoints_by_level[level_id].extend(waypoints)

    # Deduplicate waypoints per level
    for level_id in waypoints_by_level:
        waypoints_by_level[level_id] = extractor._deduplicate_waypoints(
            waypoints_by_level[level_id]
        )

    # Save to cache
    logger.info(f"Saving waypoints to {args.output_dir}...")
    extractor.save_waypoints_to_cache(waypoints_by_level, args.output_dir)

    # Print summary
    total_waypoints = sum(len(wps) for wps in waypoints_by_level.values())
    logger.info(
        f"\nExtraction complete:\n"
        f"  Levels processed: {len(waypoints_by_level)}\n"
        f"  Total waypoints: {total_waypoints}\n"
        f"  Cache directory: {args.output_dir}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
