#!/usr/bin/env python3
"""
Extract waypoints from replay files using demo checkpoint seeder.

This script processes replay files to extract trajectory data as waypoints
using the same method as the demo checkpoint seeder in test_environment.py.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from nclone.replay.demo_checkpoint_seeder import DemoCheckpointSeeder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_waypoints_from_replay(replay_path: Path) -> Dict[str, Any]:
    """
    Extract waypoints from a single replay file using the demo seeder's method.
    
    Args:
        replay_path: Path to replay file
        
    Returns:
        Dict containing level_name and extracted waypoints
    """
    try:
        # Extract level name from filename
        # Format: YYYYMMDD_HHMMSS_<level_name>.replay
        filename = replay_path.stem  # Remove .replay extension
        parts = filename.split('_', 2)  # Split on first 2 underscores
        if len(parts) >= 3:
            level_name = parts[2]
        else:
            level_name = filename
            
        logger.info(f"Processing: {replay_path.name}")
        logger.debug(f"Extracted level name '{level_name}' from '{filename}'")
        
        # Use DemoCheckpointSeeder to extract positions
        # The seeder is just for access to the extraction method
        seeder = DemoCheckpointSeeder(
            replay_dir=".",  # Not actually used for _extract_positions_from_demo_simple
            max_demos_per_level=1
        )
        
        # Extract trajectory using the simple method (expects Path object)
        checkpoints = seeder._extract_positions_from_demo_simple(replay_path)
        
        if not checkpoints:
            logger.warning(f"No checkpoints extracted from {replay_path.name}")
            return {
                "level_name": level_name,
                "waypoints": []
            }
            
        logger.info(f"  Extracted {len(checkpoints)} checkpoints")
        
        # Sample checkpoints to create waypoints
        # Take every Nth checkpoint to avoid having too many waypoints
        sample_interval = max(1, len(checkpoints) // 10)  # Get ~10 waypoints max
        sampled_waypoints = []
        
        for i in range(0, len(checkpoints), sample_interval):
            cp = checkpoints[i]
            pos = cp.get('position')
            if pos:
                sampled_waypoints.append({
                    "position": list(pos),  # Convert tuple to list for JSON
                    "tolerance": 24.0,  # Standard tolerance
                    "description": f"Checkpoint {i}"
                })
        
        logger.info(f"  Created {len(sampled_waypoints)} waypoints from checkpoints")
        
        return {
            "level_name": level_name,
            "waypoints": sampled_waypoints
        }
        
    except Exception as e:
        logger.error(f"Failed to process {replay_path.name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "level_name": replay_path.stem,
            "waypoints": []
        }


def save_waypoints_to_cache(waypoints_by_level: Dict[str, List], output_path: Path):
    """
    Save extracted waypoints to a cache file.
    
    Args:
        waypoints_by_level: Dict mapping level names to waypoint lists
        output_path: Path to save the cache file
    """
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(waypoints_by_level, f, indent=2)
    
    logger.info(f"Saved waypoint cache to {output_path}")
    logger.info(f"Total levels with waypoints: {len(waypoints_by_level)}")
    
    # Print summary
    total_waypoints = sum(len(wps) for wps in waypoints_by_level.values())
    logger.info(f"Total waypoints extracted: {total_waypoints}")


def process_replay_directory(replay_dir: Path, output_path: Path):
    """
    Process all replay files in a directory to extract waypoints.
    
    Args:
        replay_dir: Directory containing replay files
        output_path: Path to save the waypoint cache
    """
    # Find all replay files
    replay_files = list(replay_dir.glob("*.replay"))
    
    if not replay_files:
        logger.error(f"No replay files found in {replay_dir}")
        return
    
    logger.info(f"Found {len(replay_files)} replay files in {replay_dir}")
    
    # Process each replay
    waypoints_by_level = {}
    for replay_path in replay_files:
        result = extract_waypoints_from_replay(replay_path)
        
        level_name = result["level_name"]
        waypoints = result["waypoints"]
        
        if waypoints:
            # Merge with existing waypoints for this level
            if level_name in waypoints_by_level:
                logger.debug(f"Merging waypoints for level {level_name}")
                # For now, just replace with new waypoints
                # Could implement more sophisticated merging later
                waypoints_by_level[level_name] = waypoints
            else:
                waypoints_by_level[level_name] = waypoints
    
    # Save the cache
    if waypoints_by_level:
        save_waypoints_to_cache(waypoints_by_level, output_path)
    else:
        logger.warning("No waypoints extracted from any replay files")


def main():
    parser = argparse.ArgumentParser(
        description="Extract waypoints from replay files using demo checkpoint seeder"
    )
    parser.add_argument(
        "--replay-dir",
        type=Path,
        default=Path("datasets/path-replays"),
        help="Directory containing replay files (default: datasets/path-replays)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/waypoint_cache.json"),
        help="Output path for waypoint cache (default: datasets/waypoint_cache.json)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        # Suppress the detailed frame-by-frame logging from demo seeder
        logging.getLogger("nclone.replay.demo_checkpoint_seeder").setLevel(logging.WARNING)
    
    # Process replays
    if args.replay_dir.is_dir():
        process_replay_directory(args.replay_dir, args.output)
    else:
        logger.error(f"Replay directory does not exist: {args.replay_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()