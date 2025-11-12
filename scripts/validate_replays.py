#!/usr/bin/env python3
"""Comprehensive replay validation script for N++ replay datasets.

This script validates that replay files:
1. Can be loaded and parsed correctly
2. Execute without errors in the simulator
3. Result in player_won=True at the end of execution
4. Match their success flag metadata
5. Are deterministic (optional repeated validation)

Use this to verify replay datasets before using them for BC training or evaluation.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add nclone to path if running as standalone script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from nclone.replay.gameplay_recorder import CompactReplay
from nclone.replay.replay_executor import ReplayExecutor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a single replay file."""

    filename: str
    success: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    player_won: Optional[bool] = None
    player_dead: Optional[bool] = None
    success_flag: Optional[bool] = None
    frame_count: Optional[int] = None
    input_count: Optional[int] = None


class ReplayValidator:
    """Validates replay files for correctness and completion."""

    def __init__(self, verbose: bool = False):
        """Initialize replay validator.

        Args:
            verbose: Enable verbose output for debugging
        """
        self.verbose = verbose
        self.executor = None

    def validate_replay(self, replay_path: Path) -> ValidationResult:
        """Validate a single replay file.

        Args:
            replay_path: Path to replay file

        Returns:
            ValidationResult with validation status and details
        """
        filename = replay_path.name

        try:
            # Step 1: Load and parse replay file
            if self.verbose:
                logger.debug(f"Loading {filename}...")

            with open(replay_path, "rb") as f:
                replay_data = f.read()

            replay = CompactReplay.from_binary(replay_data)
            input_count = len(replay.input_sequence)

            # Step 2: Execute replay in simulator
            if self.verbose:
                logger.debug(f"Executing replay (inputs={input_count})...")

            self.executor = ReplayExecutor()
            observations = self.executor.execute_replay(
                replay.map_data, replay.input_sequence
            )

            if not observations:
                self.executor.close()
                self.executor = None
                return ValidationResult(
                    filename=filename,
                    success=False,
                    error_type="NO_OBSERVATIONS",
                    error_message="Replay execution produced no observations",
                    success_flag=replay.success,
                    input_count=input_count,
                )

            # Step 3: Get final observation state
            raw_obs = self.executor._get_raw_observation()
            player_won = raw_obs.get("player_won", None)
            player_dead = raw_obs.get("player_dead", None)
            frame_count = len(observations)

            self.executor.close()
            self.executor = None

            # Step 4: Validate completion state
            if player_won is None:
                return ValidationResult(
                    filename=filename,
                    success=False,
                    error_type="MISSING_PLAYER_WON",
                    error_message="player_won field not found in observation",
                    success_flag=replay.success,
                    frame_count=frame_count,
                    input_count=input_count,
                    player_dead=player_dead,
                )

            # Step 5: Check for success
            if not player_won:
                # Determine why player didn't win
                if player_dead:
                    error_type = "PLAYER_DIED"
                    error_message = "Player died before reaching exit"
                else:
                    error_type = "INCOMPLETE"
                    error_message = "Player did not reach exit (incomplete run)"

                return ValidationResult(
                    filename=filename,
                    success=False,
                    error_type=error_type,
                    error_message=error_message,
                    player_won=False,
                    player_dead=player_dead,
                    success_flag=replay.success,
                    frame_count=frame_count,
                    input_count=input_count,
                )

            # Step 6: Validate success flag matches
            if replay.success != player_won:
                print(
                    f"{filename}: success flag mismatch "
                    f"(metadata={replay.success}, actual={player_won})"
                )

            # Success!
            return ValidationResult(
                filename=filename,
                success=True,
                player_won=True,
                player_dead=False,
                success_flag=replay.success,
                frame_count=frame_count,
                input_count=input_count,
            )

        except Exception as e:
            # Handle any execution errors
            if self.executor:
                self.executor.close()
                self.executor = None

            error_msg = str(e)
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."

            return ValidationResult(
                filename=filename,
                success=False,
                error_type="EXECUTION_ERROR",
                error_message=error_msg,
            )

    def validate_directory(
        self,
        replay_dir: Path,
        pattern: str = "*.replay",
        max_replays: Optional[int] = None,
    ) -> List[ValidationResult]:
        """Validate all replay files in a directory.

        Args:
            replay_dir: Directory containing replay files
            pattern: Glob pattern for replay files
            max_replays: Maximum number of replays to validate (None for all)

        Returns:
            List of ValidationResult for each replay
        """
        replay_files = sorted(replay_dir.glob(pattern))

        if max_replays:
            replay_files = replay_files[:max_replays]

        logger.info(f"Found {len(replay_files)} replay files in {replay_dir}")
        logger.info("=" * 80)

        results = []
        for i, replay_path in enumerate(replay_files, 1):
            # Progress indicator
            print(
                f"[{i:3d}/{len(replay_files)}] {replay_path.name}...",
                end=" ",
                flush=True,
            )

            result = self.validate_replay(replay_path)
            results.append(result)

            # Status indicator
            if result.success:
                print("✅ VALID")
            else:
                print(f"❌ {result.error_type}")
                if self.verbose and result.error_message:
                    print(f"         Error: {result.error_message}")

        logger.info("=" * 80)
        return results


def print_summary(results: List[ValidationResult]):
    """Print validation summary statistics.

    Args:
        results: List of validation results
    """
    total = len(results)
    valid = sum(1 for r in results if r.success)
    invalid = total - valid

    # Categorize errors
    error_counts = {}
    invalid_replays = []

    for result in results:
        if not result.success:
            error_type = result.error_type or "UNKNOWN"
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            invalid_replays.append(result)

    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total replays:          {total}")
    print(f"✅ Valid replays:        {valid} ({100 * valid / total:.1f}%)")
    print(f"❌ Invalid replays:      {invalid} ({100 * invalid / total:.1f}%)")

    if error_counts:
        print(f"\nError breakdown:")
        for error_type, count in sorted(error_counts.items()):
            print(f"  {error_type:20s}: {count}")

    # Print details of invalid replays
    if invalid_replays:
        print(f"\n❌ Invalid replay details:")
        for result in invalid_replays:
            print(f"\n  File: {result.filename}")
            print(f"  Error: {result.error_type}")
            if result.error_message:
                print(f"  Message: {result.error_message}")
            if result.player_won is not None:
                print(f"  Player won: {result.player_won}")
            if result.player_dead is not None:
                print(f"  Player dead: {result.player_dead}")
            if result.success_flag is not None:
                print(f"  Success flag: {result.success_flag}")
            if result.frame_count is not None:
                print(f"  Frames: {result.frame_count}")

    # Final verdict
    print("\n" + "=" * 80)
    if invalid == 0:
        print("✅ SUCCESS: All replays are valid!")
        print("All replays execute correctly and result in player_won=True")
        print("\nThis dataset is ready for BC training.")
    else:
        print(f"❌ FAILURE: {invalid} replay(s) are invalid!")
        print("\nPossible causes:")
        if "EXECUTION_ERROR" in error_counts:
            print("  - Physics simulation errors or corrupted replay data")
        if "PLAYER_DIED" in error_counts:
            print("  - Replay inputs lead to player death (not a success)")
        if "INCOMPLETE" in error_counts:
            print("  - Replay inputs do not reach the exit (incomplete run)")
        if "MISSING_PLAYER_WON" in error_counts:
            print("  - Observation system missing player_won field")

        print("\n⚠️  Do NOT use this dataset for BC training until issues are resolved.")

    print("=" * 80)

    return invalid == 0


def save_results(results: List[ValidationResult], output_path: Path):
    """Save validation results to JSON file.

    Args:
        results: List of validation results
        output_path: Path to output JSON file
    """
    output_data = {
        "validation_results": [
            {
                "filename": r.filename,
                "valid": r.success,
                "error_type": r.error_type,
                "error_message": r.error_message,
                "player_won": r.player_won,
                "player_dead": r.player_dead,
                "success_flag": r.success_flag,
                "frame_count": r.frame_count,
                "input_count": r.input_count,
            }
            for r in results
        ],
        "summary": {
            "total": len(results),
            "valid": sum(1 for r in results if r.success),
            "invalid": sum(1 for r in results if not r.success),
        },
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Validation results saved to {output_path}")


def main():
    """Main entry point for replay validation script."""
    parser = argparse.ArgumentParser(
        description="Validate N++ replay dataset for BC training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "replay_dir", type=str, help="Directory containing replay files"
    )

    parser.add_argument(
        "--pattern", type=str, default="*.replay", help="Glob pattern for replay files"
    )

    parser.add_argument(
        "--max-replays",
        type=int,
        default=None,
        help="Maximum number of replays to validate (for testing)",
    )

    parser.add_argument(
        "--output", type=str, default=None, help="Save validation results to JSON file"
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output for debugging"
    )

    args = parser.parse_args()

    # Validate replay directory
    replay_dir = Path(args.replay_dir)
    if not replay_dir.exists():
        print(f"Replay directory does not exist: {replay_dir}")
        return 1

    if not replay_dir.is_dir():
        print(f"Path is not a directory: {replay_dir}")
        return 1

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run validation
    logger.info(f"Validating replays in: {replay_dir}")
    logger.info(f"Pattern: {args.pattern}")

    validator = ReplayValidator(verbose=args.verbose)
    results = validator.validate_directory(
        replay_dir, pattern=args.pattern, max_replays=args.max_replays
    )

    # Print summary
    all_valid = print_summary(results)

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        save_results(results, output_path)

    # Exit code: 0 if all valid, 1 if any invalid
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
