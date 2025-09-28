#!/usr/bin/env python3
"""
N++ Replay Video Generator

This module generates videos from N++ replay data using the nclone simulation environment.
It supports both JSONL replay files and binary replay files, rendering them as MP4 videos
using the actual game environment for accurate visual representation.

Usage:
    python -m nclone.replay.video_generator --input replay.jsonl --output video.mp4
    python -m nclone.replay.video_generator --input binary_replay_dir/ --output video.mp4 --binary-replay
    python -m nclone.replay.video_generator --help
"""

import argparse
import json
import logging
import numpy as np
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from ..gym_environment.npp_environment import NppEnvironment
from .binary_replay_parser import BinaryReplayParser
from .npp_attract_decoder import NppAttractDecoder

logger = logging.getLogger(__name__)


@dataclass
class ReplayFrame:
    """Structured representation of a single replay frame."""

    timestamp: float
    level_id: str
    frame_number: int
    player_position: Tuple[float, float]
    player_velocity: Tuple[float, float]
    player_state: Dict[str, Any]
    player_inputs: Dict[str, bool]
    entities: List[Dict[str, Any]]
    level_bounds: Dict[str, int]
    meta: Dict[str, Any]


class ActionMapper:
    """Maps raw player inputs to discrete action space."""

    ACTION_MAPPING = {
        (False, False, False): 0,  # no_action
        (True, False, False): 1,  # left
        (False, True, False): 2,  # right
        (False, False, True): 3,  # jump
        (True, False, True): 4,  # left_jump
        (False, True, True): 5,  # right_jump
    }

    ACTION_NAMES = {
        0: "no_action",
        1: "left",
        2: "right",
        3: "jump",
        4: "left_jump",
        5: "right_jump",
    }

    @classmethod
    def map_inputs_to_action(cls, inputs: Dict[str, bool]) -> int:
        """Convert player inputs to discrete action index."""
        left = inputs.get("left", False)
        right = inputs.get("right", False)
        jump = inputs.get("jump", False)

        # Handle invalid combinations (left + right)
        if left and right:
            left = right = False

        key = (left, right, jump)
        return cls.ACTION_MAPPING.get(key, 0)  # Default to no_action

    @classmethod
    def get_action_name(cls, action: int) -> str:
        """Get human-readable action name."""
        return cls.ACTION_NAMES.get(action, "unknown")


class ReplayValidator:
    """Validates replay data for quality and consistency."""

    @staticmethod
    def validate_frame(frame_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a single replay frame.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check required top-level fields
        required_fields = [
            "timestamp",
            "level_id",
            "frame_number",
            "player_state",
            "player_inputs",
            "entities",
            "level_bounds",
            "meta",
        ]

        for field in required_fields:
            if field not in frame_data:
                errors.append(f"Missing required field: {field}")

        if errors:
            return False, errors

        # Validate timestamp
        if (
            not isinstance(frame_data["timestamp"], (int, float))
            or frame_data["timestamp"] <= 0
        ):
            errors.append("Invalid timestamp")

        # Validate frame number
        if (
            not isinstance(frame_data["frame_number"], int)
            or frame_data["frame_number"] < 0
        ):
            errors.append("Invalid frame_number")

        # Validate player state
        player_state = frame_data["player_state"]
        if "position" not in player_state or "velocity" not in player_state:
            errors.append("Missing player position or velocity")
        else:
            pos = player_state["position"]
            vel = player_state["velocity"]

            if not all(isinstance(pos.get(k), (int, float)) for k in ["x", "y"]):
                errors.append("Invalid player position coordinates")

            if not all(isinstance(vel.get(k), (int, float)) for k in ["x", "y"]):
                errors.append("Invalid player velocity coordinates")

        # Validate player inputs
        player_inputs = frame_data["player_inputs"]
        input_keys = ["left", "right", "jump", "restart"]
        for key in input_keys:
            if key not in player_inputs or not isinstance(player_inputs[key], bool):
                errors.append(f"Invalid or missing player input: {key}")

        # Validate entities
        if not isinstance(frame_data["entities"], list):
            errors.append("Entities must be a list")
        else:
            for i, entity in enumerate(frame_data["entities"]):
                if not isinstance(entity, dict):
                    errors.append(f"Entity {i} must be a dictionary")
                    continue

                if "type" not in entity or "position" not in entity:
                    errors.append(f"Entity {i} missing type or position")

        # Validate level bounds
        level_bounds = frame_data["level_bounds"]
        if not all(
            isinstance(level_bounds.get(k), int) and level_bounds.get(k) > 0
            for k in ["width", "height"]
        ):
            errors.append("Invalid level bounds")

        return len(errors) == 0, errors

    @staticmethod
    def validate_trajectory(frames: List[ReplayFrame]) -> Tuple[bool, List[str]]:
        """
        Validate a complete trajectory for consistency.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not frames:
            return False, ["Empty trajectory"]

        # Check temporal consistency
        timestamps = [f.timestamp for f in frames]
        if not all(
            timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1)
        ):
            errors.append("Timestamps not monotonically increasing")

        # Check frame number consistency
        frame_numbers = [f.frame_number for f in frames]
        if frame_numbers != list(range(len(frame_numbers))):
            errors.append("Frame numbers not sequential")

        # Check level consistency
        level_ids = set(f.level_id for f in frames)
        if len(level_ids) > 1:
            errors.append("Multiple level IDs in single trajectory")

        return len(errors) == 0, errors


class VideoGenerator:
    """Generates videos from replay data using NppEnvironment rendering."""

    def __init__(self, fps: int = 60, resolution: Tuple[int, int] = (1056, 600)):
        """
        Initialize video generator.

        Args:
            fps: Frames per second for output video
            resolution: Video resolution (width, height)
        """
        self.fps = fps
        self.resolution = resolution
        self.env = None

    def _get_env(self, custom_map_path: Optional[str] = None) -> NppEnvironment:
        """Get or create environment instance for rendering."""
        if self.env is None:
            self.env = NppEnvironment(
                render_mode="rgb_array",
                enable_animation=True,
                enable_debug_overlay=False,
                custom_map_path=custom_map_path,
            )
        return self.env

    def generate_video_from_jsonl(
        self,
        replay_file: Path,
        output_video: Path,
        custom_map_path: Optional[str] = None,
    ) -> bool:
        """
        Generate video from JSONL replay file.

        Args:
            replay_file: Path to JSONL replay file
            output_video: Output video file path
            custom_map_path: Optional custom map to load

        Returns:
            True if video generation succeeded
        """
        try:
            # Load replay frames
            frames = self._load_jsonl_replay(replay_file)
            if not frames:
                logger.error("No valid frames found in replay file")
                return False

            # Check for map_data_path in first frame
            if custom_map_path is None and frames:
                first_frame_data = self._load_first_frame_raw(replay_file)
                if first_frame_data and "map_data_path" in first_frame_data:
                    # Resolve relative path to map data file
                    map_data_path = (
                        replay_file.parent / first_frame_data["map_data_path"]
                    )
                    if map_data_path.exists():
                        custom_map_path = str(map_data_path)
                        logger.info(f"Using map data from: {custom_map_path}")

            # Initialize environment
            env = self._get_env(custom_map_path)
            env.reset()

            # Create temporary directory for frames
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                logger.info(f"Generating {len(frames)} frames...")

                # Generate frame images
                for i, frame in enumerate(frames):
                    try:
                        # Map frame inputs to action
                        action = ActionMapper.map_inputs_to_action(frame.player_inputs)

                        # Step environment
                        obs, reward, terminated, truncated, info = env.step(action)

                        # Render frame
                        frame_image = env.render()
                        if frame_image is not None:
                            # Save frame as PNG
                            frame_file = temp_path / f"frame_{i:06d}.png"
                            self._save_frame_as_png(frame_image, frame_file)

                        if terminated or truncated:
                            logger.info(f"Episode ended at frame {i}")
                            break

                    except Exception as e:
                        logger.error(f"Error processing frame {i}: {e}")
                        continue

                # Generate video using ffmpeg
                return self._create_video_from_frames(temp_path, output_video)

        except Exception as e:
            logger.error(f"Failed to generate video: {e}")
            return False

    def generate_video_from_binary_replay(
        self,
        binary_replay_file: Path,
        output_video: Path,
        map_file: Optional[str] = None,
    ) -> bool:
        """
        Generate video from binary replay file.

        Args:
            binary_replay_file: Path to binary replay file
            output_video: Output video file path
            map_file: Optional path to N++ map definition file

        Returns:
            True if video generation succeeded
        """
        try:
            # Parse binary replay to get frames
            parser = BinaryReplayParser(map_file_path=map_file)

            # Create temporary JSONL file
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_jsonl = Path(temp_dir) / "temp_replay.jsonl"

                # Parse binary replay to JSONL
                success = parser.parse_single_replay_file_to_jsonl(
                    Path(binary_replay_file), temp_jsonl.parent
                )

                if not success:
                    logger.error("Failed to parse binary replay file")
                    return False

                # Find the generated JSONL file
                jsonl_files = list(temp_jsonl.parent.glob("*.jsonl"))
                if not jsonl_files:
                    logger.error("No JSONL file generated from binary replay")
                    return False

                # Use the first JSONL file found
                jsonl_file = jsonl_files[0]

                # Persist the generated JSONL to a non-temporary file in the same directory as the output video
                output_jsonl = output_video.with_suffix(".jsonl")
                jsonl_files = list(temp_jsonl.parent.glob("*.jsonl"))
                if not jsonl_files:
                    logger.error("No JSONL file generated from binary replay")
                    return False
                # Copy the first JSONL file to the output location
                import shutil

                shutil.copy(jsonl_files[0], output_jsonl)

                # Find corresponding map data file and persist if present
                map_files = list(temp_jsonl.parent.glob("*_map.dat"))
                custom_map_path = None
                if map_files:
                    output_map = output_video.with_name(output_video.stem + "_map.dat")
                    shutil.copy(map_files[0], output_map)
                    custom_map_path = str(output_map)
                # Generate video from the JSONL file
                return self.generate_video_from_jsonl(
                    jsonl_file, output_video, custom_map_path
                )

        except Exception as e:
            logger.error(f"Failed to generate video from binary replay: {e}")
            return False

    def generate_video_from_npp_attract(
        self,
        npp_attract_file: Path,
        output_video: Path,
    ) -> bool:
        """
        Generate video directly from npp_attract file using perfect decoder.

        Args:
            npp_attract_file: Path to npp_attract file
            output_video: Output video file path

        Returns:
            True if video generation succeeded
        """
        try:
            logger.info(f"Generating video from npp_attract file: {npp_attract_file}")
            
            # Use perfect decoder to extract map data
            npp_decoder = NppAttractDecoder()
            decoded_data = npp_decoder.decode_npp_attract_file(str(npp_attract_file))
            
            # Create temporary map file
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_map_file = Path(temp_dir) / "perfect_map.dat"
                
                # Create nclone-compatible map
                nclone_map_bytes = npp_decoder.create_nclone_map(str(npp_attract_file))
                with open(temp_map_file, 'wb') as f:
                    f.write(nclone_map_bytes)
                
                logger.info(f"Created perfect map: {len(decoded_data['tiles'])} tiles, {len(decoded_data['entities'])} entities, spawn {decoded_data['ninja_spawn']}")
                
                # Use binary replay parser to generate JSONL with perfect map
                return self.generate_video_from_binary_replay(
                    npp_attract_file, output_video, str(temp_map_file)
                )

        except Exception as e:
            logger.error(f"Failed to generate video from npp_attract file: {e}")
            return False

    def _load_first_frame_raw(self, replay_file: Path) -> Optional[Dict[str, Any]]:
        """Load the first frame from JSONL file as raw data."""
        try:
            with open(replay_file, "r") as f:
                first_line = f.readline().strip()
                if first_line:
                    return json.loads(first_line)
        except Exception as e:
            logger.error(f"Failed to load first frame: {e}")
        return None

    def _load_jsonl_replay(self, replay_file: Path) -> List[ReplayFrame]:
        """Load replay frames from JSONL file."""
        frames = []
        validator = ReplayValidator()

        try:
            with open(replay_file, "r") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        frame_data = json.loads(line.strip())

                        # Validate frame
                        is_valid, errors = validator.validate_frame(frame_data)
                        if not is_valid:
                            logger.warning(
                                f"Frame {line_num} validation failed: {errors}"
                            )
                            continue

                        # Convert to ReplayFrame
                        frame = ReplayFrame(
                            timestamp=frame_data["timestamp"],
                            level_id=frame_data["level_id"],
                            frame_number=frame_data["frame_number"],
                            player_position=(
                                frame_data["player_state"]["position"]["x"],
                                frame_data["player_state"]["position"]["y"],
                            ),
                            player_velocity=(
                                frame_data["player_state"]["velocity"]["x"],
                                frame_data["player_state"]["velocity"]["y"],
                            ),
                            player_state=frame_data["player_state"],
                            player_inputs=frame_data["player_inputs"],
                            entities=frame_data["entities"],
                            level_bounds=frame_data["level_bounds"],
                            meta=frame_data["meta"],
                        )
                        frames.append(frame)

                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error at line {line_num}: {e}")
                    except Exception as e:
                        logger.error(f"Error processing line {line_num}: {e}")

        except Exception as e:
            logger.error(f"Failed to load replay file {replay_file}: {e}")

        return frames

    def _save_frame_as_png(self, frame_array: np.ndarray, output_path: Path):
        """Save frame array as PNG image."""
        try:
            from PIL import Image

            # Handle different frame formats
            if len(frame_array.shape) == 3:
                if frame_array.shape[2] == 1:
                    # Single channel - squeeze to 2D
                    frame_2d = np.squeeze(frame_array, axis=2)
                    image = Image.fromarray(frame_2d.astype(np.uint8), mode="L")
                elif frame_array.shape[2] == 3:
                    # RGB format
                    image = Image.fromarray(frame_array.astype(np.uint8), mode="RGB")
                elif frame_array.shape[2] == 4:
                    # RGBA format
                    image = Image.fromarray(frame_array.astype(np.uint8), mode="RGBA")
                else:
                    logger.warning(
                        f"Unsupported frame format with {frame_array.shape[2]} channels"
                    )
                    return
            elif len(frame_array.shape) == 2:
                # Already 2D grayscale
                image = Image.fromarray(frame_array.astype(np.uint8), mode="L")
            else:
                logger.warning(f"Unsupported frame shape {frame_array.shape}")
                return

            image.save(output_path)

        except Exception as e:
            logger.error(f"Failed to save frame as PNG: {e}")

    def _create_video_from_frames(self, frames_dir: Path, output_video: Path) -> bool:
        """Create video from frame images using ffmpeg."""
        try:
            # Check if ffmpeg is available
            result = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, text=True
            )
            if result.returncode != 0:
                logger.error(
                    "ffmpeg not found. Please install ffmpeg to generate videos."
                )
                return False

            # Create output directory if it doesn't exist
            output_video.parent.mkdir(parents=True, exist_ok=True)

            # Build ffmpeg command
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file
                "-framerate",
                str(self.fps),
                "-i",
                str(frames_dir / "frame_%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",  # High quality
                str(output_video),
            ]

            logger.info(f"Running ffmpeg: {' '.join(cmd)}")

            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Video generated successfully: {output_video}")
                return True
            else:
                logger.error(f"ffmpeg failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Failed to create video: {e}")
            return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate videos from N++ replay data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate video from JSONL replay
  python -m nclone.replay.video_generator --input replay.jsonl --output video.mp4

  # Generate video from binary replay file
  python -m nclone.replay.video_generator --input npp_attract/1 --output video.mp4 --binary-replay
  
  # Generate video from npp_attract file (perfect decoder)
  python -m nclone.replay.video_generator --input npp_attract/0 --output video.mp4 --npp-attract

  # Generate video with custom framerate
  python -m nclone.replay.video_generator --input replay.jsonl --output video.mp4 --fps 30

  # Generate video with custom map data
  python -m nclone.replay.video_generator --input replay.jsonl --output video.mp4 --custom-map map.dat
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL replay file or binary replay file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output video file path (MP4)",
    )
    parser.add_argument(
        "--binary-replay",
        action="store_true",
        help="Input is binary replay file (not JSONL)",
    )
    parser.add_argument(
        "--npp-attract",
        action="store_true",
        help="Input is npp_attract file (uses perfect decoder)",
    )
    parser.add_argument(
        "--fps", type=int, default=60, help="Video framerate (default: 60)"
    )
    parser.add_argument(
        "--custom-map", type=str, help="Custom map file to use for video generation"
    )
    parser.add_argument(
        "--map-file",
        type=str,
        help="N++ map definition file for real map data (format: $name#mapdata#)",
    )
    parser.add_argument(
        "--official-levels",
        type=str,
        help="Directory containing official N++ level .txt files",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Validate arguments
    if not args.input.exists():
        parser.error(f"Input file does not exist: {args.input}")

    # Initialize video generator
    video_generator = VideoGenerator(fps=args.fps)

    # Determine map source (prioritize official_levels over single map_file)
    map_source = args.official_levels if args.official_levels else args.map_file

    # Generate video
    if args.npp_attract:
        success = video_generator.generate_video_from_npp_attract(
            args.input, args.output
        )
    elif args.binary_replay:
        success = video_generator.generate_video_from_binary_replay(
            args.input, args.output, map_source
        )
    else:
        # Auto-detect npp_attract files if they're in the npp_attract directory
        if "npp_attract" in str(args.input) and args.input.is_file():
            logger.info("Auto-detected npp_attract file, using perfect decoder")
            success = video_generator.generate_video_from_npp_attract(
                args.input, args.output
            )
        else:
            success = video_generator.generate_video_from_jsonl(
                args.input, args.output, args.custom_map
            )

    if success:
        logger.info(f"Video generation completed: {args.output}")
        return 0
    else:
        logger.error("Video generation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
