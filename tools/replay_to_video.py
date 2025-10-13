#!/usr/bin/env python3
"""
Compact Replay to Video Converter

Converts compact .replay files (recorded by GameplayRecorder) to MP4 videos.
Uses deterministic replay execution to regenerate the gameplay and render as video.

Usage:
    python tools/replay_to_video.py input.replay output.mp4
    python tools/replay_to_video.py input.replay output.mp4 --fps 30
"""

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nclone.replay.gameplay_recorder import CompactReplay
from nclone.replay.replay_executor import decode_input_to_controls
from nclone.nplay_headless import NPlayHeadless
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompactReplayVideoGenerator:
    """Generates videos from compact replay files."""

    def __init__(self, fps: int = 60):
        """
        Initialize video generator.

        Args:
            fps: Frames per second for output video
        """
        self.fps = fps

    def generate_video(
        self,
        replay_path: Path,
        output_video: Path,
    ) -> bool:
        """
        Generate video from compact replay file.

        Args:
            replay_path: Path to .replay file
            output_video: Output video file path

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load replay file
            logger.info(f"Loading replay: {replay_path}")
            with open(replay_path, "rb") as f:
                replay_data = f.read()

            replay = CompactReplay.from_binary(replay_data, episode_id=replay_path.stem)

            logger.info(f"Replay loaded:")
            logger.info(f"  - Episode ID: {replay.episode_id}")
            logger.info(f"  - Frames: {len(replay.input_sequence)}")
            logger.info(f"  - Map data: {len(replay.map_data)} bytes")

            # Create temporary directory for frames
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                logger.info(f"Rendering frames to temporary directory...")

                # Render all frames
                frame_count = self._render_frames(
                    replay.map_data, replay.input_sequence, temp_path
                )

                if frame_count == 0:
                    logger.error("No frames were rendered")
                    return False

                logger.info(f"Rendered {frame_count} frames")

                # Generate video using ffmpeg
                logger.info("Encoding video with ffmpeg...")
                return self._create_video_from_frames(temp_path, output_video)

        except Exception as e:
            logger.error(f"Failed to generate video: {e}", exc_info=True)
            return False

    def _render_frames(
        self,
        map_data: bytes,
        input_sequence: list,
        output_dir: Path,
    ) -> int:
        """
        Render all frames from replay.

        Args:
            map_data: Raw map data
            input_sequence: Input sequence
            output_dir: Directory to save frame images

        Returns:
            Number of frames rendered
        """
        # Create headless environment for simulation
        nplay = NPlayHeadless(
            render_mode="rgb_array",
            enable_animation=True,
            enable_logging=False,
            enable_debug_overlay=False,
            seed=42,  # Fixed seed for determinism
        )

        # Load map
        nplay.load_map_from_map_data(list(map_data))

        # Render each frame
        frame_count = 0
        for frame_idx, input_byte in enumerate(input_sequence):
            # Decode input to controls
            horizontal, jump = decode_input_to_controls(input_byte)

            # Execute simulation step
            nplay.tick(horizontal, jump)

            # Render frame
            frame = nplay.render()

            if frame is not None:
                # Save frame as PNG
                frame_path = output_dir / f"frame_{frame_idx:06d}.png"

                # Convert numpy array to PIL Image and save
                # Handle different frame formats
                if len(frame.shape) == 3 and frame.shape[2] == 1:
                    # Single channel grayscale (H, W, 1) - squeeze to 2D
                    frame_2d = np.squeeze(frame, axis=2)
                    img = Image.fromarray(frame_2d.astype(np.uint8))
                elif len(frame.shape) == 3 and frame.shape[2] == 3:
                    # RGB format (H, W, 3)
                    img = Image.fromarray(frame.astype(np.uint8))
                elif len(frame.shape) == 2:
                    # Already 2D grayscale (H, W)
                    img = Image.fromarray(frame.astype(np.uint8))
                else:
                    logger.warning(f"Unexpected frame shape: {frame.shape}")
                    continue

                img.save(frame_path)

                frame_count += 1

                # Progress indicator
                if frame_idx % 60 == 0 or frame_idx == len(input_sequence) - 1:
                    progress = (frame_idx + 1) / len(input_sequence) * 100
                    logger.info(
                        f"  Progress: {progress:.1f}% ({frame_idx + 1}/{len(input_sequence)} frames)"
                    )

        return frame_count

    def _create_video_from_frames(
        self,
        frames_dir: Path,
        output_video: Path,
    ) -> bool:
        """
        Create video from frame images using ffmpeg.

        Args:
            frames_dir: Directory containing frame images
            output_video: Output video file path

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if ffmpeg is available
            result = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, text=True
            )
            if result.returncode != 0:
                logger.error(
                    "ffmpeg not found. Please install ffmpeg to generate videos."
                )
                logger.error("On Ubuntu/Debian: sudo apt-get install ffmpeg")
                logger.error("On macOS: brew install ffmpeg")
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
                "libx264",  # H.264 codec
                "-preset",
                "medium",  # Encoding speed/quality tradeoff
                "-crf",
                "18",  # Quality (lower = better, 18 is visually lossless)
                "-pix_fmt",
                "yuv420p",  # Pixel format for compatibility
                str(output_video),
            ]

            logger.info(f"Running ffmpeg...")

            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Get file size
                file_size = output_video.stat().st_size
                file_size_mb = file_size / (1024 * 1024)

                logger.info(f"✅ Video generated successfully!")
                logger.info(f"   Output: {output_video}")
                logger.info(f"   Size: {file_size_mb:.2f} MB")
                logger.info(f"   FPS: {self.fps}")
                return True
            else:
                logger.error(f"ffmpeg failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Failed to create video: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert compact replay files to MP4 videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert replay to video at 60fps (default)
  python tools/replay_to_video.py replay.replay output.mp4
  
  # Convert at 30fps
  python tools/replay_to_video.py replay.replay output.mp4 --fps 30
  
  # Convert with custom output path
  python tools/replay_to_video.py output/20251012_180456_simple_002.replay videos/gameplay.mp4
""",
    )

    parser.add_argument("replay_file", type=str, help="Input .replay file path")

    parser.add_argument("output_video", type=str, help="Output video file path (.mp4)")

    parser.add_argument(
        "--fps", type=int, default=60, help="Video framerate (default: 60)"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate inputs
    replay_path = Path(args.replay_file)
    if not replay_path.exists():
        logger.error(f"Replay file not found: {replay_path}")
        return 1

    if not replay_path.suffix == ".replay":
        logger.warning(f"File does not have .replay extension: {replay_path}")

    output_path = Path(args.output_video)
    if not output_path.suffix in [".mp4", ".MP4"]:
        logger.warning(f"Output file does not have .mp4 extension: {output_path}")

    # Generate video
    generator = CompactReplayVideoGenerator(fps=args.fps)
    success = generator.generate_video(replay_path, output_path)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
