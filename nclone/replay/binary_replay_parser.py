#!/usr/bin/env python3
"""
N++ Binary Replay Parser

This tool converts N++ binary replay files ("trace" mode) to JSONL format
compatible with the npp-rl training pipeline. It parses the original N++
replay format and simulates the level to extract frame-by-frame data.

The parser automatically detects and handles both compressed and uncompressed
input files, following the ntrace.py pattern for zlib compression support.

Usage:
    python -m nclone.replay.binary_replay_parser --input replays/ --output datasets/raw/
    python -m nclone.replay.binary_replay_parser --help
"""

import argparse
import hashlib
import logging
import struct
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from .file_handler import FileHandler
from .map_loader import MapLoader
from .npp_attract_decoder import NppAttractDecoder
from .simulation_manager import SimulationManager

logger = logging.getLogger(__name__)


class BinaryReplayParser:
    """
    Parser for N++ binary replay files to JSONL format.

    Supports automatic detection and handling of compressed (zlib) and
    uncompressed binary replay files, following the ntrace.py pattern.
    """

    def __init__(self, map_file_path: Optional[str] = None):
        """
        Initialize the parser.

        Args:
            map_file_path: Optional path to N++ map definition file for real map data
        """
        self.stats = {
            "files_processed": 0,
            "frames_generated": 0,
            "replays_processed": 0,
            "replays_failed": 0,
        }
        self.map_loader = MapLoader(Path(map_file_path) if map_file_path else None)
        self.npp_attract_decoder = NppAttractDecoder()
        self.file_handler = FileHandler()
        self.simulation_manager = SimulationManager()

    def validate_replay_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Delegate to file handler."""
        return self.file_handler.validate_replay_file(file_path)

    def detect_trace_mode(self, replay_dir: Path) -> bool:
        """Delegate to file handler."""
        return self.file_handler.detect_trace_mode(replay_dir)

    def detect_single_replay_file(self, file_path: Path) -> bool:
        """Delegate to file handler."""
        return self.file_handler.detect_single_replay_file(file_path)

    def load_inputs_and_map(
        self, replay_dir: Path
    ) -> Tuple[List[List[int]], List[int]]:
        """Delegate to file handler."""
        return self.file_handler.load_inputs_and_map(replay_dir)

    def parse_single_replay_file(
        self, replay_file: Path
    ) -> Tuple[List[int], List[int], int, str]:
        """
        Parse a single N++ replay file (npp_attract format).

        Args:
            replay_file: Path to the single replay file

        Returns:
            Tuple of (inputs, map_data, level_id, level_name)
        """
        with open(replay_file, "rb") as f:
            data = f.read()

        logger.debug(f"Parsing single replay file: {replay_file} ({len(data)} bytes)")

        # Parse header information
        level_id = struct.unpack("<I", data[0:4])[0]
        size_checksum = struct.unpack("<I", data[4:8])[0]

        # Extract level name (starts around offset 39)
        string_start = 39
        string_end = data.find(0, string_start)
        if string_end == -1:
            string_end = min(string_start + 50, len(data))
        level_name = (
            data[string_start:string_end].decode("ascii", errors="ignore").strip()
        )

        logger.debug(
            f"Level ID: {level_id}, Size/Checksum: {size_checksum}, Name: '{level_name}'"
        )

        # Use the attract file decoder to extract demo data according to the official format
        logger.info("Extracting demo data using proper attract file format parser")

        decoded_data = self.npp_attract_decoder.decode_npp_attract_file(
            str(replay_file)
        )

        # Extract the demo data (input sequence)
        inputs = decoded_data.get("demo_data", [])

        if inputs and len(inputs) > 0:
            logger.info(
                f"Successfully extracted {len(inputs)} demo inputs using attract file format"
            )
            logger.info(
                f"  Demo sequence length: {len(inputs)} inputs = {len(inputs) / 60.0:.1f}s"
            )

            # Analyze input distribution for validation
            input_counts = Counter(inputs)
            logger.info(f"  Input distribution: {dict(input_counts)}")

            # Validate inputs are in expected range (0-7)
            invalid_inputs = [inp for inp in inputs if not (0 <= inp <= 7)]
            if invalid_inputs:
                print(
                    f"Found {len(invalid_inputs)} invalid input values, clamping to 0-7 range"
                )
                inputs = [max(0, min(7, inp)) for inp in inputs]
        else:
            raise ValueError("No valid input sequences found in attract file")

        # Create complete nclone-compatible map data
        try:
            nclone_map_bytes = self.npp_attract_decoder.create_nclone_map(
                str(replay_file)
            )
            map_data = list(nclone_map_bytes)
            logger.info(
                f"Perfect decoder extracted {len(decoded_data['tiles'])} tiles, {len(decoded_data['entities'])} entities, spawn {decoded_data['ninja_spawn']}"
            )
        except Exception as e:
            print(
                f"Perfect decoder map creation failed, falling back to basic map data: {e}"
            )
            map_data = [0] * 1335  # Standard empty map size

        logger.info(
            f"Extracted {len(inputs)} input frames and {len(map_data)} map bytes"
        )
        logger.debug(f"Input range: {min(inputs)} to {max(inputs)}")

        # Log input distribution for debugging
        input_dist = {i: inputs.count(i) for i in range(8) if i in inputs}
        logger.debug(f"Input distribution: {input_dist}")

        return inputs, map_data, level_id, level_name

    def simulate_replay(
        self,
        inputs: List[int],
        map_data: List[int],
        level_id: str,
        session_id: str,
        output_path: str = "",
        npp_level_id: Optional[int] = None,
        level_name: str = "",
    ) -> List[Dict[str, Any]]:
        """Delegate to simulation manager."""
        return self.simulation_manager.simulate_replay(
            inputs,
            map_data,
            level_id,
            session_id,
            output_path,
            npp_level_id,
            level_name,
        )

    def parse_replay_directory(
        self, replay_dir: Path, output_dir: Path, level_id: Optional[str] = None
    ) -> bool:
        """
        Parse a single replay directory and generate JSONL files.

        Args:
            replay_dir: Directory containing binary replay files
            output_dir: Output directory for JSONL files
            level_id: Optional level identifier (auto-generated if None)

        Returns:
            True if parsing succeeded
        """
        try:
            # Check if this is a trace mode directory
            if not self.detect_trace_mode(replay_dir):
                print(f"Directory {replay_dir} does not contain trace mode files")
                return False

            # Load data
            inputs_list, map_data = self.load_inputs_and_map(replay_dir)

            # Generate level ID if not provided
            if level_id is None:
                # Use directory name or generate from map data hash
                if replay_dir.name != "." and replay_dir.name:
                    level_id = replay_dir.name
                else:
                    map_hash = hashlib.md5(str(map_data).encode()).hexdigest()[:8]
                    level_id = f"level_{map_hash}"

            # Process each input sequence
            session_counter = 0
            for i, inputs in enumerate(inputs_list):
                session_id = f"{level_id}_session_{session_counter:03d}"

                logger.info(
                    f"Processing replay {i + 1}/{len(inputs_list)} (session: {session_id})"
                )

                try:
                    # Save to JSONL file
                    output_file = output_dir / f"{session_id}.jsonl"

                    # Simulate and extract frames
                    frames = self.simulate_replay(
                        inputs, map_data, level_id, session_id, str(output_file)
                    )

                    if frames:
                        self.save_frames_to_jsonl(frames, output_file)

                        self.stats["frames_generated"] += len(frames)
                        self.stats["replays_processed"] += 1
                        session_counter += 1

                except Exception as e:
                    print(f"Failed to process replay {i}: {e}")
                    self.stats["replays_failed"] += 1

            self.stats["files_processed"] += 1
            return True

        except Exception as e:
            print(f"Failed to parse replay directory {replay_dir}: {e}")
            return False

    def save_frames_to_jsonl(self, frames: List[Dict[str, Any]], output_file: Path):
        """Delegate to simulation manager."""
        self.simulation_manager.save_frames_to_jsonl(frames, output_file)

    def parse_single_replay_file_to_jsonl(
        self, replay_file: Path, output_dir: Path
    ) -> bool:
        """
        Parse a single replay file and generate JSONL output.

        Args:
            replay_file: Path to the single replay file
            output_dir: Output directory for JSONL files

        Returns:
            True if parsing succeeded
        """
        try:
            # Parse the single replay file
            inputs, map_data, npp_level_id, level_name = self.parse_single_replay_file(
                replay_file
            )

            # Generate session ID from filename
            file_id = replay_file.stem
            session_id = f"{file_id}_session_000"

            logger.info(
                f"Processing single replay file: {replay_file.name} (Level ID: {npp_level_id}, Name: '{level_name}')"
            )

            # Create output file path
            output_file = output_dir / f"{session_id}.jsonl"

            # Simulate and extract frames using enhanced map data
            frames = self.simulate_replay(
                inputs,
                map_data,
                file_id,
                session_id,
                str(output_file),
                npp_level_id,
                level_name,
            )

            if frames:
                self.save_frames_to_jsonl(frames, output_file)

                self.stats["frames_generated"] += len(frames)
                self.stats["replays_processed"] += 1
                self.stats["files_processed"] += 1
                return True
            else:
                print(f"No frames generated for {replay_file}")
                self.stats["replays_failed"] += 1
                return False

        except Exception as e:
            print(f"Failed to parse single replay file {replay_file}: {e}")
            import traceback

            print(f"Full traceback: {traceback.format_exc()}")
            self.stats["replays_failed"] += 1
            return False

    def process_directory(self, input_dir: Path, output_dir: Path):
        """
        Process all replay directories or single replay files in the input directory.

        Args:
            input_dir: Directory containing replay subdirectories or single replay files
            output_dir: Output directory for JSONL files
        """
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if input_dir itself is a replay directory
        if self.detect_trace_mode(input_dir):
            logger.info(f"Processing single replay directory: {input_dir}")
            self.parse_replay_directory(input_dir, output_dir)
        else:
            # Look for subdirectories containing replays or single replay files
            replay_dirs = []
            single_replay_files = []

            for item in input_dir.iterdir():
                if item.is_dir() and self.detect_trace_mode(item):
                    replay_dirs.append(item)
                elif item.is_file() and self.detect_single_replay_file(item):
                    single_replay_files.append(item)

            # Process trace mode directories
            if replay_dirs:
                logger.info(f"Found {len(replay_dirs)} replay directories")
                for replay_dir in replay_dirs:
                    logger.info(f"Processing replay directory: {replay_dir}")
                    self.parse_replay_directory(
                        replay_dir, output_dir, level_id=replay_dir.name
                    )

            # Process single replay files
            if single_replay_files:
                logger.info(f"Found {len(single_replay_files)} single replay files")
                for replay_file in single_replay_files:
                    self.parse_single_replay_file_to_jsonl(replay_file, output_dir)

            if not replay_dirs and not single_replay_files:
                print(f"No replay files or directories found in {input_dir}")
                return

    def print_statistics(self):
        """Print processing statistics."""
        print("\n" + "=" * 50)
        print("BINARY REPLAY PARSER STATISTICS")
        print("=" * 50)
        print(f"Directories processed: {self.stats['files_processed']}")
        print(f"Replays processed: {self.stats['replays_processed']}")
        print(f"Replays failed: {self.stats['replays_failed']}")
        print(f"Frames generated: {self.stats['frames_generated']}")

        if self.stats["replays_processed"] > 0:
            avg_frames = (
                self.stats["frames_generated"] / self.stats["replays_processed"]
            )
            print(f"Average frames per replay: {avg_frames:.1f}")

        total_replays = self.stats["replays_processed"] + self.stats["replays_failed"]
        if total_replays > 0:
            success_rate = self.stats["replays_processed"] / total_replays * 100
            print(f"Success rate: {success_rate:.1f}%")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert N++ binary replay files to JSONL format. Automatically detects and handles compressed (zlib) and uncompressed files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single replay directory (compressed or uncompressed)
  python -m nclone.replay.binary_replay_parser --input replays/level_001 --output datasets/raw/
  
  # Process multiple replay directories
  python -m nclone.replay.binary_replay_parser --input replays/ --output datasets/raw/
  
  # Enable verbose logging to see compression detection details
  python -m nclone.replay.binary_replay_parser --input replays/ --output datasets/raw/ --verbose

Note: The parser follows the ntrace.py pattern for handling compressed input files.
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory containing binary replay files",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output directory for JSONL files"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Validate arguments
    if not args.input.exists():
        parser.error(f"Input directory does not exist: {args.input}")

    # Initialize parser
    replay_parser = BinaryReplayParser()

    # Process data
    logger.info(f"Processing binary replay data from {args.input}")
    logger.info(f"Output directory: {args.output}")

    try:
        # Check if input is a single file or directory
        if args.input.is_file():
            # Single file processing
            if replay_parser.detect_single_replay_file(args.input):
                logger.info(f"Processing single replay file: {args.input}")
                args.output.mkdir(parents=True, exist_ok=True)
                replay_parser.parse_single_replay_file_to_jsonl(args.input, args.output)
            else:
                print(f"File does not appear to be a valid replay file: {args.input}")
                return 1
        else:
            # Directory processing
            replay_parser.process_directory(args.input, args.output)
    except Exception as e:
        print(f"Processing failed: {e}")
        return 1

    # Print statistics
    replay_parser.print_statistics()

    return 0


if __name__ == "__main__":
    exit(main())
