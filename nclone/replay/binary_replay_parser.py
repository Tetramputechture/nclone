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
import json
import logging
import zlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
import hashlib

from ..nsim import Simulator
from ..sim_config import SimConfig

logger = logging.getLogger(__name__)


class BinaryReplayParser:
    """
    Parser for N++ binary replay files to JSONL format.

    Supports automatic detection and handling of compressed (zlib) and
    uncompressed binary replay files, following the ntrace.py pattern.
    """

    # Input encoding dictionaries (from ntrace.py)
    HOR_INPUTS_DIC = {0: 0, 1: 0, 2: 1, 3: 1, 4: -1, 5: -1, 6: -1, 7: -1}
    JUMP_INPUTS_DIC = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1}

    # Required file names for trace mode
    RAW_INPUTS = ["inputs_0", "inputs_1", "inputs_2", "inputs_3"]
    RAW_MAP_DATA = "map_data"

    # Fixed level dimensions (as specified by user)
    LEVEL_WIDTH = 1056
    LEVEL_HEIGHT = 600

    def __init__(self):
        """Initialize the parser."""
        self.stats = {
            "files_processed": 0,
            "frames_generated": 0,
            "replays_processed": 0,
            "replays_failed": 0,
        }

    def _is_compressed_data(self, data: bytes) -> bool:
        """
        Check if data appears to be zlib compressed.

        Args:
            data: Raw bytes to check

        Returns:
            True if data appears to be zlib compressed
        """
        # Check for zlib header patterns
        if len(data) < 2:
            return False

        # zlib header: first byte should be 0x78 for common compression levels
        # Common zlib headers: 0x789C (default), 0x78DA (best compression), 0x7801 (no compression)
        first_byte = data[0]
        if first_byte == 0x78:
            return True

        # Try to decompress a small portion to verify
        try:
            zlib.decompress(data[:100])  # Try first 100 bytes
            return True
        except zlib.error:
            return False

    def _read_file_with_compression_detection(self, file_path: Path) -> List[int]:
        """
        Read a file and automatically handle compression.

        This method follows the ntrace.py pattern (lines 34-35) which uses
        zlib.decompress() to handle compressed input files, but adds automatic
        detection to support both compressed and uncompressed files.

        Args:
            file_path: Path to the file to read

        Returns:
            List of integers from the file data
        """
        with open(file_path, "rb") as f:
            raw_data = f.read()

        # Try to detect if data is compressed
        if self._is_compressed_data(raw_data):
            try:
                # Decompress the data (following ntrace pattern)
                decompressed_data = zlib.decompress(raw_data)
                return [int(b) for b in decompressed_data]
            except zlib.error as e:
                logger.warning(f"Failed to decompress {file_path}: {e}")
                # Fall back to treating as uncompressed
                return [int(b) for b in raw_data]
        else:
            # Treat as uncompressed data
            return [int(b) for b in raw_data]

    def detect_trace_mode(self, replay_dir: Path) -> bool:
        """
        Check if directory contains trace mode files.

        Args:
            replay_dir: Directory to check

        Returns:
            True if trace mode files are present
        """
        map_data_exists = (replay_dir / self.RAW_MAP_DATA).exists()
        inputs_exist = any((replay_dir / inp).exists() for inp in self.RAW_INPUTS)

        return map_data_exists and inputs_exist

    def detect_single_replay_file(self, file_path: Path) -> bool:
        """
        Check if file is a single N++ replay file (like npp_attract format).

        Args:
            file_path: Path to the file to check

        Returns:
            True if file appears to be a single N++ replay file
        """
        if not file_path.is_file():
            return False

        try:
            with open(file_path, "rb") as f:
                # Read first few bytes to check for N++ replay signature
                header = f.read(32)
                if len(header) < 32:
                    return False

                # Look for patterns that suggest this is an N++ replay
                # Based on analysis of npp_attract files:
                # - Files are typically 1000-3000 bytes
                # - Start with structured header data
                # - Contain level metadata and input sequences
                file_size = file_path.stat().st_size
                if 500 <= file_size <= 10000:  # Reasonable size range
                    return True

        except Exception:
            return False

        return False

    def load_inputs_and_map(
        self, replay_dir: Path
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Load input sequences and map data from binary files.
        Automatically detects and handles compressed files (following ntrace pattern).

        Args:
            replay_dir: Directory containing replay files

        Returns:
            Tuple of (inputs_list, map_data)
        """
        inputs_list = []

        # Load input files (following ntrace.py pattern for compressed inputs)
        # Reference: ntrace.py lines 32-37 show the original compression handling
        for input_file in self.RAW_INPUTS:
            input_path = replay_dir / input_file
            if input_path.exists():
                try:
                    inputs = self._read_file_with_compression_detection(input_path)
                    inputs_list.append(inputs)
                    logger.debug(
                        f"Loaded input file {input_file} with {len(inputs)} frames"
                    )
                except Exception as e:
                    logger.error(f"Failed to load input file {input_file}: {e}")
                    break
            else:
                break

        # Load map data (can also be compressed)
        map_path = replay_dir / self.RAW_MAP_DATA
        if not map_path.exists():
            raise FileNotFoundError(f"Map data file not found: {map_path}")

        try:
            map_data = self._read_file_with_compression_detection(map_path)
            logger.debug(f"Loaded map data with {len(map_data)} bytes")
        except Exception as e:
            logger.error(f"Failed to load map data: {e}")
            raise

        logger.info(f"Loaded {len(inputs_list)} input sequences and map data")
        return inputs_list, map_data

    def parse_single_replay_file(
        self, replay_file: Path
    ) -> Tuple[List[int], List[int]]:
        """
        Parse a single N++ replay file (npp_attract format).

        Based on comprehensive analysis, these files have two formats:
        Type 1: Header + Input data starting at offset 183
        Type 2: Header + Map data + Input data starting at variable offset

        Args:
            replay_file: Path to the single replay file

        Returns:
            Tuple of (inputs, map_data)
        """
        with open(replay_file, "rb") as f:
            data = f.read()

        logger.debug(f"Parsing single replay file: {replay_file} ({len(data)} bytes)")

        # Parse header information
        import struct

        level_id = struct.unpack("<I", data[0:4])[0]
        size_checksum = struct.unpack("<I", data[4:8])[0]
        format_flag = data[16]

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

        # Find "Metanet Software" string (should be at offset 167)
        metanet_pos = data.find(b"Metanet Software")
        if metanet_pos == -1:
            raise ValueError("Could not find 'Metanet Software' marker")

        # Determine file type based on input start position
        # Type 1: Input starts immediately after "Metanet Software\0" (offset 183)
        # Type 2: Input starts later with map data in between

        potential_input_start = (
            metanet_pos + len(b"Metanet Software") + 1
        )  # +1 for null terminator

        # Check if inputs start immediately at offset 183 (Type 1)
        if (
            potential_input_start <= 183
            and potential_input_start < len(data)
            and 0 <= data[potential_input_start] <= 7
        ):
            # Type 1: Simple format
            input_start = potential_input_start
            logger.debug(f"Detected Type 1 format: inputs start at {input_start}")

            # Map data is embedded in the header section
            map_start = 100  # After initial header fields
            map_end = metanet_pos - 5  # Before "Metanet Software"
            map_data = list(data[map_start:map_end])

        else:
            # Type 2: Extended format - find actual input start
            logger.debug("Detected Type 2 format: searching for input start")

            best_input_start = 0
            best_input_length = 0

            # Start searching after the header
            for start in range(200, len(data)):
                length = 0
                pos = start
                while pos < len(data) and 0 <= data[pos] <= 7:
                    length += 1
                    pos += 1

                if length > best_input_length:
                    best_input_start = start
                    best_input_length = length

            if best_input_length < 10:
                raise ValueError(
                    f"Could not find valid input sequence in Type 2 format"
                )

            input_start = best_input_start
            logger.debug(
                f"Type 2 format: inputs start at {input_start}, length {best_input_length}"
            )

            # Map data is between header end and input start
            map_start = 183  # After header
            map_end = input_start - 5  # Before inputs, leave buffer

            if map_end > map_start:
                map_data = list(data[map_start:map_end])
            else:
                # Fallback: use header section
                map_data = list(data[100:167])

        # Extract input sequence
        input_length = 0
        pos = input_start
        while pos < len(data) and 0 <= data[pos] <= 7:
            input_length += 1
            pos += 1

        inputs = list(data[input_start : input_start + input_length])

        # Validate and clean up map data
        if len(map_data) < 50:
            logger.warning(f"Map data seems small ({len(map_data)} bytes), padding")
            map_data.extend([1] * (50 - len(map_data)))

        # Limit map data size to reasonable bounds
        if len(map_data) > 2000:
            logger.warning(f"Map data seems large ({len(map_data)} bytes), truncating")
            map_data = map_data[:2000]

        logger.info(
            f"Extracted {len(inputs)} input frames and {len(map_data)} map bytes"
        )
        logger.debug(f"Input range: {min(inputs)} to {max(inputs)}")

        # Log input distribution for debugging
        input_dist = {i: inputs.count(i) for i in range(8) if i in inputs}
        logger.debug(f"Input distribution: {input_dist}")

        return inputs, map_data

    def decode_inputs(self, raw_inputs: List[int]) -> Tuple[List[int], List[int]]:
        """
        Decode raw input values to horizontal and jump components.

        Args:
            raw_inputs: List of raw input values (0-7)

        Returns:
            Tuple of (horizontal_inputs, jump_inputs)
        """
        hor_inputs = [self.HOR_INPUTS_DIC[inp] for inp in raw_inputs]
        jump_inputs = [self.JUMP_INPUTS_DIC[inp] for inp in raw_inputs]
        return hor_inputs, jump_inputs

    def get_entity_data(self, sim: Simulator) -> List[Dict[str, Any]]:
        """
        Extract entity information from simulator.

        Args:
            sim: Simulator instance

        Returns:
            List of entity dictionaries
        """
        entities = []

        # Iterate through all entity types
        for entity_type, entity_list in sim.entity_dic.items():
            for entity in entity_list:
                entity_data = {
                    "type": self._get_entity_type_name(entity_type),
                    "position": {"x": float(entity.xpos), "y": float(entity.ypos)},
                    "active": True,  # Default to active, could be refined based on entity state
                }

                # Add type-specific data
                if hasattr(entity, "state"):
                    entity_data["state"] = entity.state
                if hasattr(entity, "radius"):
                    entity_data["radius"] = entity.radius

                entities.append(entity_data)

        return entities

    def _get_entity_type_name(self, entity_type: int) -> str:
        """
        Convert entity type ID to name.

        Args:
            entity_type: Numeric entity type

        Returns:
            Entity type name string
        """
        # Entity type mapping based on N++ documentation
        type_mapping = {
            1: "mine",
            2: "gold",
            3: "exit_door",
            4: "exit_switch",
            5: "door",
            6: "locked_door",
            8: "trap_door",
            10: "launch_pad",
            11: "one_way_platform",
            14: "drone",
            17: "bounce_block",
            20: "thwump",
            21: "toggle_mine",
            25: "death_ball",
            26: "mini_drone",
        }

        return type_mapping.get(entity_type, f"unknown_{entity_type}")

    def simulate_replay(
        self, inputs: List[int], map_data: List[int], level_id: str, session_id: str
    ) -> List[Dict[str, Any]]:
        """
        Simulate a replay and extract frame-by-frame data.

        Args:
            inputs: Raw input sequence
            map_data: Map data
            level_id: Level identifier
            session_id: Session identifier

        Returns:
            List of frame dictionaries in JSONL format
        """
        frames = []

        # Decode inputs
        hor_inputs, jump_inputs = self.decode_inputs(inputs)
        inp_len = len(inputs)

        # Initialize simulator (disable animation for parsing)
        sim_config = SimConfig(enable_anim=False)
        sim = Simulator(sim_config)
        sim.load(map_data)

        # Track initial state
        start_time = time.time()

        # Execute simulation frame by frame
        while sim.frame < inp_len:
            # Get current inputs
            hor_input = hor_inputs[sim.frame]
            jump_input = jump_inputs[sim.frame]

            # Create frame data before simulation step
            frame_data = {
                "timestamp": start_time + (sim.frame * (1.0 / 60.0)),  # 60 FPS
                "level_id": level_id,
                "frame_number": sim.frame,
                "player_state": {
                    "position": {
                        "x": float(sim.ninja.xpos),
                        "y": float(sim.ninja.ypos),
                    },
                    "velocity": {
                        "x": float(sim.ninja.xspeed),
                        "y": float(sim.ninja.yspeed),
                    },
                    "on_ground": sim.ninja.state in [0, 1, 2],  # Ground states
                    "wall_sliding": sim.ninja.state == 5,  # Wall sliding state
                    "jump_time_remaining": max(0.0, (45 - sim.ninja.jump_timer) / 45.0)
                    if hasattr(sim.ninja, "jump_timer")
                    else 0.0,
                },
                "player_inputs": {
                    "left": hor_input == -1,
                    "right": hor_input == 1,
                    "jump": jump_input == 1,
                    "restart": False,
                },
                "entities": self.get_entity_data(sim),
                "level_bounds": {
                    "width": self.LEVEL_WIDTH,
                    "height": self.LEVEL_HEIGHT,
                },
                "meta": {
                    "session_id": session_id,
                    "player_id": "binary_replay",
                    "quality_score": 0.8,  # Default quality score
                    "completion_status": "in_progress",
                },
            }

            frames.append(frame_data)

            # Execute simulation step
            sim.tick(hor_input, jump_input)

            # Check for death or completion
            if sim.ninja.state == 6:  # Dead
                # Update final frame status
                frames[-1]["meta"]["completion_status"] = "failed"
                break
            elif sim.ninja.state == 8:  # Celebrating (completed)
                # Update final frame status
                frames[-1]["meta"]["completion_status"] = "completed"
                frames[-1]["meta"]["quality_score"] = 0.9  # Higher score for completion
                break

        logger.info(f"Generated {len(frames)} frames for replay")
        return frames

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
                logger.warning(
                    f"Directory {replay_dir} does not contain trace mode files"
                )
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
                    # Simulate and extract frames
                    frames = self.simulate_replay(
                        inputs, map_data, level_id, session_id
                    )

                    if frames:
                        # Save to JSONL file
                        output_file = output_dir / f"{session_id}.jsonl"
                        self.save_frames_to_jsonl(frames, output_file)

                        self.stats["frames_generated"] += len(frames)
                        self.stats["replays_processed"] += 1
                        session_counter += 1

                except Exception as e:
                    logger.error(f"Failed to process replay {i}: {e}")
                    self.stats["replays_failed"] += 1

            self.stats["files_processed"] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to parse replay directory {replay_dir}: {e}")
            return False

    def save_frames_to_jsonl(self, frames: List[Dict[str, Any]], output_file: Path):
        """
        Save frames to JSONL file.

        Args:
            frames: List of frame dictionaries
            output_file: Output file path
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            for frame in frames:
                f.write(json.dumps(frame) + "\n")

        logger.info(f"Saved {len(frames)} frames to {output_file}")

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
            inputs, map_data = self.parse_single_replay_file(replay_file)

            # Generate level ID from filename
            level_id = replay_file.stem
            session_id = f"{level_id}_session_000"

            logger.info(f"Processing single replay file: {replay_file.name}")

            # Simulate and extract frames
            frames = self.simulate_replay(inputs, map_data, level_id, session_id)

            if frames:
                # Save to JSONL file
                output_file = output_dir / f"{session_id}.jsonl"
                self.save_frames_to_jsonl(frames, output_file)

                self.stats["frames_generated"] += len(frames)
                self.stats["replays_processed"] += 1
                self.stats["files_processed"] += 1
                return True
            else:
                logger.error(f"No frames generated for {replay_file}")
                self.stats["replays_failed"] += 1
                return False

        except Exception as e:
            logger.error(f"Failed to parse single replay file {replay_file}: {e}")
            import traceback

            logger.debug(f"Full traceback: {traceback.format_exc()}")
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
                logger.warning(f"No replay files or directories found in {input_dir}")
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
                logger.error(
                    f"File does not appear to be a valid replay file: {args.input}"
                )
                return 1
        else:
            # Directory processing
            replay_parser.process_directory(args.input, args.output)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1

    # Print statistics
    replay_parser.print_statistics()

    return 0


if __name__ == "__main__":
    exit(main())
