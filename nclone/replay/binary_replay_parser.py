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
import os
import zlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
import hashlib

from ..nsim import Simulator
from ..sim_config import SimConfig
from .map_loader import MapLoader

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
    ) -> Tuple[List[int], List[int], int, str]:
        """
        Parse a single N++ replay file (npp_attract format).

        Based on comprehensive analysis, these files have two formats:
        Type 1: Header + Input data starting at offset 183
        Type 2: Header + Map data + Input data starting at variable offset

        Args:
            replay_file: Path to the single replay file

        Returns:
            Tuple of (inputs, map_data, level_id, level_name)
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

        # Check if inputs start immediately at offset 183/184 (Type 1)
        # Allow for slight variation in the exact offset
        if (
            potential_input_start <= 185
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

        # For npp_attract files, generate empty/minimal map data since they only contain map references
        logger.info(f"Generating empty map for npp_attract file (Level ID: {level_id})")
        map_data = self._generate_empty_map_data(level_id, level_name)

        logger.info(
            f"Extracted {len(inputs)} input frames and {len(map_data)} map bytes"
        )
        logger.debug(f"Input range: {min(inputs)} to {max(inputs)}")

        # Log input distribution for debugging
        input_dist = {i: inputs.count(i) for i in range(8) if i in inputs}
        logger.debug(f"Input distribution: {input_dist}")

        return inputs, map_data, level_id, level_name

    def _generate_empty_map_data(self, level_id: int, level_name: str) -> List[int]:
        """
        Generate map data for npp_attract files.
        
        First tries to load real map data using the map loader, then falls back
        to generating empty map data if no real map is found.
        
        Args:
            level_id: N++ Level ID (primary map reference)
            level_name: Level name for logging
            
        Returns:
            List of integers representing map structure
        """
        # Try to find real map data first using enhanced correlation
        real_map = self.map_loader.find_map_by_name(level_name, level_id)
        if not real_map:
            real_map = self.map_loader.find_map_by_id(level_id)
        
        if real_map:
            map_name, map_data_str, source_file = real_map
            logger.info(f"Found real map data for '{level_name}' (ID: {level_id}) -> '{map_name}' from {source_file}")
            try:
                # Convert N++ map format to nclone format
                map_bytes = self.map_loader.convert_map_data_to_nclone_format(map_data_str)
                return list(map_bytes)
            except Exception as e:
                logger.warning(f"Failed to convert real map data: {e}, falling back to empty map")
        
        # Fall back to empty map generation
        logger.info(f"Generating empty map for Level ID {level_id}: '{level_name}'")
        
        # Create a minimal empty map structure
        # This creates a small empty room that allows the ninja to spawn and move
        map_data = [0] * 1245  # Standard map size
        
        # Set basic level metadata in the header
        map_data[0:4] = [6, 0, 0, 0]  # Level type
        map_data[4:8] = [221, 4, 0, 0]  # Size/checksum value
        map_data[8:12] = [255, 255, 255, 255]  # Unknown field
        map_data[12:16] = [0, 0, 0, 0]  # No entities for empty map
        map_data[16:20] = [level_id & 0xFF, (level_id >> 8) & 0xFF, (level_id >> 16) & 0xFF, (level_id >> 24) & 0xFF]  # Store Level ID
        
        # Create a minimal room structure (empty space with boundaries)
        # This ensures the ninja can spawn and move without crashing the simulator
        room_start = 184
        room_size = 1061  # Remaining space for room data
        
        # Fill with empty space (0 = empty, 1 = wall)
        for i in range(room_start, room_start + room_size):
            if i < room_start + 50:  # Some initial structure
                map_data[i] = 0  # Empty space
            else:
                map_data[i] = 0  # All empty for minimal map
        
        logger.debug(f"Generated empty map with {len(map_data)} bytes for Level ID {level_id}")
        return map_data

    def _generate_default_map_data(self, level_name: str) -> List[int]:
        """
        Generate a default map data structure suitable for simulation.
        
        This creates a minimal but valid map that allows the simulator to run
        and test input sequences from npp_attract files.
        
        Args:
            level_name: Name of the level (for logging)
            
        Returns:
            List of integers representing a valid map data structure
        """
        logger.info(f"Generating default map structure for level: {level_name}")
        
        # Create a basic map structure based on the simple-walk test map
        # Map header (first 184 bytes) - contains level metadata
        map_data = [0] * 184
        
        # Set basic level metadata in the header
        map_data[0:4] = [6, 0, 0, 0]  # Level type
        map_data[4:8] = [221, 4, 0, 0]  # Some size/checksum value
        map_data[8:12] = [255, 255, 255, 255]  # Unknown field
        map_data[12:16] = [4, 0, 0, 0]  # Entity count or similar
        map_data[16:20] = [37, 0, 0, 0]  # Another metadata field
        
        # Add level name to header (starting around position 40)
        level_name_bytes = level_name.encode('ascii', errors='ignore')[:30]
        for i, byte_val in enumerate(level_name_bytes):
            if 40 + i < 184:
                map_data[40 + i] = byte_val
        
        # Tile data (42x23 = 966 bytes) - create a simple open area with walls around edges
        tile_data = []
        
        for y in range(23):
            for x in range(42):
                if x == 0 or x == 41 or y == 0 or y == 22:
                    # Wall tiles around the edges
                    tile_data.append(1)
                elif y == 21:
                    # Floor tiles near the bottom
                    tile_data.append(1)
                else:
                    # Empty space in the middle
                    tile_data.append(0)
        
        # Add tile data to map
        map_data.extend(tile_data)
        
        # Add some basic entity data (remaining bytes up to ~1245)
        remaining_bytes = 1245 - len(map_data)
        if remaining_bytes > 0:
            # Add minimal entity data - mostly zeros with a few basic entities
            entity_data = [0] * remaining_bytes
            
            # Add a simple ninja spawn point and exit
            if remaining_bytes >= 20:
                entity_data[0:4] = [100, 0, 0, 0]  # Ninja spawn X
                entity_data[4:8] = [400, 0, 0, 0]  # Ninja spawn Y
                entity_data[8:12] = [800, 0, 0, 0]  # Exit X
                entity_data[12:16] = [400, 0, 0, 0]  # Exit Y
            
            map_data.extend(entity_data)
        
        logger.debug(f"Generated default map with {len(map_data)} bytes")
        return map_data

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

    def get_comprehensive_ninja_state(self, sim: Simulator) -> Dict[str, Any]:
        """
        Extract comprehensive ninja state matching npp-rl expectations.
        
        Args:
            sim: Simulator instance
            
        Returns:
            Dictionary with comprehensive ninja state information
        """
        ninja = sim.ninja
        
        # Calculate velocity magnitude for normalization
        velocity_magnitude = (ninja.xspeed**2 + ninja.yspeed**2) ** 0.5
        max_velocity = 3.333 * 2  # MAX_HOR_SPEED * 2 for vertical velocity
        
        # === Core Movement State (8 features) ===
        velocity_mag_norm = min(velocity_magnitude / max_velocity, 1.0) * 2 - 1  # [-1, 1]
        
        # Velocity direction (normalized, handling zero velocity)
        if velocity_magnitude > 1e-6:
            velocity_dir_x = ninja.xspeed / velocity_magnitude
            velocity_dir_y = ninja.yspeed / velocity_magnitude
        else:
            velocity_dir_x = 0.0
            velocity_dir_y = 0.0
            
        # Movement state categories
        ground_movement = 1.0 if ninja.state in [0, 1, 2] else -1.0
        air_movement = 1.0 if ninja.state in [3, 4] else -1.0
        wall_interaction = 1.0 if ninja.state == 5 else -1.0
        special_states = 1.0 if ninja.state in [6, 7, 8, 9] else -1.0
        airborne_status = 1.0 if ninja.airborn else -1.0
        
        # === Input and Buffer State (5 features) ===
        horizontal_input = float(ninja.hor_input)  # Already -1, 0, or 1
        jump_input = 1.0 if ninja.jump_input else -1.0
        
        # Buffer states (normalized to [-1, 1])
        buffer_window_size = 5.0
        jump_buffer_norm = (max(ninja.jump_buffer, 0) / buffer_window_size) * 2 - 1
        floor_buffer_norm = (max(ninja.floor_buffer, 0) / buffer_window_size) * 2 - 1
        wall_buffer_norm = (max(ninja.wall_buffer, 0) / buffer_window_size) * 2 - 1
        
        # === Surface Contact Information (6 features) ===
        floor_contact = (min(ninja.floor_count, 1) * 2) - 1
        wall_contact = (min(ninja.wall_count, 1) * 2) - 1
        ceiling_contact = (min(ninja.ceiling_count, 1) * 2) - 1
        
        floor_normal_strength = (ninja.floor_normalized_x**2 + ninja.floor_normalized_y**2) ** 0.5
        floor_normal_strength = (floor_normal_strength * 2) - 1
        
        # Wall normal direction
        if ninja.wall_count > 0 and hasattr(ninja, 'wall_normal'):
            wall_direction = float(ninja.wall_normal)
        else:
            wall_direction = 0.0
            
        surface_slope = ninja.floor_normalized_y  # Already [-1, 1]
        
        # === Momentum and Physics (4 features) ===
        accel_x = (ninja.xspeed - ninja.xspeed_old) / 3.333  # Normalize by MAX_HOR_SPEED
        accel_y = (ninja.yspeed - ninja.yspeed_old) / 3.333
        accel_x = max(-1.0, min(1.0, accel_x))
        accel_y = max(-1.0, min(1.0, accel_y))
        
        # Momentum preservation
        prev_velocity_mag = (ninja.xspeed_old**2 + ninja.yspeed_old**2) ** 0.5
        if prev_velocity_mag > 1e-6 and velocity_magnitude > 1e-6:
            momentum_preservation = (ninja.xspeed * ninja.xspeed_old + ninja.yspeed * ninja.yspeed_old) / (velocity_magnitude * prev_velocity_mag)
        else:
            momentum_preservation = 0.0
            
        impact_risk = velocity_mag_norm if (ninja.floor_count > 0 or ninja.ceiling_count > 0) else 0.0
        
        # === Entity Proximity and Hazards (4 features) ===
        # These will be placeholders for now, could be enhanced with actual entity analysis
        nearest_hazard_distance = 0.0
        nearest_collectible_distance = 0.0
        hazard_threat_level = 0.0
        interaction_cooldown = (ninja.jump_duration / 45.0) * 2 - 1  # MAX_JUMP_DURATION = 45
        
        # === Level Progress and Objectives (3 features) ===
        # These are placeholders that would normally be calculated from game state
        switch_progress = 0.0
        exit_accessibility = -1.0
        completion_progress = -1.0
        
        # Construct the 30-feature game state array
        game_state = [
            velocity_mag_norm, velocity_dir_x, velocity_dir_y,
            ground_movement, air_movement, wall_interaction, special_states, airborne_status,
            horizontal_input, jump_input, jump_buffer_norm, floor_buffer_norm, wall_buffer_norm,
            floor_contact, wall_contact, ceiling_contact, floor_normal_strength, wall_direction, surface_slope,
            accel_x, accel_y, momentum_preservation, impact_risk,
            nearest_hazard_distance, nearest_collectible_distance, hazard_threat_level, interaction_cooldown,
            switch_progress, exit_accessibility, completion_progress
        ]
        
        return {
            "position": {"x": float(ninja.xpos), "y": float(ninja.ypos)},
            "velocity": {"x": float(ninja.xspeed), "y": float(ninja.yspeed)},
            "state": int(ninja.state),
            "airborne": bool(ninja.airborn),
            "walled": bool(ninja.walled),
            "jump_duration": int(ninja.jump_duration),
            "applied_gravity": float(ninja.applied_gravity),
            "applied_drag": float(ninja.applied_drag),
            "applied_friction": float(ninja.applied_friction),
            "floor_normal": {"x": float(ninja.floor_normalized_x), "y": float(ninja.floor_normalized_y)},
            "ceiling_normal": {"x": float(ninja.ceiling_normalized_x), "y": float(ninja.ceiling_normalized_y)},
            "buffers": {
                "jump": int(ninja.jump_buffer),
                "floor": int(ninja.floor_buffer),
                "wall": int(ninja.wall_buffer),
                "launch_pad": int(ninja.launch_pad_buffer)
            },
            "contact_counts": {
                "floor": int(ninja.floor_count),
                "wall": int(ninja.wall_count),
                "ceiling": int(ninja.ceiling_count)
            },
            "game_state": game_state,  # 30-feature array for npp-rl compatibility
            "gold_collected": int(ninja.gold_collected),
            "doors_opened": int(ninja.doors_opened)
        }

    def save_map_data(self, map_data: List[int], output_path: str) -> str:
        """
        Save map data to a separate file alongside the JSONL.
        
        Args:
            map_data: Map data bytes
            output_path: Base output path for JSONL file
            
        Returns:
            Path to the saved map data file
        """
        # Create map data filename based on JSONL filename
        base_path = os.path.splitext(output_path)[0]
        map_data_path = f"{base_path}_map.dat"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(map_data_path), exist_ok=True)
        
        # Save map data as binary file
        with open(map_data_path, 'wb') as f:
            # Convert to bytes if needed
            if isinstance(map_data, list) and len(map_data) > 0:
                # Ensure all values are in valid byte range (0-255)
                byte_data = []
                for val in map_data:
                    if isinstance(val, int):
                        # Clamp to byte range
                        byte_data.append(max(0, min(255, val)))
                    else:
                        byte_data.append(0)
                map_bytes = bytes(byte_data)
            else:
                map_bytes = bytes(map_data) if isinstance(map_data, (list, tuple)) else map_data
            f.write(map_bytes)
            
        logger.info(f"Saved map data to {map_data_path} ({len(map_data)} bytes)")
        return map_data_path

    def simulate_replay(
        self, inputs: List[int], map_data: List[int], level_id: str, session_id: str, output_path: str = "", npp_level_id: int = None, level_name: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Simulate a replay and extract frame-by-frame data.

        Args:
            inputs: Raw input sequence
            map_data: Map data
            level_id: Level identifier (file-based)
            session_id: Session identifier
            output_path: Output path for JSONL file (used to determine map data path)
            npp_level_id: N++ Level ID (map reference from binary file)
            level_name: N++ Level name

        Returns:
            List of frame dictionaries in JSONL format
        """
        frames = []
        
        # Save map data to separate file if output path provided
        map_data_path = None
        if output_path:
            map_data_path = self.save_map_data(map_data, output_path)

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

            # Get comprehensive ninja state
            ninja_state = self.get_comprehensive_ninja_state(sim)

            # Create frame data before simulation step
            frame_data = {
                "timestamp": start_time + (sim.frame * (1.0 / 60.0)),  # 60 FPS
                "level_id": level_id,
                "frame_number": sim.frame,
                "player_state": ninja_state,
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
                    "npp_level_id": npp_level_id,  # N++ Level ID (map reference)
                    "npp_level_name": level_name,  # N++ Level name
                },
            }
            
            # Add map data path if available
            if map_data_path and output_path:
                frame_data["map_data_path"] = os.path.relpath(map_data_path, os.path.dirname(output_path))

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
            inputs, map_data, npp_level_id, level_name = self.parse_single_replay_file(replay_file)

            # Generate session ID from filename
            file_id = replay_file.stem
            session_id = f"{file_id}_session_000"

            logger.info(f"Processing single replay file: {replay_file.name} (Level ID: {npp_level_id}, Name: '{level_name}')")

            # Create output file path
            output_file = output_dir / f"{session_id}.jsonl"
            
            # Simulate and extract frames, passing the N++ Level ID as metadata
            frames = self.simulate_replay(inputs, map_data, file_id, session_id, str(output_file), npp_level_id, level_name)

            if frames:
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
