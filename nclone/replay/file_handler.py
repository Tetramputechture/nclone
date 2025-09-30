#!/usr/bin/env python3
"""
File handling utilities for N++ binary replay parsing.

Handles file I/O, compression detection, and validation for both
trace mode directories and single replay files.
"""

import logging
import zlib
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


class FileHandler:
    """
    Handles file I/O operations for binary replay parsing.

    Supports automatic detection and handling of compressed (zlib) and
    uncompressed binary replay files, following the ntrace.py pattern.
    """

    # Required file names for trace mode
    RAW_INPUTS = ["inputs_0", "inputs_1", "inputs_2", "inputs_3"]
    RAW_MAP_DATA = "map_data"

    def validate_replay_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """
        Validate a replay file before processing.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not file_path.exists():
            errors.append(f"File does not exist: {file_path}")
            return False, errors

        try:
            file_size = file_path.stat().st_size

            # Check minimum file size
            if file_size < 1000:
                errors.append(f"File too small: {file_size} bytes")

            # Check maximum reasonable file size
            if file_size > 10000:
                errors.append(f"File too large: {file_size} bytes")

            # Try to read first few bytes
            with open(file_path, "rb") as f:
                header = f.read(10)
                if len(header) < 10:
                    errors.append("Cannot read file header")

        except Exception as e:
            errors.append(f"File access error: {e}")

        return len(errors) == 0, errors

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
