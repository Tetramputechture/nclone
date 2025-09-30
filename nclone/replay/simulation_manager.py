#!/usr/bin/env python3
"""
Simulation management utilities for N++ binary replay parsing.

Handles simulation execution, frame generation, and map data management
for converting replays to JSONL format.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..nsim import Simulator
from ..sim_config import SimConfig
from .unified_observation_extractor import UnifiedObservationExtractor
from .input_decoder import InputDecoder

logger = logging.getLogger(__name__)


class SimulationManager:
    """
    Manages simulation execution and frame data generation.

    Handles the conversion of replay inputs to frame-by-frame JSONL data
    by running the N++ simulator and extracting comprehensive state information.
    """

    # Fixed level dimensions (as specified by user)
    LEVEL_WIDTH = 1056
    LEVEL_HEIGHT = 600

    def __init__(self):
        self.observation_extractor = UnifiedObservationExtractor(
            enable_visual_observations=False
        )
        self.input_decoder = InputDecoder()

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
        with open(map_data_path, "wb") as f:
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
                map_bytes = (
                    bytes(map_data) if isinstance(map_data, (list, tuple)) else map_data
                )
            f.write(map_bytes)

        logger.info(f"Saved map data to {map_data_path} ({len(map_data)} bytes)")
        return map_data_path

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
        hor_inputs, jump_inputs = self.input_decoder.decode_inputs(inputs)
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

            # Extract frame data using unified observation system
            frame_data = self.observation_extractor.extract_legacy_frame_data(
                sim, sim.frame, level_id, session_id, npp_level_id, level_name
            )

            # Update timestamp to match expected format
            frame_data["timestamp"] = start_time + (sim.frame * (1.0 / 60.0))

            # Add map data path if available
            if map_data_path and output_path:
                frame_data["map_data_path"] = os.path.relpath(
                    map_data_path, os.path.dirname(output_path)
                )

            frames.append(frame_data)

            # Execute simulation step
            sim.tick(hor_input, jump_input)

            # Check for death or completion (status is now handled in the unified extractor)
            if sim.ninja.state == 6:  # Dead
                break
            elif sim.ninja.state == 8:  # Celebrating (completed)
                break

        logger.info(f"Generated {len(frames)} frames for replay")
        return frames

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
