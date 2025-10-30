#!/usr/bin/env python3
"""
Unified observation extraction system for N++ replay parsing.

This module integrates the NppEnvironment observation system into replay parsing
to ensure consistency between training data and replay data. It eliminates
redundant observation building code by reusing the standardized observation
processing pipeline.
"""

import logging
import numpy as np
import time
from typing import Dict, Any, Optional

from ..gym_environment.observation_processor import ObservationProcessor
from ..gym_environment.constants import MAX_TIME_IN_FRAMES
from ..nplay_headless import NPlayHeadless
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from ..constants.entity_types import EntityType
from ..constants.physics_constants import NINJA_RADIUS
from ..graph.level_data import LevelData, extract_start_position_from_map_data
from ..graph.reachability.feature_extractor import ReachabilityFeatureExtractor

logger = logging.getLogger(__name__)


class UnifiedObservationExtractor:
    """
    Unified observation extractor that uses NppEnvironment observation system.

    This class bridges the gap between replay simulation and training observation
    formats by reusing the standardized observation processing pipeline from
    NppEnvironment. This ensures consistency and eliminates code duplication.
    """

    def __init__(self, enable_visual_observations: bool = False):
        """
        Initialize the unified observation extractor.

        Args:
            enable_visual_observations: Whether to extract visual observations (player frames, global view)
                                      Set to False for replay parsing to save computation time
        """
        self.enable_visual_observations = enable_visual_observations
        self.observation_processor = ObservationProcessor(enable_augmentation=False)

        # Create a minimal NPlayHeadless instance for observation compatibility
        # We'll use this to leverage the existing observation extraction methods
        self._nplay_headless = None

        # Initialize reachability feature extractor
        self._reachability_extractor = ReachabilityFeatureExtractor()
        self._reachability_cache = {}
        self._reachability_cache_ttl = 0.1  # 100ms cache TTL
        self._last_reachability_time = 0

    def _create_nplay_headless_wrapper(self, sim) -> NPlayHeadless:
        """
        Create a minimal NPlayHeadless wrapper around the simulator.

        This allows us to reuse the existing observation extraction methods
        without duplicating the logic.

        Args:
            sim: Simulator instance

        Returns:
            NPlayHeadless wrapper instance
        """
        if self._nplay_headless is None:
            # Create headless instance with minimal configuration for observation extraction
            self._nplay_headless = NPlayHeadless(
                render_mode="grayscale_array",
                enable_animation=False,
                enable_logging=False,
                enable_debug_overlay=False,
            )

        # Replace the internal simulator with our replay simulator
        self._nplay_headless.sim = sim

        return self._nplay_headless

    def _extract_level_data(self, sim) -> LevelData:
        """
        Extract level structure data for graph construction.

        This replicates the logic from NppEnvironment._extract_level_data()
        to ensure consistency in reachability feature extraction.

        Args:
            sim: Simulator instance

        Returns:
            LevelData object containing tiles and entities
        """
        # Build level tiles as a compact 2D array of inner playable area [23 x 42]
        tile_dic = sim.tile_dic
        tiles = np.zeros((MAP_TILE_HEIGHT, MAP_TILE_WIDTH), dtype=np.int32)
        # Simulator tiles include a 1-tile border; map inner (1..42, 1..23) -> (0..41, 0..22)
        for (x, y), tile_id in tile_dic.items():
            inner_x = x - 1
            inner_y = y - 1
            if 0 <= inner_x < MAP_TILE_WIDTH and 0 <= inner_y < MAP_TILE_HEIGHT:
                tiles[inner_y, inner_x] = int(tile_id)

        # Extract entities
        entities = self._extract_graph_entities(sim)

        # Extract ninja spawn position from map_data
        start_position = extract_start_position_from_map_data(sim.map_data)

        return LevelData(
            start_position=start_position,
            tiles=tiles,
            entities=entities,
            level_id=f"level_{getattr(sim, 'frame', 0)}",
        )

    def _extract_graph_entities(self, sim) -> list:
        """
        Extract entities for graph construction.

        This replicates the logic from NppEnvironment._extract_graph_entities()
        to ensure consistency in reachability feature extraction.

        Args:
            sim: Simulator instance

        Returns:
            List of entity dictionaries with type, position, and state
        """
        entities = []

        # Add ninja as a special entity node
        ninja_pos = (sim.ninja.xpos, sim.ninja.ypos)
        entities.append(
            {
                "type": EntityType.NINJA,
                "radius": NINJA_RADIUS,
                "x": ninja_pos[0],
                "y": ninja_pos[1],
                "active": True,
                "state": 0.0,
                "entity_id": "ninja",
            }
        )

        # Extract entities from simulator entity dictionary
        # This is a simplified version - could be enhanced with full NppEnvironment logic
        for entity_type, entity_list in sim.entity_dic.items():
            for i, entity in enumerate(entity_list):
                entity_data = {
                    "type": entity_type,
                    "x": float(entity.xpos),
                    "y": float(entity.ypos),
                    "active": getattr(entity, "active", True),
                    "state": getattr(entity, "state", 0.0),
                    "entity_id": f"{self._get_entity_type_name(entity_type)}_{i}",
                }

                # Add radius if available
                if hasattr(entity, "radius"):
                    entity_data["radius"] = entity.radius
                elif hasattr(entity, "RADIUS"):
                    entity_data["radius"] = entity.RADIUS
                else:
                    entity_data["radius"] = 10.0  # Default radius

                entities.append(entity_data)

        return entities

    def _get_reachability_features(self, sim, frame_number: int) -> np.ndarray:
        """
        Extract compact reachability features from current game state.

        This replicates the logic from NppEnvironment._get_reachability_features()
        to ensure consistency with training observations.

        Args:
            sim: Simulator instance
            frame_number: Current frame number

        Returns:
            8-dimensional float32 feature vector

        Raises:
            RuntimeError: If reachability feature extraction fails
            ValueError: If features are None or have wrong shape
            TypeError: If features are not a numpy array
        """
        current_time = time.time()

        # Check cache
        ninja_pos = (sim.ninja.xpos, sim.ninja.ypos)
        cache_key = (ninja_pos, sim.frame)

        if (
            cache_key in self._reachability_cache
            and current_time - self._last_reachability_time
            < self._reachability_cache_ttl
        ):
            return self._reachability_cache[cache_key]

        # Extract current game state
        level_data = self._extract_level_data(sim)
        entities = self._extract_graph_entities(sim)

        try:
            # Extract reachability features
            features = self._reachability_extractor.extract_features(
                ninja_position=ninja_pos,
                level_data=level_data,
                entities=entities,
            )
        except Exception as e:
            logger.error(f"Failed to extract reachability features: {e}")
            logger.error(f"Ninja position: {ninja_pos}")
            logger.error(
                f"Level data: tiles shape {level_data.tiles.shape}, {len(entities)} entities"
            )
            raise RuntimeError(f"Reachability feature extraction failed: {e}") from e

        # Validate features
        if features is None:
            raise ValueError("Reachability feature extractor returned None")

        if not isinstance(features, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(features)}")

        if features.shape != (8,):
            raise ValueError(f"Expected shape (8,), got {features.shape}")

        if features.dtype != np.float32:
            logger.warning(f"Converting features from {features.dtype} to float32")
            features = features.astype(np.float32)

        # Cache the result
        self._reachability_cache[cache_key] = features
        self._last_reachability_time = current_time

        # Limit cache size
        if len(self._reachability_cache) > 100:
            # Remove oldest entries
            oldest_keys = list(self._reachability_cache.keys())[:-50]
            for key in oldest_keys:
                del self._reachability_cache[key]

        return features

    def extract_raw_observation(self, sim, frame_number: int) -> Dict[str, Any]:
        """
        Extract raw observation data from simulator using NppEnvironment methods.

        This method reuses the observation extraction logic from NppEnvironment._get_observation()
        to ensure consistency between training and replay data.

        Args:
            sim: Simulator instance
            frame_number: Current frame number

        Returns:
            Raw observation dictionary matching NppEnvironment format

        Raises:
            RuntimeError: If reachability feature extraction fails
            ValueError: If reachability features are invalid
            TypeError: If reachability features have wrong type
        """
        # Create wrapper to access NPlayHeadless observation methods
        nplay_wrapper = self._create_nplay_headless_wrapper(sim)

        # Calculate time remaining feature (matching NppEnvironment logic)
        time_remaining = (MAX_TIME_IN_FRAMES - sim.frame) / MAX_TIME_IN_FRAMES

        # Extract ninja state using the standardized method
        ninja_state = nplay_wrapper.get_ninja_state()

        # Extract entity states using the standardized method
        entity_states = nplay_wrapper.get_entity_states()

        # Combine ninja and entity states (matching NppEnvironment format)
        game_state = np.concatenate([ninja_state, entity_states])

        # Build raw observation dictionary matching NppEnvironment._get_observation()
        obs = {
            "game_state": game_state,
            "player_dead": nplay_wrapper.ninja_has_died(),
            "player_won": nplay_wrapper.ninja_has_won(),
            "player_x": nplay_wrapper.ninja_position()[0],
            "player_y": nplay_wrapper.ninja_position()[1],
            "switch_activated": nplay_wrapper.exit_switch_activated(),
            "switch_x": nplay_wrapper.exit_switch_position()[0],
            "switch_y": nplay_wrapper.exit_switch_position()[1],
            "exit_door_x": nplay_wrapper.exit_door_position()[0],
            "exit_door_y": nplay_wrapper.exit_door_position()[1],
            "time_remaining": time_remaining,
            "sim_frame": sim.frame,
            "doors_opened": nplay_wrapper.get_doors_opened(),
            "entity_states": entity_states,  # Raw entity states for PBRS
            "reachability_features": self._get_reachability_features(sim, frame_number),
        }

        # Add visual observations if enabled
        if self.enable_visual_observations:
            # Render the current frame using the wrapper
            obs["screen"] = nplay_wrapper.render()
        else:
            # Provide a minimal placeholder screen for compatibility
            obs["screen"] = np.zeros((600, 1056, 1), dtype=np.uint8)  # Level dimensions

        return obs

    def extract_processed_observation(self, sim, frame_number: int) -> Dict[str, Any]:
        """
        Extract fully processed observation data ready for training.

        This method combines raw observation extraction with the standardized
        observation processing pipeline from ObservationProcessor.

        Args:
            sim: Simulator instance
            frame_number: Current frame number

        Returns:
            Processed observation dictionary matching NppEnvironment training format
        """
        # Extract raw observation data
        raw_obs = self.extract_raw_observation(sim, frame_number)

        # Process observation using the standardized processor
        if self.enable_visual_observations:
            processed_obs = self.observation_processor.process_observation(raw_obs)
        else:
            # For replay parsing, we only need the processed game state and reachability features
            processed_obs = {
                "game_state": self.observation_processor.process_game_state(raw_obs),
                "reachability_features": raw_obs["reachability_features"],
                # Provide minimal placeholders for visual observations
                "player_frame": np.zeros(
                    (128, 128, 3), dtype=np.uint8
                ),  # Player frame placeholder
                "global_view": np.zeros(
                    (100, 176, 1), dtype=np.uint8
                ),  # Global view placeholder
            }

        return processed_obs

    def extract_legacy_frame_data(
        self,
        sim,
        frame_number: int,
        level_id: str,
        session_id: str,
        npp_level_id: Optional[int] = None,
        level_name: str = "",
    ) -> Dict[str, Any]:
        """
        Extract frame data in the legacy JSONL format for backward compatibility.

        This method maintains compatibility with existing replay parsing output format
        while using the unified observation system internally.

        Args:
            sim: Simulator instance
            frame_number: Current frame number
            level_id: Level identifier
            session_id: Session identifier
            npp_level_id: N++ Level ID (optional)
            level_name: N++ Level name (optional)

        Returns:
            Frame data dictionary in legacy JSONL format
        """
        # Extract raw observation using unified system
        raw_obs = self.extract_raw_observation(sim, frame_number)

        # Extract ninja state from the processed game state (first 26 features after redundancy removal)
        ninja_state = (
            raw_obs["game_state"][:26]
            if len(raw_obs["game_state"]) >= 26
            else raw_obs["game_state"]
        )

        # Build legacy ninja state dictionary for backward compatibility
        ninja = sim.ninja
        legacy_ninja_state = {
            "position": {"x": float(ninja.xpos), "y": float(ninja.ypos)},
            "velocity": {"x": float(ninja.xspeed), "y": float(ninja.yspeed)},
            "state": int(ninja.state),
            "airborne": bool(ninja.airborn),
            "walled": bool(ninja.walled),
            "jump_duration": int(ninja.jump_duration),
            "applied_gravity": float(ninja.applied_gravity),
            "applied_drag": float(ninja.applied_drag),
            "applied_friction": float(ninja.applied_friction),
            "floor_normal": {
                "x": float(ninja.floor_normalized_x),
                "y": float(ninja.floor_normalized_y),
            },
            "ceiling_normal": {
                "x": float(ninja.ceiling_normalized_x),
                "y": float(ninja.ceiling_normalized_y),
            },
            "buffers": {
                "jump": int(ninja.jump_buffer),
                "floor": int(ninja.floor_buffer),
                "wall": int(ninja.wall_buffer),
                "launch_pad": int(ninja.launch_pad_buffer),
            },
            "contact_counts": {
                "floor": int(ninja.floor_count),
                "wall": int(ninja.wall_count),
                "ceiling": int(ninja.ceiling_count),
            },
            "game_state": ninja_state.tolist(),  # 26-feature array for npp-rl compatibility
            "doors_opened": int(ninja.doors_opened),
        }

        # Extract entity data using existing logic (could be enhanced later)
        entities = []
        for entity_type, entity_list in sim.entity_dic.items():
            for entity in entity_list:
                entity_data = {
                    "type": self._get_entity_type_name(entity_type),
                    "position": {"x": float(entity.xpos), "y": float(entity.ypos)},
                    "active": True,  # Default to active
                }

                # Add type-specific data
                if hasattr(entity, "state"):
                    entity_data["state"] = entity.state
                if hasattr(entity, "radius"):
                    entity_data["radius"] = entity.radius

                entities.append(entity_data)

        # Build legacy frame data structure
        frame_data = {
            "timestamp": frame_number * (1.0 / 60.0),  # 60 FPS
            "level_id": level_id,
            "frame_number": frame_number,
            "player_state": legacy_ninja_state,
            "player_inputs": {
                "left": ninja.hor_input == -1,
                "right": ninja.hor_input == 1,
                "jump": ninja.jump_input == 1,
                "restart": False,
            },
            "entities": entities,
            "level_bounds": {
                "width": 1056,  # Fixed level dimensions
                "height": 600,
            },
            "meta": {
                "session_id": session_id,
                "player_id": "binary_replay",
                "quality_score": 0.8,  # Default quality score
                "completion_status": "in_progress",
                "npp_level_id": npp_level_id,
                "npp_level_name": level_name,
            },
        }

        # Update completion status based on ninja state
        if ninja.state == 6:  # Dead
            frame_data["meta"]["completion_status"] = "failed"
        elif ninja.state == 8:  # Celebrating (completed)
            frame_data["meta"]["completion_status"] = "completed"
            frame_data["meta"]["quality_score"] = 0.9

        return frame_data

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

    def reset(self) -> None:
        """Reset the observation processor state and clear caches."""
        self.observation_processor.reset()
        self._reachability_cache.clear()
        self._last_reachability_time = 0
