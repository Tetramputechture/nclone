"""
Deterministic Replay Executor

Replays stored input sequences against map data to regenerate observations.
Leverages the completely deterministic nature of N++ physics simulation.
"""

from typing import Dict, List, Any, Optional
import numpy as np

from ..nplay_headless import NPlayHeadless
from ..gym_environment.observation_processor import ObservationProcessor
from ..gym_environment.entity_extractor import EntityExtractor
from ..graph.level_data import LevelData, extract_start_position_from_map_data
from ..graph.reachability.graph_builder import GraphBuilder
from ..graph.reachability.path_distance_calculator import CachedPathDistanceCalculator
from ..graph.reachability.feature_computation import (
    compute_reachability_features_from_graph,
)
from ..constants.physics_constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


def map_input_to_action(input_byte: int) -> int:
    """
    Map N++ input byte (0-7) to discrete action (0-5).

    Inverse of map_action_to_input from gameplay_recorder.py
    """
    input_to_action_map = {
        0: 0,  # 000: NOOP
        1: 3,  # 001: JUMP
        2: 2,  # 010: RIGHT
        3: 5,  # 011: RIGHT+JUMP
        4: 1,  # 100: LEFT
        5: 4,  # 101: LEFT+JUMP
        6: 0,  # 110: Invalid (mapped to NOOP)
        7: 0,  # 111: Invalid (mapped to NOOP)
    }
    return input_to_action_map.get(input_byte, 0)


def decode_input_to_controls(input_byte: int) -> tuple:
    """
    Decode input byte to horizontal and jump controls.

    Returns:
        (horizontal, jump) where:
            horizontal: -1 (left), 0 (none), 1 (right)
            jump: 0 (no jump), 1 (jump)
    """
    jump = 1 if (input_byte & 0x01) else 0
    right = 1 if (input_byte & 0x02) else 0
    left = 1 if (input_byte & 0x04) else 0

    # Handle conflicting inputs
    if left and right:
        horizontal = 0
    elif left:
        horizontal = -1
    elif right:
        horizontal = 1
    else:
        horizontal = 0

    return horizontal, jump


class ReplayExecutor:
    """Executes replays deterministically to generate observations."""

    def __init__(
        self,
        observation_config: Optional[Dict[str, Any]] = None,
        render_mode: str = "grayscale_array",
    ):
        """Initialize replay executor.

        Args:
            observation_config: Configuration for observation processor
            render_mode: Rendering mode for environment
        """
        self.observation_config = observation_config or {}
        self.render_mode = render_mode

        # Create headless environment
        self.nplay_headless = NPlayHeadless(
            render_mode=render_mode,
            enable_animation=False,
            enable_logging=False,
            enable_debug_overlay=False,
            seed=42,  # Fixed seed for determinism
        )

        # Create observation processor
        self.obs_processor = ObservationProcessor(
            enable_augmentation=False,  # No augmentation for replay
        )

        # Initialize graph builder and path calculator for reachability features
        self.graph_builder = GraphBuilder()
        self.path_calculator = CachedPathDistanceCalculator(
            max_cache_size=200, use_astar=True
        )

        # Entity extractor for level data extraction
        self.entity_extractor = None  # Will be initialized when nplay_headless is ready

        # Cache for graph and level data (only rebuild when level changes or entity states change)
        self._cached_level_data = None
        self._cached_graph_data = None
        self._cached_level_id = None

    def execute_replay(
        self,
        map_data: bytes,
        input_sequence: List[int],
    ) -> List[Dict[str, Any]]:
        """Execute a replay and generate observations for each frame.

        Args:
            map_data: Raw map data (1335 bytes)
            input_sequence: Input sequence (1 byte per frame)

        Returns:
            List of observations, one per frame
        """
        # Load map
        self.nplay_headless.load_map_from_map_data(list(map_data))

        # Initialize entity extractor now that map is loaded
        if self.entity_extractor is None:
            self.entity_extractor = EntityExtractor(self.nplay_headless)

        # Extract level data and build graph once per replay
        level_data = self._extract_level_data()
        level_id = getattr(level_data, "level_id", None)

        # Only rebuild graph if level changed
        if level_id != self._cached_level_id or self._cached_graph_data is None:
            # Build graph once for the entire replay
            ninja_pos = (
                int(self.nplay_headless.ninja_position()[0]),
                int(self.nplay_headless.ninja_position()[1]),
            )
            self._cached_graph_data = self.graph_builder.build_graph(
                level_data, ninja_pos=ninja_pos
            )
            self._cached_level_data = level_data
            self._cached_level_id = level_id

            # Build level cache once
            adjacency = self._cached_graph_data.get("adjacency")
            if adjacency:
                self.path_calculator.build_level_cache(
                    level_data, adjacency, self._cached_graph_data
                )

        observations = []

        # Execute each input frame
        for frame_idx, input_byte in enumerate(input_sequence):
            # Decode input to controls
            horizontal, jump = decode_input_to_controls(input_byte)

            # Execute one simulation step
            self.nplay_headless.tick(horizontal, jump)

            # Get raw observation
            raw_obs = self._get_raw_observation()

            # Process observation
            processed_obs = self.obs_processor.process_observation(raw_obs)

            # Get discrete action for this frame
            action = map_input_to_action(input_byte)

            # Store observation with action
            observations.append(
                {
                    "observation": processed_obs,
                    "action": action,
                    "frame": frame_idx,
                }
            )

        return observations

    def _get_raw_observation(self) -> Dict[str, Any]:
        """Get raw observation from environment.

        Returns a complete raw observation dict with all required keys
        that the ObservationProcessor expects.

        Required keys:
        - screen: rendered frame
        - player_x, player_y: ninja position
        - switch_x, switch_y: exit switch position
        - exit_door_x, exit_door_y: exit door position
        - switch_activated: whether switch has been activated
        - game_state: ninja state (GAME_STATE_CHANNELS features)
        - reachability_features: 8-dimensional reachability vector
        - player_won: whether the player has won
        - player_dead: whether the player has died
        """
        # Render current frame
        screen = self.nplay_headless.render()

        # Get ninja position
        ninja_x, ninja_y = self.nplay_headless.ninja_position()

        # Get entity positions
        switch_x, switch_y = self._get_switch_position()
        exit_door_x, exit_door_y = self._get_exit_door_position()
        switch_activated = self._is_switch_activated()

        # Get game state using the standardized method (GAME_STATE_CHANNELS features)
        game_state = self.nplay_headless.get_ninja_state()
        # Convert to numpy array if needed
        if not isinstance(game_state, np.ndarray):
            game_state = np.array(game_state, dtype=np.float32)

        # Compute reachability features
        reachability_features = self._compute_reachability_features(
            ninja_x, ninja_y, switch_x, switch_y, exit_door_x, exit_door_y
        )

        # Get win/death state
        player_won = self.nplay_headless.ninja_has_won()
        player_dead = self.nplay_headless.sim.ninja.has_died()

        # Build complete raw observation
        obs = {
            "screen": screen,
            "player_x": ninja_x,
            "player_y": ninja_y,
            "switch_x": switch_x,
            "switch_y": switch_y,
            "exit_door_x": exit_door_x,
            "exit_door_y": exit_door_y,
            "switch_activated": switch_activated,
            "game_state": game_state,
            "reachability_features": reachability_features,
            "player_won": player_won,
            "player_dead": player_dead,
        }

        return obs

    def _get_switch_position(self) -> tuple:
        """Get exit switch position with fallback."""
        return self.nplay_headless.exit_switch_position()

    def _get_exit_door_position(self) -> tuple:
        """Get exit door position with fallback."""
        return self.nplay_headless.exit_door_position()

    def _is_switch_activated(self) -> bool:
        """Check if exit switch is activated."""
        return self.nplay_headless.exit_switch_activated()

    def _extract_level_data(self) -> LevelData:
        """
        Extract level structure data for graph construction.

        Returns:
            LevelData object containing tiles and entities
        """
        if self.entity_extractor is None:
            raise RuntimeError(
                "Entity extractor not initialized. Map must be loaded first."
            )

        # Build level tiles as a compact 2D array of inner playable area [23 x 42]
        tile_dic = self.nplay_headless.get_tile_data()
        tiles = np.zeros((MAP_TILE_HEIGHT, MAP_TILE_WIDTH), dtype=np.int32)
        # Simulator tiles include a 1-tile border; map inner (1..42, 1..23) -> (0..41, 0..22)
        for (x, y), tile_id in tile_dic.items():
            inner_x = x - 1
            inner_y = y - 1
            if 0 <= inner_x < MAP_TILE_WIDTH and 0 <= inner_y < MAP_TILE_HEIGHT:
                tiles[inner_y, inner_x] = int(tile_id)

        # Extract entities
        entities = self.entity_extractor.extract_graph_entities()

        # Extract ninja spawn position from map_data
        start_position = extract_start_position_from_map_data(
            self.nplay_headless.sim.map_data
        )

        return LevelData(
            start_position=start_position,
            tiles=tiles,
            entities=entities,
        )

    def _compute_reachability_features(
        self,
        ninja_x: float,
        ninja_y: float,
        switch_x: float,
        switch_y: float,
        exit_door_x: float,
        exit_door_y: float,
    ) -> np.ndarray:
        """
        Compute 8-dimensional reachability features using adjacency graph.

        Features:
        1. Reachable area ratio (0-1)
        2. Distance to nearest switch (normalized)
        3. Distance to exit (normalized)
        4. Reachable switches count (normalized)
        5. Reachable hazards count (normalized)
        6. Connectivity score (0-1)
        7. Exit reachable flag (0-1)
        8. Switch-to-exit path exists (0-1)
        """
        # Use cached graph and level data (built once per replay)
        if self._cached_graph_data is None or self._cached_level_data is None:
            raise RuntimeError(
                "Graph and level data not cached. Call execute_replay() first."
            )

        # Get current ninja position
        ninja_pos = (int(ninja_x), int(ninja_y))

        # Extract adjacency from cached graph
        adjacency = self._cached_graph_data.get("adjacency")
        if adjacency is None:
            raise RuntimeError("adjacency not found in cached graph_data")

        # Compute features using shared function with cached data
        features = compute_reachability_features_from_graph(
            adjacency,
            self._cached_graph_data,
            self._cached_level_data,
            ninja_pos,
            self.path_calculator,
        )

        return features

    def close(self):
        """Clean up resources."""
        if hasattr(self, "nplay_headless"):
            del self.nplay_headless
