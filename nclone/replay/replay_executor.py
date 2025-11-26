"""
Deterministic Replay Executor

Replays stored input sequences against map data to regenerate observations.
Leverages the completely deterministic nature of N++ physics simulation.
"""

from typing import Dict, List, Any, Optional
import logging
import numpy as np

from ..nplay_headless import NPlayHeadless
from ..gym_environment.observation_processor import ObservationProcessor
from ..gym_environment.entity_extractor import EntityExtractor
from ..graph.level_data import LevelData, extract_start_position_from_map_data
from ..graph.common import (
    GraphData,
    N_MAX_NODES,
    E_MAX_EDGES,
    NODE_FEATURE_DIM,
)
from ..graph.reachability.graph_builder import GraphBuilder
from ..graph.reachability.path_distance_calculator import CachedPathDistanceCalculator
from ..graph.reachability.feature_computation import (
    compute_reachability_features_from_graph,
)
from ..graph.reachability.subcell_node_lookup import SUB_NODE_SIZE
from ..constants.physics_constants import (
    MAP_TILE_WIDTH,
    MAP_TILE_HEIGHT,
)
from ..gym_environment.constants import (
    LEVEL_DIAGONAL,
    MAX_LOCKED_DOORS,
    FEATURES_PER_DOOR,
    SWITCH_STATES_DIM,
)
from ..constants.physics_constants import LEVEL_WIDTH_PX, LEVEL_HEIGHT_PX


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
        enable_rendering: bool = False,
    ):
        """Initialize replay executor.

        Args:
            observation_config: Configuration for observation processor
            render_mode: Rendering mode for environment
        """
        self.observation_config = observation_config or {}
        self.render_mode = render_mode
        self.enable_rendering = enable_rendering

        # Create headless environment
        self.nplay_headless = NPlayHeadless(
            render_mode=render_mode,
            enable_animation=False,
            enable_logging=False,
            enable_debug_overlay=False,
            seed=42,  # Fixed seed for determinism
            enable_rendering=enable_rendering,
        )

        # Create observation processor
        self.obs_processor = ObservationProcessor(
            enable_augmentation=False,  # No augmentation for replay
            enable_visual_processing=False,
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
        self._cached_switch_states: Dict[
            str, bool
        ] = {}  # Track switch states for graph rebuilding

        # Cache for GraphData objects (for graph observations)
        self._graph_data_cache: Dict[str, Any] = {}
        self._current_graph: Optional[Any] = None

        # Cache for graph observation arrays (avoid regenerating numpy arrays every frame)
        self._graph_obs_cache: Optional[Dict[str, np.ndarray]] = None
        self._graph_obs_cache_level_id: Optional[str] = None

    def _safe_path_distance(
        self,
        start_pos,
        goal_pos,
        adjacency,
        feature_name,
        level_data=None,
        graph_data=None,
        entity_radius=0.0,
    ):
        """Calculate path distance with inf handling and detailed logging.

        Returns safe max value (LEVEL_DIAGONAL * 2.0) for unreachable paths.
        Logs detailed warnings when paths are unreachable for debugging.

        Args:
            start_pos: Starting position tuple (x, y)
            goal_pos: Goal position tuple (x, y)
            adjacency: Graph adjacency structure
            feature_name: Name of feature for logging (e.g., "exit_switch", "mine")
            level_data: Optional level data for caching
            graph_data: Optional graph data for spatial indexing
            entity_radius: Collision radius of the goal entity (default 0.0)

        Returns:
            Path distance in pixels, or LEVEL_DIAGONAL * 2.0 if unreachable
        """
        from ..constants.physics_constants import NINJA_RADIUS

        base_adjacency = (
            graph_data.get("base_adjacency", adjacency) if graph_data else adjacency
        )
        distance = self.path_calculator.get_distance(
            start_pos,
            goal_pos,
            adjacency,
            base_adjacency,
            level_data=level_data,
            graph_data=graph_data,
            entity_radius=entity_radius,
            ninja_radius=NINJA_RADIUS,
        )

        if distance == float("inf"):
            print(
                f"[PATH_DISTANCE] Unreachable {feature_name}: "
                f"start={start_pos}, goal={goal_pos}, "
                f"level={getattr(level_data, 'level_id', 'unknown')}"
            )
            return LEVEL_DIAGONAL * 2.0  # Safe max value

        return distance

    def _get_reachable_area_scale(self) -> float:
        """Get reachable area scale for distance normalization.

        Uses shared cached surface area calculation from pathfinding_utils.
        Avoids duplicate flood fill logic and ensures consistency with other
        components that use the same surface area calculation.

        Returns:
            Distance scale in pixels (sqrt(surface_area) * SUB_NODE_SIZE)

        Raises:
            RuntimeError: If graph/adjacency not available (no fallback)
        """
        from nclone.graph.reachability.pathfinding_utils import (
            get_cached_surface_area,
            NODE_WORLD_COORD_OFFSET,
        )

        # Get level data
        if self._cached_level_data is None:
            raise RuntimeError(
                "Cannot compute reachable area scale: level_data not available."
            )

        level_data = self._cached_level_data

        # Get adjacency graph
        if self._cached_graph_data is None:
            raise RuntimeError(
                "Cannot compute reachable area scale: graph data not available."
            )

        adjacency = self._cached_graph_data.get("adjacency")
        if not adjacency:
            raise RuntimeError(
                "Cannot compute reachable area scale: adjacency graph is empty."
            )

        # Get player start position
        start_position = getattr(level_data, "start_position", None)
        if start_position is None:
            raise RuntimeError(
                "Cannot compute reachable area scale: level_data missing start_position."
            )

        # Update switch states from simulator to ensure cache key is correct
        self._update_level_data_switch_states(level_data)

        # Generate cache key using LevelData utility method for consistency
        cache_key = level_data.get_cache_key_for_reachability(
            include_switch_states=True
        )

        # Convert to world space
        start_pos = (
            int(start_position[0]) + NODE_WORLD_COORD_OFFSET,
            int(start_position[1]) + NODE_WORLD_COORD_OFFSET,
        )

        # Use shared cached surface area calculation
        surface_area = get_cached_surface_area(
            cache_key, start_pos, adjacency, self._cached_graph_data
        )

        # Convert to distance scale
        area_scale = np.sqrt(surface_area) * SUB_NODE_SIZE

        return area_scale

    def _update_level_data_switch_states(self, level_data):
        """Update level_data.switch_states from current simulator state.

        This ensures cache keys include current switch states for proper
        cache invalidation when switches change during replay.

        Args:
            level_data: LevelData instance to update

        Returns:
            bool: True if switch states changed, False otherwise
        """
        # Extract switch states from locked doors
        locked_doors = self.nplay_headless.locked_doors()
        current_states = {}
        for i, locked_door in enumerate(locked_doors):
            # Check if door is open (switch activated)
            is_activated = not getattr(locked_door, "active", True)
            current_states[f"locked_door_{i}"] = is_activated

        # Check if states changed
        states_changed = current_states != self._cached_switch_states

        # Update level_data and cache
        level_data.switch_states = current_states
        self._cached_switch_states = current_states.copy()

        return states_changed

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
        # Clear surface area cache at start of each replay to prevent stale data
        # from previous replays (especially important for BC pretraining)
        from nclone.graph.reachability.pathfinding_utils import clear_surface_area_cache

        clear_surface_area_cache()

        # Store input sequence for time_remaining calculation
        self._current_input_sequence = input_sequence

        # Load map
        self.nplay_headless.load_map_from_map_data(list(map_data))

        # Initialize entity extractor now that map is loaded
        if self.entity_extractor is None:
            self.entity_extractor = EntityExtractor(self.nplay_headless)

        # Extract level data and build graph once per replay
        level_data = self._extract_level_data()
        # Use consistent level_id generation (hash-based) to match _convert_graph_data_to_graphdata
        level_id = f"level_{hash(level_data.tiles.tobytes())}"

        # Only rebuild graph if level changed
        import logging

        logger = logging.getLogger(__name__)

        if level_id != self._cached_level_id or self._cached_graph_data is None:
            # Build graph once for the entire replay
            # Use flood fill filtering from spawn point to constrain graph size
            # Graph building is fast (<0.2ms), and filtering ensures we stay within
            # theoretical max nodes/edges (N_MAX_NODES=4500, E_MAX_EDGES=18500)
            ninja_pos = (
                int(self.nplay_headless.ninja_position()[0]),
                int(self.nplay_headless.ninja_position()[1]),
            )
            logger.info(f"Building graph for level {level_id}, ninja_pos={ninja_pos}")

            try:
                self._cached_graph_data = self.graph_builder.build_graph(
                    level_data,
                    ninja_pos=ninja_pos,
                    filter_by_reachability=True,  # Filter to reachable nodes only
                )
            except Exception as e:
                logger.error(f"Graph building failed for level {level_id}: {e}")
                logger.error(f"  Ninja position: {ninja_pos}")
                logger.error(f"  Level data tiles shape: {level_data.tiles.shape}")
                import traceback

                logger.error(f"  Traceback:\n{traceback.format_exc()}")
                # Create empty graph data as fallback
                self._cached_graph_data = {"adjacency": {}}

            # DEFENSIVE: Log graph build results and validate
            adjacency = self._cached_graph_data.get("adjacency")
            if adjacency:
                num_nodes = len(adjacency)
                # Count total edges
                num_edges = sum(len(neighbors) for neighbors in adjacency.values())
                logger.info(
                    f"Built graph with {num_nodes} nodes, {num_edges} edges for level {level_id}"
                )
                if num_nodes == 0:
                    logger.warning(
                        "  Graph has ZERO nodes - this will cause training issues!"
                    )
                if num_edges > E_MAX_EDGES:
                    logger.warning(
                        f"  Graph has {num_edges} edges, exceeding E_MAX_EDGES={E_MAX_EDGES}"
                    )
            else:
                logger.error(
                    f"Graph builder returned NO adjacency for level {level_id}"
                )
                logger.error(f"  Ninja position: {ninja_pos}")
                logger.error(f"  Level data: {level_data}")
                # Ensure we have an empty adjacency dict
                self._cached_graph_data = {"adjacency": {}}

            self._cached_level_data = level_data
            self._cached_level_id = level_id

            # Initialize switch states cache for new level
            self._cached_switch_states = {}

            # Clear graph observation cache for new level
            self._graph_obs_cache = None
            self._graph_obs_cache_level_id = None

            # Build level cache once
            if adjacency:
                base_adjacency = self._cached_graph_data.get(
                    "base_adjacency", adjacency
                )
                self.path_calculator.build_level_cache(
                    level_data, adjacency, base_adjacency, self._cached_graph_data
                )
        else:
            logger.info(f"Using cached graph for level {level_id}")

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
        - game_state: ninja state (NINJA_STATE_DIM=41 features, enhanced physics + time_remaining)
        - reachability_features: 6-dimensional reachability vector
        - player_won: whether the player has won
        - player_dead: whether the player has died
        """
        # Update level_data switch states BEFORE computing any features
        # CRITICAL: This must happen before any path calculations or cache key generation
        switch_states_changed = False
        if self._cached_level_data is not None:
            switch_states_changed = self._update_level_data_switch_states(
                self._cached_level_data
            )

            # If switch states changed, rebuild the graph to reflect new door states
            # This is critical for path distance calculations to work correctly
            # Use filter_by_reachability=False to avoid filtering by spawn reachability,
            # as the ninja may have moved through doors that closed behind them
            if switch_states_changed:
                ninja_pos = (
                    int(self.nplay_headless.ninja_position()[0]),
                    int(self.nplay_headless.ninja_position()[1]),
                )
                self._cached_graph_data = self.graph_builder.build_graph(
                    self._cached_level_data,
                    ninja_pos=ninja_pos,
                    filter_by_reachability=False,  # Don't filter - ninja may be past closed doors
                )

                # Rebuild level cache with new graph
                adjacency = self._cached_graph_data.get("adjacency")
                if adjacency:
                    base_adjacency = self._cached_graph_data.get(
                        "base_adjacency", adjacency
                    )
                    self.path_calculator.build_level_cache(
                        self._cached_level_data,
                        adjacency,
                        base_adjacency,
                        self._cached_graph_data,
                    )

                # Clear graph observation cache since graph changed
                self._graph_obs_cache = None
                self._graph_obs_cache_level_id = None

        # Get ninja position
        ninja_x, ninja_y = self.nplay_headless.ninja_position()

        # Get entity positions
        switch_x, switch_y = self._get_switch_position()
        exit_door_x, exit_door_y = self._get_exit_door_position()
        switch_activated = self._is_switch_activated()

        # Get ninja state (40D enhanced physics)
        ninja_state = self.nplay_headless.get_ninja_state()
        if not isinstance(ninja_state, np.ndarray):
            ninja_state = np.array(ninja_state, dtype=np.float32)

        # Calculate time_remaining (for replay, we use frame progress as approximation)
        # Note: In replay context, we don't have access to truncation limits,
        # so we use a simple frame-based approximation
        current_frame = self.nplay_headless.sim.frame
        estimated_max_frames = len(getattr(self, "_current_input_sequence", [])) or 2000
        time_remaining = max(
            0.0, (estimated_max_frames - current_frame) / estimated_max_frames
        )

        # Create 41D game_state (40D physics + 1D time_remaining) to match environment
        game_state = np.append(ninja_state[:40], time_remaining).astype(np.float32)

        # Compute reachability features (6-dim)
        reachability_features = self._compute_reachability_features(
            ninja_x, ninja_y, switch_x, switch_y, exit_door_x, exit_door_y
        )

        # Get win/death state
        player_won = self.nplay_headless.ninja_has_won()
        player_dead = self.nplay_headless.sim.ninja.has_died()

        # Generate graph observations
        # Convert cached graph_data to GraphData format for observations
        ninja_pos_int = (int(ninja_x), int(ninja_y))
        try:
            self._current_graph = self._convert_graph_data_to_graphdata(
                self._cached_graph_data, self._cached_level_data, ninja_pos_int
            )
            if self._current_graph is None:
                logger = logging.getLogger(__name__)
                logger.error(
                    f"Graph conversion returned None for level {self._cached_level_id}"
                )
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Graph conversion failed: {e}")
            import traceback

            logger.error(f"Traceback:\n{traceback.format_exc()}")
            self._current_graph = None

        # Get graph observations matching GraphMixin output
        graph_obs = self._get_graph_observations()
        # Get locked doors for switch states
        locked_doors = self.nplay_headless.locked_doors()

        # Build complete raw observation
        obs = {
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
            "locked_doors": locked_doors,
        }

        if self.enable_rendering:
            obs["screen"] = self.nplay_headless.render()

        # Add graph observations to observation dict
        obs.update(graph_obs)

        # Add switch states (for hierarchical PPO and other components)
        obs["switch_states_dict"] = self._get_switch_states_from_env()
        obs["switch_states"] = self._build_switch_states_array(obs)

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

    def _get_switch_states_from_env(self) -> Dict[str, bool]:
        """Get switch states from environment.

        Returns dictionary mapping switch IDs to their activation states.
        Based on BaseNppEnvironment._get_switch_states_from_env.
        """
        switch_states = {}

        # Check locked door switches
        locked_doors = self.nplay_headless.locked_doors()
        for i, locked_door in enumerate(locked_doors):
            # Check if door is open (switch activated)
            is_activated = not getattr(locked_door, "active", True)

            # Store switch state
            switch_states[f"locked_door_{i}"] = is_activated

            # Store switch part state (same as door for locked doors)
            switch_states[f"locked_door_switch_{i}"] = is_activated

        return switch_states

    def _extract_locked_door_positions(self, locked_door) -> tuple:
        """Extract positions from locked door entity.

        Returns (switch_x, switch_y, door_x, door_y, switch_collected).
        Based on NppEnvironment._extract_locked_door_positions.
        """
        # Switch position (entity position)
        switch_x = getattr(locked_door, "xpos", 0.0)
        switch_y = getattr(locked_door, "ypos", 0.0)

        # Door position (segment center)
        segment = getattr(locked_door, "segment", None)
        if segment:
            door_x = (segment.x1 + segment.x2) * 0.5
            door_y = (segment.y1 + segment.y2) * 0.5
        else:
            door_x, door_y = switch_x, switch_y

        # Switch collected (door open) - active=False means collected
        switch_collected = 1.0 if not getattr(locked_door, "active", True) else 0.0

        return switch_x, switch_y, door_x, door_y, switch_collected

    def _build_switch_states_array(self, obs: Dict[str, Any]) -> np.ndarray:
        """Build switch states array with detailed locked door information.

        Format per door (5 features):
        - switch_x_norm: Normalized X position of switch (0-1)
        - switch_y_norm: Normalized Y position of switch (0-1)
        - door_x_norm: Normalized X position of door (0-1)
        - door_y_norm: Normalized Y position of door (0-1)
        - collected: 1.0 if switch collected (door open), 0.0 otherwise (door closed)

        Returns:
            Array of shape (SWITCH_STATES_DIM,) for up to MAX_LOCKED_DOORS

        Based on NppEnvironment._build_switch_states_array.
        """
        switch_states_array = np.zeros(SWITCH_STATES_DIM, dtype=np.float32)
        locked_doors = obs.get("locked_doors", [])

        for i, locked_door in enumerate(locked_doors[:MAX_LOCKED_DOORS]):
            switch_x, switch_y, door_x, door_y, switch_collected = (
                self._extract_locked_door_positions(locked_door)
            )

            # Normalize positions to [0, 1]
            base_idx = i * FEATURES_PER_DOOR
            switch_states_array[base_idx + 0] = np.clip(
                switch_x / LEVEL_WIDTH_PX, 0.0, 1.0
            )
            switch_states_array[base_idx + 1] = np.clip(
                switch_y / LEVEL_HEIGHT_PX, 0.0, 1.0
            )
            switch_states_array[base_idx + 2] = np.clip(
                door_x / LEVEL_WIDTH_PX, 0.0, 1.0
            )
            switch_states_array[base_idx + 3] = np.clip(
                door_y / LEVEL_HEIGHT_PX, 0.0, 1.0
            )
            switch_states_array[base_idx + 4] = switch_collected

        return switch_states_array

    def _get_area_scale(self) -> float:
        """Get area scale for distance normalization.

        Uses the shared _get_reachable_area_scale() method which computes
        proper reachable surface area via flood fill, rather than just
        counting all nodes in adjacency.
        """
        try:
            return self._get_reachable_area_scale()
        except RuntimeError:
            # Fallback to level diagonal if computation fails
            return LEVEL_DIAGONAL

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
        Compute 6-dimensional reachability features using adjacency graph.

        Features (4 base + 2 mine context):
        1. Reachable area ratio (0-1)
        2. Distance to nearest switch (normalized, inverted)
        3. Distance to exit (normalized, inverted)
        4. Exit reachable flag (0-1)
        5. Total mines normalized (0-1)
        6. Deadly mine ratio (0-1)
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

    def _convert_graph_data_to_graphdata(
        self,
        graph_data_dict: Dict[str, Any],
        level_data: LevelData,
        ninja_pos: Optional[tuple],
    ) -> Optional[GraphData]:
        """Convert GraphBuilder dict to GraphData format for ML models.

        This conversion generates graph observations matching what GraphMixin produces.
        Uses the fast GraphBuilder adjacency dict and converts it to the
        GraphData format expected by graph neural networks.

        Args:
            graph_data_dict: Dict from GraphBuilder.build_graph() with 'adjacency' key
            level_data: Level data for feature building
            ninja_pos: Current ninja position for reachability features

        Returns:
            GraphData object or None if conversion fails
        """
        if not graph_data_dict or "adjacency" not in graph_data_dict:
            return None

        adjacency = graph_data_dict.get("adjacency", {})
        if not adjacency:
            return None

        # level_data should already be a LevelData object
        if not isinstance(level_data, LevelData):
            return None

        # Import logger first
        import logging

        logger = logging.getLogger(__name__)

        # Generate level_id from tiles hash (same strategy as GraphBuilder)
        level_id = f"level_{hash(level_data.tiles.tobytes())}"

        # Check cache for base GraphData
        cached_graph_data = self._graph_data_cache.get(level_id)
        if cached_graph_data is not None and level_id == self._cached_level_id:
            # Reuse cached GraphData since level hasn't changed
            logger.debug(f"Cache HIT for level {level_id}")
            return cached_graph_data

        logger.debug(
            f"Cache MISS for level {level_id} (cached_id={self._cached_level_id})"
        )

        # Convert adjacency dict to Edge list
        from ..graph.common import Edge

        edges = []
        for source_pos, neighbors in adjacency.items():
            # DEFENSIVE: Ensure source_pos is a 2-tuple (x, y)
            if not isinstance(source_pos, tuple) or len(source_pos) != 2:
                logger.warning(
                    f"source_pos is not a 2-tuple: {type(source_pos)} = {source_pos}"
                )
                continue

            for neighbor_info in neighbors:
                if isinstance(neighbor_info, tuple):
                    # Check if neighbor_info is (target_pos, weight, edge_type) format
                    # or just (x, y) format
                    if len(neighbor_info) >= 2:
                        first_elem = neighbor_info[0]

                        # Case 1: neighbor_info = ((x, y), ...)
                        if isinstance(first_elem, tuple) and len(first_elem) == 2:
                            target_pos = first_elem
                        # Case 2: neighbor_info = (x, y, ...)
                        elif isinstance(first_elem, (int, float)):
                            target_pos = (neighbor_info[0], neighbor_info[1])
                        else:
                            logger.warning(
                                f"Unexpected neighbor_info format: {neighbor_info}"
                            )
                            continue

                        # Ensure target_pos is a 2-tuple
                        if not isinstance(target_pos, tuple) or len(target_pos) != 2:
                            logger.warning(f"target_pos is not a 2-tuple: {target_pos}")
                            continue

                        edges.append(
                            Edge(
                                source=source_pos,
                                target=target_pos,
                            )
                        )

        # Convert edges to GraphData using existing function
        from ..graph.edge_building import create_graph_data

        try:
            # Build GraphData (will extract entity info and build features)
            graph_data = create_graph_data(edges, level_data)

            # Cache GraphData per level_id
            self._graph_data_cache[level_id] = graph_data

            return graph_data
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Exception in _convert_graph_data_to_graphdata: {e}")
            logger.error(f"  Level ID: {level_id}")
            logger.error(f"  Number of edges: {len(edges)}")
            logger.error(f"  Adjacency keys: {len(adjacency)} nodes")
            import traceback

            logger.error(f"  Traceback: {traceback.format_exc()}")
            return None

    def _get_graph_observations(self) -> Dict[str, np.ndarray]:
        """Get minimal graph observations matching GraphMixin output (GCN-optimized).

        Returns graph observations with only what GCN needs:
        - Dense format: Padded to N_MAX_NODES/E_MAX_EDGES
        - No edge features, no node/edge types (GCN doesn't use them)
        """

        graph_obs = {
            "graph_node_feats": np.zeros(
                (N_MAX_NODES, NODE_FEATURE_DIM), dtype=np.float32
            ),
            "graph_edge_index": np.zeros((2, E_MAX_EDGES), dtype=np.int32),
            "graph_node_mask": np.zeros(N_MAX_NODES, dtype=np.int32),
            "graph_edge_mask": np.zeros(E_MAX_EDGES, dtype=np.int32),
        }

        # Fill with actual graph data if available
        if self._current_graph is not None:
            graph_data = self._current_graph

            # Copy node features (up to max nodes)
            num_nodes = min(graph_data.num_nodes, N_MAX_NODES)
            if (
                hasattr(graph_data, "node_features")
                and graph_data.node_features is not None
            ):
                graph_obs["graph_node_feats"][:num_nodes] = graph_data.node_features[
                    :num_nodes
                ]

            # Copy edge index (up to max edges)
            num_edges = min(graph_data.num_edges, E_MAX_EDGES)
            if hasattr(graph_data, "edge_index") and graph_data.edge_index is not None:
                graph_obs["graph_edge_index"][:, :num_edges] = graph_data.edge_index[
                    :, :num_edges
                ]

            # Set masks
            graph_obs["graph_node_mask"][:num_nodes] = 1
            graph_obs["graph_edge_mask"][:num_edges] = 1

        return graph_obs

    def close(self):
        """Clean up resources."""
        if hasattr(self, "nplay_headless"):
            del self.nplay_headless
