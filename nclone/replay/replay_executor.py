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
from ..graph.common import (
    GraphData,
    N_MAX_NODES,
    E_MAX_EDGES,
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
)
from ..graph.reachability.graph_builder import GraphBuilder
from ..graph.reachability.path_distance_calculator import CachedPathDistanceCalculator
from ..graph.reachability.feature_computation import (
    compute_reachability_features_from_graph,
)
from ..graph.reachability.subcell_node_lookup import SUB_NODE_SIZE
from ..graph.reachability.pathfinding_utils import (
    find_closest_node_to_position,
    extract_spatial_lookups_from_graph_data,
    NODE_WORLD_COORD_OFFSET,
)
from ..constants.physics_constants import (
    MAP_TILE_WIDTH,
    MAP_TILE_HEIGHT,
    LEVEL_WIDTH_PX,
    LEVEL_HEIGHT_PX,
    NINJA_RADIUS,
)
from collections import deque
from ..constants.entity_types import EntityType
from ..gym_environment.constants import (
    LEVEL_DIAGONAL,
    NINJA_STATE_DIM,
    PATH_AWARE_OBJECTIVES_DIM,
    MINE_FEATURES_DIM,
    PROGRESS_FEATURES_DIM,
    SEQUENTIAL_GOAL_DIM,
    ACTION_DEATH_PROBABILITIES_DIM,
    GAME_STATE_CHANNELS,
    ENTITY_POSITIONS_DIM,
)


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

        # Cache for GraphData objects (for graph observations)
        self._graph_data_cache: Dict[str, Any] = {}
        self._current_graph: Optional[Any] = None

        # Cache for reachable area scale (per level ID)
        self._reachable_area_scale_cache: Dict[str, float] = {}

        # Pre-allocate observation buffers for 52-feature game_state
        self._game_state_buffer = np.zeros(GAME_STATE_CHANNELS, dtype=np.float32)
        self._path_aware_objectives_buffer = np.zeros(
            PATH_AWARE_OBJECTIVES_DIM, dtype=np.float32
        )
        self._mine_features_buffer = np.zeros(MINE_FEATURES_DIM, dtype=np.float32)
        self._progress_features_buffer = np.zeros(
            PROGRESS_FEATURES_DIM, dtype=np.float32
        )

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

        distance = self.path_calculator.get_distance(
            start_pos,
            goal_pos,
            adjacency,
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

        Computes reachable surface area using flood-fill from start position,
        then converts to distance scale: sqrt(surface_area) * SUB_NODE_SIZE.
        Caches result per level ID to avoid repeated computations.

        Returns:
            Distance scale in pixels (sqrt(surface_area) * SUB_NODE_SIZE)

        Raises:
            RuntimeError: If graph/adjacency not available (no fallback)
        """
        # Get level data
        if self._cached_level_data is None:
            raise RuntimeError(
                "Cannot compute reachable area scale: level_data not available."
            )

        level_data = self._cached_level_data

        # Generate level ID
        level_id = None
        if hasattr(level_data, "map_data_hash"):
            level_id = str(level_data.map_data_hash)
        elif hasattr(level_data, "id"):
            level_id = str(level_data.id)
        elif hasattr(level_data, "map_data"):
            import hashlib

            map_bytes = (
                bytes(level_data.map_data)
                if hasattr(level_data.map_data, "__iter__")
                else b""
            )
            level_id = hashlib.md5(map_bytes).hexdigest()
        else:
            level_id = f"level_{id(level_data)}"

        # Check cache
        if level_id in self._reachable_area_scale_cache:
            return self._reachable_area_scale_cache[level_id]

        # Get adjacency graph
        if self._cached_graph_data is None:
            raise RuntimeError(
                "Cannot compute reachable area scale: graph data not available. "
                "Graph must be built before computing reachable area scale."
            )

        adjacency = self._cached_graph_data.get("adjacency")
        if not adjacency:
            raise RuntimeError(
                "Cannot compute reachable area scale: adjacency graph is empty. "
                "Graph building must succeed before computing reachable area scale."
            )

        # Get player start position from level_data
        start_position = getattr(level_data, "start_position", None)
        if start_position is None:
            raise RuntimeError(
                "Cannot compute reachable area scale: level_data missing start_position. "
                "Surface area calculation requires player start position."
            )

        # Convert start position to integer tuple (world space)
        start_pos = (
            int(start_position[0]) + NODE_WORLD_COORD_OFFSET,
            int(start_position[1]) + NODE_WORLD_COORD_OFFSET,
        )

        # Extract spatial lookups for O(1) node finding
        spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(
            self._cached_graph_data
        )

        # Find closest node to start position
        PLAYER_RADIUS = NINJA_RADIUS
        closest_node = find_closest_node_to_position(
            start_pos,
            adjacency,
            threshold=PLAYER_RADIUS,
            spatial_hash=spatial_hash,
            subcell_lookup=subcell_lookup,
        )

        # If no node found within threshold, try with larger threshold
        if closest_node is None:
            closest_node = find_closest_node_to_position(
                start_pos,
                adjacency,
                threshold=50.0,  # Relaxed threshold
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
            )

        if closest_node is None or closest_node not in adjacency:
            raise RuntimeError(
                "Cannot compute reachable area scale: no node found near start position. "
                "Graph builder may have failed to find start position."
            )

        # Flood fill from start node to find all reachable nodes
        reachable_nodes = set()
        queue = deque([closest_node])
        visited = set([closest_node])

        while queue:
            current = queue.popleft()
            reachable_nodes.add(current)

            # Get neighbors from adjacency
            neighbors = adjacency.get(current, [])
            for neighbor_info in neighbors:
                if isinstance(neighbor_info, tuple) and len(neighbor_info) >= 2:
                    neighbor_pos = neighbor_info[0]
                    if neighbor_pos not in visited:
                        visited.add(neighbor_pos)
                        queue.append(neighbor_pos)

        total_reachable_nodes = len(reachable_nodes)

        if total_reachable_nodes == 0:
            raise RuntimeError(
                "Cannot compute reachable area scale: no nodes reachable from start position. "
                "Level may have no traversable space from spawn."
            )

        # Convert surface area (number of nodes) to distance scale
        surface_area = float(total_reachable_nodes)
        area_scale = np.sqrt(surface_area) * SUB_NODE_SIZE

        # Cache result
        self._reachable_area_scale_cache[level_id] = area_scale

        return area_scale

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

            # Initialize mine death predictor after graph is built
            self._build_mine_death_predictor()

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

        # Build entity_positions array: [ninja_x, ninja_y, switch_x, switch_y, exit_x, exit_y]
        from ..gym_environment.constants import LEVEL_WIDTH, LEVEL_HEIGHT

        entity_positions = np.zeros(ENTITY_POSITIONS_DIM, dtype=np.float32)
        entity_positions[0] = ninja_x / LEVEL_WIDTH
        entity_positions[1] = ninja_y / LEVEL_HEIGHT
        entity_positions[2] = switch_x / LEVEL_WIDTH if switch_x > 0 else 0.0
        entity_positions[3] = switch_y / LEVEL_HEIGHT if switch_y > 0 else 0.0
        entity_positions[4] = exit_door_x / LEVEL_WIDTH if exit_door_x > 0 else 0.0
        entity_positions[5] = exit_door_y / LEVEL_HEIGHT if exit_door_y > 0 else 0.0

        # Extract entities for feature computation
        entities = []
        locked_doors = []
        if hasattr(self.nplay_headless.sim, "entity_dic"):
            toggle_mines = self.nplay_headless.sim.entity_dic.get(
                EntityType.TOGGLE_MINE, []
            )
            entities.extend(toggle_mines)

            toggled_mines = self.nplay_headless.sim.entity_dic.get(
                EntityType.TOGGLE_MINE_TOGGLED, []
            )
            entities.extend(toggled_mines)

            locked_doors = self.nplay_headless.locked_doors()

        # Get base game state (29 features)
        base_game_state = self.nplay_headless.get_ninja_state()
        if not isinstance(base_game_state, np.ndarray):
            base_game_state = np.array(base_game_state, dtype=np.float32)

        # Extend game_state to 52 features
        self._game_state_buffer[:NINJA_STATE_DIM] = base_game_state[:NINJA_STATE_DIM]

        # Verify graph and path_calculator are available
        if self._cached_graph_data is None or self._cached_level_data is None:
            raise RuntimeError(
                "Graph and level data not cached. Call execute_replay() first."
            )
        if self.path_calculator is None:
            raise RuntimeError("Path calculator not initialized.")

        # Build observation dict for feature computation
        obs_for_features = {
            "entity_positions": entity_positions,
            "entities": entities,
            "locked_doors": locked_doors,
            "switch_activated": switch_activated,
        }

        # Compute additional features
        self._compute_path_aware_objectives(
            obs_for_features, self._path_aware_objectives_buffer
        )
        self._extract_mine_features(obs_for_features, self._mine_features_buffer)
        self._compute_progress_features(
            obs_for_features, self._progress_features_buffer
        )

        # Concatenate features into game_state_buffer
        idx = NINJA_STATE_DIM
        self._game_state_buffer[idx : idx + PATH_AWARE_OBJECTIVES_DIM] = (
            self._path_aware_objectives_buffer
        )
        idx += PATH_AWARE_OBJECTIVES_DIM
        self._game_state_buffer[idx : idx + MINE_FEATURES_DIM] = (
            self._mine_features_buffer
        )
        idx += MINE_FEATURES_DIM
        self._game_state_buffer[idx : idx + PROGRESS_FEATURES_DIM] = (
            self._progress_features_buffer
        )
        idx += PROGRESS_FEATURES_DIM

        # NEW: Sequential goal features
        if hasattr(self.nplay_headless, "get_sequential_goal_features"):
            seq_features = self.nplay_headless.get_sequential_goal_features()
            self._game_state_buffer[idx : idx + SEQUENTIAL_GOAL_DIM] = seq_features
        else:
            self._game_state_buffer[idx : idx + SEQUENTIAL_GOAL_DIM] = 0.0
        idx += SEQUENTIAL_GOAL_DIM

        ninja = self.nplay_headless.sim.ninja
        if (
            hasattr(ninja, "mine_death_predictor")
            and ninja.mine_death_predictor is not None
        ):
            death_prob_result = ninja.mine_death_predictor.calculate_death_probability(
                frames_to_simulate=10
            )
            self._game_state_buffer[idx : idx + ACTION_DEATH_PROBABILITIES_DIM] = (
                death_prob_result.action_death_probs
            )
        else:
            print("No mine death predictor found")
            self._game_state_buffer[idx : idx + ACTION_DEATH_PROBABILITIES_DIM] = 0.0

        # Get final 64-feature game_state (was 58)
        game_state = np.array(self._game_state_buffer, copy=True)

        # Compute reachability features
        reachability_features = self._compute_reachability_features(
            ninja_x, ninja_y, switch_x, switch_y, exit_door_x, exit_door_y
        )

        # Get win/death state
        player_won = self.nplay_headless.ninja_has_won()
        player_dead = self.nplay_headless.sim.ninja.has_died()

        # Generate graph observations
        # Convert cached graph_data to GraphData format for observations
        ninja_pos_int = (int(ninja_x), int(ninja_y))
        self._current_graph = self._convert_graph_data_to_graphdata(
            self._cached_graph_data, self._cached_level_data, ninja_pos_int
        )

        # Get graph observations matching GraphMixin output
        graph_obs = self._get_graph_observations()

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
            "entity_positions": entity_positions,
            "entities": entities,
            "locked_doors": locked_doors,
        }

        # Add graph observations to observation dict
        obs.update(graph_obs)

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

    def _compute_path_aware_objectives(
        self, obs: Dict[str, Any], buffer: np.ndarray
    ) -> np.ndarray:
        """
        Compute path-aware objective features using graph-based pathfinding.

        Returns PATH_AWARE_OBJECTIVES_DIM (15) features:
        - Exit switch (4): collected, rel_x, rel_y, path_distance
        - Exit door (3): rel_x, rel_y, path_distance
        - Nearest locked door (8): present, switch_collected, switch_rel_x, switch_rel_y,
          switch_path_distance, door_rel_x, door_rel_y, door_path_distance
        """
        features = buffer
        features.fill(0.0)

        adjacency = self._cached_graph_data.get("adjacency")
        if adjacency is None or len(adjacency) == 0:
            return features

        entity_positions = obs.get(
            "entity_positions", np.zeros(ENTITY_POSITIONS_DIM, dtype=np.float32)
        )
        ninja_pos = (
            int(entity_positions[0] * LEVEL_WIDTH_PX),
            int(entity_positions[1] * LEVEL_HEIGHT_PX),
        )

        # Exit switch [0-3]
        from ..constants.physics_constants import EXIT_SWITCH_RADIUS

        exit_switch_collected = 1.0 if obs.get("switch_activated", False) else 0.0
        exit_switch_pos = (
            int(entity_positions[2] * LEVEL_WIDTH_PX),
            int(entity_positions[3] * LEVEL_HEIGHT_PX),
        )
        rel_switch_x = (exit_switch_pos[0] - ninja_pos[0]) / (LEVEL_WIDTH_PX / 2)
        rel_switch_y = (exit_switch_pos[1] - ninja_pos[1]) / (LEVEL_HEIGHT_PX / 2)
        switch_path_dist = self._safe_path_distance(
            ninja_pos,
            exit_switch_pos,
            adjacency,
            "exit_switch",
            level_data=self._cached_level_data,
            graph_data=self._cached_graph_data,
            entity_radius=EXIT_SWITCH_RADIUS,
        )
        features[0] = exit_switch_collected
        features[1] = np.clip(rel_switch_x, -1.0, 1.0)
        features[2] = np.clip(rel_switch_y, -1.0, 1.0)
        area_scale = self._get_reachable_area_scale()
        features[3] = np.clip(switch_path_dist / area_scale, 0.0, 1.0)

        # Exit door [4-6]
        from ..constants.physics_constants import EXIT_DOOR_RADIUS

        exit_door_pos = (
            int(entity_positions[4] * LEVEL_WIDTH_PX),
            int(entity_positions[5] * LEVEL_HEIGHT_PX),
        )
        rel_door_x = (exit_door_pos[0] - ninja_pos[0]) / (LEVEL_WIDTH_PX / 2)
        rel_door_y = (exit_door_pos[1] - ninja_pos[1]) / (LEVEL_HEIGHT_PX / 2)
        door_path_dist = self._safe_path_distance(
            ninja_pos,
            exit_door_pos,
            adjacency,
            "exit_door",
            level_data=self._cached_level_data,
            graph_data=self._cached_graph_data,
            entity_radius=EXIT_DOOR_RADIUS,
        )
        features[4] = np.clip(rel_door_x, -1.0, 1.0)
        features[5] = np.clip(rel_door_y, -1.0, 1.0)
        features[6] = np.clip(door_path_dist / area_scale, 0.0, 1.0)

        # Nearest locked door [7-14]
        locked_doors = obs.get("locked_doors", [])
        if locked_doors:
            nearest_door = None
            nearest_door_dist = float("inf")

            for door in locked_doors:
                if getattr(door, "active", True):
                    door_segment = getattr(door, "segment", None)
                    if door_segment and hasattr(door_segment, "p1"):
                        door_x = (door_segment.p1[0] + door_segment.p2[0]) / 2.0
                        door_y = (door_segment.p1[1] + door_segment.p2[1]) / 2.0
                    else:
                        door_x = getattr(door, "xpos", 0.0)
                        door_y = getattr(door, "ypos", 0.0)

                    euclidean_dist = np.sqrt(
                        (door_x - ninja_pos[0]) ** 2 + (door_y - ninja_pos[1]) ** 2
                    )
                    if euclidean_dist < nearest_door_dist:
                        nearest_door_dist = euclidean_dist
                        nearest_door = door

            if nearest_door is not None:
                features[7] = 1.0
                features[8] = 0.0 if getattr(nearest_door, "active", True) else 1.0

                switch_x = getattr(
                    nearest_door, "sw_xpos", getattr(nearest_door, "xpos", 0.0)
                )
                switch_y = getattr(
                    nearest_door, "sw_ypos", getattr(nearest_door, "ypos", 0.0)
                )
                from ..constants.physics_constants import LOCKED_DOOR_SWITCH_RADIUS

                switch_pos = (int(switch_x), int(switch_y))
                rel_switch_x = (switch_pos[0] - ninja_pos[0]) / (LEVEL_WIDTH_PX / 2)
                rel_switch_y = (switch_pos[1] - ninja_pos[1]) / (LEVEL_HEIGHT_PX / 2)
                switch_path_dist = self._safe_path_distance(
                    ninja_pos,
                    switch_pos,
                    adjacency,
                    "locked_door_switch",
                    level_data=self._cached_level_data,
                    graph_data=self._cached_graph_data,
                    entity_radius=LOCKED_DOOR_SWITCH_RADIUS,
                )
                features[9] = np.clip(rel_switch_x, -1.0, 1.0)
                features[10] = np.clip(rel_switch_y, -1.0, 1.0)
                features[11] = np.clip(switch_path_dist / area_scale, 0.0, 1.0)

                door_segment = getattr(nearest_door, "segment", None)
                if door_segment and hasattr(door_segment, "p1"):
                    door_x = (door_segment.p1[0] + door_segment.p2[0]) / 2.0
                    door_y = (door_segment.p1[1] + door_segment.p2[1]) / 2.0
                else:
                    door_x = getattr(nearest_door, "xpos", 0.0)
                    door_y = getattr(nearest_door, "ypos", 0.0)

                door_pos = (int(door_x), int(door_y))
                rel_door_x = (door_pos[0] - ninja_pos[0]) / (LEVEL_WIDTH_PX / 2)
                rel_door_y = (door_pos[1] - ninja_pos[1]) / (LEVEL_HEIGHT_PX / 2)
                # Door is a line segment, use radius 0 for center point
                door_path_dist = self._safe_path_distance(
                    ninja_pos,
                    door_pos,
                    adjacency,
                    "locked_door",
                    level_data=self._cached_level_data,
                    graph_data=self._cached_graph_data,
                    entity_radius=0.0,
                )
                features[12] = np.clip(rel_door_x, -1.0, 1.0)
                features[13] = np.clip(rel_door_y, -1.0, 1.0)
                features[14] = np.clip(door_path_dist / area_scale, 0.0, 1.0)

        return features

    def _extract_mine_features(
        self, obs: Dict[str, Any], buffer: np.ndarray
    ) -> np.ndarray:
        """Extract enhanced mine features (8 dims).

        Features:
        [0-1] nearest_mine_rel_x, rel_y (normalized -1 to 1)
        [2] nearest_mine_state (0=deadly, 0.5=toggling, 1=safe, -1=none)
        [3] nearest_mine_path_distance (normalized 0-1)
        [4] deadly_mines_nearby_count (normalized 0-1)
        [5] mine_state_certainty (0=unknown to 1=recently seen)
        [6] safe_mines_nearby_count (normalized 0-1)
        [7] mine_avoidance_difficulty (0-1: spatial complexity)
        """
        NEARBY_RADIUS = 100.0
        MAX_NEARBY = 10.0
        CERTAINTY_RADIUS = 150.0

        features = buffer
        features.fill(0.0)

        adjacency = self._cached_graph_data.get("adjacency")
        if adjacency is None or len(adjacency) == 0:
            features[2] = -1.0
            features[5] = 1.0  # High certainty when no mines
            return features

        entity_positions = obs.get(
            "entity_positions", np.zeros(ENTITY_POSITIONS_DIM, dtype=np.float32)
        )
        ninja_x = entity_positions[0] * LEVEL_WIDTH_PX
        ninja_y = entity_positions[1] * LEVEL_HEIGHT_PX
        ninja_pos = (int(ninja_x), int(ninja_y))

        # Collect mines
        entities = obs.get("entities", [])
        mines = []
        deadly_nearby = 0
        safe_nearby = 0

        for entity in entities:
            entity_type = getattr(entity, "entity_type", None)
            if entity_type in (EntityType.TOGGLE_MINE, EntityType.TOGGLE_MINE_TOGGLED):
                mx = getattr(entity, "xpos", 0.0)
                my = getattr(entity, "ypos", 0.0)
                mstate = getattr(entity, "state", 1)

                if entity_type == EntityType.TOGGLE_MINE_TOGGLED:
                    mstate = 0

                dist = np.sqrt((mx - ninja_x) ** 2 + (my - ninja_y) ** 2)

                mines.append({"x": mx, "y": my, "state": mstate, "dist": dist})

                if dist < NEARBY_RADIUS:
                    if mstate == 0:
                        deadly_nearby += 1
                    elif mstate == 1:
                        safe_nearby += 1

        if not mines:
            features[2] = -1.0
            features[5] = 1.0
            return features

        # Find nearest mine
        nearest = min(mines, key=lambda m: m["dist"])

        # Features 0-1: Relative position
        area_scale = self._get_reachable_area_scale()
        features[0] = np.clip((nearest["x"] - ninja_x) / area_scale, -1.0, 1.0)
        features[1] = np.clip((nearest["y"] - ninja_y) / area_scale, -1.0, 1.0)

        # Feature 2: State
        if nearest["state"] == 0:
            features[2] = 0.0  # Deadly
        elif nearest["state"] == 2:
            features[2] = 0.5  # Toggling
        else:
            features[2] = 1.0  # Safe

        # Feature 3: Path distance
        try:
            from ..constants.physics_constants import TOGGLE_MINE_RADII

            mine_radius = TOGGLE_MINE_RADII.get(nearest["state"], 4.0)
            mine_pos = (int(nearest["x"]), int(nearest["y"]))
            path_dist = self._safe_path_distance(
                ninja_pos,
                mine_pos,
                adjacency,
                "mine",
                level_data=self._cached_level_data,
                graph_data=self._cached_graph_data,
                entity_radius=mine_radius,
            )
            features[3] = min(path_dist / area_scale, 1.0)
        except Exception:
            features[3] = min(nearest["dist"] / area_scale, 1.0)

        # Feature 4: Deadly mines nearby
        features[4] = min(deadly_nearby / MAX_NEARBY, 1.0)

        # Feature 5: State certainty (based on distance)
        features[5] = 1.0 - min(nearest["dist"] / CERTAINTY_RADIUS, 1.0)

        # Feature 6: Safe mines nearby
        features[6] = min(safe_nearby / MAX_NEARBY, 1.0)

        # Feature 7: Avoidance difficulty
        total_nearby = deadly_nearby + safe_nearby
        if total_nearby > 0:
            danger_ratio = deadly_nearby / total_nearby
            density = total_nearby / MAX_NEARBY
            features[7] = min(0.7 * danger_ratio + 0.3 * density, 1.0)

        return features

    def _compute_progress_features(
        self, obs: Dict[str, Any], buffer: np.ndarray
    ) -> np.ndarray:
        """
        Compute progress tracking features.

        Returns PROGRESS_FEATURES_DIM (3) features:
        - current_objective_type (0=switch, 0.33=door, 0.67=exit, normalized 0-1)
        - objectives_completed_ratio (0 to 1)
        - total_path_distance_remaining (normalized)
        """
        MAX_OBJECTIVE_PATHS = 3.0

        features = buffer
        features.fill(0.0)

        adjacency = self._cached_graph_data.get("adjacency")
        if adjacency is None or len(adjacency) == 0:
            return features

        entity_positions = obs.get(
            "entity_positions", np.zeros(ENTITY_POSITIONS_DIM, dtype=np.float32)
        )
        ninja_pos = (
            int(entity_positions[0] * LEVEL_WIDTH_PX),
            int(entity_positions[1] * LEVEL_HEIGHT_PX),
        )

        exit_switch_collected = obs.get("switch_activated", False)
        locked_doors = obs.get("locked_doors", [])

        completed = 0
        total = 1

        for door in locked_doors:
            total += 1
            if not getattr(door, "active", True):
                completed += 1

        if not exit_switch_collected:
            current_obj_type = 0.0
        elif any(getattr(door, "active", True) for door in locked_doors):
            current_obj_type = 0.33
        else:
            current_obj_type = 0.67

        total_path_dist = 0.0

        if not exit_switch_collected:
            from ..constants.physics_constants import EXIT_SWITCH_RADIUS

            exit_switch_pos = (
                int(entity_positions[2] * LEVEL_WIDTH_PX),
                int(entity_positions[3] * LEVEL_HEIGHT_PX),
            )
            switch_dist = self._safe_path_distance(
                ninja_pos,
                exit_switch_pos,
                adjacency,
                "progress_exit_switch",
                level_data=self._cached_level_data,
                graph_data=self._cached_graph_data,
                entity_radius=EXIT_SWITCH_RADIUS,
            )
            total_path_dist += switch_dist

        for door in locked_doors:
            if getattr(door, "active", True):
                from ..constants.physics_constants import LOCKED_DOOR_SWITCH_RADIUS

                switch_x = getattr(door, "sw_xpos", getattr(door, "xpos", 0.0))
                switch_y = getattr(door, "sw_ypos", getattr(door, "ypos", 0.0))
                switch_pos = (int(switch_x), int(switch_y))

                door_dist = self._safe_path_distance(
                    ninja_pos,
                    switch_pos,
                    adjacency,
                    "progress_locked_door_switch",
                    level_data=self._cached_level_data,
                    graph_data=self._cached_graph_data,
                    entity_radius=LOCKED_DOOR_SWITCH_RADIUS,
                )
                total_path_dist += door_dist

        from ..constants.physics_constants import EXIT_DOOR_RADIUS

        exit_door_pos = (
            int(entity_positions[4] * LEVEL_WIDTH_PX),
            int(entity_positions[5] * LEVEL_HEIGHT_PX),
        )
        exit_dist = self._safe_path_distance(
            ninja_pos,
            exit_door_pos,
            adjacency,
            "progress_exit_door",
            level_data=self._cached_level_data,
            graph_data=self._cached_graph_data,
            entity_radius=EXIT_DOOR_RADIUS,
        )
        total_path_dist += exit_dist

        features[0] = current_obj_type
        features[1] = completed / max(total, 1)
        area_scale = self._get_reachable_area_scale()
        features[2] = np.clip(
            total_path_dist / (area_scale * MAX_OBJECTIVE_PATHS), 0.0, 1.0
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

        # Generate level_id from tiles hash (same strategy as GraphBuilder)
        level_id = f"level_{hash(level_data.tiles.tobytes())}"

        # Check cache for base GraphData
        cached_graph_data = self._graph_data_cache.get(level_id)
        if cached_graph_data is not None and level_id == self._cached_level_id:
            # Reuse cached GraphData since level hasn't changed
            return cached_graph_data

        # Convert adjacency dict to Edge list
        from ..graph.common import Edge, EdgeType

        edges = []
        for source_pos, neighbors in adjacency.items():
            for neighbor_info in neighbors:
                if isinstance(neighbor_info, tuple) and len(neighbor_info) >= 2:
                    target_pos = (neighbor_info[0], neighbor_info[1])
                    weight = neighbor_info[2] if len(neighbor_info) > 2 else 1.0

                    # Determine edge type (default to ADJACENT)
                    edge_type = EdgeType.ADJACENT
                    if len(neighbor_info) > 3:
                        edge_type = neighbor_info[3]

                    edges.append(
                        Edge(
                            source=source_pos,
                            target=target_pos,
                            edge_type=edge_type,
                            weight=weight,
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
        except Exception:
            return None

    def _get_graph_observations(self) -> Dict[str, np.ndarray]:
        """Get complete graph observations matching GraphMixin output.

        Returns graph observations with proper feature dimensions:
        - node_features: NODE_FEATURE_DIM (19 dims from nclone)
        - edge_features: EDGE_FEATURE_DIM (6 dims from nclone)
        """
        # Initialize empty graph observations with full feature dimensions
        graph_obs = {
            "graph_node_feats": np.zeros(
                (N_MAX_NODES, NODE_FEATURE_DIM), dtype=np.float32
            ),
            "graph_edge_index": np.zeros((2, E_MAX_EDGES), dtype=np.int32),
            "graph_edge_feats": np.zeros(
                (E_MAX_EDGES, EDGE_FEATURE_DIM), dtype=np.float32
            ),
            "graph_node_mask": np.zeros(N_MAX_NODES, dtype=np.int32),
            "graph_edge_mask": np.zeros(E_MAX_EDGES, dtype=np.int32),
            "graph_node_types": np.zeros(N_MAX_NODES, dtype=np.int32),
            "graph_edge_types": np.zeros(E_MAX_EDGES, dtype=np.int32),
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

            # Copy edge features (up to max edges)
            if (
                hasattr(graph_data, "edge_features")
                and graph_data.edge_features is not None
            ):
                graph_obs["graph_edge_feats"][:num_edges] = graph_data.edge_features[
                    :num_edges
                ]

            # Set masks
            graph_obs["graph_node_mask"][:num_nodes] = 1
            graph_obs["graph_edge_mask"][:num_edges] = 1

            # Copy node and edge types
            if hasattr(graph_data, "node_types") and graph_data.node_types is not None:
                graph_obs["graph_node_types"][:num_nodes] = graph_data.node_types[
                    :num_nodes
                ]

            if hasattr(graph_data, "edge_types") and graph_data.edge_types is not None:
                graph_obs["graph_edge_types"][:num_edges] = graph_data.edge_types[
                    :num_edges
                ]

        return graph_obs

    def _build_mine_death_predictor(self):
        """
        Build hybrid mine death predictor for BC pretraining.

        Uses graph system's reachability data to filter mines and build
        a danger zone grid. Called once per replay after graph building is complete.
        """
        from ..mine_death_predictor import MineDeathPredictor

        # Get reachable positions from graph system
        if (
            self._cached_graph_data is None
            or "reachable" not in self._cached_graph_data
        ):
            # If graph data not available, set predictor to None
            if hasattr(self.nplay_headless, "sim") and hasattr(
                self.nplay_headless.sim, "ninja"
            ):
                self.nplay_headless.sim.ninja.mine_death_predictor = None
            return

        reachable_positions = self._cached_graph_data.get("reachable", set())

        # Skip if no reachable positions
        if not reachable_positions:
            if hasattr(self.nplay_headless, "sim") and hasattr(
                self.nplay_headless.sim, "ninja"
            ):
                self.nplay_headless.sim.ninja.mine_death_predictor = None
            return

        # Create and build predictor
        predictor = MineDeathPredictor(self.nplay_headless.sim)

        try:
            # Build danger zone grid (non-verbose for BC dataset generation)
            predictor.build_lookup_table(reachable_positions, verbose=False)

            # Attach predictor to ninja
            self.nplay_headless.sim.ninja.mine_death_predictor = predictor
        except Exception:
            # If predictor build fails, set to None to avoid errors
            # Death probabilities will default to 0.0 in this case
            if hasattr(self.nplay_headless, "sim") and hasattr(
                self.nplay_headless.sim, "ninja"
            ):
                self.nplay_headless.sim.ninja.mine_death_predictor = None

    def close(self):
        """Clean up resources."""
        if hasattr(self, "nplay_headless"):
            del self.nplay_headless
