"""
Graph functionality mixin for N++ environment.

This module contains all graph-related functionality that was previously
integrated into the main NppEnvironment class.
"""

import time
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ...graph.level_data import LevelData
from ...graph.common import (
    GraphData,
    N_MAX_NODES,
    E_MAX_EDGES,
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
)
from ...graph.reachability.graph_builder import GraphBuilder
from ...graph.reachability.subcell_node_lookup import SUB_NODE_SIZE
from ...graph.reachability.pathfinding_utils import (
    find_closest_node_to_position,
    extract_spatial_lookups_from_graph_data,
    NODE_WORLD_COORD_OFFSET,
)
from ...constants.physics_constants import NINJA_RADIUS
from collections import deque


@dataclass
class GraphUpdateInfo:
    """Container for graph update information."""

    nodes_updated: int = 0
    edges_updated: int = 0
    switch_states: Dict[int, bool] = None

    def __post_init__(self):
        if self.switch_states is None:
            self.switch_states = {}


class GraphMixin:
    """
    Mixin class providing graph functionality for N++ environment.

    This mixin handles:
    - Dynamic graph updates based on switch states
    - Graph observation generation for ML models
    - Performance tracking and statistics
    - Graph debug visualization support
    """

    def _init_graph_system(
        self,
        debug: bool = False,
    ):
        """Initialize the graph system components.

        Args:
            debug: Enable debug logging
        """
        self.debug = debug

        # Initialize graph system
        self.graph_builder = GraphBuilder()
        self.current_graph: Optional[GraphData] = None
        self.current_graph_data: Optional[Dict[str, Any]] = None
        self.last_switch_states: Dict[int, bool] = {}
        self.last_update_time = 0.0

        # Graph debug visualization state
        self._graph_debug_enabled: bool = False
        self._graph_builder: Optional[GraphBuilder] = None
        self._graph_debug_cache: Optional[Dict[str, Any]] = None

        # Cache for reachable area scale (per level ID)
        self._reachable_area_scale_cache: Dict[str, float] = {}

        # Cache for GraphData per level (static features only)
        # Dynamic features (proximity, reachability) updated each step
        self._graph_data_cache: Dict[str, GraphData] = {}
        self._current_level_id: Optional[str] = None

        # Initialize logging if debug is enabled
        if self.debug:
            logging.basicConfig(level=logging.DEBUG)
            self.logger = logging.getLogger(__name__)

    def _reset_graph_state(self):
        """Reset graph state during environment reset."""
        self.current_graph = None
        self.current_graph_data = None
        self.last_switch_states.clear()
        self.last_update_time = time.time()
        self.graph_builder.clear_cache()
        self._current_level_id = None

        # Clear per-level caches since we use random levels and won't reload the same level
        # This prevents unbounded memory growth during training
        self._reachable_area_scale_cache.clear()
        self._graph_data_cache.clear()

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
        # Get level data to compute level ID
        level_data = self._get_level_data_from_env()
        if level_data is None:
            raise RuntimeError(
                "Cannot compute reachable area scale: level_data not available."
            )

        # Generate level ID from tiles array (most reliable unique identifier)
        # PERFORMANCE: Using tiles hash ensures consistent cache hits across same levels
        level_id = hash(level_data.tiles.tobytes())

        # Check cache first (O(1) lookup)
        if level_id in self._reachable_area_scale_cache:
            return self._reachable_area_scale_cache[level_id]

        # Get adjacency graph
        if self.current_graph_data is None:
            raise RuntimeError(
                "Cannot compute reachable area scale: graph data not available. "
                "Graph must be built before computing reachable area scale."
            )

        adjacency = self.current_graph_data.get("adjacency")
        if not adjacency:
            raise RuntimeError(
                "Cannot compute reachable area scale: adjacency graph is empty. "
                "Graph building must succeed before computing reachable area scale."
            )

        # Validate graph has at least 1 node (catch completely broken graphs)
        # Note: Even very small maps (2-3 nodes) can be valid
        logger = logging.getLogger(__name__)
        MIN_GRAPH_NODES = 1
        if len(adjacency) < MIN_GRAPH_NODES:
            map_name = getattr(self, "map_loader", None)
            if map_name and hasattr(map_name, "current_map_name"):
                map_name = map_name.current_map_name
            else:
                map_name = "unknown"

            logger.error(
                f"CRITICAL: Empty graph detected with {len(adjacency)} nodes! "
                f"Map: {map_name}"
            )
            raise RuntimeError(
                f"Empty graph: {len(adjacency)} nodes. "
                f"This indicates a critical bug in graph building."
            )

        # Log warning for very small graphs (likely indicates spawn isolation issue)
        if len(adjacency) < 10:
            map_name = getattr(self, "map_loader", None)
            if map_name and hasattr(map_name, "current_map_name"):
                map_name = map_name.current_map_name
            else:
                map_name = "unknown"
            logger.warning(
                f"Small graph detected: {len(adjacency)} nodes. "
                f"Map: {map_name}. This may indicate spawn isolation or very small map."
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
            self.current_graph_data
        )

        # Find closest node to start position
        closest_node = find_closest_node_to_position(
            start_pos,
            adjacency,
            threshold=NINJA_RADIUS,
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

    def _should_update_graph(self) -> bool:
        """
        Simple check if graph update is needed.

        Only updates when switch states change - no complex event system.
        """
        # Get current switch states from environment
        current_switch_states = self._get_switch_states_from_env()

        # Check if any switch state changed
        if current_switch_states != self.last_switch_states:
            return True

        return False

    def _get_switch_states_from_env(self) -> Dict[int, bool]:
        """Extract switch states from nclone environment entities."""
        switch_states = {}

        # Get nplay_headless instance
        nplay = self.nplay_headless

        if nplay and hasattr(nplay, "sim") and hasattr(nplay.sim, "entity_dic"):
            entity_dic = nplay.sim.entity_dic

            # Extract switch states from different entity types
            # Exit switches (entity_dic key 3)
            if 3 in entity_dic:
                exit_entities = entity_dic[3]
                for i, entity in enumerate(exit_entities):
                    if (
                        hasattr(entity, "activated")
                        and type(entity).__name__ == "EntityExitSwitch"
                    ):
                        switch_states[f"exit_switch_{i}"] = bool(entity.activated)

            # Door switches (entity_dic key 4 - doors)
            if 4 in entity_dic:
                door_entities = entity_dic[4]
                for i, entity in enumerate(door_entities):
                    # Check for door state (open/closed)
                    if hasattr(entity, "open"):
                        switch_states[f"door_{i}"] = bool(entity.open)
                    elif hasattr(entity, "activated"):
                        switch_states[f"door_{i}"] = bool(entity.activated)

        return switch_states

    def _update_graph_from_env_state(self):
        """
        Update graph using GraphBuilder (simple approach).

        This uses GraphBuilder to create updated connectivity based on current game state.
        GraphBuilder returns a dict with adjacency information for pathfinding.
        """
        # Get level data from environment
        level_data = self._get_level_data_from_env()
        if level_data is None:
            return

        # Get ninja position for reachability analysis
        ninja_pos = None
        ninja_pos_tuple = self.nplay_headless.ninja_position()
        ninja_pos = (int(ninja_pos_tuple[0]), int(ninja_pos_tuple[1]))

        self.current_graph_data = self.graph_builder.build_graph(
            level_data, ninja_pos=ninja_pos
        )

        # GraphBuilder returns dict with 'adjacency', 'reachable', etc.
        # Convert to GraphData format for ML models if graph observations are enabled
        self.current_graph = self._convert_graph_data_to_graphdata(
            self.current_graph_data, level_data, ninja_pos
        )

        # Build level cache for path distances if path calculator is available
        # CRITICAL: Use _path_calculator (from ReachabilityMixin), not path_calculator
        # CRITICAL: Must build cache BEFORE any path distance calculations to avoid
        # "unreachable" warnings for valid levels on first observation
        if hasattr(self, "_path_calculator") and self._path_calculator is not None:
            adjacency = (
                self.current_graph_data.get("adjacency", {})
                if self.current_graph_data
                else {}
            )
            if adjacency and level_data is not None:
                # Pass graph_data for spatial indexing optimization
                # This builds the level cache with precomputed distances to all goals
                cache_built = self._path_calculator.build_level_cache(
                    level_data, adjacency, self.current_graph_data
                )

                if not cache_built and self._path_calculator.level_cache is None:
                    # Ensure level_cache is initialized even if build_cache returns False
                    from nclone.graph.reachability.path_distance_cache import (
                        LevelBasedPathDistanceCache,
                    )

                    self._path_calculator.level_cache = LevelBasedPathDistanceCache()
                    # Try building again
                    self._path_calculator.build_level_cache(
                        level_data, adjacency, self.current_graph_data
                    )

        # Update switch state tracking
        self.last_switch_states = self._get_switch_states_from_env()

        # Update simple statistics
        adjacency = (
            self.current_graph_data.get("adjacency", {})
            if self.current_graph_data
            else {}
        )

    def _get_level_data_from_env(self) -> Optional[LevelData]:
        """Extract level data from environment for graph building."""
        # Use existing level_data property
        return self.level_data

    def _get_graph_observations(self, use_sparse: bool = True) -> Dict[str, np.ndarray]:
        """Get complete graph observations for HGT processing.

        Args:
            use_sparse: If True, return sparse format (default). If False, return dense format.

        MEMORY OPTIMIZATION (Sparse Format):
        - Stores only valid nodes/edges without padding (~95% memory reduction)
        - Uses float32 for features (converted from float16 in current_graph if needed)
        - Edge indices remain uint16 (sufficient for max 4500 nodes)
        - Returns arrays with actual sizes, not fixed N_MAX_NODES/E_MAX_EDGES

        BACKWARD COMPATIBILITY (Dense Format):
        - Uses float16 for node/edge features to reduce memory by 50%
        - The feature extractor will cast back to float32 for training
        """

        if use_sparse:
            # SPARSE FORMAT: Return only valid nodes/edges without padding
            # This achieves ~95% memory reduction in rollout buffer
            #
            # REACHABILITY FILTERING:
            # Nodes are already filtered to only include those reachable from spawn
            # via flood-fill because GraphMixin.build_graph() uses the default
            # filter_by_reachability=True. This ensures consistency with PBRS potentials,
            # which also use flood-fill from start position for surface area calculation.
            #
            # All nodes in graph_data are guaranteed to be reachable from player spawn,
            # matching the reachability semantics used in reward shaping.
            if self.current_graph is None:
                # Return empty sparse observation with dummy dense arrays for VecEnv compatibility
                obs_dict = {
                    "graph_node_feats_sparse": np.zeros(
                        (0, NODE_FEATURE_DIM), dtype=np.float32
                    ),
                    "graph_edge_index_sparse": np.zeros((2, 0), dtype=np.uint16),
                    "graph_edge_feats_sparse": np.zeros(
                        (0, EDGE_FEATURE_DIM), dtype=np.float32
                    ),
                    "graph_node_types_sparse": np.zeros(0, dtype=np.uint8),
                    "graph_edge_types_sparse": np.zeros(0, dtype=np.uint8),
                    "graph_num_nodes": np.array([0], dtype=np.int32),
                    "graph_num_edges": np.array([0], dtype=np.int32),
                }
                # Add dummy dense arrays for VecEnv
                obs_dict.update(
                    {
                        "graph_node_feats": np.zeros(
                            (N_MAX_NODES, NODE_FEATURE_DIM), dtype=np.float16
                        ),
                        "graph_edge_index": np.zeros((2, E_MAX_EDGES), dtype=np.uint16),
                        "graph_edge_feats": np.zeros(
                            (E_MAX_EDGES, EDGE_FEATURE_DIM), dtype=np.float16
                        ),
                        "graph_node_mask": np.zeros(N_MAX_NODES, dtype=np.uint8),
                        "graph_edge_mask": np.zeros(E_MAX_EDGES, dtype=np.uint8),
                        "graph_node_types": np.zeros(N_MAX_NODES, dtype=np.uint8),
                        "graph_edge_types": np.zeros(E_MAX_EDGES, dtype=np.uint8),
                    }
                )
                return obs_dict

            graph_data = self.current_graph
            num_nodes = graph_data.num_nodes
            num_edges = graph_data.num_edges

            # Extract only valid data (no padding)
            # Note: All nodes here are already reachable-filtered by GraphBuilder
            node_features = graph_data.node_features[:num_nodes].astype(np.float32)
            edge_index = graph_data.edge_index[:, :num_edges].astype(np.uint16)
            edge_features = graph_data.edge_features[:num_edges].astype(np.float32)
            node_types = graph_data.node_types[:num_nodes].astype(np.uint8)
            edge_types = graph_data.edge_types[:num_edges].astype(np.uint8)

            # Return BOTH sparse (for memory efficiency) and dense (for VecEnv compatibility)
            # Dense arrays are empty/dummy to satisfy observation space requirements
            # Sparse rollout buffer will extract and use only the sparse keys
            obs_dict = {
                # Sparse format (actual data, memory efficient)
                "graph_node_feats_sparse": node_features,
                "graph_edge_index_sparse": edge_index,
                "graph_edge_feats_sparse": edge_features,
                "graph_node_types_sparse": node_types,
                "graph_edge_types_sparse": edge_types,
                "graph_num_nodes": np.array([num_nodes], dtype=np.int32),
                "graph_num_edges": np.array([num_edges], dtype=np.int32),
            }

            # Add dummy dense arrays to satisfy VecEnv/observation space requirements
            # These are ignored by sparse rollout buffer but needed for VecEnv compatibility
            obs_dict.update(
                {
                    "graph_node_feats": np.zeros(
                        (N_MAX_NODES, NODE_FEATURE_DIM), dtype=np.float16
                    ),
                    "graph_edge_index": np.zeros((2, E_MAX_EDGES), dtype=np.uint16),
                    "graph_edge_feats": np.zeros(
                        (E_MAX_EDGES, EDGE_FEATURE_DIM), dtype=np.float16
                    ),
                    "graph_node_mask": np.zeros(N_MAX_NODES, dtype=np.uint8),
                    "graph_edge_mask": np.zeros(E_MAX_EDGES, dtype=np.uint8),
                    "graph_node_types": np.zeros(N_MAX_NODES, dtype=np.uint8),
                    "graph_edge_types": np.zeros(E_MAX_EDGES, dtype=np.uint8),
                }
            )

            return obs_dict
        else:
            # DENSE FORMAT (backward compatibility): Padded to N_MAX_NODES/E_MAX_EDGES
            graph_obs = {
                "graph_node_feats": np.zeros(
                    (N_MAX_NODES, NODE_FEATURE_DIM),
                    dtype=np.float16,
                ),
                "graph_edge_index": np.zeros((2, E_MAX_EDGES), dtype=np.uint16),
                "graph_edge_feats": np.zeros(
                    (E_MAX_EDGES, EDGE_FEATURE_DIM),
                    dtype=np.float16,
                ),
                "graph_node_mask": np.zeros(N_MAX_NODES, dtype=np.uint8),
                "graph_edge_mask": np.zeros(E_MAX_EDGES, dtype=np.uint8),
                "graph_node_types": np.zeros(N_MAX_NODES, dtype=np.uint8),
                "graph_edge_types": np.zeros(E_MAX_EDGES, dtype=np.uint8),
            }

            # Fill with actual graph data if available
            if self.current_graph is not None:
                graph_data = self.current_graph

                # Copy node features (up to max nodes)
                num_nodes = min(graph_data.num_nodes, N_MAX_NODES)
                if (
                    hasattr(graph_data, "node_features")
                    and graph_data.node_features is not None
                ):
                    graph_obs["graph_node_feats"][:num_nodes] = (
                        graph_data.node_features[:num_nodes]
                    )

                # Copy edge index (up to max edges)
                num_edges = min(graph_data.num_edges, E_MAX_EDGES)
                if (
                    hasattr(graph_data, "edge_index")
                    and graph_data.edge_index is not None
                ):
                    graph_obs["graph_edge_index"][:, :num_edges] = (
                        graph_data.edge_index[:, :num_edges]
                    )

                # Copy edge features (up to max edges)
                if (
                    hasattr(graph_data, "edge_features")
                    and graph_data.edge_features is not None
                ):
                    graph_obs["graph_edge_feats"][:num_edges] = (
                        graph_data.edge_features[:num_edges]
                    )

                # Set masks
                graph_obs["graph_node_mask"][:num_nodes] = 1
                graph_obs["graph_edge_mask"][:num_edges] = 1

                # Copy node and edge types
                if (
                    hasattr(graph_data, "node_types")
                    and graph_data.node_types is not None
                ):
                    graph_obs["graph_node_types"][:num_nodes] = graph_data.node_types[
                        :num_nodes
                    ]

                if (
                    hasattr(graph_data, "edge_types")
                    and graph_data.edge_types is not None
                ):
                    graph_obs["graph_edge_types"][:num_edges] = graph_data.edge_types[
                        :num_edges
                    ]

            return graph_obs

    def get_graph_data(self):
        """Get the fast graph data (adjacency dict) for pathfinding/reachability."""
        return self.current_graph_data

    def get_current_graph(self) -> Optional[GraphData]:
        """Get current graph data for external use."""
        return self.current_graph

    def _convert_graph_data_to_graphdata(
        self,
        graph_data_dict: Dict[str, Any],
        level_data: Any,
        ninja_pos: Optional[Tuple[int, int]],
    ) -> Optional[GraphData]:
        """Convert GraphBuilder dict to GraphData format for ML models.

        This conversion is only done when graph observations are enabled.
        Uses the fast GraphBuilder adjacency dict and converts it to the
        GraphData format expected by graph neural networks.

        Caching strategy:
        - Base GraphData is cached per level_id (tiles hash)
        - Dynamic features (proximity, reachability, entity state) updated each step
        - GraphBuilder already caches adjacency, so this optimizes feature building

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

        # level_data should already be a LevelData object from _get_level_data_from_env()
        if not isinstance(level_data, LevelData):
            if self.debug:
                self.logger.warning(
                    f"Expected LevelData object, got {type(level_data)}. "
                    "Graph observations may be incomplete."
                )
            return None

        # Generate level_id from tiles hash (same strategy as GraphBuilder)
        level_id = f"level_{hash(level_data.tiles.tobytes())}"

        # Check cache for base GraphData
        cached_graph_data = self._graph_data_cache.get(level_id)
        if cached_graph_data is not None and level_id == self._current_level_id:
            # Update only dynamic features (proximity, reachability, entity state)
            # For now, rebuild fully since entity state changes are complex
            # TODO: Optimize to update only proximity/reachability features
            pass

        # Convert adjacency dict to Edge list
        from ...graph.common import Edge, EdgeType

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
        from ...graph.edge_building import create_graph_data

        try:
            # Build GraphData (will extract entity info and build features)
            graph_data = create_graph_data(edges, level_data)

            # Cache GraphData per level_id
            # Note: Entity states may change, but positions/types are static
            # Full caching helps avoid rebuilding structure each step
            self._graph_data_cache[level_id] = graph_data
            self._current_level_id = level_id

            return graph_data
        except Exception as e:
            if self.debug:
                self.logger.warning(f"Failed to convert graph data: {e}")
            return None

    def force_graph_update(self):
        """Force a graph update (for testing/debugging)."""
        self._update_graph_from_env_state()

    # Graph debug visualization methods
    def set_graph_debug_enabled(self, enabled: bool):
        """Enable/disable graph debug overlay visualization."""
        self._graph_debug_enabled = bool(enabled)
        # Invalidate cache so next render rebuilds with current state
        self._graph_debug_cache = None

    def _maybe_build_graph_debug(self) -> Optional[Dict[str, Any]]:
        """Build graph data for debug visualization, with dynamic caching that considers door states."""
        # Enhanced cache that considers door states and ninja position
        sim_frame = getattr(self.nplay_headless.sim, "frame", None)
        cached_frame = getattr(self, "_graph_debug_cached_frame", None)

        # Get current door states for cache invalidation
        current_door_states = self._get_door_states_signature()
        cached_door_states = getattr(self, "_graph_debug_cached_door_states", None)

        # Get ninja position for cache invalidation (sub-cell level precision)
        ninja_pos = self.nplay_headless.ninja_position()
        ninja_pos_int = (int(ninja_pos[0]), int(ninja_pos[1]))
        ninja_sub_cell = (
            int(ninja_pos[1] // 12),
            int(ninja_pos[0] // 12),
        )  # (sub_row, sub_col)
        cached_ninja_sub_cell = getattr(
            self, "_graph_debug_cached_ninja_sub_cell", None
        )

        # Check if cache is still valid
        cache_valid = (
            self._graph_debug_cache is not None
            and sim_frame == cached_frame
            and current_door_states == cached_door_states
            and ninja_sub_cell == cached_ninja_sub_cell
        )

        if cache_valid:
            return self._graph_debug_cache

        # Initialize graph builder if needed
        if self._graph_builder is None:
            self._graph_builder = GraphBuilder()

        # Use level data extraction method
        level_data = self._get_level_data_from_env()
        if level_data is None:
            return None

        # Build graph using GraphBuilder
        graph_data = self._graph_builder.build_graph(
            level_data, ninja_pos=ninja_pos_int
        )

        # Update cache with all relevant state
        self._graph_debug_cache = graph_data
        setattr(self, "_graph_debug_cached_frame", sim_frame)
        setattr(self, "_graph_debug_cached_door_states", current_door_states)
        setattr(self, "_graph_debug_cached_ninja_sub_cell", ninja_sub_cell)

        return graph_data

    def _get_door_states_signature(self) -> Tuple:
        """
        Get a signature of current door states for cache invalidation.

        Returns:
            Tuple representing current door states
        """
        # Extract door-related entities and their states
        entities = self.entity_extractor.extract_graph_entities()
        door_states = []

        for entity in entities:
            entity_type = entity.get("type", "")

            # Check for door entities
            if (isinstance(entity_type, int) and entity_type in {3, 5, 6, 8}) or any(
                door_type in str(entity_type).lower()
                for door_type in ["door", "switch"]
            ):
                # Include position and state for doors/switches
                state_tuple = (
                    entity.get("type", ""),
                    entity.get("x", 0),
                    entity.get("y", 0),
                    entity.get("active", True),
                    entity.get("closed", False),
                )
                door_states.append(state_tuple)

        return tuple(sorted(door_states))

    def _reinit_graph_system_after_unpickling(self, debug: bool = False):
        """Reinitialize graph system components after unpickling."""
        # Graph building happens if either flag is True
        if not hasattr(self, "graph_builder"):
            self.graph_builder = GraphBuilder()
