"""
Graph functionality mixin for N++ environment.

This module contains all graph-related functionality that was previously
integrated into the main NppEnvironment class.
"""

import os
import time
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ...graph.level_data import LevelData
from ...graph.common import GraphData
from ...graph.reachability.graph_builder import GraphBuilder
from ...graph.reachability.subcell_node_lookup import SUB_NODE_SIZE
from ...graph.reachability.pathfinding_utils import (
    find_closest_node_to_position,
    extract_spatial_lookups_from_graph_data,
    NODE_WORLD_COORD_OFFSET,
)
from ...constants.physics_constants import NINJA_RADIUS
from collections import deque, OrderedDict


class _GraphCache:
    """Level-keyed reachability-graph cache: a BOUNDED in-memory LRU backed by an
    optional SHARED on-disk cache.

    The reachability graph is the dominant per-reset cost (~100ms build). In
    multi-level (curriculum/pool) training the old behavior cleared the cache
    every reset (0% hit -> every reset rebuilds) to avoid unbounded memory. This
    instead caps memory with an LRU and, when ``NCLONE_GRAPH_CACHE_DIR`` is set,
    persists each built graph to disk keyed by level id — so across N parallel env
    workers each distinct level is built ONCE total and every later reset (any
    worker) loads it in ~1ms. Dict-like API so call sites are unchanged.
    """

    _MISS = object()

    def __init__(self, maxsize: int = 256, disk_dir: Optional[str] = None):
        self._mem: "OrderedDict[str, GraphData]" = OrderedDict()
        self._maxsize = max(1, int(maxsize))
        import os as _os
        self._disk = disk_dir or _os.environ.get("NCLONE_GRAPH_CACHE_DIR")
        if self._disk:
            from pathlib import Path as _Path
            _Path(self._disk).mkdir(parents=True, exist_ok=True)

    def _disk_path(self, key):
        import hashlib
        from pathlib import Path as _Path
        h = hashlib.md5(str(key).encode("utf-8")).hexdigest()
        return _Path(self._disk) / f"{h}.pkl"

    def _put_mem(self, key, val):
        self._mem[key] = val
        self._mem.move_to_end(key)
        while len(self._mem) > self._maxsize:
            self._mem.popitem(last=False)

    def get(self, key, default=None):
        if key in self._mem:
            self._mem.move_to_end(key)
            return self._mem[key]
        if self._disk:
            p = self._disk_path(key)
            if p.exists():
                try:
                    import pickle
                    with open(p, "rb") as fh:
                        g = pickle.load(fh)
                    self._put_mem(key, g)
                    return g
                except Exception:
                    pass
        return default

    def __contains__(self, key):
        return self.get(key, _GraphCache._MISS) is not _GraphCache._MISS

    def __getitem__(self, key):
        v = self.get(key, _GraphCache._MISS)
        if v is _GraphCache._MISS:
            raise KeyError(key)
        return v

    def __setitem__(self, key, val):
        self._put_mem(key, val)
        if self._disk:
            p = self._disk_path(key)
            if not p.exists():
                try:
                    import os as _os
                    import pickle
                    tmp = str(p) + f".tmp{_os.getpid()}"
                    with open(tmp, "wb") as fh:
                        pickle.dump(val, fh, protocol=pickle.HIGHEST_PROTOCOL)
                    _os.replace(tmp, p)
                except Exception:
                    pass

    def clear(self):
        # Only the in-memory tier; the on-disk tier is shared/persistent.
        self._mem.clear()


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
        use_graph_data_in_observation: bool = False,
    ):
        """Initialize the graph system components.

        Args:
            debug: Enable debug logging
            use_graph_data_in_observation: Use graph data in observation space
        """
        self.debug = debug
        self.use_graph_data_in_observation = use_graph_data_in_observation

        # Initialize graph system
        self.graph_builder = GraphBuilder()
        self.current_graph: Optional[GraphData] = None
        self.current_graph_data: Optional[Dict[str, Any]] = None
        self.last_switch_states: Dict[int, bool] = {}
        self.last_mine_states: Dict[str, int] = {}  # Add mine state tracking
        self.last_update_time = 0.0

        # Graph debug visualization state
        self._graph_debug_enabled: bool = False
        self._graph_builder: Optional[GraphBuilder] = None
        self._graph_debug_cache: Optional[Dict[str, Any]] = None

        # Cache for reachable area scale (per level ID)
        self._reachable_area_scale_cache: Dict[str, float] = {}

        # Cache for GraphData per level (static features only). Bounded LRU +
        # optional shared on-disk cache (NCLONE_GRAPH_CACHE_DIR) so multi-level
        # curriculum training doesn't rebuild the ~100ms reachability graph every
        # reset (the dominant throughput cost on a synchronous vec-env).
        # Cap default 160 -> 48 (v14 worker-RSS audit: ~16MB per entry means the
        # old cap alone held ~2.6GB per env worker across a 16-worker box; the
        # shared DISK cache makes an eviction a cheap re-load, not a rebuild).
        # NCLONE_GRAPH_MEM_LRU overrides for profiling/tuning.
        self._graph_data_cache = _GraphCache(
            maxsize=int(os.environ.get("NCLONE_GRAPH_MEM_LRU", "48"))
        )
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
        self.last_mine_states = {}  # Add mine state tracking
        self.last_update_time = time.time()

        # Single static level (custom_map_path): the geometry — and therefore the
        # whole reachability graph — is invariant across resets, so KEEP the
        # per-level caches. ``_update_graph_from_env_state`` then restores the
        # cached graph in O(1) instead of rebuilding it (the dominant reset cost
        # on mine-dense levels: ~430ms/reset). Only clear the level-keyed caches
        # when levels actually vary between episodes (curriculum / pool / random),
        # where keeping them would grow memory unbounded.
        if getattr(self, "custom_map_path", None) is None:
            self.graph_builder.clear_cache()
            self._current_level_id = None
            self._reachable_area_scale_cache.clear()
            # NOTE: do NOT clear _graph_data_cache here. It is a bounded LRU (+
            # shared on-disk) cache, so keeping it across resets is what gives the
            # huge throughput win in curriculum training (a re-seen level skips the
            # ~100ms rebuild); memory stays bounded by the LRU cap.

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

        Updates when switch states OR mine states change.
        """
        # Check switch states
        current_switch_states = self._get_switch_states_from_env()
        if current_switch_states != self.last_switch_states:
            return True

        # Check mine states
        current_mine_states = self._get_mine_states_from_env()
        if not hasattr(self, "last_mine_states"):
            self.last_mine_states = {}

        if current_mine_states != self.last_mine_states:
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

    def _get_mine_states_from_env(self) -> Dict[str, int]:
        """Extract mine states from nclone environment entities."""
        mine_states = {}

        nplay = self.nplay_headless

        if nplay and hasattr(nplay, "sim") and hasattr(nplay.sim, "entity_dic"):
            entity_dic = nplay.sim.entity_dic

            # Toggle mines (entity_dic key 1)
            if 1 in entity_dic:
                toggle_mines = entity_dic[1]
                # Key on the raw entity id rather than an f-string wrapping it:
                # the dict exists only to be compared with the previous one, and
                # `id(m) -> f"mine_{id(m)}"` is a bijection, so every equality
                # test has the same outcome. On a 110-mine level this ran twice
                # per step and the formatting alone was a measurable slice of it.
                mine_states = {
                    id(mine): int(mine.state)
                    for mine in toggle_mines
                    if hasattr(mine, "state")
                }

        return mine_states

    def _update_graph_from_env_state(self):
        """
        Update graph and build level cache.

        This is THE authoritative place for cache building during normal operation.
        Other code should NOT call build_level_cache directly - they should rely
        on this method being called first (during reset or when graph needs updating).

        Implementation notes:
        - Uses GraphBuilder to create updated connectivity based on current game state
        - Builds level cache for path distances (via _path_calculator)
        - Also builds reward calculator's separate path calculator cache
        - GraphBuilder returns a dict with adjacency information for pathfinding
        """
        # Get level data from environment
        level_data = self._get_level_data_from_env()
        if level_data is None:
            return

        # OPTIMIZATION: Cache graph per level to skip expensive rebuilds
        # For single-level training, this eliminates graph building on every reset.
        #
        # KEY BY LEVEL CONTENT, not by map_loader.current_map_name. The name is a
        # mutable label that DIFFERENT levels can collide on (e.g. the benchmark
        # eval reuses integer level ids / may not refresh the name per level), and
        # a colliding name served a PREVIOUS level's graph for the current one —
        # the ninja was then checked against a foreign graph's bounds and flagged
        # "far outside graph bounds → inf" (an OOB flood during test-split eval).
        # ``level_data.level_id`` is a content hash (tiles+entities+start+switch/
        # exit, auto-set in LevelData.__post_init__) so the cache can never desync
        # from the level actually loaded. Fall back to the name only if absent.
        level_id = getattr(level_data, "level_id", None) or (
            self.map_loader.current_map_name if hasattr(self, "map_loader") else None
        )

        # Check if we have cached graph for this level (without switch states for now)
        # Note: This caches the base graph structure, not dynamic switch-dependent connectivity
        if level_id and level_id in self._graph_data_cache:
            self.current_graph_data = self._graph_data_cache[level_id]
            # Keep the hit path complete (mirrors the build path below): the
            # GraphData conversion and switch/mine bookkeeping otherwise only
            # ran on a cache MISS, leaving ``current_graph`` None (breaks
            # graph-tensor observations when enabled) and the state maps stale.
            if self.use_graph_data_in_observation:
                ninja_pos_tuple = self.nplay_headless.ninja_position()
                ninja_pos = (int(ninja_pos_tuple[0]), int(ninja_pos_tuple[1]))
                t_convert = self._profile_start("graph_convert")
                self.current_graph = self._convert_graph_data_to_graphdata(
                    self.current_graph_data, level_data, ninja_pos
                )
                self._profile_end("graph_convert", t_convert)
            self.last_switch_states = self._get_switch_states_from_env()
            self.last_mine_states = self._get_mine_states_from_env()
            return  # Skip expensive rebuild

        # Get ninja position for reachability analysis
        ninja_pos = None
        ninja_pos_tuple = self.nplay_headless.ninja_position()
        ninja_pos = (int(ninja_pos_tuple[0]), int(ninja_pos_tuple[1]))

        self.current_graph_data = self.graph_builder.build_graph(
            level_data, ninja_pos=ninja_pos
        )

        # Cache the built graph for this level
        if level_id:
            self._graph_data_cache[level_id] = self.current_graph_data

        _logger = logging.getLogger(__name__)

        # Validation: Ensure base_adjacency is present (node_physics for physics-aware pathfinding)
        if self.current_graph_data:
            has_base_adj = "base_adjacency" in self.current_graph_data
            if not has_base_adj:
                raise RuntimeError(
                    f"Graph data missing base_adjacency required for pathfinding."
                )

            # # Log successful validation with warning level for visibility
            # base_adj_size = len(self.current_graph_data.get("base_adjacency", {}))
            # physics_size = len(self.current_graph_data.get("node_physics", {}))
            # _logger.warning(
            #     f"[GRAPH_BUILD] Graph built with physics data: "
            #     f"{base_adj_size} base edges, {physics_size} node physics entries"
            # )

        # GraphBuilder returns dict with 'adjacency', 'reachable', etc.
        # Convert to GraphData format for ML models if graph observations are enabled
        if self.use_graph_data_in_observation:
            t_convert = self._profile_start("graph_convert")
            self.current_graph = self._convert_graph_data_to_graphdata(
                self.current_graph_data, level_data, ninja_pos
            )
            self._profile_end("graph_convert", t_convert)

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
                base_adjacency = self.current_graph_data.get(
                    "base_adjacency", adjacency
                )
                self._path_calculator.build_level_cache(
                    level_data, adjacency, base_adjacency, self.current_graph_data
                )

                # Also build the reward calculator's path_calculator cache when it is a
                # second instance (tests / legacy). Usually it aliases _path_calculator.
                if hasattr(self, "reward_calculator") and hasattr(
                    self.reward_calculator, "path_calculator"
                ):
                    reward_path_calc = self.reward_calculator.path_calculator
                    if (
                        reward_path_calc is not None
                        and reward_path_calc is not self._path_calculator
                    ):
                        reward_cache_built = reward_path_calc.build_level_cache(
                            level_data,
                            adjacency,
                            base_adjacency,
                            self.current_graph_data,
                        )
                        if reward_cache_built:
                            level_id = getattr(level_data, "level_id", "unknown")
                            _logger.info(
                                f"Built reward calculator's path cache for level {level_id}"
                            )

        # Update switch and mine state tracking
        self.last_switch_states = self._get_switch_states_from_env()
        self.last_mine_states = self._get_mine_states_from_env()

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

        # OPTIMIZATION: Check cache for base GraphData
        # Graph structure (nodes, edges) is static per level, only entity states change
        cached_graph_data = self._graph_data_cache.get(level_id)
        if cached_graph_data is not None and level_id == self._current_level_id:
            # FAST PATH: Return cached GraphData without rebuilding
            # Entity state changes are handled by the ML model, not the graph structure
            # This saves ~93ms per call (40 calls per 4000 steps = 3.7s total savings)
            return cached_graph_data

        # SLOW PATH: Build GraphData from scratch (only on first encounter of level)
        # Convert adjacency dict to Edge list
        # Adjacency format: {(x, y): [((nx, ny), cost), ...]}
        # Simplified: All edges are simple adjacency between reachable nodes
        from ...graph.common import Edge

        edges = []
        for source_pos, neighbors in adjacency.items():
            for neighbor_info in neighbors:
                # neighbor_info is a 2-tuple: ((x, y), cost)
                if isinstance(neighbor_info, tuple) and len(neighbor_info) >= 2:
                    target_pos = neighbor_info[0]  # Already a (x, y) tuple
                    # Cost is ignored - all edges are equal (simple adjacency)

                    edges.append(
                        Edge(
                            source=source_pos,
                            target=target_pos,
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
            int(ninja_pos[1] // 6),
            int(ninja_pos[0] // 6),
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
