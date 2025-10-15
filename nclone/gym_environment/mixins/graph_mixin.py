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

from ...graph.hierarchical_builder import HierarchicalGraphBuilder
from ...graph.level_data import LevelData
from ...graph.common import (
    GraphData,
    N_MAX_NODES,
    E_MAX_EDGES,
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
)


@dataclass
class GraphUpdateInfo:
    """Container for graph update information."""

    nodes_updated: int = 0
    edges_updated: int = 0
    update_time_ms: float = 0.0
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
        self, enable_graph_updates: bool = True, debug: bool = False
    ):
        """Initialize the graph system components."""
        self.enable_graph_updates = enable_graph_updates
        self.debug = debug

        # Initialize graph system
        self.graph_builder = HierarchicalGraphBuilder(debug=debug)
        self.current_graph: Optional[GraphData] = None
        self.current_hierarchical_graph = None
        self.last_switch_states: Dict[int, bool] = {}
        self.last_update_time = 0.0

        # Graph performance tracking
        self.graph_update_stats = {
            "total_updates": 0,
            "avg_update_time_ms": 0.0,
            "last_update_info": GraphUpdateInfo(),
        }

        # Graph debug visualization state
        self._graph_debug_enabled: bool = False
        self._graph_builder: Optional[HierarchicalGraphBuilder] = None
        self._graph_debug_cache: Optional[GraphData] = None

        # Initialize logging if debug is enabled
        if self.debug:
            logging.basicConfig(level=logging.DEBUG)
            self.logger = logging.getLogger(__name__)

    def _reset_graph_state(self):
        """Reset graph state during environment reset."""
        if self.enable_graph_updates:
            self.current_graph = None
            self.current_hierarchical_graph = None
            self.last_switch_states.clear()
            self.last_update_time = time.time()

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

            # One-way platforms (entity_dic key 5)
            if 5 in entity_dic:
                platform_entities = entity_dic[5]
                for i, entity in enumerate(platform_entities):
                    if hasattr(entity, "activated"):
                        switch_states[f"platform_{i}"] = bool(entity.activated)

            # Other interactive entities
            for entity_type_id, entities in entity_dic.items():
                if entity_type_id not in [3, 4, 5]:  # Skip already processed types
                    for i, entity in enumerate(entities):
                        if hasattr(entity, "activated"):
                            switch_states[f"entity_{entity_type_id}_{i}"] = bool(
                                entity.activated
                            )
                        elif hasattr(entity, "open"):
                            switch_states[f"entity_{entity_type_id}_{i}"] = bool(
                                entity.open
                            )

        return switch_states

    def _update_graph_from_env_state(self):
        """
        Update graph using nclone's graph builder (simple approach).

        This uses nclone's hierarchical graph builder to create updated
        connectivity based on current game state. No complex event processing.
        """
        # Get level data from environment
        level_data = self._get_level_data_from_env()
        if level_data is None:
            return

        # Use nclone's graph builder - proper abstraction
        start_time = time.time()
        self.current_hierarchical_graph = self.graph_builder.build_graph(level_data)
        build_time = (time.time() - start_time) * 1000

        # Extract the fine-resolution graph as the primary graph for compatibility
        if self.current_hierarchical_graph:
            self.current_graph = self.current_hierarchical_graph.fine_graph
        else:
            self.current_graph = None

        # Update switch state tracking
        self.last_switch_states = self._get_switch_states_from_env()

        # Update simple statistics
        total_nodes = (
            self.current_hierarchical_graph.total_nodes
            if self.current_hierarchical_graph
            else 0
        )
        total_edges = (
            self.current_hierarchical_graph.total_edges
            if self.current_hierarchical_graph
            else 0
        )

        update_info = GraphUpdateInfo(
            nodes_updated=total_nodes,
            edges_updated=total_edges,
            update_time_ms=build_time,
            switch_states=self.last_switch_states.copy(),
        )
        self.graph_update_stats["last_update_info"] = update_info

        if self.debug:
            self.logger.debug(
                f"Hierarchical graph rebuilt: {total_nodes} total nodes, "
                f"{total_edges} total edges in {build_time:.2f}ms"
            )
            if self.current_hierarchical_graph:
                self.logger.debug(
                    f"  Fine: {self.current_hierarchical_graph.fine_graph.num_nodes} nodes, "
                    f"{self.current_hierarchical_graph.fine_graph.num_edges} edges"
                )
                self.logger.debug(
                    f"  Medium: {self.current_hierarchical_graph.medium_graph.num_nodes} nodes, "
                    f"{self.current_hierarchical_graph.medium_graph.num_edges} edges"
                )
                self.logger.debug(
                    f"  Coarse: {self.current_hierarchical_graph.coarse_graph.num_nodes} nodes, "
                    f"{self.current_hierarchical_graph.coarse_graph.num_edges} edges"
                )

    def _get_level_data_from_env(self) -> Optional[LevelData]:
        """Extract level data from environment for graph building."""
        # Use existing level_data property
        return self.level_data

    def _get_graph_observations(self) -> Dict[str, np.ndarray]:
        """Get complete graph observations for HGT processing with full 56/6 features."""

        # Initialize empty graph observations with full feature dimensions
        graph_obs = {
            "graph_node_feats": np.zeros(
                (N_MAX_NODES, NODE_FEATURE_DIM), dtype=np.float32
            ),  # 56 dims
            "graph_edge_index": np.zeros((2, E_MAX_EDGES), dtype=np.int32),
            "graph_edge_feats": np.zeros(
                (E_MAX_EDGES, EDGE_FEATURE_DIM), dtype=np.float32
            ),  # 6 dims
            "graph_node_mask": np.zeros(N_MAX_NODES, dtype=np.int32),
            "graph_edge_mask": np.zeros(E_MAX_EDGES, dtype=np.int32),
            "graph_node_types": np.zeros(N_MAX_NODES, dtype=np.int32),
            "graph_edge_types": np.zeros(E_MAX_EDGES, dtype=np.int32),
        }

        # Fill with actual graph data if available
        if self.current_graph is not None:
            # Use the fine-resolution graph for primary observations
            graph_data = self.current_graph

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

    def get_hierarchical_graph_data(self):
        """Get the full hierarchical graph data for advanced processing."""
        return self.current_hierarchical_graph

    def _update_graph_performance_stats(self, update_time_ms: float):
        """Update simple performance statistics."""
        self.graph_update_stats["total_updates"] += 1

        # Simple rolling average
        alpha = 0.1
        self.graph_update_stats["avg_update_time_ms"] = (
            alpha * update_time_ms
            + (1 - alpha) * self.graph_update_stats["avg_update_time_ms"]
        )

    def get_current_graph(self) -> Optional[GraphData]:
        """Get current graph data for external use."""
        return self.current_graph

    def get_graph_performance_stats(self) -> Dict[str, Any]:
        """Get simple performance statistics."""
        return self.graph_update_stats.copy()

    def force_graph_update(self):
        """Force a graph update (for testing/debugging)."""
        if self.enable_graph_updates:
            self._update_graph_from_env_state()

    # Graph debug visualization methods
    def set_graph_debug_enabled(self, enabled: bool):
        """Enable/disable graph debug overlay visualization."""
        self._graph_debug_enabled = bool(enabled)
        # Invalidate cache so next render rebuilds with current state
        self._graph_debug_cache = None

    def _maybe_build_graph_debug(self) -> Optional[GraphData]:
        """Build GraphData for the current state, with dynamic caching that considers door states."""
        # Enhanced cache that considers door states and ninja position
        sim_frame = getattr(self.nplay_headless.sim, "frame", None)
        cached_frame = getattr(self, "_graph_debug_cached_frame", None)

        # Get current door states for cache invalidation
        current_door_states = self._get_door_states_signature()
        cached_door_states = getattr(self, "_graph_debug_cached_door_states", None)

        # Get ninja position for cache invalidation (sub-cell level precision)
        ninja_pos = self.nplay_headless.ninja_position()
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

        # Use centralized extraction logic
        level_data = self._extract_level_data()

        hierarchical_data = self._graph_builder.build_graph(level_data, ninja_pos)

        # Extract sub-cell graph for debug visualization (maintains backward compatibility)
        graph = hierarchical_data.sub_cell_graph

        # Update cache with all relevant state
        self._graph_debug_cache = graph
        setattr(self, "_graph_debug_cached_frame", sim_frame)
        setattr(self, "_graph_debug_cached_door_states", current_door_states)
        setattr(self, "_graph_debug_cached_ninja_sub_cell", ninja_sub_cell)

        return graph

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
        if self.enable_graph_updates and not hasattr(self, "graph_builder"):
            self.graph_builder = HierarchicalGraphBuilder(debug=debug)
