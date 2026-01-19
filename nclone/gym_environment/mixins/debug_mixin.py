"""
Debug and visualization functionality mixin for N++ environment.

This module contains all debug overlay and visualization functionality that was
previously integrated into the main NppEnvironment class.
"""

from typing import Dict, Any, Optional


class DebugMixin:
    """
    Mixin class providing debug and visualization functionality for N++ environment.

    This mixin handles:
    - Debug overlay information generation
    - Graph debug visualization
    - Exploration debug visualization
    - Grid debug visualization
    - Tile type debug visualization
    - Reachability debug visualization
    - Subgoal visualization
    - Mine SDF (Signed Distance Field) visualization
    """

    def _init_debug_system(self, enable_debug_overlay: bool = False):
        """Initialize the debug system components."""
        self._enable_debug_overlay = enable_debug_overlay

        # Debug visualization state
        self._grid_debug_enabled: bool = False
        self._tile_types_debug_enabled: bool = False
        self._reachability_debug_enabled: bool = False
        self._reachability_state = None
        self._reachability_subgoals = []
        self._reachability_frontiers = []

        # Path-aware debug visualization state
        self._adjacency_graph_debug_enabled: bool = False
        self._blocked_entities_debug_enabled: bool = False
        self._show_paths_to_goals: bool = False
        self._path_segments_debug_enabled: bool = False
        self._path_aware_graph_data = None
        self._path_aware_entity_mask = None
        self._path_aware_level_data = None

        # Action mask debug visualization state
        self._action_mask_debug_enabled: bool = False
        self._last_action_taken: Optional[int] = None

        # Reachable wall debug visualization state
        self._reachable_walls_debug_enabled: bool = False

        # Mine SDF (Signed Distance Field) debug visualization state
        self._mine_sdf_debug_enabled: bool = False

        # Demo checkpoint visualization state
        self._demo_checkpoints_debug_enabled: bool = False
        self._demo_checkpoints_data = None

    def _debug_info(self) -> Optional[Dict[str, Any]]:
        """Returns a dictionary containing debug information to be displayed on the screen."""
        info: Dict[str, Any] = {}

        # Add reachability visualization payload if enabled (independent of general debug overlay)
        if self._reachability_debug_enabled and self._reachability_state:
            info["reachability"] = {
                "state": self._reachability_state,
                "subgoals": self._reachability_subgoals,
                "frontiers": self._reachability_frontiers,
            }

        # Add path-aware visualization payload if enabled (independent of general debug overlay)
        if (
            self._adjacency_graph_debug_enabled
            or self._blocked_entities_debug_enabled
            or self._show_paths_to_goals
        ):
            info["path_aware"] = {
                "show_adjacency": self._adjacency_graph_debug_enabled,
                "show_blocked": self._blocked_entities_debug_enabled,
                "show_paths": self._show_paths_to_goals,
                "show_segments": self._show_paths_to_goals,  # Auto-enable when paths shown
                "graph_data": self._path_aware_graph_data,
                "entity_mask": self._path_aware_entity_mask,
                "level_data": self._path_aware_level_data
                or (self.level_data if hasattr(self, "level_data") else None),
                "ninja_position": self.nplay_headless.ninja_position(),
                "ninja_velocity": self.nplay_headless.ninja_velocity(),
                "entities": self.level_data.entities
                if hasattr(self, "level_data")
                else [],
            }

        # Add mine SDF visualization payload if enabled
        if self._mine_sdf_debug_enabled:
            mine_sdf_data = self._get_mine_sdf_data()
            if mine_sdf_data:
                info["mine_sdf"] = mine_sdf_data

        # Add demo checkpoint visualization payload if enabled
        if self._demo_checkpoints_debug_enabled and self._demo_checkpoints_data:
            info["demo_checkpoints"] = {
                "checkpoints": self._demo_checkpoints_data.get("checkpoints", []),
                "ninja_position": self.nplay_headless.ninja_position(),
                "grid_size": 12,  # Match Go-Explore checkpoint discretization
            }

        # Add other debug info only if general debug overlay is enabled
        if self._enable_debug_overlay:
            # Basic environment info
            ninja_x, ninja_y = self.nplay_headless.ninja_position()

            env_info = {
                "frame": self.nplay_headless.sim.frame,
                "current_ep_reward": self.current_ep_reward,
                "current_map_name": self.current_map_name,
                "ninja_position": self.nplay_headless.ninja_position(),
                "ninja_velocity": self.nplay_headless.ninja_velocity(),
            }

            # Add PBRS surface area if available
            pbrs_surface_area = self._get_pbrs_surface_area()
            if pbrs_surface_area is not None:
                env_info["pbrs_surface_area"] = pbrs_surface_area

            info.update(env_info)

            # Add grid outline debug info if enabled
            if self._grid_debug_enabled:
                info["grid_outline"] = True

            # Add tile types debug info if enabled
            if self._tile_types_debug_enabled:
                info["tile_types"] = True

        # Add action mask visualization if enabled
        if self._action_mask_debug_enabled:
            # Get action mask from cached observation or fresh observation
            action_mask = None
            if hasattr(self, "_cached_observation") and self._cached_observation:
                action_mask = self._cached_observation.get("action_mask")
            elif hasattr(self, "nplay_headless") and hasattr(
                self.nplay_headless, "sim"
            ):
                sim = self.nplay_headless.sim
                if hasattr(sim, "ninja"):
                    action_mask = sim.ninja.get_valid_action_mask()

            if action_mask is not None:
                info["action_mask"] = {
                    "action_mask": action_mask,
                    "ninja_position": self.nplay_headless.ninja_position(),
                    "last_action": self._last_action_taken,
                }

        return info if info else None  # Return None if no debug info is to be shown

    def _get_pbrs_surface_area(self) -> Optional[float]:
        """Get PBRS calculated surface area if available.

        Returns:
            Surface area (number of reachable sub-nodes) if available, None otherwise.
            Surface area is computed when PBRS potential is first calculated for a level.
        """
        if hasattr(self, "reward_calculator") and hasattr(
            self.reward_calculator, "pbrs_calculator"
        ):
            pbrs_calc = self.reward_calculator.pbrs_calculator
            if hasattr(pbrs_calc, "_cached_surface_area"):
                return pbrs_calc._cached_surface_area
        return None

    def set_grid_debug_enabled(self, enabled: bool):
        """Enable/disable grid outline debug overlay visualization."""
        self._grid_debug_enabled = bool(enabled)

    def set_tile_types_debug_enabled(self, enabled: bool):
        """Enable/disable tile type overlay visualization."""
        self._tile_types_debug_enabled = bool(enabled)

    def set_tile_rendering_enabled(self, enabled: bool):
        """Enable/disable tile rendering."""
        if hasattr(self.nplay_headless, "sim_renderer"):
            self.nplay_headless.sim_renderer.tile_rendering_enabled = bool(enabled)

    def set_reachability_debug_enabled(self, enabled: bool):
        """Enable/disable reachability analysis debug overlay visualization."""
        self._reachability_debug_enabled = bool(enabled)

    def set_reachability_data(self, reachability_state, subgoals=None, frontiers=None):
        """Set reachability analysis data for visualization."""
        self._reachability_state = reachability_state
        self._reachability_subgoals = subgoals or []
        self._reachability_frontiers = frontiers or []

    @property
    def debug_overlay_renderer(self):
        """Access the debug overlay renderer."""
        if hasattr(self.nplay_headless, "sim_renderer") and hasattr(
            self.nplay_headless.sim_renderer, "debug_overlay_renderer"
        ):
            return self.nplay_headless.sim_renderer.debug_overlay_renderer
        return None

    def set_adjacency_graph_debug_enabled(self, enabled: bool):
        """Enable/disable adjacency graph debug visualization."""
        self._adjacency_graph_debug_enabled = bool(enabled)

    def set_blocked_entities_debug_enabled(self, enabled: bool):
        """Enable/disable blocked entities debug visualization."""
        self._blocked_entities_debug_enabled = bool(enabled)

    def set_show_paths_to_goals(self, enabled: bool):
        """Enable/disable path to goals visualization."""
        self._show_paths_to_goals = bool(enabled)

    def set_path_segments_debug_enabled(self, enabled: bool):
        """Enable/disable path segment visualization."""
        self._path_segments_debug_enabled = bool(enabled)

    def set_path_aware_data(self, graph_data=None, entity_mask=None, level_data=None):
        """Set path-aware graph and entity mask data for visualization."""
        self._path_aware_graph_data = graph_data
        self._path_aware_entity_mask = entity_mask
        self._path_aware_level_data = level_data

    def set_action_mask_debug_enabled(self, enabled: bool):
        """Enable/disable action mask debug visualization."""
        self._action_mask_debug_enabled = bool(enabled)

    def _record_action_for_debug(self, action: int):
        """Record the action taken for debug visualization."""
        self._last_action_taken = action

    def set_mine_sdf_debug_enabled(self, enabled: bool):
        """Enable/disable mine SDF (Signed Distance Field) debug visualization.

        Shows a heatmap overlay of the precomputed distance field to deadly mines.
        Red = danger zone (close to mines), Green = safe (far from mines).
        Also shows escape direction arrows at each tile.
        """
        self._mine_sdf_debug_enabled = bool(enabled)

    def set_demo_checkpoints_debug_enabled(self, enabled: bool):
        """Enable/disable demo checkpoint heatmap visualization.

        Shows expert demonstration trajectories as a heatmap overlay,
        with colors indicating cumulative reward progression.
        """
        self._demo_checkpoints_debug_enabled = bool(enabled)

    def set_demo_checkpoints_data(self, checkpoints: list):
        """Set demo checkpoint data for visualization.

        Args:
            checkpoints: List of checkpoint dicts with keys:
                - cell: (x, y) discretized position
                - cumulative_reward: Reward value at this checkpoint
                - position: (x, y) actual position in pixels
        """
        self._demo_checkpoints_data = {"checkpoints": checkpoints}

    def _get_mine_sdf_data(self) -> Optional[Dict[str, Any]]:
        """Get mine SDF data for visualization.

        Returns:
            Dictionary with SDF grid, gradient grid, ninja position, and
            deadly mine positions, or None if SDF is not available.
        """
        # Try to get the mine SDF from the path calculator
        if not hasattr(self, "_path_calculator") or self._path_calculator is None:
            return None

        if not hasattr(self._path_calculator, "mine_sdf"):
            return None

        mine_sdf = self._path_calculator.mine_sdf
        if mine_sdf.sdf_grid is None:
            return None

        # Extract deadly mine positions from level_data
        deadly_mine_positions = []
        if hasattr(self, "level_data") and self.level_data is not None:
            from ...constants.entity_types import EntityType

            mines = self.level_data.get_all_toggle_mines()

            for mine in mines:
                mine_state = mine.get("state", 0)
                # State 0 = deadly (toggled), per EntityToggleMine
                if mine_state == 0:
                    mine_x = mine.get("x", 0)
                    mine_y = mine.get("y", 0)
                    deadly_mine_positions.append((mine_x, mine_y))

        return {
            "sdf_grid": mine_sdf.sdf_grid,
            "gradient_grid": mine_sdf.gradient_grid,
            "danger_radius": mine_sdf.danger_radius,
            "ninja_position": self.nplay_headless.ninja_position(),
            "deadly_mine_positions": deadly_mine_positions,
        }
