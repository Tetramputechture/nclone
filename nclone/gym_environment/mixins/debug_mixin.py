"""
Debug and visualization functionality mixin for N++ environment.

This module contains all debug overlay and visualization functionality that was
previously integrated into the main NppEnvironment class.
"""

import numpy as np
from typing import Dict, Any, Optional


class DebugMixin:
    """
    Mixin class providing debug and visualization functionality for N++ environment.

    This mixin handles:
    - Debug overlay information generation
    - Graph debug visualization
    - Exploration debug visualization
    - Grid debug visualization
    - Reachability debug visualization
    - Subgoal visualization
    """

    def _init_debug_system(self, enable_debug_overlay: bool = False):
        """Initialize the debug system components."""
        self._enable_debug_overlay = enable_debug_overlay

        # Debug visualization state
        self._exploration_debug_enabled: bool = False
        self._grid_debug_enabled: bool = False
        self._reachability_debug_enabled: bool = False
        self._reachability_state = None
        self._reachability_subgoals = []
        self._reachability_frontiers = []

    def _debug_info(self) -> Optional[Dict[str, Any]]:
        """Returns a dictionary containing debug information to be displayed on the screen."""
        info: Dict[str, Any] = {}

        # Add graph visualization payload if enabled (independent of general debug overlay)
        if self._graph_debug_enabled:
            if self._graph_builder is None:
                from ...graph.hierarchical_builder import HierarchicalGraphBuilder

                self._graph_builder = HierarchicalGraphBuilder()
            graph_data = self._maybe_build_graph_debug()
            if graph_data is not None:
                info["graph"] = {
                    "data": graph_data,
                }

        # Add reachability visualization payload if enabled (independent of general debug overlay)
        if self._reachability_debug_enabled and self._reachability_state:
            info["reachability"] = {
                "state": self._reachability_state,
                "subgoals": self._reachability_subgoals,
                "frontiers": self._reachability_frontiers,
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
            info.update(env_info)

            # Add exploration debug info if enabled
            if self._exploration_debug_enabled:
                exploration_info = self._get_exploration_debug_info(ninja_x, ninja_y)
                info["exploration"] = exploration_info

            # Add grid outline debug info if enabled
            if self._grid_debug_enabled:
                info["grid_outline"] = True

        return info if info else None  # Return None if no debug info is to be shown

    def _get_exploration_debug_info(
        self, ninja_x: float, ninja_y: float
    ) -> Dict[str, Any]:
        """Get exploration debug information."""
        cell_x, cell_y = self.reward_calculator.exploration_calculator._get_cell_coords(
            ninja_x, ninja_y
        )
        area_4x4_x = cell_x // 4
        area_4x4_y = cell_y // 4
        area_8x8_x = cell_x // 8
        area_8x8_y = cell_y // 8
        area_16x16_x = cell_x // 16
        area_16x16_y = cell_y // 16

        return {
            "current_cell": (cell_x, cell_y),
            "current_4x4_area": (area_4x4_x, area_4x4_y),
            "current_8x8_area": (area_8x8_x, area_8x8_y),
            "current_16x16_area": (area_16x16_x, area_16x16_y),
            "visited_cells": self.reward_calculator.exploration_calculator.visited_cells,
            "visited_4x4": self.reward_calculator.exploration_calculator.visited_4x4,
            "visited_8x8": self.reward_calculator.exploration_calculator.visited_8x8,
            "visited_16x16": self.reward_calculator.exploration_calculator.visited_16x16,
            "visited_cells_count": np.sum(
                self.reward_calculator.exploration_calculator.visited_cells
            ),
            "visited_4x4_count": np.sum(
                self.reward_calculator.exploration_calculator.visited_4x4
            ),
            "visited_8x8_count": np.sum(
                self.reward_calculator.exploration_calculator.visited_8x8
            ),
            "visited_16x16_count": np.sum(
                self.reward_calculator.exploration_calculator.visited_16x16
            ),
        }

    def set_exploration_debug_enabled(self, enabled: bool):
        """Enable/disable exploration debug overlay visualization."""
        self._exploration_debug_enabled = bool(enabled)

    def set_grid_debug_enabled(self, enabled: bool):
        """Enable/disable grid outline debug overlay visualization."""
        self._grid_debug_enabled = bool(enabled)

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

    def set_subgoal_debug_enabled(self, enabled: bool):
        """Enable or disable subgoal visualization."""
        renderer = self.debug_overlay_renderer
        if renderer:
            renderer.set_subgoal_debug_enabled(enabled)

    def set_subgoal_visualization_mode(self, mode: str):
        """Set subgoal visualization mode."""
        renderer = self.debug_overlay_renderer
        if renderer:
            renderer.set_subgoal_visualization_mode(mode)

    def set_subgoal_data(self, subgoals, plan=None, reachable_positions=None):
        """Set subgoal data for visualization."""
        renderer = self.debug_overlay_renderer
        if renderer:
            renderer.set_subgoal_data(subgoals, plan, reachable_positions)

    def export_subgoal_visualization(self, filename: str) -> bool:
        """Export subgoal visualization to image file."""
        renderer = self.debug_overlay_renderer
        if renderer:
            return renderer.export_subgoal_visualization(filename)
        return False
