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
    - Tile type debug visualization
    - Reachability debug visualization
    - Subgoal visualization
    """

    def _init_debug_system(self, enable_debug_overlay: bool = False):
        """Initialize the debug system components."""
        self._enable_debug_overlay = enable_debug_overlay

        # Debug visualization state
        self._exploration_debug_enabled: bool = False
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
        self._path_aware_graph_data = None
        self._path_aware_entity_mask = None
        self._path_aware_level_data = None

        # Mine predictor debug visualization state
        self._mine_predictor_debug_enabled: bool = False
        self._death_probability_debug_enabled: bool = False
        self._death_probability_frames: int = 10  # Number of frames to simulate

        # Terminal velocity predictor debug visualization state
        self._terminal_velocity_probability_debug_enabled: bool = False
        self._terminal_velocity_probability_frames: int = (
            10  # Number of frames to simulate
        )

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
                "graph_data": self._path_aware_graph_data,
                "entity_mask": self._path_aware_entity_mask,
                "level_data": self._path_aware_level_data
                or (self.level_data if hasattr(self, "level_data") else None),
                "ninja_position": self.nplay_headless.ninja_position(),
                "entities": self.level_data.entities
                if hasattr(self, "level_data")
                else [],
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

            # Add exploration debug info if enabled
            if self._exploration_debug_enabled:
                exploration_info = self._get_exploration_debug_info(ninja_x, ninja_y)
                info["exploration"] = exploration_info

            # Add grid outline debug info if enabled
            if self._grid_debug_enabled:
                info["grid_outline"] = True

            # Add tile types debug info if enabled
            if self._tile_types_debug_enabled:
                info["tile_types"] = True

        # Add mine predictor visualization if enabled (independent of general debug overlay)
        if self._mine_predictor_debug_enabled:
            predictor = None
            if hasattr(self, "nplay_headless") and hasattr(self.nplay_headless, "sim"):
                sim = self.nplay_headless.sim
                if hasattr(sim, "ninja") and hasattr(sim.ninja, "mine_death_predictor"):
                    predictor = sim.ninja.mine_death_predictor

            if predictor:
                info["mine_predictor"] = {
                    "mine_positions": predictor.mine_positions,
                    "danger_zone_cells": predictor.danger_zone_cells,
                    "stats": predictor.stats,
                    "ninja_position": self.nplay_headless.ninja_position(),
                }

        # Add mine death probability visualization if enabled
        if self._death_probability_debug_enabled:
            predictor = None
            if hasattr(self, "nplay_headless") and hasattr(self.nplay_headless, "sim"):
                sim = self.nplay_headless.sim
                if hasattr(sim, "ninja") and hasattr(sim.ninja, "mine_death_predictor"):
                    predictor = sim.ninja.mine_death_predictor

            if predictor and predictor.mine_positions:
                # Calculate death probability for current state
                death_prob_result = predictor.calculate_death_probability(
                    frames_to_simulate=self._death_probability_frames
                )
                info["death_probability"] = {
                    "result": death_prob_result,
                    "ninja_position": self.nplay_headless.ninja_position(),
                }

        # Add terminal velocity death probability visualization if enabled
        if self._terminal_velocity_probability_debug_enabled:
            terminal_velocity_predictor = None
            if hasattr(self, "nplay_headless") and hasattr(self.nplay_headless, "sim"):
                sim = self.nplay_headless.sim
                if hasattr(sim, "ninja") and hasattr(
                    sim.ninja, "terminal_velocity_predictor"
                ):
                    terminal_velocity_predictor = sim.ninja.terminal_velocity_predictor

            if terminal_velocity_predictor:
                # Calculate terminal velocity death probability for current state
                terminal_velocity_death_prob_result = (
                    terminal_velocity_predictor.calculate_death_probability(
                        frames_to_simulate=self._terminal_velocity_probability_frames
                    )
                )
                info["terminal_velocity_probability"] = {
                    "result": terminal_velocity_death_prob_result,
                    "ninja_position": self.nplay_headless.ninja_position(),
                }

        return info if info else None  # Return None if no debug info is to be shown

    def _get_pbrs_surface_area(self) -> Optional[float]:
        """Get PBRS calculated surface area if available.

        Returns:
            Surface area (number of reachable sub-nodes) if available, None otherwise.
            Surface area is computed when PBRS potential is first calculated for a level.
        """
        try:
            if hasattr(self, "reward_calculator") and hasattr(
                self.reward_calculator, "pbrs_calculator"
            ):
                pbrs_calc = self.reward_calculator.pbrs_calculator
                if hasattr(pbrs_calc, "_cached_surface_area"):
                    return pbrs_calc._cached_surface_area
        except Exception:
            # Silently fail if PBRS calculator not available or not initialized
            pass
        return None

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

    def set_adjacency_graph_debug_enabled(self, enabled: bool):
        """Enable/disable adjacency graph debug visualization."""
        self._adjacency_graph_debug_enabled = bool(enabled)

    def set_blocked_entities_debug_enabled(self, enabled: bool):
        """Enable/disable blocked entities debug visualization."""
        self._blocked_entities_debug_enabled = bool(enabled)

    def set_show_paths_to_goals(self, enabled: bool):
        """Enable/disable path to goals visualization."""
        self._show_paths_to_goals = bool(enabled)

    def set_path_aware_data(self, graph_data=None, entity_mask=None, level_data=None):
        """Set path-aware graph and entity mask data for visualization."""
        self._path_aware_graph_data = graph_data
        self._path_aware_entity_mask = entity_mask
        self._path_aware_level_data = level_data

    def set_mine_predictor_debug_enabled(self, enabled: bool):
        """Enable/disable mine death predictor debug visualization."""
        self._mine_predictor_debug_enabled = bool(enabled)

    def set_death_probability_debug_enabled(self, enabled: bool):
        """Enable/disable mine death probability debug visualization."""
        self._death_probability_debug_enabled = bool(enabled)

    def set_death_probability_frames(self, frames: int):
        """Set number of frames to simulate for mine death probability calculation."""
        self._death_probability_frames = max(1, min(frames, 30))  # Clamp to [1, 30]

    def set_terminal_velocity_probability_debug_enabled(self, enabled: bool):
        """Enable/disable terminal velocity death probability debug visualization."""
        self._terminal_velocity_probability_debug_enabled = bool(enabled)

    def set_terminal_velocity_probability_frames(self, frames: int):
        """Set number of frames to simulate for terminal velocity death probability calculation."""
        self._terminal_velocity_probability_frames = max(
            1, min(frames, 30)
        )  # Clamp to [1, 30]
