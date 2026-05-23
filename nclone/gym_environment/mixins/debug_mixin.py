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

        # Combined graph + mine SDF + path mine proximity (single toggle, test_environment H)
        self._hazard_debug_combo_enabled: bool = False

        # Demo checkpoint visualization state
        self._demo_checkpoints_debug_enabled: bool = False
        self._demo_checkpoints_data = None

        # Value function (PBRS cost-to-go) visualization state
        self._value_function_debug_enabled: bool = False
        self._value_function_pbrs = None  # SurfaceGraphPBRS or TwoPhaseValuePBRS

        # Analytical reachability overlay state
        self._reachability_overlay_enabled: bool = False
        self._hull_collection = None  # JumpHullCollection (floor)
        self._wall_hull_collection = None  # WallHullCollection (wall jumps)
        self._last_reachability_overlay_data: Optional[Dict[str, Any]] = None

        # PBRS path visualization state
        self._pbrs_path_overlay_enabled: bool = False
        self._pbrs_path_cached_data: Optional[Dict[str, Any]] = None

        # Step-on-input overlay state (input sequence for frame-perfect testing)
        self._input_sequence_overlay_enabled: bool = False
        self._input_sequence_overlay_data: list = []

    def set_input_sequence_overlay(self, sequence: list, enabled: bool):
        """Set input sequence overlay data for step-on-input mode.

        When enabled, displays the sequence of actions (frame-perfect input log)
        in the debug overlay. Used by test_environment.py in --step-on-input mode.

        Args:
            sequence: List of action indices (0-5) executed so far this episode.
            enabled: Whether to show the overlay.
        """
        self._input_sequence_overlay_enabled = bool(enabled)
        self._input_sequence_overlay_data = list(sequence) if sequence else []

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
            or self._hazard_debug_combo_enabled
        ):
            # Get straightness masking debug data if available
            straightness_debug = None
            if hasattr(self.nplay_headless, "sim") and hasattr(
                self.nplay_headless.sim, "_straightness_masking_debug"
            ):
                straightness_debug = self.nplay_headless.sim._straightness_masking_debug

            # CRITICAL: Use current_graph_data to ensure visualization matches level_cache
            # The level_cache (used for rewards and masking) is built from current_graph_data,
            # so visualization must use the same graph to show consistent paths
            graph_data_for_viz = getattr(
                self, "current_graph_data", self._path_aware_graph_data
            )

            show_adjacency = (
                self._adjacency_graph_debug_enabled
                or self._hazard_debug_combo_enabled
            )
            mine_proximity_by_node = None
            if (
                self._hazard_debug_combo_enabled
                or (
                    self._mine_sdf_debug_enabled
                    and self._adjacency_graph_debug_enabled
                )
            ):
                mine_proximity_by_node = self._get_mine_proximity_node_multipliers()

            info["path_aware"] = {
                "show_adjacency": show_adjacency,
                "show_blocked": self._blocked_entities_debug_enabled,
                "show_paths": self._show_paths_to_goals,
                "show_segments": self._show_paths_to_goals,  # Auto-enable when paths shown
                "graph_data": graph_data_for_viz,  # Use current_graph_data for consistency
                "entity_mask": self._path_aware_entity_mask,
                "level_data": self._path_aware_level_data
                or (self.level_data if hasattr(self, "level_data") else None),
                "ninja_position": self.nplay_headless.ninja_position(),
                "ninja_velocity": self.nplay_headless.ninja_velocity(),
                "entities": self.level_data.entities
                if hasattr(self, "level_data")
                else [],
                "cached_reward_paths": getattr(self, "_cached_reward_paths", None),
                "straightness_masking_debug": straightness_debug,
                "hazard_combo": self._hazard_debug_combo_enabled,
                "mine_proximity_by_node": mine_proximity_by_node,
            }

        # Add mine SDF visualization payload if enabled
        if self._mine_sdf_debug_enabled or self._hazard_debug_combo_enabled:
            mine_sdf_data = self._get_mine_sdf_data()
            if mine_sdf_data:
                info["mine_sdf"] = mine_sdf_data

        # Add demo checkpoint visualization payload if enabled
        if self._demo_checkpoints_debug_enabled and self._demo_checkpoints_data:
            info["demo_checkpoints"] = {
                "checkpoints": self._demo_checkpoints_data.get("checkpoints", []),
                "ninja_position": self.nplay_headless.ninja_position(),
                "grid_size": 6,  # Match sub-node grid resolution
            }

        # Add analytical reachability overlay if enabled
        if self._reachability_overlay_enabled:
            reachability_data = self._get_reachability_overlay_data()
            if reachability_data:
                info["reachability_overlay"] = reachability_data

        # Add value function (PBRS cost-to-go) visualization if enabled
        if self._value_function_debug_enabled and self._value_function_pbrs:
            vf_info = self._get_value_function_debug_info()
            if vf_info:
                info["value_function"] = vf_info

        # Add PBRS path overlay if enabled
        if self._pbrs_path_overlay_enabled:
            pbrs_path_data = self._get_pbrs_path_overlay_data()
            if pbrs_path_data:
                info["pbrs_path_overlay"] = pbrs_path_data

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

        # Add step-on-input overlay if enabled
        if self._input_sequence_overlay_enabled:
            info["input_sequence"] = {
                "sequence": self._input_sequence_overlay_data,
                "step_count": len(self._input_sequence_overlay_data),
                "ninja_position": self.nplay_headless.ninja_position(),
            }

        # Mechanic benchmark HUD (e.g. test_environment sets _benchmark_hud_lines)
        bench = getattr(self, "_benchmark_hud_lines", None)
        if bench:
            info["benchmark_hud"] = list(bench)

        return info if info else None  # Return None if no debug info is to be shown

    def _get_pbrs_surface_area(self) -> Optional[float]:
        """Get cached level surface area if available.

        Returns the module-level cached value from
        ``nclone.graph.reachability.surface_area`` when this env's current level
        is represented there; otherwise returns None.
        """
        try:
            level_data = getattr(self, "level_data", None)
            if level_data is None:
                return None
            from ...graph.reachability.surface_area import _surface_area_cache

            cache_key = level_data.get_cache_key_for_reachability(
                include_switch_states=True
            )
            return _surface_area_cache.get(cache_key)
        except Exception:
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

    def set_path_aware_data(
        self, graph_data=None, entity_mask=None, level_data=None, cached_paths=None
    ):
        """Set path-aware graph and entity mask data for visualization.

        Args:
            graph_data: Graph adjacency data
            entity_mask: Entity blocking data
            level_data: Level data for mine avoidance
            cached_paths: Pre-computed paths from reward calculator (ensures viz matches rewards)
        """
        self._path_aware_graph_data = graph_data
        self._path_aware_entity_mask = entity_mask
        self._path_aware_level_data = level_data
        self._cached_reward_paths = cached_paths

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

    def set_hazard_debug_combo_enabled(self, enabled: bool):
        """Enable combined hazard overlay: adjacency graph, mine SDF, and path mine cost multipliers on nodes."""
        self._hazard_debug_combo_enabled = bool(enabled)

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

    def set_pbrs_path_overlay_enabled(self, enabled: bool):
        """Enable/disable PBRS shortest-path edge visualization.

        Shows the Dijkstra shortest path from the ninja's current position
        to the active goal (switch or exit), color-coded by edge type
        (walk vs jump), plus a cost-to-go heatmap on all surface nodes.

        Requires the reachability PBRS to be built (use_reachability_pbrs=True
        or manually build via build_reachability_pbrs()).
        """
        self._pbrs_path_overlay_enabled = bool(enabled)
        if not enabled:
            self._pbrs_path_cached_data = None
            self._pbrs_path_last_pos = None
            self._pbrs_path_last_phase = None

    def set_reachability_overlay_enabled(self, enabled: bool):
        """Enable/disable hull-based reachability overlay.

        Shows the precomputed jump hull as a shaded area, with two layers:
        - Full physics hull (everywhere the ninja CAN go, ignoring geometry)
        - Level-masked subset (hull intersected with reachable node set)

        Initializes the hull collection on first enable (~1.5 s).
        """
        self._reachability_overlay_enabled = bool(enabled)
        if not enabled:
            self._last_reachability_overlay_data = None

        # Lazy-initialize hull collections on first enable (~1.5 s floor, ~2.3 s wall)
        if enabled:
            if self._hull_collection is None:
                from nclone.analytical_physics.jump_hull import get_or_build_hulls

                self._hull_collection = get_or_build_hulls()
            if self._wall_hull_collection is None:
                from nclone.analytical_physics.jump_hull import get_or_build_wall_hulls

                self._wall_hull_collection = get_or_build_wall_hulls()

    def _get_reachability_overlay_data(self) -> Optional[Dict[str, Any]]:
        """Build hull-based reachability overlay payload for the debug renderer.

        Looks up the precomputed hull for the ninja's current surface type and
        velocity bin (floor) or wall side/jump type/vy bin (wall), translates to
        absolute coordinates, and computes the level-masked subset using the
        entity-masked adjacency graph.
        """
        if not hasattr(self.nplay_headless, "sim") or self.nplay_headless.sim is None:
            return None

        sim = self.nplay_headless.sim
        if not hasattr(sim, "ninja"):
            return None

        ninja = sim.ninja
        ninja_pos = (ninja.xpos, ninja.ypos)
        ninja_vel = (ninja.xspeed, ninja.yspeed)

        from nclone.analytical_physics.jump_hull import GRID_CELL
        from nclone.analytical_physics.reachability_graph import velocity_to_bin

        graph_data = getattr(self, "current_graph_data", self._path_aware_graph_data)
        reachable_set = (
            set(graph_data["adjacency"].keys())
            if graph_data and "adjacency" in graph_data
            else set()
        )

        # Wall hull: when ninja is walled (state 5 or grounded+walled)
        if ninja.walled and self._wall_hull_collection is not None:
            wall_normal_sign = int(ninja.wall_normal)
            # Slide jump: state 5 and pressing into wall (hor_input * wall_normal <= 0)
            jump_type = (
                "slide"
                if ninja.state == 5 and ninja.hor_input * ninja.wall_normal <= 0
                else "regular"
            )
            vy_bin = velocity_to_bin(ninja.yspeed, False)  # is_floor=False for wall
            hull = self._wall_hull_collection.get_hull(
                wall_normal_sign,
                jump_type,
                vy_bin,
            )
            if hull:
                sx, sy = ninja.xpos, ninja.ypos
                hull_cells = []
                for (cdx, cdy), cell in hull.items():
                    abs_x = sx + cdx * GRID_CELL
                    abs_y = sy + cdy * GRID_CELL
                    hull_cells.append((abs_x, abs_y, cell.min_frames))

                masked_cells = []
                for abs_x, abs_y, min_frames in hull_cells:
                    key = (round(abs_x), round(abs_y))
                    if key in reachable_set:
                        masked_cells.append((abs_x, abs_y, min_frames))

                wall_side = "L" if wall_normal_sign > 0 else "R"
                contact_type = f"wall-{wall_side} {jump_type}"
                data = {
                    "hull_cells": hull_cells,
                    "masked_cells": masked_cells,
                    "ninja_position": ninja_pos,
                    "ninja_velocity": ninja_vel,
                    "state": ninja.state,
                    "contact_type": contact_type,
                    "surface_type": -1,  # No floor surface for wall
                    "walled": ninja.walled,
                    "wall_normal": ninja.wall_normal,
                    "jump_type": jump_type,
                    "airborn": ninja.airborn,
                }
                self._last_reachability_overlay_data = data
                return data

        # Floor hull: when grounded (states 0, 1, 2)
        if ninja.state in (0, 1, 2) and self._hull_collection is not None:
            from nclone.analytical_physics.jump_hull import classify_surface

            surface_type = classify_surface(
                ninja.floor_normalized_x,
                ninja.floor_normalized_y,
            )
            vb = velocity_to_bin(ninja.xspeed, True)  # is_floor=True
            hull = self._hull_collection.get_hull(surface_type, vb)

            if not hull:
                return None

            sx, sy = ninja.xpos, ninja.ypos
            hull_cells = []
            for (cdx, cdy), cell in hull.items():
                abs_x = sx + cdx * GRID_CELL
                abs_y = sy + cdy * GRID_CELL
                hull_cells.append((abs_x, abs_y, cell.min_frames))

            masked_cells = []
            for abs_x, abs_y, min_frames in hull_cells:
                key = (round(abs_x), round(abs_y))
                if key in reachable_set:
                    masked_cells.append((abs_x, abs_y, min_frames))

            contact_names = {
                0: "immobile",
                1: "floor",
                2: "floor",
                3: "jumping",
                4: "falling",
                5: "wall",
            }
            contact_type = contact_names.get(ninja.state, "unknown")

            data = {
                "hull_cells": hull_cells,
                "masked_cells": masked_cells,
                "ninja_position": ninja_pos,
                "walled": ninja.walled,
                "wall_normal": ninja.wall_normal if ninja.walled else None,
                "jump_type": "floor",
                "airborn": ninja.airborn,
                "ninja_velocity": ninja_vel,
                "state": ninja.state,
                "contact_type": contact_type,
                "surface_type": surface_type,
            }
            self._last_reachability_overlay_data = data
            return data

        # Airborne: show last hull from ground/wall, with current position/velocity
        if self._last_reachability_overlay_data is not None:
            cached = dict(self._last_reachability_overlay_data)
            cached["ninja_position"] = ninja_pos
            cached["ninja_velocity"] = ninja_vel
            cached["state"] = ninja.state
            cached["contact_type"] = f"airborne (from {cached['contact_type']})"
            cached["walled"] = ninja.walled
            cached["wall_normal"] = ninja.wall_normal if ninja.walled else None
            cached["airborn"] = ninja.airborn
            return cached

        return None

    def set_value_function_debug_enabled(self, enabled: bool):
        """Enable/disable value function (PBRS cost-to-go) overlay.

        Shows real-time potential and cost-to-go to help understand
        how PBRS shapes rewards during training.
        """
        self._value_function_debug_enabled = bool(enabled)

    def set_value_function_pbrs(self, pbrs):
        """Set the PBRS instance for value function visualization.

        Args:
            pbrs: SurfaceGraphPBRS or TwoPhaseValuePBRS instance.
        """
        self._value_function_pbrs = pbrs

    def _get_value_function_debug_info(self) -> Optional[Dict[str, Any]]:
        """Build value function debug payload for overlay."""
        if not self._value_function_pbrs:
            return None

        pbrs = self._value_function_pbrs
        inner = getattr(pbrs, "_inner", pbrs)

        ninja_pos = self.nplay_headless.ninja_position()
        ninja_vel = self.nplay_headless.ninja_velocity()
        switch_activated = self.nplay_headless.exit_switch_activated()

        state_code = 2
        if hasattr(self.nplay_headless, "sim") and self.nplay_headless.sim:
            sim = self.nplay_headless.sim
            if hasattr(sim, "get_ninja_state_code"):
                state_code = sim.get_ninja_state_code()
            elif hasattr(sim, "ninja") and hasattr(sim.ninja, "state"):
                state_code = sim.ninja.state

        from ...value_function.physics_step_wrapper import PhysicsStepWrapper

        contact_state = PhysicsStepWrapper.derive_contact_state(state_code)

        state_dict = {
            "player_x": ninja_pos[0],
            "player_y": ninja_pos[1],
            "player_xspeed": ninja_vel[0],
            "player_yspeed": ninja_vel[1],
            "switch_activated": switch_activated,
            "contact_state": contact_state,
        }

        potential = pbrs.get_potential(state_dict)

        from ...value_function.precomputed_pbrs import C_PHASE

        d_max = inner.d_max_exit if switch_activated else inner.d_max_switch
        phi_phase = C_PHASE if switch_activated else 0.0
        phi_state = potential - phi_phase
        cost_frames = max(0.0, -phi_state * d_max) if d_max > 0 else float("inf")

        gradient = None
        try:
            gradient = pbrs.get_gradient(state_dict, epsilon=6.0)
        except Exception:
            pass

        return {
            "potential": potential,
            "cost_frames": cost_frames,
            "phase": "exit" if switch_activated else "switch",
            "contact_state": contact_state,
            "ninja_position": ninja_pos,
            "gradient": gradient,
            "show_heatmap": False,
        }

    def _get_mine_proximity_node_multipliers(self) -> Optional[Dict[tuple, float]]:
        """Node positions -> mine path cost multiplier from CachedPathDistanceCalculator, if any."""
        if not hasattr(self, "_path_calculator") or self._path_calculator is None:
            return None
        mpc = getattr(self._path_calculator, "mine_proximity_cache", None)
        if mpc is None or not hasattr(mpc, "cache"):
            return None
        cache = getattr(mpc, "cache", None)
        if not cache:
            return None
        return dict(cache)

    def _get_mine_sdf_data(self) -> Optional[Dict[str, Any]]:
        """Get mine SDF data for visualization.

        Returns:
            Dictionary with SDF grid, gradient grid, ninja position, and
            deadly mine positions, or None if SDF is not available.
        """
        from nclone.graph.reachability.mine_proximity_cache import SDF_CELL_SIZE

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
            "sdf_cell_size": SDF_CELL_SIZE,
            "ninja_position": self.nplay_headless.ninja_position(),
            "deadly_mine_positions": deadly_mine_positions,
        }

    def _get_pbrs_path_overlay_data(self) -> Optional[Dict[str, Any]]:
        """Build PBRS shortest-path overlay data for the debug renderer.

        Traces the Dijkstra shortest path from the ninja's current position
        to the active goal. Recomputed only when the ninja moves at least
        12px (one graph cell) or the goal phase changes.
        """
        pbrs = getattr(self, "_current_reachability_pbrs", None)
        if pbrs is None:
            return None

        if not hasattr(self.nplay_headless, "sim") or self.nplay_headless.sim is None:
            return None

        sim = self.nplay_headless.sim
        if not hasattr(sim, "ninja"):
            return None

        ninja = sim.ninja

        # Distance-based throttle: recompute when ninja moves >= 12px
        # or when the goal phase changes (switch collected).
        current_pos = (ninja.xpos, ninja.ypos)
        current_phase = (
            "exit" if self.nplay_headless.exit_switch_activated() else "switch"
        )
        last_pos = getattr(self, "_pbrs_path_last_pos", None)
        last_phase = getattr(self, "_pbrs_path_last_phase", None)
        cached = getattr(self, "_pbrs_path_cached_data", None)

        if cached is not None and last_pos is not None and last_phase == current_phase:
            dx = current_pos[0] - last_pos[0]
            dy = current_pos[1] - last_pos[1]
            if dx * dx + dy * dy < 144.0:  # 12^2 = 144
                return cached

        self._pbrs_path_last_pos = current_pos
        self._pbrs_path_last_phase = current_phase

        try:
            from nclone.analytical_physics.reachability_graph import (
                trace_shortest_path,
                find_nearest_node,
            )

            graph = pbrs.graph

            # Determine phase and select appropriate d_star
            switch_activated = self.nplay_headless.exit_switch_activated()
            if switch_activated:
                d_star = pbrs.d_star_exit
                d_max = pbrs.d_max_exit
                goal_pos = self.nplay_headless.exit_door_position()
                phase = "exit"
            else:
                d_star = pbrs.d_star_switch
                d_max = pbrs.d_max_switch
                goal_pos = self.nplay_headless.exit_switch_position()
                phase = "switch"

            # Find nearest node to ninja
            # Graph positions use inner-grid coords (playable area only); ninja uses
            # full-map coords. Convert: graph_space = full_map - TILE_PIXEL_SIZE.
            from nclone.constants.physics_constants import TILE_PIXEL_SIZE

            ninja_state = ninja.state
            is_floor = ninja_state in (0, 1, 2)  # immobile, running, ground_sliding
            vel = ninja.xspeed if is_floor else ninja.yspeed
            gx = ninja.xpos - TILE_PIXEL_SIZE
            gy = ninja.ypos - TILE_PIXEL_SIZE
            start_nid = find_nearest_node(graph, gx, gy, vel, is_floor)

            # Trace shortest path
            path_edges = []
            if start_nid is not None:
                path_edges = trace_shortest_path(graph, d_star, start_nid)

            # Get current potential
            state_dict = {
                "player_x": ninja.xpos,
                "player_y": ninja.ypos,
                "player_xspeed": ninja.xspeed,
                "player_yspeed": ninja.yspeed,
                "switch_activated": switch_activated,
            }
            if ninja_state == 0:
                state_dict["contact_state"] = 0
            elif ninja_state == 5:
                state_dict["contact_state"] = 1
            else:
                state_dict["contact_state"] = 2

            potential = pbrs.get_potential(state_dict)

            result = {
                "path_edges": path_edges,
                "ninja_position": (ninja.xpos, ninja.ypos),
                "goal_position": goal_pos,
                "phase": phase,
                "potential": potential,
                "d_max": d_max,
                "n_surface_points": graph.n_points,
                "n_edges": len(graph.edges_src),
            }
            self._pbrs_path_cached_data = result
            return result

        except Exception as e:
            import logging

            logging.getLogger(__name__).debug(f"PBRS path overlay error: {e}")
            self._pbrs_path_cached_data = None
            return None

    def _get_deadly_mine_positions(self) -> list[tuple[float, float]]:
        """Extract positions of deadly (state=0) mines from level data.

        Returns list of (x, y) for deadly mines, or empty list.
        """
        try:
            from nclone.analytical_physics.landing_filter import (
                extract_deadly_mine_positions,
            )

            return extract_deadly_mine_positions(self.level_data)
        except Exception:
            return []
