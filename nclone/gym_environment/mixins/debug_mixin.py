"""
Debug and visualization functionality mixin for N++ environment.

This module contains all debug overlay and visualization functionality that was
previously integrated into the main NppEnvironment class.
"""

import threading
import time
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# Async jump cost-to-go field build (keep the GUI responsive).
#
# Building a JumpDistanceSource is SLOW (6-11s/level: validating every trajectory
# per support cell) AND pure-Python (jump-edge enumeration + Dijkstra), so it
# HOLDS THE GIL the whole time. Running it on a background *thread* therefore
# still starves the main pygame event loop -> the window freezes. We instead run
# it in a SEPARATE PROCESS (spawn) whose only job is to populate the shared
# on-disk jumpfield cache; the main process then reads that cache (a fast numpy/
# pickle load) once the subprocess exits. Everything is keyed by a hash of
# map_data:
#   _JUMP_BUILD_RESULTS[hash] -> JumpDistanceSource (or None if build failed)
#   _JUMP_BUILD_INFLIGHT[hash] -> build start time (frames show "computing… (Xs)")
#   _JUMP_BUILD_PROCS[hash]    -> the live multiprocessing.Process building it
# A hash present in RESULTS means "done" (value may be None = failed/unavailable).
# A hash in INFLIGHT but not RESULTS means "still building". Module-level so the
# work is shared across env instances; guarded by a single lock. Changing levels
# just means a different hash, so stale in-flight builds are harmless (ignored).
# spawn (not fork) so the child is a clean interpreter that never inherits/touches
# the parent's pygame display (important on WSL2).
# ---------------------------------------------------------------------------
_JUMP_BUILD_LOCK = threading.Lock()
_JUMP_BUILD_RESULTS: Dict[str, Any] = {}
_JUMP_BUILD_INFLIGHT: Dict[str, float] = {}
_JUMP_BUILD_PROCS: Dict[str, Any] = {}


def _jump_build_subprocess(map_data: list) -> None:
    """Subprocess (spawn) entry: build + disk-cache the jump field, then exit.

    Runs in a SEPARATE PROCESS so the GIL-holding pure-Python build can't starve
    the main pygame loop. No return value — the build's side effect is the shared
    on-disk ``jumpfield`` cache, which the parent reads once we exit.
    """
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # never touch a display
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        from nclone.value_function.jump_distance import get_jump_distance_source

        get_jump_distance_source(list(map_data))
    except Exception:
        pass


def _load_jump_source_from_cache(map_data: list) -> Any:
    """Load a JumpDistanceSource ONLY if its field is already cached (mem/disk).

    Returns the source on a cache hit, or ``None`` when nothing is cached (e.g.
    the subprocess build failed). Critically, this never triggers a synchronous
    6-11s rebuild on the main thread — a cache miss simply yields ``None``.
    """
    import os as _os

    try:
        from ...value_function.jump_distance import (
            _JUMP_MEM_CACHE,
            _jump_cache_dir,
            _jump_cache_key,
            get_jump_distance_source,
        )

        key = _jump_cache_key(list(map_data))
    except Exception:
        return None

    in_mem = key in _JUMP_MEM_CACHE
    cache_dir = _jump_cache_dir()
    path = _os.path.join(cache_dir, key + ".pkl") if cache_dir else None
    on_disk = bool(path and _os.path.exists(path))
    if not in_mem and not on_disk:
        return None  # subprocess produced nothing -> don't rebuild synchronously
    try:
        return get_jump_distance_source(list(map_data))  # fast: hits the cache
    except Exception:
        return None


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

        # Threat time-to-lethal field state. Native-backend only: the field is
        # computed inside the Rust simulator (see `threat_field.rs`) precisely so
        # no per-entity data crosses FFI, and the pure-Python sim has no
        # equivalent. The overlay reports that rather than silently drawing
        # nothing.
        self._threat_ttl_debug_enabled: bool = False
        self._threat_ttl_max_frames: int = 64
        self._threat_ttl_stride: int = 4

        # Jump-aware cost-to-go (physics-faithful) heatmap state. The source build
        # is SLOW (6-11s/level), so it is built OFF-THREAD (see the module-level
        # _JUMP_BUILD_* caches) and keyed by a map_data hash; the main loop never
        # blocks on it. While building, the overlay just shows a "computing" label.
        self._jump_cost_to_go_debug_enabled: bool = False
        self._physics_cost_to_go_debug_enabled: bool = False
        # Overlay-only toggle: draw the GEOMETRIC free-space field instead of the
        # anisotropic physics one, so the two can be A/B'd on the same level
        # without restarting. Does not touch what training reads.
        self._physics_cost_to_go_geometric: bool = False

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

        # Ghost-arc (Audit II P0-1) overlay state: draws the three ballistic
        # projections (u = -1/0/+1) the obs bundle is computed from, with
        # contact class / normal / clear-width / impact annotations.
        self._ghost_arcs_debug_enabled: bool = False

        # Cardinal-raycast overlay state: draws the 8 compass surface-distance
        # rays (game_state RAYCAST block, [E,NE,N,NW,W,SW,S,SE]) the policy sees.
        self._raycasts_debug_enabled: bool = False

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

    def set_tas_overlay(self, payload: Optional[dict], enabled: bool = True):
        """Set TAS editor overlay data (trajectory + piano-roll HUD).

        When enabled, the debug overlay renderer draws the ninja trajectory over
        the level plus a frame-by-frame timeline panel. Used by the ``--tas``
        editor in test_environment.py. ``payload`` is built by
        :func:`nclone.tas_tool.overlay.build_tas_payload`.
        """
        self._tas_overlay_enabled = bool(enabled) and payload is not None
        self._tas_overlay_data = payload

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

        # Add ghost-arc (ballistic projection) overlay payload if enabled
        if getattr(self, "_ghost_arcs_debug_enabled", False):
            ghost_data = self._get_ghost_arc_debug_data()
            if ghost_data:
                info["ghost_arcs"] = ghost_data

        # Add cardinal-raycast overlay payload if enabled
        if getattr(self, "_raycasts_debug_enabled", False):
            raycast_data = self._get_raycasts_debug_data()
            if raycast_data:
                info["raycasts"] = raycast_data

        # Add mine SDF visualization payload if enabled
        if self._mine_sdf_debug_enabled or self._hazard_debug_combo_enabled:
            mine_sdf_data = self._get_mine_sdf_data()
            if mine_sdf_data:
                info["mine_sdf"] = mine_sdf_data

        # Add threat time-to-lethal payload if enabled
        if self._threat_ttl_debug_enabled:
            ttl_data = self._get_threat_ttl_data()
            if ttl_data:
                info["threat_ttl"] = ttl_data

        # Add jump-aware cost-to-go heatmap payload if enabled
        if self._jump_cost_to_go_debug_enabled:
            jump_data = self._get_jump_cost_to_go_data()
            if jump_data:
                info["jump_cost_to_go"] = jump_data

        # Add physics free-space cost-to-go field + descent route if enabled
        if self._physics_cost_to_go_debug_enabled:
            phys_data = self._get_physics_cost_to_go_data()
            if phys_data:
                info["physics_cost_to_go"] = phys_data

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

        # TAS editor overlay (trajectory + piano-roll), gated independently.
        if getattr(self, "_tas_overlay_enabled", False):
            info["tas"] = self._tas_overlay_data

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

    def set_ghost_arcs_debug_enabled(self, enabled: bool):
        """Enable/disable the ghost-arc (Audit II P0-1) overlay.

        Draws the three ballistic projections (u = -1 / 0 / +1) that the
        27-dim game_state ghost-arc bundle is computed from: the exact native
        polyline to first contact, the contact class (floor / wall / ceiling /
        deadly mine / none), the surface normal, and per-arc annotations
        (time-to-contact, impact-speed ratio vs survivable, clear surface
        width). Faithful to the agent's obs: arcs are shown only while the
        ninja is AIRBORNE (grounded frames feed the fast-path zeros bundle,
        and the overlay says so instead).
        """
        self._ghost_arcs_debug_enabled = bool(enabled)

    def set_raycasts_debug_enabled(self, enabled: bool):
        """Enable/disable the cardinal-raycast overlay.

        Draws the 8 compass surface-distance rays the policy sees in the
        game_state RAYCAST block ([E, NE, N, NW, W, SW, S, SE]): each ray marches
        the tile grid from the ninja to the first solid surface (or 500px), and
        the overlay draws it to the SAME normalized distance the obs carries
        (0 = adjacent surface, 1.0 = nothing within 500px), with the compass
        label + normalized value. Hit rays are solid with a filled endpoint;
        no-surface rays fade out to the 500px horizon with a hollow endpoint.
        """
        self._raycasts_debug_enabled = bool(enabled)

    def _get_raycasts_debug_data(self) -> Optional[Dict[str, Any]]:
        """Build the cardinal-raycast overlay payload.

        Reads the SAME 8 normalized distances the obs carries
        (``_compute_surface_raycasts``, order [E, NE, N, NW, W, SW, S, SE]) and
        turns each into a draw segment: ninja -> ninja + unit(dir) * dist*NORM.
        So the drawn ray length is literally the un-normalized obs value, and a
        ray that saturates at 1.0 (no surface within NORM px) is flagged as a
        miss. Direction vectors + normalizer come from the env constants so the
        overlay can never drift from the feature.
        """
        try:
            dirs = getattr(self, "_RAYCAST_DIRS", None)
            norm = float(getattr(self, "_RAYCAST_NORM_PX", 500.0))
            if dirs is None:
                return None
            dists = self._compute_surface_raycasts()
            px, py = self.nplay_headless.ninja_position()
            labels = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")
            rays = []
            for i, (dtx, dty) in enumerate(dirs):
                mag = (dtx * dtx + dty * dty) ** 0.5 or 1.0
                ux, uy = dtx / mag, dty / mag
                d = float(dists[i]) if i < len(dists) else 1.0
                reach = d * norm
                rays.append({
                    "label": labels[i] if i < len(labels) else str(i),
                    "start": (float(px), float(py)),
                    "end": (float(px) + ux * reach, float(py) + uy * reach),
                    "dist_norm": d,
                    "hit": d < 0.999,  # <1.0 => a surface was found within NORM px
                })
            return {"ninja": (float(px), float(py)), "rays": rays, "norm_px": norm}
        except Exception:
            return None

    def _get_ghost_arc_debug_data(self) -> Optional[Dict[str, Any]]:
        """Build the per-frame ghost-arc overlay payload (see renderer).

        Uses the SAME projector instance, jump-gravity derivation, and frame
        horizon as the obs bundle (`_fill_ghost_arc_bundle`), so what is drawn
        is exactly what the agent sees — plus the frame-by-frame polyline the
        native kernel integrated (project_with_path).
        """
        try:
            from ..constants import GHOST_ARC_MAX_FRAMES

            ninja = self.nplay_headless.sim.ninja
            airborne = bool(getattr(ninja, "airborn", False))
            px, py = self.nplay_headless.ninja_position()
            payload: Dict[str, Any] = {
                "airborne": airborne,
                "ninja": (float(px), float(py)),
                "arcs": [],
            }
            get_proj = getattr(self, "_get_trajectory_projector", None)
            proj = get_proj() if get_proj is not None else None
            if proj is None or not hasattr(proj, "project_with_path"):
                payload["unavailable"] = True
                return payload
            vx, vy = self.nplay_headless.ninja_velocity()

            def _project_row(x0, y0, vx0, vy0, jf, key):
                rows = payload.setdefault(key, [])
                for u in (-1, 0, 1):
                    contact, path = proj.project_with_path(
                        float(x0), float(y0), float(vx0), float(vy0),
                        int(u), int(jf), GHOST_ARC_MAX_FRAMES,
                    )
                    # v15: contact_vx, contact_vy, min_mine_clearance_px.
                    # v16 (feature C): min_goal_clearance_px appended (inf/None
                    # when no goal is set).
                    (
                        cls, cx, cy, nx, ny, tau, imp, thr, cw,
                        cvx, cvy, mclr, gclr,
                    ) = contact
                    surface_class = (
                        1 if int(cls) == 0 else -1 if int(cls) == 2 else 0
                    )  # 0=floor->+1, 2=ceiling->-1, wall/mine/none->0
                    rows.append(
                        {
                            "u": int(u),
                            "path": path,
                            "class": int(cls),
                            "contact": (float(cx), float(cy)),
                            "normal": (float(nx), float(ny)),
                            "tau": float(tau),
                            "impact_ratio": (
                                float(imp) / float(thr) if thr > 1e-9 else 0.0
                            ),
                            "clear_width": int(cw),
                            "contact_vel": (float(cvx), float(cvy)),
                            "surface_class": surface_class,
                            "mine_clearance": float(mclr),  # px, inf = clear
                            "goal_clearance": float(gclr),  # v16 C: px, inf = no goal
                        }
                    )

            if airborne:
                # Committed arcs — same jump-hold gravity derivation as
                # _fill_ghost_arc_bundle.
                from ...constants.physics_constants import (
                    JUMP_WALL_REGULAR_X_MULTIPLIER,
                    JUMP_WALL_REGULAR_Y,
                    MAX_JUMP_DURATION,
                )

                jf = 0
                if (
                    getattr(ninja, "state", 4) == 3
                    and getattr(ninja, "jump_input_old", 0) != 0
                ):
                    jf = max(
                        0,
                        MAX_JUMP_DURATION - int(getattr(ninja, "jump_duration", 0)),
                    )
                _project_row(px, py, vx, vy, jf, "arcs")
                # Passive-release arc (v16 batch A): "if I let go of everything
                # NOW" -> u=0 (no steer), jf=0 (jump released). The single arc
                # the game_state passive block is computed from; drawn distinct
                # (white dashed) with an explicit fatal flag so over-jump /
                # no-input-fall trajectories are legible.
                pcontact, ppath = proj.project_with_path(
                    float(px), float(py), float(vx), float(vy),
                    0, 0, GHOST_ARC_MAX_FRAMES,
                )
                (
                    pcls, pcx, pcy, pnx, pny, ptau, pimp, pthr, pcw,
                    pcvx, pcvy, pmclr, pgclr,
                ) = pcontact
                p_fatal = int(pcls) == 3 or (
                    int(pcls) in (0, 1, 2) and pthr > 1e-9 and pimp >= pthr
                )
                payload["passive_arc"] = {
                    "path": ppath,
                    "class": int(pcls),
                    "contact": (float(pcx), float(pcy)),
                    "contact_vel": (float(pcvx), float(pcvy)),
                    "tau": float(ptau),
                    "impact_ratio": (
                        float(pimp) / float(pthr) if pthr > 1e-9 else 0.0
                    ),
                    "fatal": bool(p_fatal),
                    "is_none": int(pcls) == 4,
                    "mine_clearance": float(pmclr),
                    "goal_clearance": float(pgclr),  # v16 C
                }
                # Wall-jump-now row (v14): counterfactual regular wall-jump
                # kick while walled — same impulse as _fill_ghost_arc_bundle.
                wn = float(getattr(ninja, "wall_normal", 0.0) or 0.0)
                if getattr(ninja, "walled", False) and wn != 0.0:
                    vxw = 0.0 if float(vx) * wn < 0 else float(vx)
                    vyw = min(float(vy), 0.0)
                    vxw += JUMP_WALL_REGULAR_X_MULTIPLIER * wn
                    vyw += JUMP_WALL_REGULAR_Y
                    _project_row(
                        px + JUMP_WALL_REGULAR_X_MULTIPLIER * wn,
                        py + JUMP_WALL_REGULAR_Y,
                        vxw, vyw, MAX_JUMP_DURATION, "wall_arcs",
                    )
            else:
                # Grounded: the two counterfactual takeoff rows the obs exposes
                # (same flat-floor impulse as _fill_ghost_arc_bundle's grounded
                # branch). jump_arcs = v13 full-hold (jf = MAX_JUMP_DURATION);
                # tap_arcs = v14 B1 tap (jf = 0 -> full gravity after impulse).
                from ...constants.physics_constants import (
                    JUMP_FLOOR_Y,
                    MAX_JUMP_DURATION,
                )

                vyj = min(float(vy), 0.0) + JUMP_FLOOR_Y
                _project_row(px, py + JUMP_FLOOR_Y, vx, vyj,
                             MAX_JUMP_DURATION, "jump_arcs")
                _project_row(px, py + JUMP_FLOOR_Y, vx, vyj, 0, "tap_arcs")
            return payload
        except Exception:
            return None

    def set_mine_sdf_debug_enabled(self, enabled: bool):
        """Enable/disable mine SDF (Signed Distance Field) debug visualization.

        Shows a heatmap overlay of the precomputed distance field to deadly mines.
        Red = danger zone (close to mines), Green = safe (far from mines).
        Also shows escape direction arrows at each tile.
        """
        self._mine_sdf_debug_enabled = bool(enabled)

    def set_threat_ttl_debug_enabled(self, enabled: bool):
        """Enable/disable the threat time-to-lethal overlay.

        Draws, per 12 px cell, WHEN that cell next becomes lethal rather than how
        far it is from a threat -- red = lethal now, through orange/yellow for
        "lethal within a few frames", to unshaded for "safe for the whole
        window". Cyan hatching marks cells reachable only by a REACTIVE threat
        (chaser drone, death ball, eye bat), whose timing is a worst-case
        envelope rather than an exact forecast.

        This is the field itself, read straight out of the simulator -- not a
        re-derivation -- so what the overlay shows is exactly what the policy
        would receive.
        """
        self._threat_ttl_debug_enabled = bool(enabled)

    def _get_threat_ttl_data(self) -> Optional[Dict[str, Any]]:
        """Threat TTL payload for the debug renderer, or None if unavailable.

        Returns the raw `(4, H, W)` field plus the window it was computed over,
        so the overlay can label absolute frame counts instead of normalised
        fractions -- "lethal in 12 frames" is actionable, "0.1875" is not.
        """
        try:
            import numpy as np

            max_frames = self._threat_ttl_max_frames
            stride = self._threat_ttl_stride

            native = getattr(self.nplay_headless, "_native_sim", None)
            if native is not None and hasattr(native, "threat_ttl"):
                raw = native.threat_ttl(max_frames, stride)
                field = np.frombuffer(raw, dtype=np.float32).reshape(4, 46, 84).copy()
            else:
                # The dev tool runs the PYTHON backend -- the native sim has no
                # renderer at all -- so the overlay would otherwise never be
                # visible where it is most useful. `threat_ttl_reference` computes
                # the identical field from the Python sim's own drones, and
                # `test_threat_ttl_parity.py` pins it against the Rust engine so
                # the two cannot drift.
                from ..threat_ttl_reference import compute_threat_ttl_reference

                field = compute_threat_ttl_reference(
                    self.nplay_headless, max_frames=max_frames, stride=stride
                )
            return {
                "field": field,
                "max_frames": max_frames,
                "stride": stride,
                "cell_size": 12,
                # World px of the field's (0, 0) corner. The grid covers the
                # PLAYABLE area, which begins one 24px border tile in, so the
                # renderer must not assume it starts at the window origin.
                "origin": 24.0,
            }
        except Exception:
            return None

    def set_jump_cost_to_go_debug_enabled(self, enabled: bool):
        """Enable/disable the jump-aware cost-to-go heatmap visualization.

        Shows a heatmap of the physics-faithful cost-to-go field (in frames)
        toward the current objective (pre-switch -> switch, post-switch -> exit).
        Green = near goal (low cost), red = far (high cost), unreachable cells
        are skipped. This is the field that feeds the agent's cost_to_go_map obs
        + PBRS reward, so the overlay shows that the gradient routes over
        platforms rather than through lethal gaps. The underlying JumpDistanceSource
        is cached per-level (keyed by map_data hash) because its build is slow.
        """
        self._jump_cost_to_go_debug_enabled = bool(enabled)

    def set_physics_cost_to_go_debug_enabled(self, enabled: bool):
        """Enable/disable the free-space PHYSICS cost-to-go overlay.

        Draws the anisotropic frame-cost field toward the current objective plus
        the steepest-descent route through it, with each route segment coloured
        by the move class the cost model charged for (run / air / fall / jump /
        wall). Unlike the jump cost-to-go overlay this needs no background build
        -- the native flood is ~10ms -- so it repaints on the spot.

        Use ``set_physics_cost_to_go_geometric(True)`` to draw the legacy
        isotropic field instead; toggling between them on one level is the
        quickest way to see what the physics weights bought.
        """
        self._physics_cost_to_go_debug_enabled = bool(enabled)

    def set_physics_cost_to_go_geometric(self, geometric: bool):
        """Draw the geometric (isotropic) field instead of the physics one."""
        self._physics_cost_to_go_geometric = bool(geometric)

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

    def _get_jump_cost_to_go_data(self) -> Optional[Dict[str, Any]]:
        """Build the jump-aware cost-to-go heatmap payload for the renderer.

        NON-BLOCKING: the JumpDistanceSource for the CURRENT level is built in a
        daemon thread (keyed by a map_data hash) so the GUI never freezes on the
        6-11s build. This method returns immediately every frame:
        - build not started -> spawn the daemon thread, return a "computing" payload
        - build in progress -> return a "computing" payload (with elapsed seconds)
        - build done -> select the field toward the current objective (pre-switch
          -> ``_d_switch``, post-switch -> ``_d_exit``) and return it for the heatmap.

        The dense fields are ``(W, H)`` (x-major) float32 cost-to-go in FRAMES at
        6px resolution, ``inf`` where unreachable. The per-level result is cached
        (module-level), so toggling off/on or revisiting the level is instant.
        """
        import hashlib

        import numpy as np

        map_data = getattr(self.nplay_headless, "current_map_data", None)
        if not map_data:
            return None

        try:
            key = hashlib.md5(
                np.asarray([int(v) for v in map_data], dtype=np.int64).tobytes()
            ).hexdigest()
        except Exception:
            return None

        switch_activated = self.nplay_headless.exit_switch_activated()
        phase = "exit" if switch_activated else "switch"

        with _JUMP_BUILD_LOCK:
            done = key in _JUMP_BUILD_RESULTS
            src = _JUMP_BUILD_RESULTS.get(key)
            inflight_start = _JUMP_BUILD_INFLIGHT.get(key)
            proc = _JUMP_BUILD_PROCS.get(key)

            if not done and inflight_start is None:
                # First time we've seen this level: kick off the OFF-PROCESS build
                # (spawn) so the GIL-holding build can't freeze the GUI.
                import multiprocessing as _mp

                inflight_start = time.time()
                try:
                    ctx = _mp.get_context("spawn")
                    proc = ctx.Process(
                        target=_jump_build_subprocess,
                        args=(list(map_data),),
                        daemon=True,
                    )
                    proc.start()
                    _JUMP_BUILD_INFLIGHT[key] = inflight_start
                    _JUMP_BUILD_PROCS[key] = proc
                except Exception:
                    # Couldn't spawn -> mark as failed (None) rather than block.
                    _JUMP_BUILD_RESULTS[key] = None
                    done = True
                    src = None
            elif not done and proc is not None and not proc.is_alive():
                # Subprocess finished: reap it and load the result from the cache
                # it populated (fast). A cache miss means the build failed -> None.
                try:
                    proc.join(timeout=0)
                except Exception:
                    pass
                src = _load_jump_source_from_cache(map_data)
                _JUMP_BUILD_RESULTS[key] = src
                _JUMP_BUILD_INFLIGHT.pop(key, None)
                _JUMP_BUILD_PROCS.pop(key, None)
                done = True

        if not done:
            # Still building -> draw only a lightweight "computing" indicator.
            elapsed = time.time() - inflight_start if inflight_start else 0.0
            return {
                "computing": True,
                "elapsed": elapsed,
                "phase": phase,
                "map_hash": key,
                "cell_px": 6,
            }

        if src is None or not getattr(src, "available", False):
            return None

        # (W, H) x-major: pre-switch route to the switch, post-switch route to exit.
        field = src._d_exit if switch_activated else src._d_switch
        if field is None:
            return None

        return {
            "field": field,
            "switch_activated": switch_activated,
            "phase": phase,
            "map_hash": key,
            "ninja_position": self.nplay_headless.ninja_position(),
            "cell_px": 6,
        }

    def _get_physics_cost_to_go_data(self) -> Optional[Dict[str, Any]]:
        """Build the free-space cost-to-go field + descent-route overlay payload.

        Synchronous, unlike ``_get_jump_cost_to_go_data``: the native flood costs
        ~10ms, so there is no reason for the subprocess machinery. Sources are
        memoized per (map hash, cost model) since the field is a pure function of
        the level -- only the route is recomputed per frame, and that is ~1ms.
        """
        import hashlib

        import numpy as np

        map_data = getattr(self.nplay_headless, "current_map_data", None)
        if not map_data:
            return None
        try:
            key = hashlib.md5(
                np.asarray([int(v) for v in map_data], dtype=np.int64).tobytes()
            ).hexdigest()
        except Exception:
            return None

        geometric = bool(getattr(self, "_physics_cost_to_go_geometric", False))
        cache = getattr(self, "_physics_ctg_cache", None)
        if cache is None:
            cache = self._physics_ctg_cache = {}
        cache_key = (key, geometric)
        src = cache.get(cache_key)
        if src is None:
            try:
                from ...value_function.free_space_distance import (
                    FreeSpaceDistanceSource,
                )

                src = FreeSpaceDistanceSource(
                    list(map_data), physics_cost=not geometric
                )
            except Exception:
                return None
            if len(cache) > 8:
                cache.clear()
            cache[cache_key] = src
        if not getattr(src, "available", False):
            return None

        switch_activated = self.nplay_headless.exit_switch_activated()
        field = src._d_exit if switch_activated else src._d_switch
        if field is None:
            return None

        nx, ny = self.nplay_headless.ninja_position()
        try:
            route = src.descent_path(float(nx), float(ny), bool(switch_activated))
        except Exception:
            route = None

        return {
            "field": field,
            "route": route,
            "phase": "exit" if switch_activated else "switch",
            "model": "geometric" if geometric else "physics",
            "map_hash": key,
            "ninja_position": (nx, ny),
            "goal": src.exit if switch_activated else src.switch,
            # Coverage is the headline diagnostic: a field that only reaches a
            # sliver of free space is a dead PBRS signal on that level.
            "coverage": (
                float(np.isfinite(field).sum()) / float(max(int(src._free.sum()), 1))
            ),
            "cell_px": 6,
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
