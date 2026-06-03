"""
Reachability functionality mixin for N++ environment.

This module contains all reachability analysis functionality that was previously
integrated into the main NppEnvironment class.
"""

import logging
import numpy as np
from typing import Tuple

from ...graph.reachability.reachability_system import ReachabilitySystem
from ...graph.reachability.feature_computation import (
    compute_reachability_features_from_graph,
)
from ...graph.reachability.path_distance_calculator import (
    CachedPathDistanceCalculator,
)


class ReachabilityMixin:
    """
    Mixin class providing reachability analysis functionality for N++ environment.

    This mixin handles:
    - Simplified reachability feature extraction
    - Flood-fill based reachability analysis
    - Performance tracking and caching
    - Entity type classification for reachability
    """

    def _init_reachability_system(self, debug: bool = False):
        """Initialize the reachability system components."""
        self.debug = debug

        # Initialize reachability system
        self._reachability_system = None
        self._reachability_cache = {}
        self._reachability_cache_ttl = 0.1  # 100ms cache TTL
        self._last_reachability_time = 0

        self._reachability_system = ReachabilitySystem()

        # OPTIMIZATION: Cell-based feature caching
        # Most frames ninja is in same cell, so we can reuse features
        self._last_reachability_key = None
        self._cached_reachability_features = None

        # Get shared_level_cache from parent environment if available
        shared_cache = getattr(self, "shared_level_cache", None)

        self._path_calculator = CachedPathDistanceCalculator(
            max_cache_size=200,
            shared_level_cache=shared_cache,
        )

        # Lattice d* reward-potential source (substitute + geometric fallback).
        # Built lazily per unique map; the per-episode use decision is made on
        # the first reachability call (needs the spawn position).  See
        # ReachabilityConfig.use_lattice_potential / lattice_distance.py.
        self._lattice_distance_source = None
        self._lattice_map_name = None
        self._use_lattice_this_episode = False
        self._lattice_episode_decided = False
        # Free-space grid potential: fallback for airborne/mine-threading levels
        # where the surface lattice has no finite cost-to-go (severed by mines).
        self._freespace_source = None
        self._freespace_map_name = None
        self._use_freespace_this_episode = False
        self._freespace_episode_decided = False
        # Cached physics-aware nav hint for the optional ``lattice_nav`` obs key.
        self._cached_lattice_nav = np.zeros(4, dtype=np.float32)
        # Number of NECESSARY locked doors opened so far (from the lattice
        # door-dependency plan); drives the locked-door reward + HWM sub-goal
        # phase. 0 when there is no plan / no necessary doors.
        self._necessary_doors_open = 0

        # Initialize logging if debug is enabled
        if self.debug:
            logging.basicConfig(level=logging.DEBUG)
            self.logger = logging.getLogger(__name__)

    def _reset_reachability_state(self):
        """Reset reachability state during environment reset."""
        if hasattr(self, "_path_calculator"):
            self._path_calculator.clear_cache()
        # Re-decide lattice-vs-geometric on the next reachability call (the
        # source itself is rebuilt only when the map identity changes).
        self._lattice_episode_decided = False
        self._freespace_episode_decided = False
        self._cached_lattice_nav = np.zeros(4, dtype=np.float32)
        self._necessary_doors_open = 0

    def _build_lattice_source(self, reach_cfg):
        """Build a LatticeDistanceSource from the current map_data (cached on
        disk by map_data hash).  Returns None on any failure so the caller
        falls back to the geometric A* distances."""
        try:
            nplay = getattr(self, "nplay_headless", None)
            map_data = getattr(nplay, "current_map_data", None)
            if not map_data:
                return None
            from ...value_function.lattice_distance import LatticeDistanceSource
            return LatticeDistanceSource(
                map_data,
                w_risk=getattr(reach_cfg, "lattice_w_risk", None),
                w_loop=getattr(reach_cfg, "lattice_w_loop", None),
                risk_cap=getattr(reach_cfg, "lattice_risk_cap", None),
                use_native=getattr(reach_cfg, "lattice_use_native", True),
                use_cache=True,
                name=self._lattice_map_name,
            )
        except Exception:
            return None

    def _maybe_substitute_lattice_distance(self, ninja_pos, switch_activated):
        """Replace the geometric HWM distances with the lattice d* cost-to-go.

        No-op unless ``ReachabilityConfig.use_lattice_potential`` is set.  The
        source is rebuilt only when the map identity changes; the lattice-vs-
        geometric choice is made once per episode (on the first call, when the
        ninja is at spawn) and only commits to the lattice when the spawn has a
        finite plan — so we never mix the two scales within an episode.  Any
        failure leaves the geometric distances untouched.
        """
        cfg = getattr(self, "config", None)
        reach_cfg = getattr(cfg, "reachability", None)
        if reach_cfg is None:
            return
        want_reward = bool(getattr(reach_cfg, "use_lattice_potential", False))
        want_obs = bool(getattr(reach_cfg, "expose_lattice_features", False))
        # The locked-door reward needs the door-dependency plan (which defines
        # "necessary"); building it requires the lattice source. So enable the
        # build/plan when door rewards are on, even if the potential/obs aren't.
        want_door_reward = bool(
            getattr(getattr(cfg, "reward_config", None),
                    "enable_locked_door_rewards", False)
        )
        if not (want_reward or want_obs or want_door_reward):
            return
        try:
            map_name = getattr(
                getattr(self, "map_loader", None), "current_map_name", None
            )
            if (map_name != self._lattice_map_name
                    or self._lattice_distance_source is None):
                self._lattice_map_name = map_name
                self._lattice_distance_source = self._build_lattice_source(reach_cfg)
                self._lattice_episode_decided = False

            nx, ny = float(ninja_pos[0]), float(ninja_pos[1])
            src = self._lattice_distance_source
            if src is None or not getattr(src, "available", False):
                self._use_lattice_this_episode = False
                self._lattice_episode_decided = True
                self._freespace_substitute(nx, ny, switch_activated, want_reward)
                return

            if not self._lattice_episode_decided:
                self._use_lattice_this_episode = src.spawn_finite(nx, ny)
                self._lattice_episode_decided = True
            if not self._use_lattice_this_episode:
                # Surface lattice severed at spawn (airborne / mine-threading) ->
                # use the free-space grid potential instead of the geometric A*.
                self._freespace_substitute(nx, ny, switch_activated, want_reward)
                return

            # Locked-door phase: which doors are open now, and how many of the
            # NECESSARY ones (for the door reward + N-phase potential select).
            open_door_indices = self._lattice_open_door_indices()
            plan = getattr(src, "_door_plan", None)
            if plan is not None:
                self._necessary_doors_open = sum(
                    1 for d in plan.necessary_order if d in open_door_indices
                )
            else:
                self._necessary_doors_open = 0

            # Policy nav hint (independent of the reward substitution). When the
            # gold potential is on, make the hint orienteering-aware so it bends
            # toward worthwhile gold (same 4 dims; parity when gold is off).
            want_gold_nav = bool(getattr(reach_cfg, "lattice_gold_potential", False))
            if want_obs:
                if want_gold_nav:
                    rem_gold_nav = [
                        (float(g.xpos), float(g.ypos))
                        for g in (getattr(self, "_cached_gold", None) or [])
                        if getattr(g, "active", True)
                    ]
                    gfv = float(getattr(reach_cfg, "lattice_gold_value", 120.0))
                    self._cached_lattice_nav = src.nav_feature(
                        nx, ny, switch_activated,
                        remaining_gold=rem_gold_nav, gold_frame_value=gfv,
                        open_door_indices=open_door_indices)
                else:
                    self._cached_lattice_nav = src.nav_feature(
                        nx, ny, switch_activated,
                        open_door_indices=open_door_indices)

            # Reward distance substitution: phase-0 metric -> d_comb
            # (cost-to-completion via switch), phase-1 -> d_ex.  The reward picks
            # the right one by switch state.  With lattice_gold_potential on, add
            # the orienteering tail-cost (detour through only WORTHWHILE remaining
            # gold, i.e. gold worth its <120-frame detour) so the potential guides
            # toward gold worth collecting (depends on remaining-gold SET => PBRS).
            if want_reward:
                want_gold = bool(getattr(reach_cfg, "lattice_gold_potential", False))
                if want_gold:
                    rem_gold = [
                        (float(g.xpos), float(g.ypos))
                        for g in (getattr(self, "_cached_gold", None) or [])
                        if getattr(g, "active", True)
                    ]
                    gw = float(getattr(reach_cfg, "lattice_gold_weight", 1.0))
                    gfv = float(getattr(reach_cfg, "lattice_gold_value", 120.0))
                    c_sw = src.gold_cost_to_go(nx, ny, False, rem_gold, gw, gfv)
                    c_ex = src.gold_cost_to_go(nx, ny, True, rem_gold, gw, gfv)
                else:
                    c_sw = src.cost_to_go(
                        nx, ny, False, open_door_indices=open_door_indices)
                    c_ex = src.cost_to_go(
                        nx, ny, True, open_door_indices=open_door_indices)
                if np.isfinite(c_sw):
                    self._cached_switch_distance = float(c_sw)
                if np.isfinite(c_ex):
                    self._cached_exit_distance = float(c_ex)
        except Exception:
            # Never let the potential source break an episode.
            self._use_lattice_this_episode = False

    def _lattice_open_door_indices(self) -> set:
        """Indices of currently-OPEN locked doors, aligned with the lattice door
        plan's door ordering.

        Both the plan (``_extract_locked_doors`` in lattice_audit) and the runtime
        ``nplay_headless.locked_doors()`` enumerate type-6 entities in map-load
        order, so the i-th locked door here is door index ``i`` in the plan. A door
        is open once its switch is collected (``closed`` is False)."""
        try:
            doors = self.nplay_headless.locked_doors()
        except Exception:
            return set()
        return {
            i for i, d in enumerate(doors)
            if not bool(getattr(d, "closed", True))
        }

    def _freespace_substitute(self, nx, ny, switch_activated, want_reward):
        """Fallback potential for levels where the surface lattice is severed
        (airborne / dense-mine corridors): a mine-aware free-space grid cost-to-go.
        Substitutes the reward distances so PBRS/HWM still get a dense, corridor-
        following signal (the RL agent learns the frame-precise execution). The
        per-episode use decision mirrors the lattice gate (commit only if the spawn
        has a finite free-space route). No-op unless a reward potential is wanted."""
        if not want_reward:
            return
        try:
            map_name = getattr(
                getattr(self, "map_loader", None), "current_map_name", None
            )
            if (map_name != self._freespace_map_name
                    or self._freespace_source is None):
                self._freespace_map_name = map_name
                nplay = getattr(self, "nplay_headless", None)
                md = getattr(nplay, "current_map_data", None)
                if md:
                    from ...value_function.free_space_distance import (
                        FreeSpaceDistanceSource,
                    )
                    self._freespace_source = FreeSpaceDistanceSource(md, name=map_name)
                else:
                    self._freespace_source = None
                self._freespace_episode_decided = False

            fs = self._freespace_source
            if fs is None or not getattr(fs, "available", False):
                self._use_freespace_this_episode = False
                self._freespace_episode_decided = True
                return
            if not self._freespace_episode_decided:
                self._use_freespace_this_episode = fs.spawn_finite(nx, ny)
                self._freespace_episode_decided = True
            if not self._use_freespace_this_episode:
                return
            c_sw = fs.cost_to_go(nx, ny, False)
            c_ex = fs.cost_to_go(nx, ny, True)
            if np.isfinite(c_sw):
                self._cached_switch_distance = float(c_sw)
            if np.isfinite(c_ex):
                self._cached_exit_distance = float(c_ex)
        except Exception:
            self._use_freespace_this_episode = False

    def _compute_reachability(self, ninja_pos: Tuple[int, int]) -> np.ndarray:
        """
        Compute 38-dimensional reachability features using adjacency graph (expanded in Phases 1-5).

        Base features (4 dims):
        0. Reachable area ratio (0-1)
        1. Distance to nearest switch (normalized, inverted)
        2. Distance to exit (normalized, inverted)
        3. Exit reachable flag (0-1)

        Path distances (2 dims):
        4. Path distance to switch (normalized 0-1)
        5. Path distance to exit (normalized 0-1)

        Direction vectors (4 dims):
        6. Direction to switch X (-1 to 1)
        7. Direction to switch Y (-1 to 1)
        8. Direction to exit X (-1 to 1)
        9. Direction to exit Y (-1 to 1)

        Mine context features (2 dims):
        10. Total mines normalized (0-1)
        11. Deadly mine ratio (0-1)

        Phase indicator (1 dim):
        12. Switch activated flag (0-1) - CRITICAL for Markov property

        Path direction features (8 dims) - Phase 1.1:
        13. Next hop direction X (-1 to 1) - optimal path direction
        14. Next hop direction Y (-1 to 1)
        15. Waypoint direction X (-1 to 1) - toward discovered waypoint
        16. Waypoint direction Y (-1 to 1)
        17. Waypoint distance (0-1) - normalized
        18. Path requires detour (0-1) - flag if must go away from goal
        19. Mine clearance direction X (-1 to 1) - safe direction
        20. Mine clearance direction Y (-1 to 1)

        Path difficulty (1 dim) - Phase 3.3:
        21. Path difficulty ratio (0-1) - physics_cost/geometric_distance

        Path curvature features (3 dims) - Phase 3.4:
        22. Multi-hop direction X (-1 to 1) - 8-hop weighted lookahead
        23. Multi-hop direction Y (-1 to 1) - anticipates upcoming path curvature
        24. Path curvature (0-1) - dot product alignment (1=straight, 0=90° turn, -1=180° turn)

        Exit lookahead (5 dims) - Phase 4:
        25. Exit next hop direction X (-1 to 1)
        26. Exit next hop direction Y (-1 to 1)
        27. Exit multi-hop direction X (-1 to 1)
        28. Exit multi-hop direction Y (-1 to 1)
        29. Near-switch transition indicator (0-1)

        Directional connectivity (8 dims) - Phase 5:
        30-37. Platform distance in 8 directions [E, NE, N, NW, W, SW, S, SE] (0-1)

        """
        # OPTIMIZATION: Cell-based caching - most frames ninja is in same cell
        # Cache key: (ninja_cell, switch_activated) - only recompute when these change
        ninja_cell = (int(ninja_pos[0] // 24), int(ninja_pos[1] // 24))
        switch_activated = (
            self.nplay_headless.exit_switch_activated()
            if hasattr(self, "nplay_headless")
            else False
        )
        # When the gold-aware lattice potential is on, the substituted distance
        # depends on the remaining-gold SET, so a gold pickup must invalidate the
        # cell cache (else the potential wouldn't drop until the cell changes,
        # making it history-dependent and no longer a valid potential).
        gold_key = 0
        _reach_cfg = getattr(getattr(self, "config", None), "reachability", None)
        if _reach_cfg is not None and getattr(_reach_cfg, "lattice_gold_potential", False):
            gold_key = sum(
                1 for g in (getattr(self, "_cached_gold", None) or [])
                if getattr(g, "active", True)
            )
        cache_key = (ninja_cell, switch_activated, gold_key)

        if (
            cache_key == self._last_reachability_key
            and self._cached_reachability_features is not None
        ):
            # Cache hit - return cached features
            return self._cached_reachability_features.copy()

        # Get graph data from GraphMixin (required)
        if not hasattr(self, "current_graph_data") or self.current_graph_data is None:
            raise RuntimeError(
                "current_graph_data not available. Graph must be built before computing reachability features."
            )

        # Get level data
        level_data = self.level_data

        # Extract adjacency from graph data
        adjacency = self.current_graph_data.get("adjacency")
        if adjacency is None:
            raise RuntimeError("adjacency not found in current_graph_data")

        # Ensure level cache is built (defensive, but now cheap with fast rejection)
        # Normally built by _update_graph_from_env_state, but this handles edge cases
        # like checkpoint resets where the normal flow may be bypassed.
        # With the fast rejection optimization in build_level_cache, redundant calls
        # are essentially free (single string comparison).
        base_adjacency = self.current_graph_data.get("base_adjacency", adjacency)
        self._path_calculator.build_level_cache(
            level_data, adjacency, base_adjacency, self.current_graph_data
        )

        # OPTIMIZATION: Get pre-computed goal positions (O(1) entity_dic lookups)
        # This eliminates O(n) entity iteration in feature computation
        goal_positions = None
        if hasattr(self, "nplay_headless") and hasattr(
            self.nplay_headless, "get_goal_positions_for_features"
        ):
            goal_positions = self.nplay_headless.get_goal_positions_for_features()

        # Compute all reachability features (22 dims)
        # Includes: 4 base + 2 path distances + 4 direction vectors + 2 mine context + 1 phase indicator + 8 path direction + 1 difficulty
        reach_cfg = getattr(
            getattr(self, "config", None), "reachability", None
        )
        include_navigation = bool(
            reach_cfg is not None
            and getattr(reach_cfg, "compute_navigation_features", False)
        )
        t_reach = self._profile_start("reach_features")
        features, switch_distance_raw, exit_distance_raw = (
            compute_reachability_features_from_graph(
                adjacency,
                self.current_graph_data,
                level_data,
                ninja_pos,
                self._path_calculator,
                goal_positions=goal_positions,  # Pass pre-computed positions
                include_navigation_features=include_navigation,
            )
        )
        self._profile_end("reach_features", t_reach)

        # OPTIMIZATION: Cache raw distances for PBRS to avoid duplicate computation
        # Store in instance variable to be added to observation in _get_observation
        self._cached_switch_distance = switch_distance_raw
        self._cached_exit_distance = exit_distance_raw

        # Substitute the physics-aware lattice d* cost-to-go for the geometric A*
        # distance signal where we have a finite plan (handoff step #1: replace
        # the "misleading 6px potential").  No-op unless
        # ReachabilityConfig.use_lattice_potential is set; falls back to the
        # geometric distances above on no-route levels / build failure.
        self._maybe_substitute_lattice_distance(ninja_pos, switch_activated)

        # Feature experiment B: optionally exclude path distances from actor obs
        # (PBRS still uses raw distances; actor learns from reward gradient alone)
        if getattr(
            getattr(self, "config", None),
            "exclude_path_distances_from_actor_obs",
            False,
        ):
            features = features.copy()
            features[0] = 0.0  # path_dist_to_switch_norm
            features[1] = 0.0  # path_dist_to_exit_norm

        # OPTIMIZATION: Cache features for next call if ninja is still in same cell
        self._last_reachability_key = cache_key
        self._cached_reachability_features = features.copy()

        return features

    def _reinit_reachability_system_after_unpickling(self):
        """Reinitialize reachability system components after unpickling."""
        if not hasattr(self, "_reachability_system"):
            self._reachability_system = ReachabilitySystem()
