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
from ..constants import LEVEL_DIAGONAL, WAYPOINT_NAV_DIM

# Process-wide LRU of free-space distance sources for the default-on
# ``free_space_nav`` policy hint, keyed by a hash of the level's map_data. The
# build is ~300ms, so this amortizes it to once-per-distinct-level within a
# worker (curriculum pools revisit the same levels constantly across resets).
# Bounded by a simple clear-on-overflow to cap memory.
_FREE_SPACE_NAV_CACHE: dict = {}
_FREE_SPACE_NAV_CACHE_CAP = 512


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
        # Mine-threading potential: the FAITHFUL fallback for mine-saturated levels
        # where the surface lattice is severed. Hard 14px mine constraint + tile-soft
        # cost over a fine 3px grid -> connected, mine-faithful cost-to-go (the route
        # threads between mines). Preferred over the mine-SOFT free-space grid.
        self._minethreading_source = None
        self._minethreading_map_name = None
        self._use_minethreading_this_episode = False
        self._minethreading_episode_decided = False
        # Free-space grid potential: last-resort fallback (mine-SOFT) for severed
        # levels where even the mine-threading region has no finite spawn route.
        self._freespace_source = None
        self._freespace_map_name = None
        self._use_freespace_this_episode = False
        self._freespace_episode_decided = False
        # True while the reward distances currently in ``_cached_switch_distance``/
        # ``_cached_exit_distance`` come from a COMMITTED substitution source
        # (lattice / mine-threading / free-space cost-to-completion). False means
        # the raw per-phase A* fallback is live — the reward calculator then
        # composes its own continuous basis (or degrades to the HWM ratchet) so
        # PBRS never flips basis at switch activation. Exposed into the obs dict
        # as ``_potential_substituted`` (reward metadata, stripped from policy obs).
        self._potential_substituted = False
        # Cached physics-aware nav hint for the optional ``lattice_nav`` obs key.
        self._cached_lattice_nav = np.zeros(4, dtype=np.float32)

        # Default-on free-space nav hint (folded into game_state). A dedicated
        # FreeSpaceDistanceSource per level (LRU-cached process-wide) gives a
        # physics-free steepest-descent direction toward the current objective
        # that rounds obstacles. Rebuilt only on map identity change.
        self._free_space_nav_source = None
        self._free_space_nav_map_name = None
        # (map_data, source_cls_name, use_hard, source) memo -- see
        # _get_free_space_nav_source.
        self._free_space_nav_memo = None
        self._cached_free_space_nav = np.zeros(4, dtype=np.float32)

        # Lattice-backed waypoint subgoals (long-horizon distance fix). A
        # WaypointTracker advances through the lattice route's subgoals; its cached
        # fields feed the reward (``waypoint_leg_distance``/``waypoint_index`` for the
        # per-leg PBRS) and the optional ``waypoint_nav`` policy obs. Rebuilt per
        # episode (cheap when waypoints are injected; lattice build is disk-cached).
        self._waypoint_planner = None
        self._waypoint_tracker = None
        self._injected_waypoints = None  # set via set_level_waypoints() at scale
        self._cached_waypoint_nav = np.zeros(WAYPOINT_NAV_DIM, dtype=np.float32)
        self._waypoint_leg_distance = None
        self._waypoint_index = None
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
        self._minethreading_episode_decided = False
        self._freespace_episode_decided = False
        self._potential_substituted = False
        self._cached_lattice_nav = np.zeros(4, dtype=np.float32)
        # Force a per-episode rebind of the free-space nav source (cheap: the
        # source object itself is LRU-cached, only the map-identity check resets).
        self._free_space_nav_map_name = None
        self._free_space_nav_source = None
        self._free_space_nav_memo = None
        self._cached_free_space_nav = np.zeros(4, dtype=np.float32)
        self._necessary_doors_open = 0
        # Force a per-episode tracker rebuild (picks up newly injected waypoints /
        # the new map). The lattice build is disk-cached so repeat maps are cheap.
        self._waypoint_tracker = None
        self._cached_waypoint_nav = np.zeros(WAYPOINT_NAV_DIM, dtype=np.float32)
        self._waypoint_leg_distance = None
        self._waypoint_index = None

    def set_level_waypoints(self, waypoints):
        """Inject precomputed subgoals (list[Waypoint]) for the current level.

        The scalable path for procgen: a level source attaches pool-precomputed
        waypoints (see ``tools/export_subgoals.py``) so the env never builds a
        lattice on the hot path. The tracker rebuilds from these on the next step.
        Passing ``None``/empty clears them (tracker falls back to on-reset build if
        ``waypoint_build_on_reset`` is set, else becomes a no-op).
        """
        self._injected_waypoints = list(waypoints) if waypoints else None
        self._waypoint_tracker = None

    def _resolve_waypoints(self, reach_cfg):
        """Injected waypoints if present; else build on reset (when allowed).

        Prefers the CHEAP free-space waypoint planner (steepest-descent on the
        disk-cached free-space cost-to-go — ~ms, scales to procgen) and falls back
        to the physics-aware lattice planner only if free-space is unavailable.
        """
        if self._injected_waypoints:
            return self._injected_waypoints
        if not getattr(reach_cfg, "waypoint_build_on_reset", True):
            return None
        nplay = getattr(self, "nplay_headless", None)
        map_data = getattr(nplay, "current_map_data", None)
        if not map_data:
            return None
        name = getattr(self, "_lattice_map_name", None) or getattr(
            getattr(self, "map_loader", None), "current_map_name", None
        )
        max_leg = getattr(reach_cfg, "waypoint_max_leg_px", None)
        # Free-space planner (cheap, default).
        if getattr(reach_cfg, "waypoint_use_free_space", True):
            try:
                from ...value_function.waypoint_planner import FreeSpaceWaypointPlanner

                pos = nplay.ninja_position()
                if pos is not None:
                    fp = FreeSpaceWaypointPlanner(
                        map_data, (float(pos[0]), float(pos[1])),
                        name=name, max_leg_px=(max_leg or 48.0),
                    )
                    if fp.available and fp.waypoints:
                        self._waypoint_planner = fp
                        return fp.waypoints
            except Exception:
                pass
        # Lattice fallback (physics-aware but precompute-heavy).
        try:
            from ...value_function.waypoint_planner import LatticeWaypointPlanner

            p = LatticeWaypointPlanner(
                map_data, use_cache=True, name=name, max_leg_px=max_leg,
            )
            self._waypoint_planner = p
            return p.waypoints if p.available else None
        except Exception:
            return None

    def _maybe_update_waypoints(self, ninja_pos, switch_activated):
        """Advance the WaypointTracker and cache the reward/obs fields.

        No-op unless ``enable_waypoint_tracker`` or ``expose_waypoint_features`` is
        set. Builds the tracker once per episode (lazy, on first call). Any failure
        leaves the cached fields as their safe defaults (``None`` distance/index ->
        the per-leg PBRS term simply doesn't fire).
        """
        cfg = getattr(self, "config", None)
        reach_cfg = getattr(cfg, "reachability", None)
        if reach_cfg is None:
            return
        if not (
            getattr(reach_cfg, "enable_waypoint_tracker", False)
            or getattr(reach_cfg, "expose_waypoint_features", False)
        ):
            return

        if self._waypoint_tracker is None:
            from ...value_function.waypoint_planner import WaypointTracker

            waypoints = self._resolve_waypoints(reach_cfg)
            self._waypoint_tracker = WaypointTracker(waypoints) if waypoints else None

        t = self._waypoint_tracker
        if t is None:
            return
        nx, ny = float(ninja_pos[0]), float(ninja_pos[1])
        self._cached_waypoint_nav = t.nav_feature(nx, ny, bool(switch_activated))
        self._waypoint_leg_distance = float(t.leg_distance(nx, ny))
        self._waypoint_index = int(t.index)

    def _potential_source_cls(self, reach_cfg):
        """Class used for the free-space-style PBRS reward / cost_to_go_map source.

        ``ReachabilityConfig.use_jump_potential`` swaps the geometric free-space
        flood for the physics-faithful jump-aware cost-to-go (a drop-in subclass,
        same API). Default returns the free-space source so nothing changes unless
        the flag is set."""
        if reach_cfg is not None and getattr(reach_cfg, "use_jump_potential", False):
            from ...value_function.jump_distance import JumpDistanceSource
            return JumpDistanceSource
        from ...value_function.free_space_distance import FreeSpaceDistanceSource
        return FreeSpaceDistanceSource

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
        # use_jump_potential routes through the same free-space-primary path (the
        # jump source is a drop-in subclass); _freespace_substitute picks the class.
        want_free_space = bool(
            getattr(reach_cfg, "use_free_space_potential", False)
            or getattr(reach_cfg, "use_jump_potential", False)
        )
        if not (want_reward or want_obs or want_door_reward or want_free_space):
            return

        # Substitution status is re-derived on every call; each commit path below
        # sets it back to True. When it stays False the raw per-phase A* distances
        # are live and the reward calculator applies its own continuity fallback.
        self._potential_substituted = False

        # Free-space-PRIMARY potential: skip the (often infeasible) surface-lattice
        # build entirely and take the reward distances straight from the cheap
        # mine-aware free-space grid. Only valid when the policy obs / locked-door
        # plan don't also need the full lattice (those fall through below).
        if want_free_space and not (want_obs or want_door_reward):
            nx, ny = float(ninja_pos[0]), float(ninja_pos[1])
            self._use_lattice_this_episode = False
            committed_src = None
            use_jump = bool(getattr(reach_cfg, "use_jump_potential", False))
            if use_jump:
                # Audit II F4 fix: with --use-jump-potential the jump-aware
                # field runs FIRST — including on mined levels. Previously the
                # mine-threading source pre-empted it whenever the level had
                # mines, so the jump field was silently unused on exactly the
                # mined precision levels it exists for. The jump field is now
                # mine-HARD itself (jump5: deadly kill-radius cells block
                # support/wall/arcs/fall-fill), so precedence is safe; the
                # threading source remains the fallback when the hardened jump
                # field has no finite spawn route.
                if self._freespace_substitute(nx, ny, switch_activated, True):
                    committed_src = self._freespace_source
                elif self._level_has_mines() and self._minethreading_substitute(
                    nx, ny, switch_activated
                ):
                    committed_src = self._minethreading_source
            else:
                # Mine-HARD threading source first when the level has mines,
                # mirroring the severed-lattice chain below: the mine-SOFT
                # free-space grid only soft-penalizes mine cells (cost
                # x(1+8*mden)), so on wall-of-mines geometry its potential can
                # route THROUGH a lethal wall when the detour is long enough.
                # The threading source's hard 14px mine mask cannot.
                if self._level_has_mines() and self._minethreading_substitute(
                    nx, ny, switch_activated
                ):
                    committed_src = self._minethreading_source
                elif self._freespace_substitute(nx, ny, switch_activated, True):
                    committed_src = self._freespace_source
            if committed_src is not None:
                self._maybe_ballistic_reevaluate(
                    committed_src, nx, ny, switch_activated
                )
            self._potential_substituted = committed_src is not None
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
                self._severed_lattice_substitute(nx, ny, switch_activated, want_reward)
                return

            if not self._lattice_episode_decided:
                self._use_lattice_this_episode = src.spawn_finite(nx, ny)
                self._lattice_episode_decided = True
            if not self._use_lattice_this_episode:
                # Surface lattice severed at spawn (airborne / mine-threading) -> use
                # the mine-faithful threading potential (then free-space) instead of
                # the geometric A*.
                self._severed_lattice_substitute(nx, ny, switch_activated, want_reward)
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
                    # Live per-tick gold (backend-uniform; the once-cached
                    # _cached_gold is stale on the native backend — see
                    # NPlayHeadless.get_gold_state).
                    rem_gold_nav = [
                        (float(gx), float(gy))
                        for (gx, gy, active) in self.nplay_headless.get_gold_state()
                        if active
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
                        (float(gx), float(gy))
                        for (gx, gy, active) in self.nplay_headless.get_gold_state()
                        if active
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
                # Committed only when BOTH phases carry the lattice scale — a
                # partial write would mix bases across the switch flip.
                self._potential_substituted = bool(
                    np.isfinite(c_sw) and np.isfinite(c_ex)
                )
        except Exception:
            # Never let the potential source break an episode.
            self._use_lattice_this_episode = False
            self._potential_substituted = False

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

    def _level_has_mines(self) -> bool:
        """True when the current level contains any toggle mines (type 1 or 21).

        Cheap gate for preferring the mine-HARD threading potential over the
        mine-SOFT free-space grid; on mine-free levels both are identical, so
        the threading build is skipped entirely."""
        try:
            toggle_mines, toggled_mines = self.nplay_headless.get_mine_entities()
            return bool(toggle_mines) or bool(toggled_mines)
        except Exception:
            return False

    def _severed_lattice_substitute(self, nx, ny, switch_activated, want_reward):
        """Fallback chain when the surface lattice is severed at spawn:
        mine-threading (faithful hard-mine) first, then the free-space grid
        (mine-soft) if the threading region has no finite spawn route. No-op unless
        a reward potential is wanted. Records whether any source committed via
        ``_potential_substituted``."""
        if not want_reward:
            return
        if self._minethreading_substitute(nx, ny, switch_activated):
            self._potential_substituted = True
            return
        self._potential_substituted = bool(
            self._freespace_substitute(nx, ny, switch_activated, want_reward)
        )

    def _maybe_ballistic_reevaluate(self, src, nx, ny, switch_activated) -> None:
        """Audit II P0-2 (--ballistic-air-potential): while AIRBORNE, evaluate
        the committed potential at the point where the CURRENT no-input arc
        actually lands, instead of at the ninja's air position (whose value is
        a straight-down, velocity-blind column projection). This is the only
        change that makes controlling flight toward the route visible to the
        reward: an unchanged arc gives dPhi = 0 (no dawdle pay — hovering is
        impossible), a 1-frame trim that moves the landing to a better cell
        pays immediately, and grounded states degenerate to today's Phi
        exactly (this method is a no-op on the ground). Still a true state
        potential (a function of (x, y, vx, vy) only) => telescopes; the death
        clawback and D0 normalization are untouched downstream.

        Overwrites ``_cached_switch_distance`` / ``_cached_exit_distance``
        (already written by the committing substitute at the ninja position)
        only when the ballistic landing evaluates finite — a degraded
        projection (mine landing in an inf pocket, no native projector, no
        contact within horizon toward the border) keeps the standard values.
        """
        reach_cfg = getattr(getattr(self, "config", None), "reachability", None)
        if not bool(getattr(reach_cfg, "ballistic_air_potential", False)):
            return
        try:
            ninja = self.nplay_headless.sim.ninja
            if not bool(getattr(ninja, "airborn", False)):
                return
            get_proj = getattr(self, "_get_trajectory_projector", None)
            proj = get_proj() if get_proj is not None else None
            if proj is None:
                return
            vx, vy = self.nplay_headless.ninja_velocity()
            jf = 0
            if (
                getattr(ninja, "state", 4) == 3
                and getattr(ninja, "jump_input_old", 0) != 0
            ):
                from ...constants.physics_constants import MAX_JUMP_DURATION

                jf = max(
                    0, MAX_JUMP_DURATION - int(getattr(ninja, "jump_duration", 0))
                )
            from ..constants import GHOST_ARC_MAX_FRAMES

            if bool(getattr(reach_cfg, "ballistic_air_best_arc", False)):
                # v13 best-arc: evaluate all three steering arcs and take the
                # best SURVIVING landing per target (min finite cost). An arc
                # is discarded if it ends in a deadly mine (class 3), never
                # contacts within the horizon (class 4), or lands hard enough
                # to die (impact_speed > impact_threshold). Still a pure
                # function of (pos, vel, jf) => a valid state potential.
                best_sw = float("inf")
                best_ex = float("inf")
                for u in (-1, 0, 1):
                    cls_, lx, ly, _nx2, _ny2, _tau, imp, thr, _cw, *_rest = proj.project(
                        float(nx),
                        float(ny),
                        float(vx),
                        float(vy),
                        u,
                        jf,
                        GHOST_ARC_MAX_FRAMES,
                    )
                    if cls_ in (3, 4) or (thr > 0.0 and imp > thr):
                        continue
                    c_sw = src.cost_to_completion(float(lx), float(ly), False)
                    c_ex = src.cost_to_go(float(lx), float(ly), True)
                    if np.isfinite(c_sw):
                        best_sw = min(best_sw, float(c_sw))
                    if np.isfinite(c_ex):
                        best_ex = min(best_ex, float(c_ex))
                if np.isfinite(best_sw):
                    self._cached_switch_distance = best_sw
                if np.isfinite(best_ex):
                    self._cached_exit_distance = best_ex
                # All arcs doomed/inf => keep the standard (ninja-position)
                # values, same degradation contract as the u=0 path below.
                return

            lx, ly, _cls, _tau = proj.project_landing(
                float(nx), float(ny), float(vx), float(vy), jf, GHOST_ARC_MAX_FRAMES
            )
            c_sw = src.cost_to_completion(float(lx), float(ly), False)
            c_ex = src.cost_to_go(float(lx), float(ly), True)
            if np.isfinite(c_sw):
                self._cached_switch_distance = float(c_sw)
            if np.isfinite(c_ex):
                self._cached_exit_distance = float(c_ex)
        except Exception:
            # Never degrade the standard substitution over a projection issue.
            pass

    def _minethreading_substitute(self, nx, ny, switch_activated) -> bool:
        """Substitute the reward distances with the mine-faithful threading
        cost-to-completion. Returns True iff it committed to the threading source
        this episode (spawn has a finite hard-mine route). Built once per map; the
        per-episode use decision mirrors the lattice gate."""
        try:
            map_name = getattr(
                getattr(self, "map_loader", None), "current_map_name", None
            )
            if (map_name != self._minethreading_map_name
                    or self._minethreading_source is None):
                self._minethreading_map_name = map_name
                nplay = getattr(self, "nplay_headless", None)
                md = getattr(nplay, "current_map_data", None)
                if md:
                    from ...value_function.mine_threading_distance import (
                        MineThreadingDistanceSource,
                    )
                    self._minethreading_source = MineThreadingDistanceSource(
                        md, name=map_name)
                else:
                    self._minethreading_source = None
                self._minethreading_episode_decided = False

            mt = self._minethreading_source
            if mt is None or not getattr(mt, "available", False):
                self._use_minethreading_this_episode = False
                self._minethreading_episode_decided = True
                return False
            if not self._minethreading_episode_decided:
                self._use_minethreading_this_episode = mt.spawn_finite(nx, ny)
                self._minethreading_episode_decided = True
            if not self._use_minethreading_this_episode:
                return False
            # Pre-switch distance composed to cost-to-COMPLETION (continuous across
            # switch activation), exactly like the free-space/lattice substitution.
            c_sw = mt.cost_to_completion(nx, ny, False)
            c_ex = mt.cost_to_go(nx, ny, True)
            if np.isfinite(c_sw):
                self._cached_switch_distance = float(c_sw)
            if np.isfinite(c_ex):
                self._cached_exit_distance = float(c_ex)
            return True
        except Exception:
            self._use_minethreading_this_episode = False
            return False

    def _freespace_substitute(self, nx, ny, switch_activated, want_reward) -> bool:
        """Fallback potential for levels where the surface lattice is severed
        (airborne / dense-mine corridors): a mine-aware free-space grid cost-to-go.
        Substitutes the reward distances so PBRS/HWM still get a dense, corridor-
        following signal (the RL agent learns the frame-precise execution). The
        per-episode use decision mirrors the lattice gate (commit only if the spawn
        has a finite free-space route). No-op unless a reward potential is wanted.

        Returns True iff it committed to the free-space source this episode AND
        wrote both phase distances (a consistent cost-to-completion basis)."""
        if not want_reward:
            return False
        try:
            reach_cfg = getattr(getattr(self, "config", None), "reachability", None)
            map_name = getattr(
                getattr(self, "map_loader", None), "current_map_name", None
            )
            if (map_name != self._freespace_map_name
                    or self._freespace_source is None):
                self._freespace_map_name = map_name
                nplay = getattr(self, "nplay_headless", None)
                md = getattr(nplay, "current_map_data", None)
                if md:
                    # Jump-aware source when use_jump_potential is set, else the
                    # geometric free-space flood (default). Both share the same API
                    # and a (distinct-namespace) two-tier disk cache.
                    cls = self._potential_source_cls(reach_cfg)
                    self._freespace_source = cls(md, name=map_name)
                else:
                    self._freespace_source = None
                self._freespace_episode_decided = False

            fs = self._freespace_source
            if fs is None or not getattr(fs, "available", False):
                self._use_freespace_this_episode = False
                self._freespace_episode_decided = True
                return False
            if not self._freespace_episode_decided:
                self._use_freespace_this_episode = fs.spawn_finite(nx, ny)
                self._freespace_episode_decided = True
            if not self._use_freespace_this_episode:
                return False
            # Pre-switch distance is composed to cost-to-COMPLETION (cost-to-switch
            # + constant exit tail) so the substituted distance is one continuous
            # potential across switch activation, matching the lattice's d_comb.
            # Without this the PBRS term would see a basis change at the switch and
            # punish activating it (see FreeSpaceDistanceSource.cost_to_completion).
            c_sw = fs.cost_to_completion(nx, ny, False)
            c_ex = fs.cost_to_go(nx, ny, True)
            if np.isfinite(c_sw):
                self._cached_switch_distance = float(c_sw)
            if np.isfinite(c_ex):
                self._cached_exit_distance = float(c_ex)
            return bool(np.isfinite(c_sw) and np.isfinite(c_ex))
        except Exception:
            self._use_freespace_this_episode = False
            return False

    def _refresh_free_space_nav(self, nx, ny, switch_activated) -> None:
        """Refresh the default-on ``free_space_nav`` hint (folded into game_state).

        Builds (LRU-cached per level) a mine-aware free-space cost-to-go and reads
        a 4-dim steepest-descent hint ``[dx_hop, dy_hop, cost_norm, phase]`` toward
        the current objective. Physics-free and policy-invariant (an OBSERVATION
        only — do NOT also use it as a reward potential, that would double-count
        against the existing PBRS). Any failure leaves the hint at zeros."""
        try:
            map_name = getattr(
                getattr(self, "map_loader", None), "current_map_name", None
            )
            if (map_name != self._free_space_nav_map_name
                    or self._free_space_nav_source is None):
                self._free_space_nav_map_name = map_name
                self._free_space_nav_source = self._get_free_space_nav_source()
            src = self._free_space_nav_source
            if src is not None and getattr(src, "available", False):
                self._cached_free_space_nav = np.asarray(
                    src.nav_feature(float(nx), float(ny), bool(switch_activated)),
                    dtype=np.float32,
                )
            else:
                self._cached_free_space_nav = np.zeros(4, dtype=np.float32)
        except Exception:
            self._cached_free_space_nav = np.zeros(4, dtype=np.float32)

    def _get_free_space_nav_source(self):
        """Get/build the per-level FreeSpaceDistanceSource for the nav hint.

        Keyed in a process-wide LRU by a hash of the map_data so repeat levels
        (the common curriculum case) skip the ~300ms rebuild."""
        nplay = getattr(self, "nplay_headless", None)
        md = getattr(nplay, "current_map_data", None)
        if not md:
            return None
        reach_cfg = getattr(getattr(self, "config", None), "reachability", None)
        cls = self._potential_source_cls(reach_cfg)
        # v24: route the POLICY-OBSERVED nav/cost-to-go through the MINE-FAITHFUL
        # free-space field (deadly-mine cells cut, so the gradient the policy
        # follows routes AROUND mines) when ``mine_faithful_obs`` is set. Only the
        # plain geometric FreeSpaceDistanceSource takes the mine_hard kwarg; the
        # jump-potential source keeps its own physics-faithful handling.
        mine_hard = bool(getattr(getattr(self, "config", None),
                                 "mine_faithful_obs", False))
        use_hard = mine_hard and cls.__name__ == "FreeSpaceDistanceSource"
        # Per-env memo in front of the process-wide LRU, keyed on the IDENTITY of
        # the map_data list. Every observation calls this (the ghost-arc bundle
        # needs the cost field), and building the LRU key hashes all ~1900 map
        # bytes through a Python generator -- on OFFICIAL levels that one line was
        # the single most-executed thing in the step. ``current_map_data`` is only
        # ever REPLACED on load, never mutated in place, so identity implies same
        # level; the class/mine_hard pair is memoized alongside it because both are
        # config-derived and must not alias.
        memo = getattr(self, "_free_space_nav_memo", None)
        if (
            memo is not None
            and memo[0] is md
            and memo[1] == cls.__name__
            and memo[2] == bool(use_hard)
        ):
            return memo[3]
        # Key on the source CLASS + mine_hard so variants never alias in the LRU.
        key = (cls.__name__, bool(use_hard), hash(tuple(int(v) for v in md)))
        src = _FREE_SPACE_NAV_CACHE.get(key)
        if src is None:
            # The jump source is slow to build but disk-cached (jumpfield/ subdir),
            # so a cold worker pays once per level; the free-space source builds in
            # ~0.1-0.5 s. Both expose the identical nav_feature / cost-to-go API.
            if use_hard:
                src = cls(list(md), name=self._free_space_nav_map_name,
                          mine_hard=True)
            else:
                src = cls(list(md), name=self._free_space_nav_map_name)
            if len(_FREE_SPACE_NAV_CACHE) >= _FREE_SPACE_NAV_CACHE_CAP:
                _FREE_SPACE_NAV_CACHE.clear()
            _FREE_SPACE_NAV_CACHE[key] = src
        self._free_space_nav_memo = (md, cls.__name__, bool(use_hard), src)
        return src

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
            # Live active-gold count (backend-uniform); _cached_gold is stale on
            # native so it would never invalidate the cell cache on a pickup.
            gold_key = sum(
                1 for (_gx, _gy, active) in self.nplay_headless.get_gold_state()
                if active
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
        self._sanitize_cached_distances(ninja_pos, switch_activated)

        # Substitute the physics-aware lattice d* cost-to-go for the geometric A*
        # distance signal where we have a finite plan (handoff step #1: replace
        # the "misleading 6px potential").  No-op unless
        # ReachabilityConfig.use_lattice_potential is set; falls back to the
        # geometric distances above on no-route levels / build failure.
        self._maybe_substitute_lattice_distance(ninja_pos, switch_activated)
        self._maybe_update_waypoints(ninja_pos, switch_activated)

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

    def _refresh_reward_distances(self, ninja_pos: Tuple[float, float]) -> None:
        """Recompute the PBRS reward distances EVERY frame at sub-node resolution.

        ``_compute_reachability`` caches its (unused-in-Layer-1) feature vector AND
        the reward distances by a 24px ninja cell, which froze the reward distance
        for every frame the ninja stayed in a cell — so slow (<=1px) movement
        produced no PBRS/HWM gradient. This computes ONLY the two distances, fresh
        each frame, via the path calculator's existing sub-node-resolved
        ``get_path_cost`` (continuous in the exact ninja position). The expensive
        cost-to-go field is still cached per level by the calculator's level cache,
        so this is lean: a ninja-node lookup + field read + projection per goal.

        Sets ``_cached_switch_distance``/``_cached_exit_distance`` (consumed by the
        reward) and applies the lattice substitution, mirroring
        ``_compute_reachability`` exactly but without the cell cache.
        """
        if not hasattr(self, "current_graph_data") or self.current_graph_data is None:
            return
        adjacency = self.current_graph_data.get("adjacency")
        if adjacency is None:
            return
        base_adjacency = self.current_graph_data.get("base_adjacency", adjacency)
        level_data = self.level_data
        self._path_calculator.build_level_cache(
            level_data, adjacency, base_adjacency, self.current_graph_data
        )

        # Minimal goal dict: the distances-only path needs only switch/exit
        # positions + activation. Avoid the full get_goal_positions_for_features()
        # (it also gathers locked doors + mines and builds the proxy entity_dic,
        # none of which the distance computation uses). The switch/door accessors
        # are O(1) (cached per tick on native).
        nh = self.nplay_headless
        goal_positions = {
            "switch_pos": nh.exit_switch_position(),
            "exit_pos": nh.exit_door_position(),
            "switch_activated": nh.exit_switch_activated(),
            "locked_door_switches": [],
        }

        _features, switch_distance_raw, exit_distance_raw = (
            compute_reachability_features_from_graph(
                adjacency,
                self.current_graph_data,
                level_data,
                ninja_pos,
                self._path_calculator,
                goal_positions=goal_positions,
                include_navigation_features=False,
                distances_only=True,  # fresh sub-node distances, no feature vector
            )
        )
        self._cached_switch_distance = switch_distance_raw
        self._cached_exit_distance = exit_distance_raw

        switch_activated = (
            self.nplay_headless.exit_switch_activated()
            if hasattr(self, "nplay_headless")
            else False
        )
        self._sanitize_cached_distances(ninja_pos, switch_activated)
        self._maybe_substitute_lattice_distance(ninja_pos, switch_activated)
        self._maybe_update_waypoints(ninja_pos, switch_activated)

    def _sanitize_cached_distances(self, ninja_pos, switch_activated: bool) -> None:
        """Guarantee FINITE reward distances; never let the graph's ``inf`` reach
        the reward.

        The reachability GRAPH (``path_distance_calculator``) returns ``inf`` for a
        query position outside its spawn-reachable bounding box. Start-state
        curricula (post-switch / goal-distance reverse curriculum) place the ninja
        at mid-route positions validated only against the looser FREE-SPACE model,
        so a legitimately-placed ninja can land graph-OOB. That ``inf`` flows into
        the HWM term (``hwm_alpha * hwm_progress * beta`` -> ``0*inf = NaN`` even
        when ``hwm_alpha`` is 0 under free-space PBRS) and blows up training.

        Fix the modeling mismatch at the source: when a cached distance is
        non-finite, substitute the FREE-SPACE cost-to-go (the SAME model the
        curriculum validates spawns against), scaled grid-cells -> pixels for unit
        consistency with the graph distance. Ultimate fallback: a large finite
        constant. Fires only on the rare OOB step, so it never perturbs normal
        in-graph distances.
        """
        sw = self._cached_switch_distance
        ex = self._cached_exit_distance
        sw_bad = sw is not None and not np.isfinite(sw)
        ex_bad = ex is not None and not np.isfinite(ex)
        if not (sw_bad or ex_bad):
            return

        # Capture ONLY a non-finite CURRENT-goal distance — that is the real
        # anomaly (bad procgen OR a graph/grid logic error worth offline debug).
        # The OFF-goal distance going inf is BENIGN and expected: once the switch
        # is activated its node drops from the graph so dist_switch reads inf,
        # but the goal is now the exit (and vice-versa pre-switch). Capturing
        # those would flood the dumps with non-bugs, so gate on the live goal.
        current_goal_bad = ex_bad if switch_activated else sw_bad
        if current_goal_bad:
            self._capture_invalid_distance(ninja_pos, switch_activated, sw, ex)

        src = getattr(self, "_free_space_nav_source", None)
        if src is None and hasattr(self, "_get_free_space_nav_source"):
            try:
                src = self._get_free_space_nav_source()
            except Exception:
                src = None
        from ...value_function.free_space_distance import _CELL

        big = float(LEVEL_DIAGONAL * 2.0)
        x, y = float(ninja_pos[0]), float(ninja_pos[1])

        # Lever 4a: straight-line (Euclidean) distance to the phase goal, the
        # LAST-RESORT finite substitute when BOTH the lattice/graph AND the
        # free-space cost-to-go are non-finite at this position — the ~8% OPEN
        # official maps (e.g. "288 ironic nostalgia", "096 look around you") where
        # the surface lattice is severed and the free-space flood cannot reach the
        # ninja's region. The prior fallback was a per-level CONSTANT (``big``, or
        # the field's flat ``dmax`` filler), which left ``hwm_progress`` and the
        # PBRS potential FLAT — zero dense progress signal. Euclidean VARIES with
        # position (monotone-decreasing toward the goal center), so a gradient is
        # restored; it is always finite so no NaN/inf reaches the reward. Applied
        # ONLY when free-space genuinely fails, so maps where the lattice/free-space
        # cost-to-go is available are untouched.
        nh = getattr(self, "nplay_headless", None)

        def _goal(activated: bool):
            if nh is None:
                return None
            try:
                return (
                    nh.exit_door_position() if activated
                    else nh.exit_switch_position()
                )
            except Exception:
                return None

        def _eucl(activated: bool) -> float:
            g = _goal(activated)
            if g is None:
                return big
            e = ((x - float(g[0])) ** 2 + (y - float(g[1])) ** 2) ** 0.5
            return float(e) if np.isfinite(e) else big

        def _fs(activated: bool) -> float:
            # Use the free-space cost-to-go only where it GENUINELY reaches this
            # position. ``cost_at_nearest_free`` returns inf (not the flat ``dmax``
            # filler ``cost_to_go`` substitutes) on a real severance, so it cleanly
            # distinguishes "free-space has a real answer here" from "free-space
            # failed" — only the latter falls through to the Euclidean gradient.
            if src is not None and getattr(src, "available", False):
                try:
                    if np.isfinite(src.cost_at_nearest_free(x, y, activated)):
                        c = float(src.cost_to_go(x, y, activated)) * float(_CELL)
                        if np.isfinite(c):
                            return c
                except Exception:
                    pass
            return _eucl(activated)

        if sw_bad:
            # phase-0 cost-to-go = distance to the exit SWITCH (switch_activated=False)
            self._cached_switch_distance = _fs(False)
        if ex_bad:
            # phase-1 cost-to-go = distance to the EXIT door (switch_activated=True)
            self._cached_exit_distance = _fs(True)

    def _capture_invalid_distance(self, ninja_pos, switch_activated, sw, ex) -> None:
        """Save a debug dump (map image + entities + context) when the graph
        produced a non-finite distance. Best-effort; never raises."""
        try:
            from ..invalid_value_capture import capture_invalid_value

            nh = getattr(self, "nplay_headless", None)
            map_data = getattr(nh, "current_map_data", None)
            bounds = getattr(self._path_calculator, "_adjacency_bounds", None)
            switch_pos = nh.exit_switch_position() if nh is not None else None
            exit_pos = nh.exit_door_position() if nh is not None else None
            context = {
                "ninja_pos": list(ninja_pos),
                "switch_activated": bool(switch_activated),
                "switch_pos": list(switch_pos) if switch_pos is not None else None,
                "exit_pos": list(exit_pos) if exit_pos is not None else None,
                "cached_switch_distance": sw,
                "cached_exit_distance": ex,
                "adjacency_bounds": list(bounds) if bounds is not None else None,
                "step": int(getattr(self, "_episode_step", -1)),
            }
            capture_invalid_value("oob_graph_distance", map_data, context)
        except Exception:
            pass

    def _reinit_reachability_system_after_unpickling(self):
        """Reinitialize reachability system components after unpickling."""
        if not hasattr(self, "_reachability_system"):
            self._reachability_system = ReachabilitySystem()
