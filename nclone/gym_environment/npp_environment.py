"""
Refactored N++ environment class using mixins.

This environment provides a Gym interface for the N++ game, allowing reinforcement
learning agents to learn to play levels. The functionality is now organized using
mixins for better code organization and maintainability.
"""

import logging
import math
import os
import numpy as np
from gymnasium.spaces import box, Dict as SpacesDict
from typing import Dict, Any, Optional, Tuple


from ..constants import MAP_TILE_HEIGHT, MAP_TILE_WIDTH
from ..ninja import jump_is_useless as _jump_is_useless
from ..constants.physics_constants import (
    LEVEL_WIDTH_PX,
    LEVEL_HEIGHT_PX,
    MAX_HOR_SPEED,
    MAX_VER_SPEED,
    MAX_JUMP_DURATION,
    MAX_SURVIVABLE_IMPACT,
    JUMP_FLOOR_Y,
    JUMP_WALL_REGULAR_X_MULTIPLIER,
    JUMP_WALL_REGULAR_Y,
    TILE_PIXEL_SIZE,
)
from .constants import (
    GAME_STATE_CHANNELS,
    GAME_STATE_SWITCH_ACTIVATED_IDX,
    GAME_STATE_SWITCH_VEC_IDX,
    GAME_STATE_EXIT_VEC_IDX,
    GAME_STATE_RAYCAST_IDX,
    GAME_STATE_RAYCAST_DIM,
    GAME_STATE_FREE_SPACE_NAV_IDX,
    GAME_STATE_NEAREST_MINE_IDX,
    GAME_STATE_CLOSING_SPEED_IDX,
    GAME_STATE_JUMP_HELD_IDX,
    GAME_STATE_JUMP_USELESS_IDX,
    GAME_STATE_LAUNCHPAD_BUFFER_IDX,
    GAME_STATE_LANDING_NORMAL_IDX,
    GAME_STATE_LANDING_DIST_IDX,
    GAME_STATE_VERTICAL_TERMINAL_IDX,
    GAME_STATE_TIME_TO_CONTACT_IDX,
    GAME_STATE_FRAMES_AIRBORNE_IDX,
    GAME_STATE_ACCEL_IDX,
    GAME_STATE_SCANDOTS_IDX,
    GAME_STATE_SCANDOTS_COLUMNS,
    GAME_STATE_SCANDOTS_DIM,
    GAME_STATE_GHOST_ARC_IDX,
    GAME_STATE_GHOST_ARC_ARCS,
    GAME_STATE_GHOST_ARC_PER_ARC,
    GAME_STATE_GHOST_ARC_DIM,
    GAME_STATE_JUMP_ARC_IDX,
    GAME_STATE_JUMP_ARC_DIM,
    GAME_STATE_TAP_ARC_IDX,
    GAME_STATE_TAP_ARC_DIM,
    GAME_STATE_WALL_ARC_IDX,
    GAME_STATE_WALL_ARC_DIM,
    GAME_STATE_PASSIVE_ARC_IDX,
    GAME_STATE_PASSIVE_ARC_DIM,
    GAME_STATE_ARC_PHI_IDX,
    GAME_STATE_ARC_PHI_DIM,
    ARC_PHI_NORM_COST,
    GHOST_ARC_MAX_FRAMES,
    EGO_CROP_COARSE_SIZE,
    EGO_CROP_COARSE_POOL_TILES,
    EGO_CROP_COARSE_CHANNELS,
    EGO_CROP_COARSE_ROUTE_CHANNELS,
    EGO_CROP_ROUTE_CHANNELS,
    EGO_CROP_ROUTE_FINE_CHANNELS,
    EGO_CROP_ROUTE_FINE_SIZE,
    EGO_CROP_MINES_FINE_CHANNELS,
    EGO_CROP_MINES_FINE_SIZE,
    EGO_CROP_FALL_FINE_CHANNELS,
    EGO_CROP_FALL_FINE_SIZE,
    MINE_PROXIMITY_DIM,
    THREAT_CLASS_OF_TYPE,
    THREAT_PROXIMITY_DIM,
    THREAT_PROXIMITY_RANGE_PX,
    THREAT_PROXIMITY_SPEED_NORM_PX,
    THREAT_PROX_FEATURES,
    MINE_PROXIMITY_K,
    MINE_PROXIMITY_FEATURES,
    MINE_PROXIMITY_KILL_RADIUS_PX,
    MINE_PROXIMITY_RANGE_PX,
    MINE_PROXIMITY_SPEED_NORM_PX,
    MINE_PROXIMITY_TTC_TAU_FRAMES,
    LEVEL_WIDTH,
    LEVEL_HEIGHT,
    LEVEL_DIAGONAL,
    HAZARD_MAP_CHANNELS,
    HAZARD_MAP_HEIGHT,
    HAZARD_MAP_WIDTH,
    DRONE_HAZARD_MAP_CHANNELS,
    DRONE_HAZARD_HORIZONS,
    DRONE_KILL_RADIUS_PX,
    MINI_DRONE_KILL_RADIUS_PX,
    HAZARD_MAP_OBS_DOWNSAMPLE,
    HAZARD_MAP_OBS_WIDTH,
    THREAT_TTL_CHANNELS,
    SOLID_MAP_CHANNELS,
    HAZARD_MAP_OBS_HEIGHT,
    COST_TO_GO_MAP_DOWNSAMPLE,
    COST_TO_GO_MAP_WIDTH,
    COST_TO_GO_MAP_HEIGHT,
    VISITATION_MAP_WIDTH,
    VISITATION_MAP_HEIGHT,
    VISITATION_MAP_CHANNELS,
    VISITATION_CELL_PX,
    VISITATION_ORIGIN_PX,
    VISIT_DIRECTED_FLOOR,
    FRONTIER_MAP_WIDTH,
    FRONTIER_MAP_HEIGHT,
    FRONTIER_MAP_CHANNELS,
    LATTICE_NAV_DIM,
    WAYPOINT_NAV_DIM,
    MAX_LOCKED_DOORS,
    FEATURES_PER_DOOR,
    SWITCH_STATES_DIM,
    MAX_LOCKED_DOORS_ATTENTION,
    NUM_TILE_TYPES,
    EGO_CROP_SIZE_DEFAULT,
    EGO_CROP_PAD_TILE,
)
from ..constants.entity_types import EntityType
from .base_environment import BaseNppEnvironment
from .reflection import quantize_cell, reflection_odd_direction
from .entity_overlay_utils import (
    classify_mines_by_state,
    ENTITY_OVERLAY_CH_MINE_DEADLY,
    ENTITY_OVERLAY_CH_MINE_SAFE,
)
from .fall_lethality import (
    BORDER_CELLS as FALL_BORDER_CELLS,
    CELL_PX as FALL_CELL_PX,
    FALL_LETHALITY_CHANNELS,
    compute_fall_lethality_map,
    crop_fall_lethality,
)
from .layer3_macro_action_mask import Layer3MacroActionMasker
from .obs_map_pooling import pool_channels
from .region_novelty import label_regions

# Cache sentinel for "no level has been loaded into this cache slot yet".
_UNSET_MAP_NAME = object()
from .mixins import GraphMixin, ReachabilityMixin, DebugMixin
from .config import EnvironmentConfig
from .precomputed_door_features import PrecomputedDoorFeatureCache
from nclone_native import HazardSdf, MineThreatMap

try:  # Ghost-arc projector (Audit II P0-1); absent on pre-rebuild native wheels.
    from nclone_native import TrajectoryProjector
except ImportError:  # pragma: no cover - stale native build
    TrajectoryProjector = None

logger = logging.getLogger(__name__)


#: The 8 neighbour offsets for `_compute_ego_crop_route_fine`'s steepest descent,
#: in the ORDER the scalar `for dx in (-1,0,1): for dy in (-1,0,1)` loop visited
#: them. The order is load-bearing: the loop took a neighbour on a strict
#: `score < best`, so the FIRST of any tied minima won, and `np.argmin` also
#: returns the first minimum. Reordering this silently changes which descent
#: direction is emitted on ties.
_EGO_ROUTE_NEIGHBOUR_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)
#: Euclidean step length per offset, shaped to broadcast over (8, S, S).
_EGO_ROUTE_NEIGHBOUR_STEPS = np.array(
    [math.hypot(dx, dy) for dx, dy in _EGO_ROUTE_NEIGHBOUR_OFFSETS],
    dtype=np.float64,
).reshape(-1, 1, 1)
#: Unit descent vector per offset, indexed by the argmin result.
_EGO_ROUTE_UNIT_DX = np.array(
    [dx / math.hypot(dx, dy) for dx, dy in _EGO_ROUTE_NEIGHBOUR_OFFSETS],
    dtype=np.float64,
)
_EGO_ROUTE_UNIT_DY = np.array(
    [dy / math.hypot(dx, dy) for dx, dy in _EGO_ROUTE_NEIGHBOUR_OFFSETS],
    dtype=np.float64,
)


def _cell_x(px: float, cell_px: float, odd_dir: int) -> int:
    """Ninja x -> grid column, equivariant under a left-right reflection.

    Thin wrapper over :func:`.reflection.quantize_cell` so the call sites read as
    the plain ``floor`` they replace. With ``odd_dir`` at its ``+1`` default this
    IS the plain floor; only the exact-cell-boundary case differs, which is the
    only case where reflection equivariance can break (see reflection.py).
    """
    return quantize_cell(px, cell_px, odd_dir)


def _map_data_curriculum_key(map_data) -> str:
    """Stable content key for a level's map_data, used to match demo-built
    reverse-curricula to the current level regardless of how the loader named
    it (the packed pool names by integer level_id, not the map name). MUST stay
    byte-for-byte identical to build_demo_curricula._map_data_curriculum_key so
    the keys agree across the builder, the trainer loader, and this lookup."""
    import hashlib

    b = bytes(int(v) & 0xFF for v in map_data)
    return "md:" + hashlib.md5(b).hexdigest()


class NppEnvironment(BaseNppEnvironment, GraphMixin, ReachabilityMixin, DebugMixin):
    """
    Consolidated N++ environment class using mixins.

    This environment provides a Gym interface for the N++ game, allowing reinforcement
    learning agents to learn to play levels. We use a headless version of the game
    to speed up training.

    Features:
    - Multiple observation profiles (minimal/rich)
    - HWM A* distance shaping (single unified reward path)
    - Frame stacking support
    - Graph-based planning and visualization
    - Episode truncation based on progress
    - Reachability analysis
    - Debug overlays and visualization
    """

    def __init__(self, config: EnvironmentConfig):
        """
        Initialize the N++ environment.

        Args:
            config: Environment configuration object
        """
        _logger = logging.getLogger(__name__)

        self.config = config

        super().__init__(
            render_mode=self.config.render.render_mode,
            enable_animation=self.config.render.enable_animation,
            enable_logging=self.config.enable_logging,
            enable_debug_overlay=self.config.render.enable_debug_overlay,
            panel_width=self.config.render.panel_width,
            seed=self.config.seed,
            eval_mode=self.config.eval_mode,
            pbrs_gamma=self.config.pbrs.pbrs_gamma,
            custom_map_path=self.config.custom_map_path,
            test_dataset_path=self.config.test_dataset_path,
            defer_initial_map_load=self.config.defer_initial_map_load,
            enable_profiling=self.config.enable_profiling,
            profiling_console_interval=self.config.profiling_console_interval,
            reward_config=self.config.reward_config,
            frame_skip=self.config.frame_skip,
            shared_level_cache=self.config.shared_level_cache,
            sim_backend=self.config.sim_backend,
            enable_layer3_macro_action_mask=self.config.enable_layer3_macro_action_mask,
            layer3_duration_bins=self.config.layer3_duration_bins,
            layer3_all_masked_fallback=self.config.layer3_all_masked_fallback,
            enable_route_visualization=self.config.enable_route_visualization,
        )

        self.layer3_masker: Optional[Layer3MacroActionMasker]
        if self.config.enable_layer3_macro_action_mask:
            self.layer3_masker = Layer3MacroActionMasker(
                duration_bins=self.config.layer3_duration_bins,
                fallback_mode=self.config.layer3_all_masked_fallback,
                include_lethal_impact=getattr(
                    self.config, "layer3_mask_lethal_impact", False
                ),
                lethal_impact_lookahead=getattr(
                    self.config, "layer3_lethal_impact_lookahead", 0
                ),
            )
        else:
            self.layer3_masker = None

        # Initialize mixin systems using config
        self._init_graph_system(
            debug=self.config.graph.debug,
        )
        self._init_reachability_system(self.config.reachability.debug)
        # Single CachedPathDistanceCalculator: reward and reachability share SDF, mine
        # proximity, and level path caches (avoids duplicate rebuilds on graph/mine changes).
        self.reward_calculator.path_calculator = self._path_calculator
        self._init_debug_system(self.config.render.enable_debug_overlay)

        # Update configuration flags with new options
        self.config_flags.update(
            {"debug": self.config.graph.debug or self.config.reachability.debug}
        )

        # Rust mine SDF engines powering the 2-channel `hazard_map` Layer 1 key.
        # One engine per class (deadly / safe) so each keeps its own change-detect
        # state; both emit a row-major (H, W) buffer the env transposes to (W, H).
        self._mine_threat_map_deadly_rs = MineThreatMap()
        self._mine_threat_map_safe_rs = MineThreatMap()
        # Per-class transposed (W, H) channel caches and the stacked (2, W, H) map.
        self._hazard_deadly_chan = np.zeros(
            (HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT), dtype=np.float32
        )
        self._hazard_safe_chan = np.zeros(
            (HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT), dtype=np.float32
        )
        self._hazard_map_cache = np.zeros(
            (HAZARD_MAP_CHANNELS, HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT),
            dtype=np.float32,
        )
        # Cheap mine-state signature for the hazard map. When it is unchanged
        # (the common case — a static mine field never toggles) we skip the whole
        # per-step prep (classify + build the (x,y) lists over ~1.7k mines + the
        # Rust change-detect) and return the cached SDF directly.
        self._hazard_sig = None

        # Rust SDF engines for the optional `drone_hazard_map` key. One per drone
        # class because the contact-death radius differs (zap 17.5 px, mini 14 px);
        # the two fields are combined by an element-wise min, which is the correct
        # worst-case merge for a signed distance where lower means nearer death.
        # Built only when the flag is on: each is a stateless ~15k-cell engine, but
        # the per-frame compute is the expensive part and must not run unrequested.
        self._drone_hazard_enabled = bool(
            getattr(
                getattr(self.config, "reachability", None),
                "enable_drone_hazard",
                False,
            )
        )
        if self._drone_hazard_enabled:
            self._drone_sdf_zap_rs = HazardSdf(DRONE_KILL_RADIUS_PX)
            self._drone_sdf_mini_rs = HazardSdf(MINI_DRONE_KILL_RADIUS_PX)
        else:
            self._drone_sdf_zap_rs = None
            self._drone_sdf_mini_rs = None
        # Phase 1 keys. Both come straight from the native simulator; on the
        # pure-Python backend they are simply unavailable and the env declines to
        # declare them rather than emitting a zero field that would read as "no
        # threats anywhere" / "no surfaces anywhere".
        _reach = getattr(self.config, "reachability", None)
        self._threat_ttl_enabled = bool(getattr(_reach, "enable_threat_ttl", False))
        self._threat_ttl_max_frames = int(getattr(_reach, "threat_ttl_max_frames", 64))
        self._threat_ttl_stride = int(getattr(_reach, "threat_ttl_stride", 4))
        self._solid_map_enabled = bool(getattr(_reach, "enable_solid_map", False))
        if self._solid_map_enabled and getattr(self, "sim_backend", None) != "native":
            # Refuse rather than emit zeros. A zeroed occupancy map is not a
            # degraded observation, it is a WRONG one: it says "there are no
            # surfaces here" on a level full of thwumps. If training runs native
            # and an eval runs Python, the policy silently receives a different
            # observation than it was trained on, and the eval score drops for a
            # reason nothing reports. `threat_ttl` has a pinned numpy reference
            # so it can fall back honestly; `solid_map` has none yet.
            raise ValueError(
                "enable_solid_map requires sim_backend='native' (got "
                f"{getattr(self, 'sim_backend', None)!r}): there is no Python "
                "reference implementation, and emitting zeros would misreport "
                "every solid entity as absent."
            )

        self._drone_hazard_cache = np.ones(
            (DRONE_HAZARD_MAP_CHANNELS, HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT),
            dtype=np.float32,
        )

        # Deadly-mine position array (Nx2) + its mine signature, for the scalar
        # nearest-deadly-mine game_state feature. Recomputed only when the deadly
        # set changes (a toggle); otherwise the per-step cost is a vectorized min.
        self._deadly_mine_xy: Optional[np.ndarray] = None
        self._deadly_xy_sig = None
        # Per-frame memo for _mine_lists_and_signature: several observation
        # features key their own caches on the same (n_toggle, n_toggled,
        # n_deadly) triple, and each was re-counting every mine.
        self._mine_sig_frame = -1
        self._mine_sig_cache = None

        # Cached tile grid for the current level (computed once per level load, reused each step)
        self._cached_tile_grid: Optional[np.ndarray] = None
        # v26 W6a: static fall-lethality field. Depends only on tile geometry, so
        # it is computed once per level load and invalidated with the map name —
        # the same treatment the tile grid gets. (2, HAZARD_W, HAZARD_H).
        self._fall_lethality_map: Optional[np.ndarray] = None
        # Sentinel, not None: a level can legitimately have current_map_name None
        # and must still build (and then reuse) its field.
        self._fall_lethality_map_name: object = _UNSET_MAP_NAME
        # Stable integer level ID derived from map name hash (for cache keying in the encoder)
        self._cached_level_id_int: Optional[np.ndarray] = None

        # v17 episodic visitation tracking (CHAMBER_MAZE exploration). One per-
        # episode int32 visit-count grid at 12px cells, co-registered with
        # cost_to_go_map, feeding BOTH the count-based reward bonus and the
        # optional visitation_map obs channel. Allocated lazily / reset per episode.
        self._visit_counts: Optional[np.ndarray] = None
        self._visit_novelty: float = 0.0  # 1/sqrt(count_after) for THIS step's cell
        self._visit_new_cell: bool = False  # was this step's cell first-visited?
        self._visit_cells: int = 0  # distinct cells visited this episode
        self._visit_coverage: float = 0.0  # visited / reachable-cell fraction
        self._visitation_bonus: float = 0.0  # coef * novelty * directed added this step
        self._visit_directed_wt: float = 1.0  # P2 d*-directed weight this step
        # (src identity, reachable count, pooled (84,46) bool mask or None)
        self._visit_reachable_cache: tuple = (None, 0, None)
        # Enabled iff either the reward bonus or the obs channel is requested.
        _reach_cfg = getattr(self.config, "reachability", None)
        # v26 W4: room-discovery bonus coefficient (0 = off).
        self._region_novelty_coef = float(
            getattr(self.config, "region_novelty_coef", 0.0)
        )
        self._regions_visited: set = set()
        self._region_new = False
        self._region_bonus = 0.0
        self._region_coverage = 0.0
        self._region_count = 0
        self._region_label_cache = None
        self._visitation_bonus_coef = float(
            getattr(self.config, "visitation_bonus_coef", 0.0)
        )
        self._visitation_directed = bool(
            getattr(self.config, "visitation_directed", False)
        )
        self._expose_visitation_map = bool(
            getattr(_reach_cfg, "expose_visitation_map", False)
        )
        # v19 R4a: recency (time-decay) visitation channel. tau > 0 => the
        # visitation_map value becomes exp(-(t - t_last_visit)/tau) so corridor
        # ORDER/recency is visible to the feedforward policy (a memoryless policy
        # otherwise cannot represent "try corridors in order"). 0 = legacy
        # 1/sqrt(count) count-decay. BREAKING obs semantics (same shape).
        self._visit_recency_tau = float(
            getattr(_reach_cfg, "visitation_recency_tau", 0.0)
        )
        # v18 C1 spatial frontier map + go-to-frontier shaping. frontier =
        # reachable AND unvisited on the same 12px grid, so the feature rides
        # the v17 visit-count grid (forced on below when frontier is enabled).
        self._expose_frontier_map = bool(
            getattr(_reach_cfg, "expose_frontier_map", False)
        )
        self._frontier_map_weighted = bool(
            getattr(_reach_cfg, "frontier_map_weighted", False)
        )
        self._frontier_approach_coef = float(
            getattr(self.config, "frontier_approach_bonus", 0.0)
        )
        self._frontier_enabled = bool(
            self._expose_frontier_map or self._frontier_approach_coef > 0.0
        )
        self._frontier_bonus: float = 0.0  # reward term added this step
        self._frontier_dist: float = -1.0  # cells to nearest frontier (-1 = n/a)
        self._frontier_hwm: float = float("inf")  # HWM (min approach) this epoch
        self._frontier_cells: int = 0  # frontier cells remaining
        self._frontier_frac: float = 0.0  # frontier cells / reachable cells
        self._visit_map_enabled = bool(
            self._visitation_bonus_coef > 0.0
            or self._expose_visitation_map
            or self._frontier_enabled
        )
        # v18 Component 3: cost_to_go_map normalization mode (see config.py).
        self._cost_map_norm = str(
            getattr(_reach_cfg, "cost_map_norm", "fixed_k")
        )
        # v19 R4b: window for the ego_residual cost-map mode.
        self._cost_map_ego_window = float(
            getattr(_reach_cfg, "cost_map_ego_window", 8.0)
        )

        # Extend observation space with graph, reachability features
        self.observation_space = self._build_extended_observation_space()

        # Pre-allocate observation buffers
        self._game_state_buffer = np.zeros(GAME_STATE_CHANNELS, dtype=np.float32)
        # v22 §3-B blocked-toward-goal state (cost-to-go trend + stashed feature).
        self._prev_cost_to_go = None
        self._blocked_goal_feat = np.zeros(1, dtype=np.float32)

        # Cache for locked door features (keyed by switch state hash + ninja grid cell)
        #   1. Switch states (rarely change)
        #   2. Ninja grid cell (24px quantization to balance cache hits vs accuracy)
        self._locked_door_cache: Dict[str, np.ndarray] = {}
        self._last_switch_state_hash: Optional[str] = None
        self._last_ninja_grid_cell: Optional[tuple] = None

        # Cache invalidation flag (set by entity interactions)
        self._switch_states_changed: bool = True  # Start True to compute initial state

        # Precomputed door feature cache (aggressive optimization)
        # PERFORMANCE: Precomputes ALL path distances at level load for O(1) lookup
        self.door_feature_cache = PrecomputedDoorFeatureCache()

        # Level-specific locked door caches (static for level duration)
        self._has_locked_doors: bool = False  # Flag: does level have locked doors?
        self._cached_locked_doors: list = []  # Cached locked door entities (static)
        self._cached_switch_states: Optional[np.ndarray] = None

        # Exit features cache (position-based invalidation)
        # Avoids recomputing path distances when ninja hasn't moved significantly
        self._last_exit_cache_ninja_pos: Optional[tuple] = None
        self._exit_cache_grid_size: int = 24

        # Set environment reference in simulator for cache invalidation
        self.nplay_headless.sim.gym_env = self

    def _safe_path_distance(
        self,
        start_pos,
        goal_pos,
        adjacency,
        feature_name,
        level_data=None,
        graph_data=None,
        entity_radius=0.0,
    ):
        """Calculate path distance with inf handling and detailed logging.

        Returns safe max value (LEVEL_DIAGONAL * 2.0) for unreachable paths.
        Logs detailed warnings when paths are unreachable for debugging.

        Args:
            start_pos: Starting position tuple (x, y)
            goal_pos: Goal position tuple (x, y)
            adjacency: Graph adjacency structure
            feature_name: Name of feature for logging (e.g., "exit_switch", "mine")
            level_data: Optional level data for caching
            graph_data: Optional graph data for spatial indexing
            entity_radius: Collision radius of the goal entity (default 0.0)

        Returns:
            Path distance in pixels, or LEVEL_DIAGONAL * 2.0 if unreachable
        """
        from ..constants.physics_constants import NINJA_RADIUS

        base_adjacency = (
            graph_data.get("base_adjacency", adjacency) if graph_data else adjacency
        )
        distance = self._path_calculator.get_path_cost(
            start_pos,
            goal_pos,
            adjacency,
            base_adjacency,
            level_data=level_data,
            graph_data=graph_data,
            entity_radius=entity_radius,
            ninja_radius=NINJA_RADIUS,
        )

        if distance == float("inf"):
            logger.warning(
                f"[PATH_DISTANCE] Unreachable {feature_name}: "
                f"start={start_pos}, goal={goal_pos}"
            )
            return LEVEL_DIAGONAL * 2.0  # Safe max value

        return distance

    def _build_extended_observation_space(self) -> SpacesDict:
        """Build the Layer 1 policy observation space.

        Exactly four structured tensors plus control masks:

        * ``tile_grid``       — inherited from base env
        * ``entity_overlays`` — inherited from base env
        * ``hazard_map``      — 1-channel deadly-mine SDF (NPP-46)
        * ``game_state``      — inherited from base env (overwritten with 28 dims)
        * ``action_mask`` / ``duration_mask`` — control masks

        Graph tensors, reachability vectors, spatial_context, mine_sdf_features,
        level_id and switch_states are kept internally by reward-side
        infrastructure but are NOT exposed to the policy.
        """
        obs_spaces = dict(self.observation_space.spaces)

        # Emitted at HAZARD_MAP_OBS_* (pooled from the 6px field the env computes
        # internally); the ego crops and PBRS still read full resolution.
        obs_spaces["hazard_map"] = box.Box(
            low=-1.0,
            high=1.0,
            shape=(HAZARD_MAP_CHANNELS, HAZARD_MAP_OBS_WIDTH, HAZARD_MAP_OBS_HEIGHT),
            dtype=np.float32,
        )

        # Optional patrol-drone SDF, same grid/units as `hazard_map`. Off by
        # default so the strict Layer 1 contract and existing checkpoints are
        # unaffected; enabled via EnvConfig.enable_drone_hazard.
        if self._threat_ttl_enabled:
            obs_spaces["threat_ttl"] = box.Box(
                low=-1.0,
                high=1.0,
                shape=(
                    THREAT_TTL_CHANNELS,
                    HAZARD_MAP_OBS_WIDTH,
                    HAZARD_MAP_OBS_HEIGHT,
                ),
                dtype=np.float32,
            )
        if self._solid_map_enabled:
            obs_spaces["solid_map"] = box.Box(
                low=-1.0,
                high=1.0,
                shape=(
                    SOLID_MAP_CHANNELS,
                    HAZARD_MAP_OBS_WIDTH,
                    HAZARD_MAP_OBS_HEIGHT,
                ),
                dtype=np.float32,
            )
        if self._drone_hazard_enabled:
            obs_spaces["drone_hazard_map"] = box.Box(
                low=-1.0,
                high=1.0,
                shape=(
                    DRONE_HAZARD_MAP_CHANNELS,
                    HAZARD_MAP_OBS_WIDTH,
                    HAZARD_MAP_OBS_HEIGHT,
                ),
                dtype=np.float32,
            )

        # Optional physics-aware navigation hint from the lattice d* plan
        # (``[dx_hop, dy_hop, cost_to_go_norm, phase]``).  Off by default so the
        # strict Layer-1 contract / existing checkpoints are unaffected; enabled
        # via ReachabilityConfig.expose_lattice_features.  See lattice_distance.
        if getattr(getattr(self.config, "reachability", None),
                   "expose_lattice_features", False):
            obs_spaces["lattice_nav"] = box.Box(
                low=-1.0, high=1.0, shape=(LATTICE_NAV_DIM,), dtype=np.float32,
            )

        # Optional waypoint-subgoal nav hint (same 4-dim contract as lattice_nav).
        # Off by default; enabled via ReachabilityConfig.expose_waypoint_features.
        if getattr(getattr(self.config, "reachability", None),
                   "expose_waypoint_features", False):
            obs_spaces["waypoint_nav"] = box.Box(
                low=-1.0, high=1.0, shape=(WAYPOINT_NAV_DIM,), dtype=np.float32,
            )

        # Optional cost-to-go map (Phase-2 O2): the free-space cost-to-completion
        # field as low-res spatial channels co-registered with hazard_map. 3 ch:
        #   0 = ACTIVE-phase cost-to-go (cost-to-switch pre-activation, else exit),
        #   1 = OTHER-phase cost-to-go (the full switch->exit route is visible at
        #       once -> directly targets long-horizon FAR routing),
        #   2 = reachable mask (1 where the objective is reachable through free
        #       space, 0 = unreachable pocket / solid).
        if getattr(getattr(self.config, "reachability", None),
                   "expose_cost_to_go_map", False):
            obs_spaces["cost_to_go_map"] = box.Box(
                low=0.0, high=1.0,
                shape=(3, COST_TO_GO_MAP_WIDTH, COST_TO_GO_MAP_HEIGHT),
                dtype=np.float32,
            )

        # v17 episodic visitation map: per-episode recency-decayed "where have I
        # been" grid, co-registered with cost_to_go_map (same 12px cells / frame).
        # 1 channel in [0, 1]: 1.0 = just visited, decaying toward 0 with recency,
        # 0 = never visited this episode. See _compute_visitation_map.
        if getattr(getattr(self.config, "reachability", None),
                   "expose_visitation_map", False):
            obs_spaces["visitation_map"] = box.Box(
                low=0.0, high=1.0,
                shape=(VISITATION_MAP_CHANNELS,
                       VISITATION_MAP_WIDTH, VISITATION_MAP_HEIGHT),
                dtype=np.float32,
            )

        # v18 C1 spatial frontier map: reachable AND unvisited-this-episode, on
        # the same co-registered 12px grid as cost_to_go_map / visitation_map.
        # 1.0 (or the region-density weight) at frontier cells, 0 elsewhere.
        # See _compute_frontier_map.
        if getattr(getattr(self.config, "reachability", None),
                   "expose_frontier_map", False):
            obs_spaces["frontier_map"] = box.Box(
                low=0.0, high=1.0,
                shape=(FRONTIER_MAP_CHANNELS,
                       FRONTIER_MAP_WIDTH, FRONTIER_MAP_HEIGHT),
                dtype=np.float32,
            )

        # Optional ego-centric high-res crop of the tile grid (``ego_crop``), a
        # square window of raw tile IDs re-centered on the ninja each step. Same
        # dtype/bounds as tile_grid; the extractor embeds it just like tile_grid.
        # Off by default so the strict Layer-1 contract / existing checkpoints are
        # unaffected; enabled via ReachabilityConfig.enable_ego_crop.
        if getattr(getattr(self.config, "reachability", None),
                   "enable_ego_crop", False):
            _sz = int(getattr(self.config.reachability, "ego_crop_size",
                              EGO_CROP_SIZE_DEFAULT))
            obs_spaces["ego_crop"] = box.Box(
                low=0, high=NUM_TILE_TYPES - 1,
                shape=(_sz, _sz),
                dtype=np.int8,
            )
            # Mine-density crop (deadly, safe) co-registered with ego_crop.
            obs_spaces["ego_crop_mines"] = box.Box(
                low=0.0, high=np.inf,
                shape=(2, _sz, _sz),
                dtype=np.float32,
            )
            # Route-gradient crop (v14 B2): local steepest-descent unit vector
            # of the active-phase cost-to-go, co-registered with ego_crop.
            obs_spaces["ego_crop_route"] = box.Box(
                low=-1.0, high=1.0,
                shape=(EGO_CROP_ROUTE_CHANNELS, _sz, _sz),
                dtype=np.float32,
            )

        # Audit II P1-6: coarse far ego crop — a second ninja-centered window
        # at 2-tile pooling (+-21 tiles Chebyshev) with [solid fraction,
        # deadly-mine density, safe-mine density, objective marker] channels.
        # Bridges the fine crop's +-10-tile horizon (== max survivable free
        # fall) and the 168-336 px/cell global readouts.
        if getattr(getattr(self.config, "reachability", None),
                   "enable_ego_crop_coarse", False):
            _cc_route = getattr(getattr(self.config, "reachability", None),
                                "ego_crop_coarse_route", False)
            _cc_ch = EGO_CROP_COARSE_CHANNELS + (
                EGO_CROP_COARSE_ROUTE_CHANNELS if _cc_route else 0
            )
            # low=-1 when route (unit-vector) channels are appended; density /
            # solidity / marker channels are all >=0 so -1 is a valid superset.
            obs_spaces["ego_crop_coarse"] = box.Box(
                low=-1.0 if _cc_route else 0.0, high=np.inf,
                shape=(_cc_ch, EGO_CROP_COARSE_SIZE, EGO_CROP_COARSE_SIZE),
                dtype=np.float32,
            )

        # v22 §3-C: fine (6px, un-pooled) ego route-gradient crop. Independent of
        # enable_ego_crop (own flag) so it can be added on its own.
        if getattr(getattr(self.config, "reachability", None),
                   "expose_ego_crop_route_fine", False):
            obs_spaces["ego_crop_route_fine"] = box.Box(
                low=-1.0, high=1.0,
                shape=(EGO_CROP_ROUTE_FINE_CHANNELS,
                       EGO_CROP_ROUTE_FINE_SIZE, EGO_CROP_ROUTE_FINE_SIZE),
                dtype=np.float32,
            )

        # v24: fine ego MINE crop (2ch signed-distance, ninja-centered 6px window).
        if getattr(getattr(self.config, "reachability", None),
                   "expose_ego_crop_mines_fine", False):
            obs_spaces["ego_crop_mines_fine"] = box.Box(
                low=-1.0, high=1.0,
                shape=(EGO_CROP_MINES_FINE_CHANNELS,
                       EGO_CROP_MINES_FINE_SIZE, EGO_CROP_MINES_FINE_SIZE),
                dtype=np.float32,
            )

        # v26 W6a: static fall-lethality field (global 6px + fine ego crop).
        if getattr(getattr(self.config, "reachability", None),
                   "expose_fall_lethality_map", False):
            obs_spaces["fall_lethality_map"] = box.Box(
                low=-1.0, high=1.0,
                shape=(FALL_LETHALITY_CHANNELS,
                       HAZARD_MAP_OBS_WIDTH, HAZARD_MAP_OBS_HEIGHT),
                dtype=np.float32,
            )
        if getattr(getattr(self.config, "reachability", None),
                   "expose_ego_crop_fall_fine", False):
            obs_spaces["ego_crop_fall_fine"] = box.Box(
                low=-1.0, high=1.0,
                shape=(EGO_CROP_FALL_FINE_CHANNELS,
                       EGO_CROP_FALL_FINE_SIZE, EGO_CROP_FALL_FINE_SIZE),
                dtype=np.float32,
            )

        # v25: directional nearest-mine bearing + closing speed.
        if getattr(getattr(self.config, "reachability", None),
                   "expose_mine_proximity", False):
            obs_spaces["mine_proximity"] = box.Box(
                low=-1.0, high=1.0,
                shape=(MINE_PROXIMITY_DIM,),
                dtype=np.float32,
            )

        # Shooter telegraph: per-super-class bearing + wind-up countdown. Gated
        # on the same flag as the threat field, because it is the half of that
        # signal a field cannot carry (see constants.THREAT_PROXIMITY_DIM).
        if self._threat_ttl_enabled:
            obs_spaces["threat_proximity"] = box.Box(
                low=-1.0, high=1.0,
                shape=(THREAT_PROXIMITY_DIM,),
                dtype=np.float32,
            )

        # Gold-fraction-remaining scalar, appended UNCONDITIONALLY right after the
        # base game_state (docs/GOLD_AND_SCORE_OPTIMIZATION.md part b): a global
        # "how much gold is left" signal complementing the spatial gold-density
        # overlay channel. Always present (independent of enable_gold_rewards) so
        # the obs contract is fixed; reads 0.0 on levels with no gold. This is the
        # FIRST game_state append, so the value layout is
        # [... base ..., gold_frac, blocked_goal?, wall_tau?, precision_state?, time_phase?].
        _gs = obs_spaces["game_state"]
        obs_spaces["game_state"] = box.Box(
            low=-1.0, high=1.0,
            shape=(int(_gs.shape[0]) + 1,),
            dtype=np.float32,
        )

        # Precision-state block appended to game_state (+PRECISION_STATE_DIM).
        if getattr(getattr(self.config, "reachability", None),
                   "expose_precision_state", False):
            _gs = obs_spaces["game_state"]
            obs_spaces["game_state"] = box.Box(
                low=-1.0, high=1.0,
                shape=(int(_gs.shape[0]) + self.PRECISION_STATE_DIM,),
                dtype=np.float32,
            )

        # Frames-to-wall-contact scalar appended to game_state (+1).
        if getattr(getattr(self.config, "reachability", None),
                   "expose_wall_contact_tau", False):
            _gs = obs_spaces["game_state"]
            obs_spaces["game_state"] = box.Box(
                low=-1.0, high=1.0,
                shape=(int(_gs.shape[0]) + 1,),
                dtype=np.float32,
            )

        # v22 §3-B: append the derived blocked-toward-goal scalar to game_state.
        if getattr(getattr(self.config, "reachability", None),
                   "expose_blocked_goal", False):
            _gs = obs_spaces["game_state"]
            obs_spaces["game_state"] = box.Box(
                low=-1.0, high=1.0,
                shape=(int(_gs.shape[0]) + 1,),
                dtype=np.float32,
            )

        # EXECUTION_PRECISION_PLAN §6: time/phase block (+TIME_PHASE_DIM). Grown
        # LAST so the documented game_state layout stays
        # [... base ..., blocked_goal?, wall_tau?, precision_state?, time_phase?].
        if getattr(getattr(self.config, "reachability", None),
                   "expose_time_phase", False):
            _gs = obs_spaces["game_state"]
            obs_spaces["game_state"] = box.Box(
                low=-1.0, high=1.0,
                shape=(int(_gs.shape[0]) + self.TIME_PHASE_DIM,),
                dtype=np.float32,
            )

        # v26 W6c: commitment/irreversibility block (+COMMITMENT_STATE_DIM).
        # Grown after time_phase, matching the re-attach order.
        if getattr(getattr(self.config, "reachability", None),
                   "expose_commitment_state", False):
            _gs = obs_spaces["game_state"]
            obs_spaces["game_state"] = box.Box(
                low=-1.0, high=1.0,
                shape=(int(_gs.shape[0]) + self.COMMITMENT_STATE_DIM,),
                dtype=np.float32,
            )

        return SpacesDict(obs_spaces)

    def _post_action_hook(self):
        """Update graph after action execution if needed."""
        # Graph building happens if either flag is True
        should_update = self._should_update_graph()

        if should_update:
            self._update_graph_from_env_state()

    def _extend_info_hook(self, info: Dict[str, Any]):
        """Add NppEnvironment-specific info fields."""
        # Call parent to add diagnostic and observation metrics
        super()._extend_info_hook(info)

        # These fields are removed during observation processing but needed for reward calculation
        info["player_x"] = self.nplay_headless.ninja_position()[0]
        info["player_y"] = self.nplay_headless.ninja_position()[1]

        # Curriculum-seeded episode-start telemetry (set in _maybe_apply_start_state
        # each reset). npp-rl consumes these EXACT key names to exclude seeded
        # (post-switch / RFCL / reverse-curriculum) episodes from the phase gate.
        # start_state_difficulty is 1.0 when the episode starts at the natural spawn.
        info["curriculum_seeded_start"] = bool(
            getattr(self, "_curriculum_seeded_start", False)
        )
        # v16 takeoff-arc gap mask: jump-veto count this step (0 when disabled /
        # not grounded / no fatal takeoff). Lets npp-rl log how often the veto
        # fires, i.e. how often the agent was about to jump into a mine gap.
        info["takeoff_gap_masked"] = int(
            getattr(self, "_takeoff_gap_mask_count", 0)
        )
        self._takeoff_gap_mask_count = 0
        info["start_state_difficulty"] = float(
            getattr(self, "_episode_start_difficulty", 1.0)
        )
        # Which reverse-curriculum leg seeded this episode ("pre"/"post"/None).
        # The trainer's ReverseCurriculumCallback routes wins to per-leg deques so
        # the approach and return legs anneal independently.
        info["seeded_leg"] = getattr(self, "_episode_seeded_leg", None)

        # v17 episodic visitation telemetry (updated in _pre_reward_hook this step).
        # visit_novelty = 1/sqrt(count) for this step's cell (exploration signal);
        # visit_new_cell = first visit this episode; visit_coverage = distinct
        # reachable cells seen so far / reachable cells; visitation_bonus = the
        # reward term actually added this step. All 0 when the feature is off.
        info["visit_novelty"] = float(getattr(self, "_visit_novelty", 0.0))
        info["visit_new_cell"] = bool(getattr(self, "_visit_new_cell", False))
        info["visit_cells"] = int(getattr(self, "_visit_cells", 0))
        info["visit_coverage"] = float(getattr(self, "_visit_coverage", 0.0))
        info["visitation_bonus"] = float(getattr(self, "_visitation_bonus", 0.0))
        # v26 W4: room-discovery telemetry. region_coverage is the metric that
        # says whether maze exploration is actually working — cell coverage can
        # look healthy while the agent loops one wing of the maze forever.
        # v26 W5: the level's optimal route length, in FRAMES. gamma = 0.998 gives
        # an effective horizon of ~500 frames against 1612-frame LOOP_CORRIDOR T4
        # routes and 3027-frame mazes, so a horizon-matched per-level gamma needs
        # to know how long THIS level's route actually is. The reward calculator
        # already has the d* cost in pixels; converting with the same par speed
        # the completion-speed bonus uses keeps the two definitions of "par" from
        # drifting apart. 0.0 = unknown (the consumer falls back to a fixed gamma).
        _opl = getattr(self.reward_calculator, "optimal_path_length", None)
        if _opl:
            _par_px = max(
                float(
                    getattr(self.config, "completion_speed_par_speed_px", 1.0)
                    or 1.0
                ),
                1e-6,
            )
            info["optimal_path_frames"] = float(_opl) / _par_px
        else:
            info["optimal_path_frames"] = 0.0

        info["region_bonus"] = float(getattr(self, "_region_bonus", 0.0))
        info["region_new"] = bool(getattr(self, "_region_new", False))
        info["region_coverage"] = float(getattr(self, "_region_coverage", 0.0))
        info["region_count"] = int(getattr(self, "_region_count", 0))
        info["visit_directed_wt"] = float(getattr(self, "_visit_directed_wt", 1.0))

        # v18 C1 frontier telemetry (updated in _pre_reward_hook this step).
        # frontier_frac = frontier cells / reachable cells (falls toward 0 as
        # the episode covers the level); frontier_distance = cells to the
        # nearest frontier (-1 when none / unavailable); frontier_bonus = the
        # go-to-frontier reward term actually added this step.
        info["frontier_cells"] = int(getattr(self, "_frontier_cells", 0))
        info["frontier_frac"] = float(getattr(self, "_frontier_frac", 0.0))
        info["frontier_distance"] = float(getattr(self, "_frontier_dist", -1.0))
        info["frontier_bonus"] = float(getattr(self, "_frontier_bonus", 0.0))

        # v19 R3: waypoint telemetry. The route-checkpoint subsystem was "flying
        # blind" in v18 (no TB tags; it fails silently to safe defaults when the
        # tracker/planner build fails, so the per-leg PBRS just doesn't fire). The
        # trainer aggregates these into waypoint/active_frac, waypoint/max_index_frac,
        # waypoint/reach_bonus_mean per bucket.
        _wp_tracker = getattr(self, "_waypoint_tracker", None)
        _wp_count = len(_wp_tracker.waypoints) if _wp_tracker is not None else 0
        info["waypoint_active"] = bool(_wp_tracker is not None and _wp_count > 0)
        _wp_idx = getattr(self, "_waypoint_index", None)
        info["waypoint_index"] = int(_wp_idx) if _wp_idx is not None else -1
        info["waypoint_count"] = int(_wp_count)
        info["waypoint_reach_bonus"] = float(
            getattr(self.reward_calculator, "step_waypoint_reach_bonus", 0.0)
        )
        # v19 R3: spawn-finite tripwire. On ~5-10% of CHAMBER_MAZE levels the
        # physics-faithful jump field is INFINITE at spawn while the free-space
        # field is finite -> the PBRS potential clamps to -1 (zero gradient) and the
        # cost_to_go_map degrades on a level the free-space planner considers
        # routable. Surface it so the trainer can log pbrs/spawn_cost_finite_frac.
        info["pbrs_spawn_cost_finite"] = self._pbrs_spawn_cost_finite()

        # --- Locked-door progress ------------------------------------------
        # There was NO telemetry for locked doors at all, which is why two whole
        # runs could complete 0/338 door levels with the milestone reward off and
        # nothing in the logs said so. Named under ``door_*`` and surfaced under
        # ``door/`` in TB -- deliberately NOT ``phase/door_*``, which is the
        # EXIT-switch -> EXIT-door metric and has nothing to do with these.
        #
        # Emitted on every level, door or not: ``door_level`` is what lets the
        # consumer split the door rows out, and a rate averaged over the whole
        # corpus would be diluted by the 77% of levels with no door to open.
        _n_doors = int(getattr(self, "_locked_door_count", 0))
        info["door_level"] = bool(_n_doors > 0)
        info["door_count"] = _n_doors
        if _n_doors:
            _nec_total = int(getattr(self, "_necessary_door_total", 0))
            _nec_open = int(getattr(self, "_necessary_doors_open", 0))
            info["door_necessary_total"] = _nec_total
            info["door_necessary_open"] = _nec_open
            info["door_necessary_solvable"] = bool(
                getattr(self, "_necessary_door_solvable", True)
            )
            # Any locked door open at all -- including a non-necessary one. Kept
            # separate from the necessary count because "never finds a switch"
            # and "opens the wrong doors" are different failures.
            try:
                info["door_any_open"] = len(self._lattice_open_door_indices())
            except Exception:
                info["door_any_open"] = 0

        # Add level spawn position for corruption detection in Go-Explore
        # This helps detect checkpoints with spawn position but long action sequences
        if hasattr(self, "level_data") and self.level_data is not None:
            spawn_pos = self.level_data.start_position
            info["spawn_x"] = float(spawn_pos[0])
            info["spawn_y"] = float(spawn_pos[1])

        # CRITICAL: Use cached positions instead of direct simulator calls
        # Direct calls may return (0, 0) if entities not loaded during curriculum resets
        # The cached values are validated in base_environment._get_observation()
        switch_pos = self._cached_switch_pos
        if switch_pos is None:
            switch_pos = self.nplay_headless.exit_switch_position()
            # Log if we're falling back to (0, 0)
            if switch_pos == (0, 0):
                import logging

                _logger = logging.getLogger(__name__)
                _logger.warning(
                    f"_extend_info_hook: switch_pos is (0, 0)! "
                    f"_cached_switch_pos was None, entities may not be loaded. "
                    f"frame={self.nplay_headless.sim.frame}"
                )
        exit_pos = self._cached_exit_pos
        if exit_pos is None:
            exit_pos = self.nplay_headless.exit_door_position()
            if exit_pos == (0, 0):
                import logging

                _logger = logging.getLogger(__name__)
                _logger.warning(
                    "_extend_info_hook: exit_pos is (0, 0)! "
                    "_cached_exit_pos was None, entities may not be loaded."
                )

        info["switch_x"] = switch_pos[0]
        info["switch_y"] = switch_pos[1]
        info["switch_activated"] = self.nplay_headless.exit_switch_activated()
        info["exit_door_x"] = exit_pos[0]
        info["exit_door_y"] = exit_pos[1]
        info["player_dead"] = self.nplay_headless.ninja_has_died()
        info["player_won"] = self.nplay_headless.ninja_has_won()
        info["death_cause"] = self.nplay_headless.ninja_death_cause()

        # Additional fields for Go-Explore checkpoint system
        # These are needed for doomed checkpoint invalidation (falling into mines)
        ninja_vel = self.nplay_headless.ninja_velocity()
        info["player_xspeed"] = ninja_vel[0]
        info["player_yspeed"] = ninja_vel[1]
        # Post-step ninja position: consumed by FiGAR's hindsight landing-
        # prediction aux labels (Audit II P1-10) — contact offsets are computed
        # trainer-side from these per-frame positions.
        ninja_pos = self.nplay_headless.ninja_position()
        info["player_x"] = float(ninja_pos[0])
        info["player_y"] = float(ninja_pos[1])
        info["airborne"] = self.nplay_headless.sim.ninja.airborn
        info["ninja_state"] = self.nplay_headless.sim.ninja.state
        # True whenever the ninja is touching a wall (wall_normal != 0), in ANY
        # state — the real wall-jump precondition (a REGULAR wall jump fires while
        # walled even when not in the wall-slide state). FiGAR uses this to
        # interrupt macros on wall contact so wall-jumps stay frame-actionable.
        info["ninja_walled"] = self.nplay_headless.sim.ninja.walled
        # v13 doomed-arc flag (same-frame with the obs bundle: _get_observation
        # runs before this hook). FiGAR's arc-degradation interrupt uses the
        # False->True edge to force a decision point the frame the flight
        # becomes unrecoverable-by-constant-input (a well-timed steer/hold
        # change on THAT frame is often still recoverable — the arcs are
        # constant-input projections, not a full reachability proof).
        info["arc_doomed"] = bool(getattr(self, "_arc_doomed", False))

        # Mine approach distance for FiGAR macro-action safety interrupts
        info["nearest_approaching_deadly_mine_distance"] = (
            self._compute_nearest_approaching_deadly_mine_distance()
        )
        # Ungated (velocity-independent) nearest deadly-mine distance in px. The
        # approaching-only distance above misses a ballistic arc that will LAND on
        # a mine while the current velocity vector is not yet pointing at it; the
        # airborne mine interrupt (FiGAR) uses this to force a decision point
        # mid-arc so the policy can react before landing on a mine. inf = none.
        info["nearest_deadly_mine_distance"] = (
            self._compute_nearest_deadly_mine_distance_px()
        )
        # Frames until the look-ahead landing-surface contact at the current
        # closing speed (inf = far / not approaching). FiGAR's approaching-
        # contact interrupt (Audit II P1-8) zeroes an airborne macro's committed
        # duration when this drops below its threshold, so landing re-decision
        # happens BEFORE contact (one frame after is too late for impact/
        # overshoot control). Stashed by the obs pass this same step.
        info["wall_contact_tau_frames"] = float(
            getattr(self, "_last_wall_contact_tau_frames", float("inf"))
        )
        info["time_to_contact_frames"] = float(
            getattr(self, "_last_time_to_contact_frames", float("inf"))
        )
        # v17 A: frames-to-contact ONLY when the current airborne velocity would
        # make the imminent landing fatal-if-uncorrected (gated by the exact sim
        # death condition, closing >= MAX_SURVIVABLE_IMPACT - 4/3*|ny|); inf
        # otherwise. FiGAR's lethal-landing interrupt edge-triggers a decision
        # point when this drops below its tau threshold, giving the policy a
        # frame to brake/wall-grab before a fatal vertical impact (the maze
        # impact-death failure mode: ~15% of CHAMBER_MAZE deaths in a
        # hazard-free bucket). Stashed by the obs pass this same step.
        info["lethal_landing_tau"] = float(
            getattr(self, "_lethal_landing_tau", float("inf"))
        )

        # v26 W6c: the point-of-no-return bit (every applicable arc from this
        # state is fatal). Emitted regardless of whether the obs block is on, so
        # a probe run can measure how often the agent is already doomed WITHOUT
        # paying the breaking obs change. Always False when the block is off.
        info["commitment_no_escape"] = bool(
            getattr(self, "_commitment_no_escape", False)
        )

        # Graph size info for level-size-aware optimizations
        # Used by Go-Explore to scale checkpoint thresholds for larger levels
        if hasattr(self, "current_graph") and self.current_graph is not None:
            info["graph_num_nodes"] = self.current_graph.num_nodes
        else:
            info["graph_num_nodes"] = 0

        # PRECISION mine-death trajectory capture (no-op unless armed this reset).
        self._precision_capture_step(info)

    def _init_precision_capture(self):
        """Arm (or disarm) the PRECISION mine-death trajectory buffer for the
        episode that is about to start. Enabled only when the dump dir is
        configured (``NCLONE_PRECISION_DUMP_DIR`` / graph-cache fallback) AND
        the current bucket is PRECISION, so normal training pays nothing."""
        from collections import deque
        try:
            from .precision_failure_capture import is_enabled as _pf_enabled
            on = bool(_pf_enabled()) and (self._current_bucket == "PRECISION")
        except Exception:
            on = False
        self._precision_capture_on = on
        if on:
            try:
                n = int(os.environ.get("NCLONE_PRECISION_DUMP_FRAMES", "45"))
            except ValueError:
                n = 45
            self._precision_traj = deque(maxlen=max(4, n))
        else:
            self._precision_traj = None

    def _precision_capture_step(self, info):
        """Append this step's ninja telemetry to the PRECISION buffer and, on a
        mine death, flush a dump. One dump per episode (disarms after flush)."""
        if not getattr(self, "_precision_capture_on", False):
            return
        buf = getattr(self, "_precision_traj", None)
        if buf is None:
            return
        ninja = getattr(getattr(self.nplay_headless, "sim", None), "ninja", None)
        ji = float(getattr(ninja, "jump_input", 0) or 0) if ninja is not None else 0.0
        jd = float(getattr(ninja, "jump_duration", 0) or 0) if ninja is not None else 0.0
        buf.append((
            float(info.get("player_x", 0.0)),
            float(info.get("player_y", 0.0)),
            float(info.get("player_xspeed", 0.0)),
            float(info.get("player_yspeed", 0.0)),
            1.0 if info.get("airborne") else 0.0,
            float(info.get("ninja_state", 0) or 0),
            ji,
            jd,
        ))
        if info.get("player_dead") and info.get("death_cause") == "mine":
            self._flush_precision_failure(info)
            self._precision_capture_on = False  # one dump per episode

    def _flush_precision_failure(self, info):
        """Write the buffered trajectory + map_data for this PRECISION mine death."""
        try:
            from .precision_failure_capture import capture_precision_failure
            md = getattr(self.nplay_headless, "current_map_data", None)
            if md is None:
                md = getattr(getattr(self, "map_loader", None), "map_data", None)
            ninja = getattr(getattr(self.nplay_headless, "sim", None), "ninja", None)
            context = {
                "bucket": self._current_bucket,
                "level_id": getattr(
                    getattr(self, "map_loader", None), "current_map_name", None
                ),
                "step": int(getattr(self, "_episode_step", 0) or 0),
                "death_cause": info.get("death_cause"),
                "x": float(getattr(ninja, "death_xpos", info.get("player_x", 0.0))),
                "y": float(getattr(ninja, "death_ypos", info.get("player_y", 0.0))),
                "vx": float(getattr(ninja, "death_xspeed", info.get("player_xspeed", 0.0))),
                "vy": float(getattr(ninja, "death_yspeed", info.get("player_yspeed", 0.0))),
                "switch_x": info.get("switch_x"),
                "switch_y": info.get("switch_y"),
                "switch_activated": info.get("switch_activated"),
                "exit_door_x": info.get("exit_door_x"),
                "exit_door_y": info.get("exit_door_y"),
                "curriculum_seeded_start": info.get("curriculum_seeded_start"),
            }
            capture_precision_failure(md, context, list(self._precision_traj))
        except Exception:
            pass  # diagnostics must never break training

    def _build_door_feature_cache(self):
        """
        Build precomputed door feature cache.

        Precomputes path distances from all reachable grid cells to all doors/switches.
        This is a one-time cost at level load that eliminates expensive runtime path
        distance calculations.

        PERFORMANCE: Build time ~50-200ms, saves ~25s per 858 steps (26x speedup)
        """
        # CRITICAL: Invalidate cached observation to ensure fresh positions
        # This prevents using stale positions from previous level
        if hasattr(self, "_cached_observation"):
            self._cached_observation = None

        # Get locked doors from current observation
        obs = super()._get_observation()
        locked_doors = obs.get("locked_doors", [])

        if not locked_doors:
            # No doors to cache
            return

        # Ensure graph is built
        if self.current_graph_data is None:
            logger.warning(
                "Graph data not available for door feature cache. "
                "Cache will not be built."
            )
            return

        adjacency = self.current_graph_data.get("adjacency")
        if not adjacency:
            logger.warning("No adjacency graph. Door feature cache will not be built.")
            return

        # Get area scale for normalization
        try:
            area_scale = self._get_reachable_area_scale()
        except RuntimeError as e:
            logger.warning(f"Could not compute area scale: {e}. Using default.")
            area_scale = 1.0

        # Build the cache
        self.door_feature_cache.build_cache(
            locked_doors=locked_doors,
            adjacency=adjacency,
            path_calculator=self._path_calculator,
            level_data=self.level_data,
            graph_data=self.current_graph_data,
            area_scale=area_scale,
            verbose=False,
        )

    def _initialize_locked_door_caches(self):
        """
        Initialize locked door entity and switch state caches.

        Caches static locked door entities and initializes switch collection states.
        Called once per level after map load. Door positions don't change, and
        switches can only be activated (not deactivated) by the player.

        PERFORMANCE: Eliminates per-step loops and observation calls for door features.
        """
        # CRITICAL: Invalidate cached observation to ensure fresh positions
        # This prevents using stale positions from previous level
        if hasattr(self, "_cached_observation"):
            self._cached_observation = None

        # Get locked doors from observation
        obs = super()._get_observation()
        locked_doors = obs.get("locked_doors", [])

        # Cache level door state
        self._has_locked_doors = len(locked_doors) > 0
        self._cached_locked_doors = locked_doors[:MAX_LOCKED_DOORS_ATTENTION]

        if self._has_locked_doors:
            # Initialize all switches as uncollected (0.0)
            self._cached_switch_states = np.zeros(
                len(self._cached_locked_doors), dtype=np.float32
            )
        else:
            self._cached_switch_states = None

        # Mark switch states as fresh (no changes yet)
        self._switch_states_changed = False

    def _compute_level_surface_area(self) -> Optional[float]:
        """Compute reachable surface area (node count) for the current level.

        Used to set a complexity-scaled truncation limit at level load.
        Returns ``None`` if graph data is not yet available.
        """
        graph_data = getattr(self, "current_graph_data", None)
        if not graph_data:
            return None
        adjacency = graph_data.get("adjacency")
        level_data = self.level_data
        if not adjacency or level_data is None:
            return None
        start_pos = getattr(level_data, "start_position", None)
        if start_pos is None:
            return None
        try:
            from ..graph.reachability.surface_area import get_cached_surface_area

            cache_key = level_data.get_cache_key_for_reachability(
                include_switch_states=True
            )
            return get_cached_surface_area(
                cache_key, start_pos, adjacency, graph_data
            )
        except Exception as e:
            logger.debug(f"Failed to compute level surface area: {e}")
            return None

    def _spawn_cost_to_completion(self) -> Optional[float]:
        """Best-effort free-space cost-to-completion from spawn (~pixels).

        Used to give long, winding levels a path-length-aware truncation limit.
        Builds the (disk-cached, ~ms) free-space source for the current map and
        queries the spawn cost. Returns ``None`` on any failure or non-finite cost
        so the caller falls back to the surface-area formula.
        """
        try:
            md = getattr(self.nplay_headless, "current_map_data", None)
            pos = self.nplay_headless.ninja_position()
            if not md or pos is None:
                return None
            from ..value_function.free_space_distance import FreeSpaceDistanceSource

            fs = FreeSpaceDistanceSource(md)
            if not getattr(fs, "available", False):
                return None
            c = fs.cost_to_completion(float(pos[0]), float(pos[1]), False)
            return float(c) if c is not None and c < float("inf") else None
        except Exception as e:  # noqa: BLE001
            logger.debug(f"spawn cost-to-completion unavailable: {e}")
            return None

    def reset(self, seed=None, options=None):
        """Reset the environment with planning components and visualization."""

        # v22 §3-B: clear the cost-to-go trend so blocked-toward-goal never diffs
        # across an episode boundary (first step of a new episode is un-stalled).
        self._prev_cost_to_go = None

        # Start profiling total reset time
        # reset_start_time = self._profile_start("reset_total")

        # Handle reinitialization after unpickling
        if hasattr(self, "_needs_reinit") and self._needs_reinit:
            self.observation_processor.reset()
            self.reward_calculator.reset()
            self._needs_reinit = False

        # === FAST RESET PATH DETECTION ===
        # Detect if this is a same-level reset (enables optimized reset path)
        same_level_reset = self._is_same_level_reset(options)

        # Check if map loading should be skipped and if this is a new level
        skip_map_load = False
        new_level = False  # True if a different level was loaded externally
        map_name = None
        option_map_data = None
        if options is not None and isinstance(options, dict):
            skip_map_load = options.get("skip_map_load", False)
            # When skip_map_load=True, assume new level unless explicitly told otherwise
            new_level = options.get("new_level", skip_map_load)
            map_name = options.get("map_name", None)
            option_map_data = options.get("map_data", None)
            if option_map_data is not None:
                # Explicit payload-driven load: force the standard (non-fast)
                # reset path so caches get cleared and the new map sticks.
                new_level = True

        # Track the current level's bucket (from reset options) so the
        # goal-distance reverse curriculum can be scoped to specific buckets
        # (e.g. PLATFORM_STAIRS). Re-resolve the effective fraction now that the
        # bucket for this reset is known. Both fast and slow reset paths run this.
        if options is not None and isinstance(options, dict):
            self._current_bucket = options.get("bucket")
        else:
            self._current_bucket = None
        self._apply_goal_distance_scope()
        # PRECISION mine-death diagnostic: arm a fresh rolling telemetry buffer
        # for this episode iff the dump dir is configured AND this is a PRECISION
        # level. Zero overhead otherwise (flag stays False, no buffer appends).
        self._init_precision_capture()
        # v17: zero the per-episode visitation grid on EVERY reset (both fast and
        # slow paths run this line before the branch), keeping the signal episodic.
        self._reset_visit_counts()

        # === FAST RESET PATH (2-3x faster for single-level training) ===
        # Use fast reset when:
        # 1. Same level detected (no map change)
        # 2. Not a new level load
        # 3. Entities already exist (not first reset)
        use_fast_reset = (
            same_level_reset
            and not new_level
            and hasattr(self.nplay_headless, "sim")
            and self.nplay_headless.sim.ninja is not None
        )

        if use_fast_reset:
            if self.enable_logging:
                logger.info(
                    "[FAST_RESET] Using optimized reset path for same-level training"
                )

            # Fast reset: entities in-place, no recreation
            # fast_sim_start = self._profile_start("reset_fast_sim")
            self.nplay_headless.fast_reset()
            # self._profile_end("reset_fast_sim", fast_sim_start)

            # Clear only episode-specific caches (not level-persistent ones)
            from nclone.cache_management import clear_episode_caches_only

            clear_episode_caches_only(self)

            # Reset observation processor (buffer clearing, no reallocation)
            self.observation_processor.reset()

            # ============================================================
            # Reset tracking variables from BaseNppEnvironment
            # ============================================================
            # Mirror the base reset path for per-episode state so fast resets
            # don't retain step-by-step history across the whole training run.
            self.last_action = None
            self._action_sequence = []
            self._checkpoint_replay_in_progress = False
            self._checkpoint_route = []
            self._checkpoint_source = None
            self._from_checkpoint = False
            self._checkpoint_base_reward = 0.0
            self._checkpoint_source_frame_skip = self.frame_skip
            self._checkpoint_replay_frame_count = 0
            self._switch_activation_frame = None
            self._switch_activation_step = None
            self.current_route = []
            self._warned_about_position = False
            # Force a hazard-map recompute on the first step of the new episode
            # (the level/mine field may have changed) — content-based after that.
            self._hazard_sig = None
            self._deterministic_mask_applied_count = 0
            self._total_steps_count = 0

            self._reset_reward_calculator()

            self.truncation_checker.reset()

            self.current_ep_reward = 0

            # Clear per-step path caches (ninja position resets to spawn).
            # Level-scoped caches remain valid because entity positions are unchanged.
            if hasattr(self, "reward_calculator") and hasattr(
                self.reward_calculator, "path_calculator"
            ):
                self.reward_calculator.path_calculator.clear_step_cache()

            # Reset graph state (fast - graph structure unchanged)
            # graph_start = self._profile_start("reset_fast_graph")
            self._reset_graph_state()
            # self._reset_reachability_state()

            # Update graph from current env state (fast - uses cached graph structure)
            self._update_graph_from_env_state()
            # self._profile_end("reset_fast_graph", graph_start)

            # Optional start-state curriculum: restore a mid-route ninja state.
            self._maybe_apply_start_state(options)

            # Get initial observation and return
            # obs_start = self._profile_start("reset_fast_observation")
            initial_obs = self._get_observation()
            processed_obs = self._process_observation(initial_obs)
            # self._profile_end("reset_fast_observation", obs_start)

            # Track current map name for fast reset detection
            self._last_reset_map_name = self.map_loader.current_map_name

            # End profiling and return
            # self._profile_end("reset_total", reset_start_time)

            return (processed_obs, {})

        # === STANDARD RESET PATH (full reset) ===
        # Only reached if NOT using fast reset
        else:
            # Reset observation processor
            self.observation_processor.reset()

            # ============================================================
            # CRITICAL: Reset tracking variables from BaseNppEnvironment
            # NppEnvironment doesn't call super().reset(), so we must do it here
            # ============================================================

            # Reset switch activation tracking (BaseNppEnvironment.reset line 1519)
            self._switch_activation_frame = None
            self._switch_activation_step = None

            # Reset position tracking for route visualization (BaseNppEnvironment.reset line 1515-1516)
            self.current_route = []
            self._warned_about_position = False
            # Force a hazard-map recompute on the first step of the new episode
            # (the level/mine field may have changed) — content-based after that.
            self._hazard_sig = None

            self._reset_reward_calculator()

            self.truncation_checker.reset()

            self.current_ep_reward = 0

            # Clear caches appropriately based on whether this is a new level
            from nclone.cache_management import (
                clear_all_caches_for_reset,
                clear_all_caches_for_new_level,
            )

            if new_level:
                # New level loaded externally - clear ALL caches including level-persistent ones
                clear_all_caches_for_new_level(self)
            else:
                # Same level being reset - clear only episode-specific caches
                clear_all_caches_for_reset(self)

        if option_map_data is not None:
            # Benchmark-driven training / eval: the level payload is supplied
            # directly via reset options. Bypass every dataset map-selection
            # branch and load the payload.
            effective_name = map_name if map_name is not None else "benchmark_level"
            self.map_loader.load_map_data(option_map_data, str(effective_name))
        elif not skip_map_load:
            self.map_loader.load_map()
        else:
            # Map loading skipped
            # Update map_loader.current_map_name if provided
            # This is critical for cache keying and momentum waypoint loading
            if map_name is not None:
                self.map_loader.current_map_name = map_name
                if self.enable_logging:
                    logger.debug(f"Updated map_loader.current_map_name to: {map_name}")

            # Only reset the sim if a map wasn't just loaded externally
            # (sim.load() already calls sim.reset(), so we shouldn't reset again)
            map_just_loaded = getattr(self.nplay_headless, "_map_just_loaded", False)
            if not map_just_loaded:
                # If map loading is skipped but no new map loaded, reset to spawn
                # This handles checkpoint replay and same-level resets
                self.nplay_headless.reset()
            else:
                # Clear the flag after checking it
                self.nplay_headless._map_just_loaded = False
                if self.enable_logging:
                    logger.debug(
                        "Skipped nplay_headless.reset() - map was just loaded externally"
                    )

        # CRITICAL: Reset all entity activation states after map/sim operations
        # Checkpoint replay and previous episodes can leave switches activated
        # New episodes must always start with switches deactivated
        try:
            sim = self.nplay_headless.sim

            # Reset exit DOOR (type 3) - switch_hit=False and remove from grid_entity
            exit_doors = sim.entity_dic.get(3, [])
            for door in exit_doors:
                if type(door).__name__ == "EntityExit":
                    # Reset switch_hit flag
                    if hasattr(door, "switch_hit"):
                        door.switch_hit = False
                    # CRITICAL: Remove exit door from grid_entity if present
                    # The door should only be in grid after switch collection
                    if (
                        door.cell in sim.grid_entity
                        and door in sim.grid_entity[door.cell]
                    ):
                        sim.grid_entity[door.cell].remove(door)
                        logger.debug(f"Removed exit door from grid_entity[{door.cell}]")

            # Reset exit SWITCH (type 4) - active=True (not collected)
            exit_switches = sim.entity_dic.get(4, [])
            for switch in exit_switches:
                if type(switch).__name__ == "EntityExitSwitch":
                    if hasattr(switch, "active"):
                        switch.active = True  # True = not collected
                    logger.debug("Reset exit switch: active=True")

            # Reset locked door switches to deactivated
            locked_door_entities = sim.entity_dic.get(6, [])
            for door in locked_door_entities:
                if hasattr(door, "active"):
                    door.active = True  # True = not collected
                if hasattr(door, "open"):
                    door.open = False  # False = door closed

            if self.enable_logging:
                logger.debug("Reset all entity activation states for new episode")
        except Exception as e:
            logger.warning(f"Failed to reset entity states: {e}")

        # CRITICAL: Reset and rebuild graph BEFORE getting observation
        self._reset_graph_state()
        self._reset_reachability_state()

        # Build graph from the newly loaded map (with curriculum entity positions)
        self._update_graph_from_env_state()

        # Trigger feature extractor cache precomputation (for hybrid_cached_graph)
        # This precomputes static graph embeddings once per level for O(1) runtime extraction
        if hasattr(self, "_feature_extractor_cache_callback"):
            level_id = self.map_loader.current_map_name
            self._feature_extractor_cache_callback(level_id, self.current_graph_data)

        # Build precomputed door feature cache after graph is ready
        # PERFORMANCE: Precomputes ALL path distances for O(1) runtime lookup
        self._build_door_feature_cache()

        # Initialize locked door caches for this level
        # PERFORMANCE: Cache static door entities and initialize switch states
        self._initialize_locked_door_caches()

        # Set dynamic truncation limit based on level complexity (reachable surface area).
        # Computed once per level via the shared module-level surface-area cache.
        surface_area = self._compute_level_surface_area()
        if surface_area is not None:
            # The 0 is ``reachable_mine_count``, and it is deliberate rather than a
            # missing plumb: ``truncation_calculator.TIME_PER_MINE`` is 0.0 and the
            # mine term is retired. Reviving it would add ~39,750 frames to a
            # PLATFORMING episode, the opposite of the measured need. See that
            # module's docstring before "fixing" this.
            truncation_limit = self.truncation_checker.set_level_truncation_limit(
                surface_area, 0, cost_to_completion=self._spawn_cost_to_completion()
            )
            if self.enable_logging:
                logger.info(
                    f"Level truncation limit: {truncation_limit} frames "
                    f"(surface_area={surface_area:.0f})"
                )

        # Optional start-state curriculum: restore a mid-route ninja state.
        self._maybe_apply_start_state(options)

        # NOW get initial observation (with valid graph data)
        initial_obs = self._get_observation()
        processed_obs = self._process_observation(initial_obs)

        # Track current map name for fast reset detection
        self._last_reset_map_name = self.map_loader.current_map_name

        # End profiling (standard reset path)
        # self._profile_end("reset_total", reset_start_time)

        return (processed_obs, {})

    # Frame-input table for start-action prefixes (matches the planner ACT map).
    _START_ACT = {0: (0, 0), 1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (1, 1), 5: (-1, 1)}

    def _maybe_apply_start_state(self, options):
        """Optional start-state curriculum hook (general, sim-agnostic).

        ``options['start_state']`` is a curriculum start state — a dict with
        ``start_actions`` (a frame-input prefix, list of action ids 0-5) — or
        ``options['start_state_curriculum']`` is an object with ``.sample(difficulty)``
        returning one. After the normal reset we **replay that prefix on the env's
        own sim**, so the episode begins mid-route and the state is faithful to the
        env's physics by construction (works whether the env is Python- or
        native-backed; no save/restore or cross-engine parity concern). If the
        prefix dies partway (e.g. a native-found prefix drifting on the Python sim),
        we stop at the last surviving frame — always a valid start state.

        This is the basis for a reverse curriculum (start near the goal, schedule
        the start back toward spawn) for fine-tuning hard levels, and for
        restart-state diversity in generalized training."""
        # Curriculum-seeded-start telemetry: reset per episode (both reset paths
        # call this method) and emitted into every step info dict — see
        # _extend_info_hook. npp-rl uses it to exclude seeded episodes from the
        # curriculum phase gate. Difficulty is 1.0 when not seeded.
        self._curriculum_seeded_start = False
        self._episode_start_difficulty = 1.0
        # Which reverse-curriculum leg seeded this episode ("pre"=approach,
        # "post"=return, None=natural spawn). Emitted in info so the callback can
        # anneal the two legs on independent success deques.
        self._episode_seeded_leg = None
        ss = None
        difficulty = 1.0
        if options and isinstance(options, dict):
            difficulty = float(options.get("start_state_difficulty", 1.0))
            ss = options.get("start_state")
            if ss is None:
                cur = options.get("start_state_curriculum")
                if cur is not None and hasattr(cur, "sample"):
                    ss = cur.sample(difficulty)
        # Lazy PRE-SWITCH reverse curriculum (v16; PRECISION execution wall): on
        # the first reset of a gated-bucket level, build (disk-cached) a per-level
        # pool of mid-route (toward-the-switch) start states and seed from it, so
        # the agent practises the hard final approach directly instead of dying
        # long before it.
        # v21d/L5: a level gated by BOTH curricula (LOOP_CORRIDOR round-trips are
        # in start_state_curriculum_buckets AND post_switch_curriculum_buckets)
        # must practise BOTH legs. The default pre-switch-first ordering shadows
        # post-switch entirely (pre-switch always returns a seed for a gated
        # level), so the return leg (switch->exit) is never practised and true
        # completion stays 0 even when switch_reached is high. Flip a coin per
        # reset so ~half of such episodes seed from the post-switch (return) pool
        # instead. For a level in only one set the other sampler returns None, so
        # the ordering is a no-op there (FAR_OBJECTIVE behaviour is unchanged).
        _pre_on = getattr(self.config, "start_state_curriculum", False)
        _post_on = getattr(self.config, "post_switch_curriculum", False)
        # NATURAL-SPAWN RESERVE: skip seeding on a fraction of gated resets so the
        # natural-spawn pool accumulates (the phase gate needs a per-bucket episode
        # floor) and full-task progress is measured continuously. Leaves ss=None ->
        # natural spawn, seeded_leg stays None (ignored by the anneal).
        _nat_frac = float(getattr(self.config, "start_state_natural_fraction", 0.0))
        if (_pre_on or _post_on) and _nat_frac > 0.0 and self.rng.random() < _nat_frac:
            _pre_on = False
            _post_on = False
        _post_first = _pre_on and _post_on and self.rng.random() < 0.5
        if ss is None and _post_first:
            ss = self._maybe_sample_post_switch(options)
            if ss:
                difficulty = float(self._post_switch_difficulty())
                self._episode_seeded_leg = "post"
        if ss is None and _pre_on:
            ss = self._maybe_sample_pre_switch(options)
            if ss:
                difficulty = float(self._pre_switch_difficulty())
                self._episode_seeded_leg = "pre"
        # Lazy POST-SWITCH reverse curriculum (FAR_OBJECTIVE / LOOP_CORRIDOR return
        # leg): on the first reset of a gated-bucket level, build (disk-cached) a
        # per-level pool of switch-already-activated start states and seed from it.
        # Fixes the return-to-exit leg never being practised (exit_given_switch==0).
        if ss is None and _post_on:
            ss = self._maybe_sample_post_switch(options)
            if ss:
                difficulty = float(self._post_switch_difficulty())
                self._episode_seeded_leg = "post"
        # Fall back to a STORED curriculum (set via set_start_state_curriculum) so
        # the reverse curriculum also fires on VecEnv AUTO-resets, where SB3 calls
        # reset() with no options -- this is what drives it during RL rollouts.
        if ss is None:
            cur = getattr(self, "_ssc", None)
            # Which store the curriculum came from decides which difficulty knob
            # anneals it. Tracked explicitly rather than inferred from whether a
            # demo map exists at all — both stores can be populated at once.
            from_demo_map = False
            if cur is None:
                ssc_map = getattr(self, "_ssc_map", None)
                if ssc_map:
                    from_demo_map = True
                    lvl = getattr(
                        getattr(self, "map_loader", None), "current_map_name", None
                    )
                    cur = ssc_map.get(lvl)
                    # v24 lever-2: fall back to a map_data CONTENT-HASH key. The
                    # packed-pool training path keys current_map_name by the integer
                    # pool level_id (str), NOT the map name the demo curricula were
                    # built under, so a name lookup misses. A content hash matches
                    # the level regardless of how the loader named it. Keys are
                    # "md:<md5(map_data bytes)>" (see build_demo_curricula._index +
                    # the trainer's curricula loader).
                    if cur is None:
                        md = getattr(
                            getattr(self, "nplay_headless", None),
                            "current_map_data", None,
                        )
                        if md is not None:
                            cur = ssc_map.get(_map_data_curriculum_key(md))
            # v25: a demo-curriculum entry may carry BOTH legs
            # ({"switch": cur, "exit": cur}); alternate per reset so the
            # switch->exit RETURN leg is practised too. The builder always wrote
            # both legs but only the approach leg was ever loaded, so the return
            # leg (exit_given_switch = 0.65) was never seeded.
            demo_leg = None
            if isinstance(cur, dict):
                legs = [k for k in ("switch", "exit") if cur.get(k) is not None]
                if legs:
                    demo_leg = legs[self.rng.randrange(len(legs))]
                    cur = cur[demo_leg]
                else:
                    cur = None
            if cur is None:
                from_demo_map = False
            if cur is not None and hasattr(cur, "sample"):
                if from_demo_map:
                    # Demo curricula (verified official routes) anneal on their
                    # OWN difficulty knob. Sharing the pre-switch knob let the
                    # easy procgen buckets ratchet it to 1.0 and switch the
                    # official curriculum off — see
                    # set_start_state_difficulty_demo.
                    difficulty = float(self._demo_curriculum_difficulty())
                    self._episode_seeded_leg = (
                        "demo_post" if demo_leg == "exit" else "demo_pre"
                    )
                else:
                    # Stored single-level curricula (PLATFORM_STAIRS etc.) are
                    # toward-goal/approach-style -> treat as the pre-switch leg.
                    # For single-leg levels the two knobs move together
                    # (set_start_state_difficulty sets both), so this is
                    # unchanged for them.
                    difficulty = float(self._pre_switch_difficulty())
                    self._episode_seeded_leg = "pre"
                ss = cur.sample(difficulty)
        if not ss:
            return
        actions = ss.get("start_actions") if isinstance(ss, dict) else ss
        if not actions:
            return
        nh = self.nplay_headless
        applied = True
        try:
            for a in actions:
                h, j = self._START_ACT.get(int(a), (0, 0))
                nh.tick(h, j)
                if nh.ninja_has_died() or nh.ninja_has_won():
                    break
        except Exception:
            return
        # TERMINAL-PREFIX GUARD: if the replay itself killed the ninja or won the
        # level (prefix drift onto a mine / into the exit), the seeded start is
        # unusable — the episode would begin already-terminal. Revert to the
        # natural spawn, exactly like the reachability guard below.
        try:
            if nh.ninja_has_died() or nh.ninja_has_won():
                logger.debug(
                    "Start-state prefix replay terminated (died=%s won=%s); "
                    "reverting to spawn.",
                    nh.ninja_has_died(), nh.ninja_has_won(),
                )
                nh.fast_reset()
                applied = False
        except Exception:
            pass
        # RFCL REACHABILITY GUARD (belt-and-suspenders for replay drift): the
        # curriculum only stores goal-reachable states, but replaying an RRT prefix
        # on the env's full physics can drift (the switch-goal RRT explores over
        # mines-only physics). If the replayed start lands OUTSIDE the objective's
        # reachable region (infinite free-space cost-to-go), revert to the spawn
        # rather than begin an unsolvable, inf-PBRS episode that floods logs with
        # "position far outside graph bounds". Never raises; on any error the
        # already-applied start state stands.
        if applied:
            try:
                src = self._get_free_space_nav_source()
                if src is not None and getattr(src, "available", False):
                    nx, ny = nh.ninja_position()
                    sw = bool(nh.exit_switch_activated())
                    if not math.isfinite(src.cost_to_go(float(nx), float(ny), sw)):
                        logger.debug(
                            "RFCL start state landed goal-unreachable "
                            "(pos=%.0f,%.0f sw=%s); reverting to spawn.", nx, ny, sw
                        )
                        nh.fast_reset()
                        applied = False
            except Exception:
                pass
        if applied:
            self._curriculum_seeded_start = True
            self._episode_start_difficulty = difficulty
        else:
            # A guard reverted the seed to the natural spawn: this is no longer a
            # leg-seeded episode, so it must not feed either leg's anneal deque.
            self._episode_seeded_leg = None
        # Re-decide per-episode potential gates at the new start position.
        for attr in ("_lattice_episode_decided", "_freespace_episode_decided"):
            if hasattr(self, attr):
                setattr(self, attr, False)

    def _maybe_sample_pre_switch(self, options):
        """Lazily build+sample a per-level PRE-SWITCH start state (v16).

        The pre-switch sibling of :meth:`_maybe_sample_post_switch`. Gated to the
        buckets in ``config.start_state_curriculum_buckets`` (PRECISION by
        default). The per-level pool of mid-route (toward-the-switch) states is
        generated on first encounter and disk-cached (keyed by map_data hash), so
        multi-level training pays the RRT-build cost once per level. Returns a
        start-state dict (``{'start_actions': [...], ...}``) or ``None`` when the
        level isn't gated, has no map_data this reset, or yields no prefix."""
        if not options or not isinstance(options, dict):
            return None
        bucket = options.get("bucket")
        gated = getattr(self.config, "start_state_curriculum_buckets", ())
        if bucket is None or str(bucket) not in {str(b) for b in gated}:
            return None
        map_data = options.get("map_data")
        if not map_data:
            return None
        try:
            from .pre_switch_curriculum_cache import get_or_build

            cur = get_or_build(
                list(map_data),
                n_states=int(
                    getattr(self.config, "start_state_curriculum_n_states", 400)
                ),
                max_iter=int(
                    getattr(self.config, "start_state_curriculum_max_iter", 200000)
                ),
            )
        except Exception:
            return None
        if not cur or len(cur) == 0:
            return None
        # PRE-switch (approach) leg has its OWN difficulty knob, decoupled from the
        # post-switch (return) leg (set_start_state_difficulty_pre; 1.0 == full route
        # from spawn -> 0.0 anneals the start toward the switch/goal). The legs must
        # anneal independently: a pre-switch WIN also needs the full return leg, so
        # with a SHARED knob the pre-leg can't win until the return is mastered,
        # which deadlocks the pooled anneal (v21d: difficulty frozen at 0.1).
        return cur.sample(float(self._pre_switch_difficulty()))

    def _maybe_sample_post_switch(self, options):
        """Lazily build+sample a per-level POST-SWITCH start state.

        Gated to the buckets in ``config.post_switch_curriculum_buckets`` (the
        FAR_OBJECTIVE loop levels). The per-level pool is generated on first
        encounter and disk-cached (keyed by map_data hash), so multi-level
        training pays the RRT-build cost once per level, shared across workers.
        Returns a start-state dict (``{'start_actions': [...], ...}``) or ``None``
        when the level isn't gated, has no map_data this reset, or yields no
        switch-consuming prefix."""
        if not options or not isinstance(options, dict):
            return None
        bucket = options.get("bucket")
        gated = getattr(self.config, "post_switch_curriculum_buckets", ())
        if bucket is None or str(bucket) not in {str(b) for b in gated}:
            return None
        map_data = options.get("map_data")
        if not map_data:
            return None
        try:
            from .post_switch_curriculum_cache import get_or_build

            cur = get_or_build(
                list(map_data),
                n_states=int(getattr(self.config, "post_switch_curriculum_n_states", 300)),
                max_iter=int(getattr(self.config, "post_switch_curriculum_max_iter", 120000)),
            )
        except Exception:
            return None
        if not cur or len(cur) == 0:
            return None
        # POST-switch (return) leg has its OWN difficulty knob, annealed by the
        # trainer callback via set_start_state_difficulty_post (default 1.0 == full
        # return leg). Independent of the pre-switch (approach) knob so the return
        # leg can be mastered first (it wins easily) before the approach extends.
        return cur.sample(float(self._post_switch_difficulty()))

    def set_start_state_curriculum(self, cur) -> None:
        """Store a single StartStateCurriculum (single-level reverse curriculum).
        Used on VecEnv auto-resets via the stored-curriculum fallback above."""
        self._ssc = cur

    def set_start_state_curricula(self, mapping: dict) -> None:
        """Store the multi-level/curriculum reverse curriculum.

        Accepts either ``{level_id: StartStateCurriculum}`` (single leg, legacy)
        or ``{level_id: {"switch": cur, "exit": cur}}`` (v25, per-leg). The env
        applies the curriculum matching the level it just loaded, so only the
        hard levels we precomputed routes for get it. With both legs present the
        env alternates per reset, so the switch->exit RETURN leg is practised too
        — the demo builder always wrote both, but only the approach leg was ever
        loaded, leaving exit_given_switch (0.65) unaddressed."""
        self._ssc_map = dict(mapping or {})

    def set_start_state_difficulty(self, d: float) -> None:
        """Reverse-curriculum difficulty in [0,1] (0=near goal, 1=spawn). The
        callback anneals this UP as the agent's success on the hard bucket rises.
        Sets BOTH legs (backward compat for single-leg curricula — PRECISION,
        PLATFORM_STAIRS, FAR_OBJECTIVE — and the stored single-level fallback)."""
        d = float(d)
        self._ssc_difficulty = d
        self._ssc_pre_difficulty = d
        self._ssc_post_difficulty = d

    def set_start_state_difficulty_pre(self, d: float) -> None:
        """PRE-switch (approach) leg difficulty only. Used by the reverse-curriculum
        callback to anneal the approach leg independently of the return leg — a
        pre-switch win requires the return leg to be mastered, so the legs must not
        share one knob (else the pooled rate deadlocks the anneal)."""
        self._ssc_pre_difficulty = float(d)

    def set_start_state_difficulty_post(self, d: float) -> None:
        """POST-switch (return) leg difficulty only. Anneals to 1.0 first (the
        return is independently learnable) so pre-switch episodes can then win."""
        self._ssc_post_difficulty = float(d)

    def _pre_switch_difficulty(self) -> float:
        return float(getattr(self, "_ssc_pre_difficulty",
                             getattr(self, "_ssc_difficulty", 1.0)))

    def _post_switch_difficulty(self) -> float:
        return float(getattr(self, "_ssc_post_difficulty",
                             getattr(self, "_ssc_difficulty", 1.0)))

    def set_start_state_difficulty_demo(self, d: float) -> None:
        """DEMO-curriculum (``set_start_state_curricula``) difficulty only.

        v25 fix. The demo curricula are built from verified winning routes on the
        OFFICIAL maps, while the pre/post knobs above are driven by the procgen
        start-state pools (PRECISION / CHAMBER_MAZE / LOOP_CORRIDOR). Sharing one
        knob was a measured failure: the easy procgen buckets won ~0.92 of their
        seeded episodes and ratcheted the shared difficulty to 1.0 (= start at
        spawn), which silently switched the OFFICIAL demo curriculum off long
        before the agent could use it — while official seeded episodes were
        themselves filtered out of the anneal signal by the bucket filter. Giving
        the demo leg its own knob makes official difficulty anneal on official
        success."""
        self._ssc_demo_difficulty = float(d)

    def _demo_curriculum_difficulty(self) -> float:
        # Falls back to the pre-switch knob (then the global one) so an older
        # callback that never calls the demo setter behaves exactly as before.
        return float(getattr(self, "_ssc_demo_difficulty",
                             self._pre_switch_difficulty()))

    def _get_observation(self) -> Dict[str, Any]:
        """Get the current observation from the game state."""
        t_sup = self._profile_start("obs_super_layers")
        obs = super()._get_observation()
        self._profile_end("obs_super_layers", t_sup)

        from .constants import NINJA_STATE_DIM, PHYSICS_DERIVED_DIM

        # Phase 4: Only keep ninja physics state (40 enhanced dims)
        # All other features (path objectives, mines, progress, etc.) are now in the graph
        base_game_state = obs["game_state"]
        self._game_state_buffer[:NINJA_STATE_DIM] = base_game_state[:NINJA_STATE_DIM]

        # Append physics-derived features: horizontal_range, max_jump_height, time_to_ground
        t_phys = self._profile_start("obs_npp_physics_derived")
        self._game_state_buffer[
            NINJA_STATE_DIM : NINJA_STATE_DIM + PHYSICS_DERIVED_DIM
        ] = self._compute_physics_derived_features()
        self._profile_end("obs_npp_physics_derived", t_phys)

        # L1.4: phase indicator for policy (same source as obs["switch_activated"])
        switch_activated = bool(self.nplay_headless.exit_switch_activated())
        self._game_state_buffer[GAME_STATE_SWITCH_ACTIVATED_IDX] = float(
            switch_activated
        )

        # Goal geometry: ninja-relative vectors to the exit switch and exit door,
        # normalized by level extent and clipped to [-1, 1]. Static entities sit on
        # a 6px grid, so these are exact (the 24px overlay tile would otherwise
        # quantize the goal to 1 of 16 sub-positions). Exposing both keeps the
        # policy's goal geometry continuous across the switch->exit phase change.
        nx, ny = self.nplay_headless.ninja_position()
        sx = float(obs.get("switch_x", nx))
        sy = float(obs.get("switch_y", ny))
        ex = float(obs.get("exit_door_x", nx))
        ey = float(obs.get("exit_door_y", ny))
        self._game_state_buffer[GAME_STATE_SWITCH_VEC_IDX] = np.clip(
            (sx - nx) / LEVEL_WIDTH, -1.0, 1.0
        )
        self._game_state_buffer[GAME_STATE_SWITCH_VEC_IDX + 1] = np.clip(
            (sy - ny) / LEVEL_HEIGHT, -1.0, 1.0
        )
        self._game_state_buffer[GAME_STATE_EXIT_VEC_IDX] = np.clip(
            (ex - nx) / LEVEL_WIDTH, -1.0, 1.0
        )
        self._game_state_buffer[GAME_STATE_EXIT_VEC_IDX + 1] = np.clip(
            (ey - ny) / LEVEL_HEIGHT, -1.0, 1.0
        )

        # HORIZONTAL_AUGMENTATION_PLAN: one reflection-ODD sign for the whole
        # observation, resolving the exact-cell-boundary tie in every
        # ninja-centred readout below (crops, raycast march, scandot fan, visit
        # cell). It MUST be computed before any of them and shared, or two
        # readouts could disagree about which side of a boundary the ninja is on.
        # Off => +1, which is the plain floor these call sites had before.
        self._mirror_odd_dir = (
            reflection_odd_direction(
                self.nplay_headless.ninja_velocity()[0],
                (ex if switch_activated else sx) - nx,
                float(getattr(self.nplay_headless.sim.ninja, "floor_normalized_x", 0.0)),
            )
            if self._mirror_equivariant_obs
            else 1
        )

        # Surface ray-casts: distance to the nearest grounded platform in 8 compass
        # directions (blind-jump / gap verification beyond the 24px tile view).
        t_ray = self._profile_start("obs_npp_raycasts")
        self._game_state_buffer[
            GAME_STATE_RAYCAST_IDX : GAME_STATE_RAYCAST_IDX + GAME_STATE_RAYCAST_DIM
        ] = self._compute_surface_raycasts()
        self._profile_end("obs_npp_raycasts", t_ray)

        # Free-space nav hint: physics-free steepest-descent direction toward the
        # current objective that rounds obstacles (vs the straight-line goal vectors
        # above). Observation only — never reused as a reward potential.
        t_fsn = self._profile_start("obs_npp_free_space_nav")
        self._refresh_free_space_nav(nx, ny, switch_activated)
        self._game_state_buffer[
            GAME_STATE_FREE_SPACE_NAV_IDX : GAME_STATE_FREE_SPACE_NAV_IDX + 4
        ] = getattr(self, "_cached_free_space_nav", np.zeros(4, dtype=np.float32))
        self._profile_end("obs_npp_free_space_nav", t_fsn)

        # Scalar nearest-deadly-mine distance (danger gauge for the MLP stream).
        self._game_state_buffer[GAME_STATE_NEAREST_MINE_IDX] = (
            self._compute_nearest_deadly_mine_distance_norm()
        )

        # Closing speed toward the current objective (switch pre-activation, else
        # exit): the rate at which the ninja is approaching the live goal.
        gx, gy = (ex, ey) if switch_activated else (sx, sy)
        gdx, gdy = gx - nx, gy - ny
        gmag = (gdx * gdx + gdy * gdy) ** 0.5
        if gmag > 1e-6:
            vx, vy = self.nplay_headless.ninja_velocity()
            closing = (vx * gdx + vy * gdy) / gmag  # px/frame toward goal
            self._game_state_buffer[GAME_STATE_CLOSING_SPEED_IDX] = float(
                np.clip(closing / MAX_HOR_SPEED, -1.0, 1.0)
            )
        else:
            self._game_state_buffer[GAME_STATE_CLOSING_SPEED_IDX] = 0.0

        # Jump-execution features (precise jump / wall-jump timing). The feed-forward
        # policy can't track its own action history, so expose: whether jump was held
        # last frame (jumps fire only on the rising edge), whether a jump now would be
        # useless, and the launch-pad buffer window (the 4th buffer). Read straight off
        # the ninja (both python + native expose these fields).
        _ninja = self.nplay_headless.sim.ninja
        self._game_state_buffer[GAME_STATE_JUMP_HELD_IDX] = (
            1.0 if _ninja.jump_input_old != 0 else -1.0
        )
        # jump-useless (idx 47) and the launch-pad buffer (idx 48) below are
        # CORRECTNESS fixes that change the VALUE of two long-standing
        # game_state slots, so they are gated on expose_precision_state rather
        # than applied unconditionally. Existing checkpoints (v22 and every run
        # warm-started from it) were trained against the legacy values; flipping
        # them for free would feed a subtly different distribution into two slots
        # of an otherwise-unchanged policy. Runs that opt into the precision
        # block already re-initialize the game_state MLP, so they absorb the
        # corrected semantics cleanly.
        _precision = getattr(
            getattr(self.config, "reachability", None), "expose_precision_state", False
        )
        if _precision:
            # Shared predicate: the old local copy added a ``jump_input_old != 0``
            # term the ACTION MASK does not have, so the policy could perceive
            # "a jump would work" on frames the mask had already forbidden it.
            _jump_useless_val = _jump_is_useless(_ninja)
        else:
            _has_buf = (
                -1 < _ninja.jump_buffer < 5
                or -1 < _ninja.floor_buffer < 5
                or -1 < _ninja.wall_buffer < 5
                or -1 < _ninja.launch_pad_buffer < 4
            )
            _jump_useless_val = (
                bool(_ninja.airborn)
                and _ninja.state != 3
                and _ninja.jump_input_old != 0
                and not _has_buf
            )
        self._game_state_buffer[GAME_STATE_JUMP_USELESS_IDX] = (
            1.0 if _jump_useless_val else -1.0
        )
        # (b + 1)/(window + 1) like the jump/floor/wall buffers so INACTIVE (-1)
        # maps to -1.0 and "just armed" (b=0, ACTIVE) stays distinct from it.
        # Window is 4 for the launch pad (``-1 < lp < 4``, ninja.think), NOT the
        # 5 used by jump/floor/wall; dividing by 5.0 capped the encoded value at
        # 0.6 of its range. Gated for the same backward-compatibility reason as
        # the jump-useless fix above.
        _lp_window = 4.0 if _precision else 5.0
        self._game_state_buffer[GAME_STATE_LAUNCHPAD_BUFFER_IDX] = (
            (max(_ninja.launch_pad_buffer, -1) + 1) / _lp_window
        ) * 2 - 1

        # Slope/curve EXECUTION features. The look-ahead landing-surface normal
        # (exact tile geometry of the slope/curve the ninja is moving toward) lets
        # the airborne policy pre-orient for a sloped landing; the signed vertical
        # terminal-velocity signal exposes the (otherwise saturated) vertical speed.
        ln_x, ln_y, ln_dist = self._compute_landing_surface()
        self._game_state_buffer[GAME_STATE_LANDING_NORMAL_IDX] = ln_x
        self._game_state_buffer[GAME_STATE_LANDING_NORMAL_IDX + 1] = ln_y
        self._game_state_buffer[GAME_STATE_LANDING_DIST_IDX] = ln_dist
        _vx, _vy = self.nplay_headless.ninja_velocity()
        self._game_state_buffer[GAME_STATE_VERTICAL_TERMINAL_IDX] = float(
            np.clip(-_vy / MAX_VER_SPEED, -1.0, 1.0)
        )

        # Execution-timing scalars. tau = px distance to the landing surface /
        # closing speed into it (normalized; 1 = far/not approaching).
        _closing = -(ln_x * _vx + ln_y * _vy)
        if _closing > 1e-3:
            _tau = (ln_dist * 500.0) / _closing
            self._game_state_buffer[GAME_STATE_TIME_TO_CONTACT_IDX] = float(
                np.clip(_tau / 60.0, 0.0, 1.0)
            )
            # Raw frames-to-contact for the FiGAR approaching-contact interrupt
            # (Audit II P1-8); emitted as info["time_to_contact_frames"] by
            # _extend_info_hook (obs is computed before the hook each step).
            self._last_time_to_contact_frames = float(_tau)
            # v17 A: lethal-landing signal. ``_closing`` is the speed INTO the
            # landing surface below; the sim kills the ninja when it exceeds that
            # surface's survivable threshold (ninja.py: MAX_SURVIVABLE_IMPACT -
            # 4/3*|ny|). When airborne and the CURRENT velocity would make the
            # imminent landing fatal-if-uncorrected, expose the frames-to-contact
            # so FiGAR can interrupt the macro EARLY (a decision point to brake /
            # wall-grab), targeting the maze impact-death failure mode. inf when
            # survivable / grounded / not approaching a surface.
            _thr = MAX_SURVIVABLE_IMPACT - (4.0 / 3.0) * abs(ln_y)
            if getattr(_ninja, "airborn", False) and _thr > 1e-9 and _closing >= _thr:
                self._lethal_landing_tau = float(_tau)
            else:
                self._lethal_landing_tau = float("inf")
        else:
            self._game_state_buffer[GAME_STATE_TIME_TO_CONTACT_IDX] = 1.0
            self._last_time_to_contact_frames = float("inf")
            self._lethal_landing_tau = float("inf")
        self._game_state_buffer[GAME_STATE_FRAMES_AIRBORNE_IDX] = float(
            np.clip(getattr(_ninja, "frames_airborne", 0) / 60.0, 0.0, 1.0)
        )
        # Per-frame acceleration (velocity delta), normalized by 2 px/frame.
        _ax = _vx - getattr(_ninja, "xspeed_old", _vx)
        _ay = _vy - getattr(_ninja, "yspeed_old", _vy)
        self._game_state_buffer[GAME_STATE_ACCEL_IDX] = float(np.clip(_ax / 2.0, -1.0, 1.0))
        self._game_state_buffer[GAME_STATE_ACCEL_IDX + 1] = float(np.clip(_ay / 2.0, -1.0, 1.0))

        # Forward-biased terrain scandots (floor-gap, ceiling-gap per column).
        _scan = self._compute_scandots(self._scandot_direction(_vx, gdx))
        self._game_state_buffer[
            GAME_STATE_SCANDOTS_IDX:GAME_STATE_SCANDOTS_IDX + GAME_STATE_SCANDOTS_DIM
        ] = _scan

        # Ghost-arc trajectory-projection bundle (Audit II P0-1): the current
        # route segment's TARGET, observable nowhere else — where each
        # constant-input arc (u in {-1,0,+1}) actually lands, what is there
        # (deadly mine / nothing / a surface + its normal), how wide the
        # mine-free landing is, whether the impact speed is survivable, and
        # when contact happens. The +-1 arcs are the bang-bang reachable-
        # landing interval endpoints. Native integration of the real airborne
        # recurrence (~us-scale); grounded frames take the fast-path fill.
        self._fill_ghost_arc_bundle(_ninja, _vx, _vy)

        obs["game_state"] = np.array(self._game_state_buffer, copy=True)

        # Precision-state block (stashed; appended in _process_observation).
        if getattr(getattr(self.config, "reachability", None),
                   "expose_precision_state", False):
            self._precision_state_feat = self._compute_precision_state(_ninja)

        # v26 W6c commitment/irreversibility block. MUST run after
        # _fill_ghost_arc_bundle — it reads the arc rows straight out of the
        # game_state buffer that call just filled.
        if getattr(getattr(self.config, "reachability", None),
                   "expose_commitment_state", False):
            self._commitment_state_feat = self._compute_commitment_state(_ninja)
            self._commitment_no_escape = bool(
                self._commitment_state_feat[4] >= 0.5
            )

        # Frames-to-WALL-contact scalar (stashed; appended in _process_observation).
        if getattr(getattr(self.config, "reachability", None),
                   "expose_wall_contact_tau", False):
            _wt = self._compute_wall_contact_tau(nx, ny, _vx, _vy, _ninja)
            self._wall_tau_feat = np.array([_wt], dtype=np.float32)
            # Raw FRAMES-to-wall-contact for the FiGAR wall-approach interrupt.
            # The normalized scalar is the obs; the interrupt needs real frames so
            # its threshold is expressed in the same units as the other interrupts.
            self._last_wall_contact_tau_frames = float(
                _wt * self.WALL_TAU_LOOKAHEAD_FRAMES
            )

        # EXECUTION_PRECISION_PLAN §6 time/phase block (stashed; appended in
        # _process_observation).
        if getattr(getattr(self.config, "reachability", None),
                   "expose_time_phase", False):
            self._time_phase_feat = self._compute_time_phase()

        # v22 §3-B: derived blocked-toward-goal scalar (stashed; appended to the
        # policy game_state in _process_observation, after the base truncation).
        if getattr(getattr(self.config, "reachability", None),
                   "expose_blocked_goal", False):
            self._blocked_goal_feat = np.array(
                [self._compute_blocked_toward_goal(nx, ny, switch_activated, _ninja)],
                dtype=np.float32,
            )

        obs["_impact_risk"] = self._impact_risk(ln_x, ln_y, ln_dist, _vx, _vy)
        # v13 arc-risk PBRS input: True while airborne with all three steering
        # arcs ending in a mine/lethal impact (set by _fill_ghost_arc_bundle).
        obs["_arc_doomed"] = bool(getattr(self, "_arc_doomed", False))

        # Populate the internal A*-path-distance cache used by PBRS. The feature
        # vector is discarded; only `_cached_switch_distance` / `_cached_exit_distance`
        # (set as a side-effect of `_compute_reachability`) matter for the reward
        # path. These are not exposed to the policy.
        t_path = self._profile_start("obs_npp_path_dist")
        self._update_path_distance_cache()
        self._profile_end("obs_npp_path_dist", t_path)
        if hasattr(self, "_cached_switch_distance"):
            obs["_cached_switch_distance"] = self._cached_switch_distance
        if hasattr(self, "_cached_exit_distance"):
            obs["_cached_exit_distance"] = self._cached_exit_distance
        # Whether those distances come from a COMMITTED substitution source
        # (cost-to-completion basis, continuous across switch activation). When
        # False the reward calculator composes its own continuous basis or
        # degrades to the HWM ratchet — see RewardCalculator PBRS episode-init.
        obs["_potential_substituted"] = bool(
            getattr(self, "_potential_substituted", False)
        )
        # Number of necessary locked doors opened (from the lattice door plan);
        # consumed by the reward calculator for the locked-door-switch bonus +
        # HWM sub-goal phase. 0 when no plan / no necessary doors.
        obs["necessary_doors_open"] = int(getattr(self, "_necessary_doors_open", 0))
        # How many the level requires in total. The reward needs it to bound the
        # episode's milestone payout (see R_LOCKED_DOOR_TOTAL_CAP): a flat
        # per-door bonus on a 71-door level outpays completion 35.5 to 10.0.
        obs["necessary_doors_total"] = int(getattr(self, "_necessary_door_total", 0))

        # Extract locked door switch states from environment — kept internally
        # for the reward path even though `switch_states` is no longer in the
        # policy obs dict.
        t_sa = self._profile_start("obs_npp_switch_adj")
        switch_states_dict = self._get_switch_states_from_env()
        if hasattr(self, "level_data") and self.level_data is not None:
            self.level_data.switch_states = switch_states_dict
        obs["switch_states_dict"] = switch_states_dict

        obs["level_data"] = self.level_data

        # Add adjacency graph and full graph data for PBRS path-aware reward shaping.
        adjacency = self._get_adjacency_for_rewards()
        if adjacency is None:
            raise RuntimeError("Adjacency graph not available. Graph building failed.")
        obs["_adjacency_graph"] = adjacency
        obs["_graph_data"] = self.current_graph_data
        self._profile_end("obs_npp_switch_adj", t_sa)

        t_haz = self._profile_start("obs_npp_hazard")
        # Both channels are SDFs where LOWER means nearer a mine, so min-pooling
        # keeps the nearest hazard in each block (never averages one away).
        obs["hazard_map"] = pool_channels(
            self._compute_hazard_map(),
            HAZARD_MAP_OBS_DOWNSAMPLE,
            ("min", "min"),
        )
        # Same convention: the drone SDF is lower nearer a drone, so min-pooling
        # keeps the nearest threat in each block rather than averaging it away.
        # Phase 1 keys. Already emitted on the 12px observation grid by the
        # native engine, so there is no pooling pass -- the scatter writes
        # straight into it, which is exactly equal to pooling the 6px field.
        if self._threat_ttl_enabled:
            obs["threat_ttl"] = self._read_native_field(
                "threat_ttl", THREAT_TTL_CHANNELS,
                self._threat_ttl_max_frames, self._threat_ttl_stride,
            )
        if self._solid_map_enabled:
            obs["solid_map"] = self._read_native_field(
                "solid_map", SOLID_MAP_CHANNELS
            )
        if self._drone_hazard_enabled:
            obs["drone_hazard_map"] = pool_channels(
                self._compute_drone_hazard_map(),
                HAZARD_MAP_OBS_DOWNSAMPLE,
                ("min",) * DRONE_HAZARD_MAP_CHANNELS,
            )
        self._profile_end("obs_npp_hazard", t_haz)

        # Optional physics-aware nav hint (lattice d*).  ``_cached_lattice_nav``
        # is refreshed by `_compute_reachability` (called via
        # `_update_path_distance_cache` above) on cell change; zeros when no plan.
        if getattr(getattr(self.config, "reachability", None),
                   "expose_lattice_features", False):
            obs["lattice_nav"] = np.array(
                getattr(self, "_cached_lattice_nav", np.zeros(LATTICE_NAV_DIM)),
                copy=True, dtype=np.float32,
            )

        # Waypoint subgoal feed (long-horizon distance fix).  Reward-only distance +
        # index drive the per-leg waypoint PBRS (stripped from the policy obs in
        # _process_observation); the 4-dim nav hint is exposed to the policy only when
        # expose_waypoint_features is set.  See ReachabilityConfig.enable_waypoint_tracker.
        if getattr(self, "_waypoint_leg_distance", None) is not None:
            obs["waypoint_leg_distance"] = self._waypoint_leg_distance
            obs["waypoint_index"] = self._waypoint_index
        if getattr(getattr(self.config, "reachability", None),
                   "expose_waypoint_features", False):
            _wp_nav = np.asarray(
                getattr(self, "_cached_waypoint_nav", None)
                if getattr(self, "_cached_waypoint_nav", None) is not None
                else np.zeros(WAYPOINT_NAV_DIM),
                dtype=np.float32,
            )
            # P2-11 upgrade guard: a stale 4-dim cache (pre-annotation tracker)
            # zero-pads to the 9-dim contract.
            if _wp_nav.shape[0] < WAYPOINT_NAV_DIM:
                _wp_nav = np.pad(_wp_nav, (0, WAYPOINT_NAV_DIM - _wp_nav.shape[0]))
            obs["waypoint_nav"] = np.array(_wp_nav, copy=True, dtype=np.float32)

        # Cost-to-go map (Phase-2 O2): global route field for the policy.
        if getattr(getattr(self.config, "reachability", None),
                   "expose_cost_to_go_map", False):
            obs["cost_to_go_map"] = self._compute_cost_to_go_map(switch_activated)

        # v17 episodic visitation map: per-episode recency-decayed visited grid
        # (the grid itself is advanced in _pre_reward_hook each step; here we just
        # render the current state, co-registered with cost_to_go_map).
        if getattr(self, "_expose_visitation_map", False):
            obs["visitation_map"] = self._compute_visitation_map()

        # v18 C1 frontier map: reachable-and-unvisited grid (advanced with the
        # same per-episode visit counts; rendered fresh each step).
        if getattr(self, "_expose_frontier_map", False):
            obs["frontier_map"] = self._compute_frontier_map()

        # Ego-centric high-res tile crop centered on the ninja.
        if getattr(getattr(self.config, "reachability", None),
                   "enable_ego_crop", False):
            obs["ego_crop"] = self._compute_ego_crop(nx, ny)
            # Mine-density crop co-registered with the tile crop (deadly/safe).
            obs["ego_crop_mines"] = self._compute_ego_crop_mines(
                nx, ny, obs.get("entity_overlays")
            )
            # Route-gradient crop (v14 B2): local grad(d*) unit vectors.
            obs["ego_crop_route"] = self._compute_ego_crop_route(
                nx, ny, switch_activated
            )

        # v22 §3-C: fine (6px, un-pooled) route-gradient crop — own flag, so it can
        # be exposed independently of the tile ego crop.
        if getattr(getattr(self.config, "reachability", None),
                   "expose_ego_crop_route_fine", False):
            obs["ego_crop_route_fine"] = self._compute_ego_crop_route_fine(
                nx, ny, switch_activated
            )

        # v24: fine (6px) ego MINE-SDF crop — sub-tile mine geometry around the ninja.
        if getattr(getattr(self.config, "reachability", None),
                   "expose_ego_crop_mines_fine", False):
            obs["ego_crop_mines_fine"] = self._compute_ego_crop_mines_fine(nx, ny)

        # v26 W6a: static fall-lethality field. Impact deaths are ~40% of v25's
        # losses and are committed at TAKEOFF, so this is the one impact signal
        # that arrives while the decision is still open.
        _reach_cfg = getattr(self.config, "reachability", None)
        if getattr(_reach_cfg, "expose_fall_lethality_map", False) or getattr(
            _reach_cfg, "expose_ego_crop_fall_fine", False
        ):
            _fall = self._get_fall_lethality_map()
            if getattr(_reach_cfg, "expose_fall_lethality_map", False):
                # ch0 = signed survivable margin (lower = more lethal) -> min;
                # ch1 = normalized drop distance (higher = longer fall) -> max.
                # The ego crop below keeps the full-resolution field.
                obs["fall_lethality_map"] = pool_channels(
                    _fall, HAZARD_MAP_OBS_DOWNSAMPLE, ("min", "max")
                )
            if getattr(_reach_cfg, "expose_ego_crop_fall_fine", False):
                obs["ego_crop_fall_fine"] = crop_fall_lethality(
                    _fall, nx, ny, EGO_CROP_FALL_FINE_SIZE, self._mirror_odd_dir
                )

        # v25: directional mine bearing + CLOSING SPEED — the motion-relative
        # terms the (still-frame) mine crop above cannot carry.
        if getattr(getattr(self.config, "reachability", None),
                   "expose_mine_proximity", False):
            obs["mine_proximity"] = self._compute_mine_proximity(nx, ny)

        # Shooter telegraph — the wind-up the hitscan field cannot show in time.
        if self._threat_ttl_enabled:
            obs["threat_proximity"] = self._compute_threat_proximity(nx, ny)

        # Coarse far ego crop (Audit II P1-6): +-21-tile Chebyshev window at
        # 2-tile pooling — see _compute_ego_crop_coarse.
        if getattr(getattr(self.config, "reachability", None),
                   "enable_ego_crop_coarse", False):
            obs["ego_crop_coarse"] = self._compute_ego_crop_coarse(
                nx, ny, obs.get("entity_overlays"), switch_activated,
                obs.get("switch_x"), obs.get("switch_y"),
                obs.get("exit_door_x"), obs.get("exit_door_y"),
            )

        # Return full observation - conversion to Layer 1 dict happens in
        # _process_observation(). This ensures the reward calculator has
        # access to all metadata.
        return obs

    def _update_path_distance_cache(self) -> None:
        """Run the A* path distance computation to populate reward caches.

        This used to be part of `_get_reachability_features`, which also
        produced a feature vector for the policy. The feature vector is no
        longer part of the observation (Layer 1), but the cached path
        distances (`_cached_switch_distance`, `_cached_exit_distance`) are
        still consumed by ``RewardCalculator._get_distances`` to avoid a
        duplicate A* call. Any exception here is non-fatal since the reward
        calculator falls back to direct path cost queries.
        """
        try:
            # Compute the reward distances EVERY frame at sub-node resolution
            # (not the 24px-cell-cached _compute_reachability, whose feature vector
            # is unused in Layer 1). This keeps the PBRS/HWM gradient alive during
            # slow <=1px movement while skipping the wasted feature-vector work.
            self._refresh_reward_distances(self.nplay_headless.ninja_position())
        except Exception:
            pass

    def _compute_physics_derived_features(self) -> np.ndarray:
        """Compute 3 physics-derived features for value function alignment with Φ.

        Features: horizontal_range_norm, max_jump_height_norm, time_to_ground_norm.
        Time-to-ground uses geometry-accurate downward raycast via adjacency graph.
        """
        import math
        from .constants import PHYSICS_DERIVED_DIM
        from ..constants.physics_constants import (
            DRAG_REGULAR,
            GRAVITY_FALL,
            GRAVITY_JUMP,
            MAX_HOR_SPEED,
            MAX_JUMP_DURATION,
        )
        from ..graph.reachability.directional_connectivity import find_platform_in_direction
        from ..graph.reachability.path_distance_cache import NODE_WORLD_COORD_OFFSET

        features = np.zeros(PHYSICS_DERIVED_DIM, dtype=np.float32)
        ninja = self.nplay_headless.sim.ninja
        vx, vy = ninja.xspeed, ninja.yspeed
        g = ninja.applied_gravity

        # time_to_ground: geometry-accurate via downward raycast
        if ninja.floor_count > 0:
            time_to_ground = 0.0
        else:
            graph_data = getattr(self, "current_graph_data", None) or {}
            spatial_hash = graph_data.get("spatial_hash")
            physics_cache = graph_data.get("node_physics")
            base_adjacency = graph_data.get("base_adjacency")

            if spatial_hash is not None and physics_cache is not None and base_adjacency:
                node_pos = (
                    int(ninja.xpos) - NODE_WORLD_COORD_OFFSET,
                    int(ninja.ypos) - NODE_WORLD_COORD_OFFSET,
                )
                # The downward raycast result is a pure function of node_pos and
                # the (static, per-level) graph, but it costs ~40 spatial-hash
                # queries. Memoize it per level keyed by node_pos. ``graph_data``
                # is rebuilt (fresh dict) on level change, so the memo auto-
                # invalidates. Bit-identical: only the raycast distance is cached;
                # the min() with the continuous height bound is applied after.
                memo = graph_data.get("_downcast_memo")
                if memo is None:
                    memo = {}
                    graph_data["_downcast_memo"] = memo
                dy = memo.get(node_pos)
                if dy is None:
                    dy = find_platform_in_direction(
                        node_pos,
                        (0, 1),  # South = downward in screen coords
                        base_adjacency,
                        physics_cache,
                        spatial_hash,
                        max_distance=500.0,
                        step_size=12,
                    )
                    memo[node_pos] = dy
                dy = min(dy, max(0.0, LEVEL_HEIGHT_PX - ninja.ypos))
            else:
                dy = max(0.0, LEVEL_HEIGHT_PX - ninja.ypos)

            discriminant = vy * vy + 2.0 * g * dy
            if g > 1e-9 and discriminant >= 0:
                time_to_ground = max(0.0, (-vy + math.sqrt(discriminant)) / g)
            else:
                time_to_ground = 90.0

        time_to_ground_norm = min(time_to_ground / 90.0, 1.0)
        features[2] = float(time_to_ground_norm)

        # horizontal_range_norm: 0 when grounded, else abs(vx)*time*drag^(t/2)
        if ninja.floor_count > 0:
            horizontal_range_norm = 0.0
        else:
            drag_factor = DRAG_REGULAR ** (time_to_ground / 2.0) if time_to_ground > 0 else 1.0
            horizontal_range = abs(vx) * time_to_ground * drag_factor
            max_range = MAX_HOR_SPEED * 90.0
            horizontal_range_norm = min(horizontal_range / max_range, 1.0)
        features[0] = float(horizontal_range_norm)

        # max_jump_height_norm: remaining upward displacement if in jump state
        if ninja.state == 3 and ninja.jump_duration < MAX_JUMP_DURATION and vy < 0:
            frames_left = MAX_JUMP_DURATION - ninja.jump_duration
            # Discrete: vy (negative) decreases toward 0 by GRAVITY_JUMP each frame.
            # Approx displacement = avg_|vy| * frames_left
            remaining_height = abs(vy) * frames_left * 0.5
            max_jump_height_norm = min(remaining_height / 60.0, 1.0)
        else:
            max_jump_height_norm = 0.0
        features[1] = float(max_jump_height_norm)

        return features

    # 8 compass directions in (tile_dx, tile_dy) screen space (+x east, +y south),
    # order [E, NE, N, NW, W, SW, S, SE] to match COMPASS_DIRECTIONS.
    _RAYCAST_DIRS = (
        (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1),
    )
    _RAYCAST_NORM_PX = 500.0  # distance normalizer; 1.0 = nearest surface >= 500px

    def _compute_surface_raycasts(self) -> np.ndarray:
        """8-dir normalized distance to the nearest solid surface (wall / floor).

        Order [E, NE, N, NW, W, SW, S, SE]; 0.0 = adjacent surface, 1.0 = none
        within 500px. Ray-marches the cached tile grid (cheap: <=42 int8 lookups
        per direction) from the ninja's tile to the first non-empty tile, treating
        the enclosing border as a wall. Lets the policy sense walls / verify
        landing surfaces and gaps beyond the local view. Memoized per ninja tile
        (24px) in ``current_graph_data`` (auto-invalidated on level change)."""
        from ..constants.physics_constants import (
            MAP_TILE_HEIGHT,
            MAP_TILE_WIDTH,
            TILE_PIXEL_SIZE,
        )

        grid = getattr(self, "_cached_tile_grid", None)
        if grid is None:
            return np.ones(GAME_STATE_RAYCAST_DIM, dtype=np.float32)

        ninja = self.nplay_headless.sim.ninja
        ntx = _cell_x(ninja.xpos, TILE_PIXEL_SIZE, self._mirror_odd_dir) - 1
        nty = int(ninja.ypos / TILE_PIXEL_SIZE) - 1

        graph_data = getattr(self, "current_graph_data", None)
        memo = None
        if graph_data is not None:
            memo = graph_data.get("_raycast_memo")
            if memo is None:
                memo = {}
                graph_data["_raycast_memo"] = memo
            cached = memo.get((ntx, nty))
            if cached is not None:
                return cached

        out = np.ones(GAME_STATE_RAYCAST_DIM, dtype=np.float32)
        ts = float(TILE_PIXEL_SIZE)
        for i, (dtx, dty) in enumerate(self._RAYCAST_DIRS):
            tx, ty, steps = ntx, nty, 0
            while True:
                tx += dtx
                ty += dty
                steps += 1
                if not (0 <= tx < MAP_TILE_WIDTH and 0 <= ty < MAP_TILE_HEIGHT):
                    break  # enclosing border acts as a wall
                if grid[tx, ty] != 0:
                    break  # first non-empty tile = surface
            diag = 1.41421356 if (dtx and dty) else 1.0
            out[i] = min(steps * ts * diag / self._RAYCAST_NORM_PX, 1.0)

        if memo is not None:
            memo[(ntx, nty)] = out
        return out

    # 8 tile-direction unit steps ordered by angle for velocity quantization:
    # atan2(vy, vx) bucket -> (dtx, dty), screen coords (y DOWN).
    _LANDING_DIRS = (
        (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1),
    )
    _LANDING_MAX_TILES = 12

    def _compute_landing_surface(self):
        """Look-ahead surface the ninja is moving toward: outward normal + distance.

        Ray-marches the cached tile grid from the ninja's tile along the velocity
        direction (default straight down when ~stationary, since gravity pulls the
        landing downward) to the first non-empty tile, then reads that tile's EXACT
        outward surface normal from :mod:`nclone.tile_surface_normals` (slopes via
        the diagonal segment, curves via the arc; flat/half solids -> ``(0, -1)``).
        Returns ``(nx, ny, dist_norm)``; ``(0, 0, 1)`` if no surface within range.
        Lets the airborne policy anticipate the angle of a sloped/curved landing.
        """
        import math

        from ..constants.physics_constants import (
            MAP_TILE_HEIGHT,
            MAP_TILE_WIDTH,
            TILE_PIXEL_SIZE,
        )
        from ..tile_surface_normals import tile_surface_normal

        grid = getattr(self, "_cached_tile_grid", None)
        if grid is None:
            return (0.0, 0.0, 1.0)

        ninja = self.nplay_headless.sim.ninja
        ntx = _cell_x(ninja.xpos, TILE_PIXEL_SIZE, self._mirror_odd_dir) - 1
        nty = int(ninja.ypos / TILE_PIXEL_SIZE) - 1
        vx, vy = self.nplay_headless.ninja_velocity()
        if (vx * vx + vy * vy) < 0.09:  # ~stationary -> look straight down
            dtx, dty = 0, 1
        else:
            idx = int(round(math.atan2(vy, vx) / (math.pi / 4.0))) % 8
            dtx, dty = self._LANDING_DIRS[idx]

        tx, ty, steps = ntx, nty, 0
        t_type = 0
        while steps < self._LANDING_MAX_TILES:
            tx += dtx
            ty += dty
            steps += 1
            if not (0 <= tx < MAP_TILE_WIDTH and 0 <= ty < MAP_TILE_HEIGHT):
                t_type = 1  # enclosing border acts as a flat wall/floor
                break
            t_type = int(grid[tx, ty])
            if t_type != 0:
                break
        if t_type == 0:
            return (0.0, 0.0, 1.0)

        nx, ny = tile_surface_normal(t_type)
        if nx == 0.0 and ny == 0.0:
            nx, ny = 0.0, -1.0  # flat / half solid -> upward floor normal
        diag = 1.41421356 if (dtx and dty) else 1.0
        dist = min(steps * float(TILE_PIXEL_SIZE) * diag / 500.0, 1.0)
        return (float(nx), float(ny), float(dist))

    _SCANDOT_COL_OFFSETS = (-1, 0, 1, 2, 3, 4)  # tile columns, * travel direction
    _SCANDOT_MAX_TILES = 12.0

    def _scandot_direction(self, vx: float, goal_dx: float) -> int:
        """Travel direction the forward-biased scandots fan out along.

        The legacy rule is ``sign(vx) or 1``. That tie-break is EVEN: at
        ``vx == 0`` it resolves to +1 in a level and in its mirror image alike,
        so the two runs sample columns that are not each other's reflection —
        and ``vx == 0`` is not a corner case (every spawn frame, every frame the
        ninja stands still). Since scandots are defined RELATIVE to the travel
        direction, that is the difference between the block being invariant
        under reflection and being quietly wrong.

        With ``mirror_equivariant_obs`` this returns the same reflection-odd
        sign every other ego-centric readout ties on (``self._mirror_odd_dir``),
        so the scandot fan and the crop windows can never disagree about which
        way the ninja is facing at rest.
        """
        d = int(np.sign(vx))
        if d:
            return d
        if self._mirror_equivariant_obs:
            return self._mirror_odd_dir
        return 1

    def _compute_scandots(self, facing: int) -> np.ndarray:
        """Forward-biased terrain height samples: per column, tile-distance DOWN to
        the next floor and UP to the next ceiling, relative to the ninja, normalized.

        Ego-centric (offsets are relative to the ninja and biased by travel
        direction), so it generalizes across absolute position. 1.0 = no
        surface within range. Returns a flat (COLUMNS*2,) array [floor,ceil,...].
        """
        from ..constants.physics_constants import (
            MAP_TILE_HEIGHT,
            MAP_TILE_WIDTH,
            TILE_PIXEL_SIZE,
        )

        out = np.ones(GAME_STATE_SCANDOTS_DIM, dtype=np.float32)
        grid = getattr(self, "_cached_tile_grid", None)
        if grid is None:
            return out
        ninja = self.nplay_headless.sim.ninja
        ntx = _cell_x(ninja.xpos, TILE_PIXEL_SIZE, self._mirror_odd_dir) - 1
        nty = int(ninja.ypos / TILE_PIXEL_SIZE) - 1
        mx = self._SCANDOT_MAX_TILES
        mx_i = int(mx)
        values = [1.0] * GAME_STATE_SCANDOTS_DIM
        for i, off in enumerate(self._SCANDOT_COL_OFFSETS):
            cx = ntx + facing * off
            if not (0 <= cx < MAP_TILE_WIDTH):
                values[2 * i] = 0.0  # off-map column reads as adjacent wall
                values[2 * i + 1] = 0.0
                continue
            # One numpy->list conversion per column, then a pure-Python walk: the
            # ray can take up to 2*mx steps per column and an element-wise
            # `grid[cx, ty]` builds a numpy scalar on every one of them.
            col = grid[cx].tolist()
            # distance DOWN to floor
            d = 0
            ty = nty
            while d < mx_i:
                ty += 1
                d += 1
                if not (0 <= ty < MAP_TILE_HEIGHT) or col[ty] != 0:
                    break
            values[2 * i] = d / mx if d < mx_i else 1.0
            # distance UP to ceiling
            u = 0
            ty = nty
            while u < mx_i:
                ty -= 1
                u += 1
                if not (0 <= ty < MAP_TILE_HEIGHT) or col[ty] != 0:
                    break
            values[2 * i + 1] = u / mx if u < mx_i else 1.0
        out[:] = values
        return out

    def _impact_risk(self, ln_x: float, ln_y: float, ln_dist: float, vx: float, vy: float) -> float:
        """Impact-survivability risk in [0, 1] for the optional impact PBRS.

        The speed INTO the look-ahead landing surface relative to that surface's
        survivable-impact threshold (ninja.py: death when impact_vel >
        MAX_SURVIVABLE_IMPACT - 4/3*|floor_normalized_y|), zero below a safe
        fraction of the threshold and weighted by proximity (1 - dist). 0 = safe,
        1 = imminent fatal impact. Steeper landing surfaces (smaller |ny|) tolerate
        higher impact, so the threshold rises with surface tilt.

        v11: the safe fraction was 0.5 (risk turned positive at HALF the survivable
        landing speed), so normal COMMITTED-but-survivable jumps onto small
        platforms were taxed by the impact PBRS -> the agent decelerated before
        landing = under-committed jumps. Raised to _impact_risk_safe_fraction
        (default 0.8) so only NEAR-LETHAL impacts (>=80% of threshold) count; the
        anti-fatal-impact signal (the ~95%-impact-death failure mode) is preserved
        in the truly dangerous last band, but survivable commits read risk 0.
        """
        safe_frac = float(getattr(self, "_impact_risk_safe_fraction", 0.8))
        impact_into = -(ln_x * vx + ln_y * vy)
        threshold = MAX_SURVIVABLE_IMPACT - (4.0 / 3.0) * abs(ln_y)
        safe = safe_frac * threshold
        if impact_into <= safe or threshold <= safe:
            return 0.0
        risk = min((impact_into - safe) / (threshold - safe), 1.0) * (1.0 - ln_dist)
        return float(max(0.0, min(1.0, risk)))

    def _get_trajectory_projector(self):
        """Ghost-arc projector for the current level (Audit II P0-1/§2).

        Level geometry loads once per level (keyed on the map-data object
        identity, like the tile-grid cache); the DEADLY mine set refreshes on
        the same cheap count signature the hazard map uses (mines toggle
        within an episode). Returns None when the native build predates
        TrajectoryProjector or no map is loaded — callers fall back to the
        grounded fast-path fill (zeros), never raise.
        """
        if TrajectoryProjector is None:
            return None
        map_data = getattr(self.nplay_headless, "current_map_data", None)
        if map_data is None:
            return None
        if getattr(self, "_ghost_arc_map_ref", None) is not map_data:
            try:
                self._ghost_arc_projector = TrajectoryProjector.load(
                    [int(v) for v in map_data]
                )
            except Exception:
                logger.warning(
                    "TrajectoryProjector.load failed; ghost-arc obs will be "
                    "zeros for this level.", exc_info=True,
                )
                self._ghost_arc_projector = None
            self._ghost_arc_map_ref = map_data
            self._ghost_arc_mine_sig = None
        proj = getattr(self, "_ghost_arc_projector", None)
        if proj is None:
            return None
        toggle_mines, toggled_mines, sig = self._mine_lists_and_signature()
        if sig != getattr(self, "_ghost_arc_mine_sig", None):
            deadly, _safe = classify_mines_by_state(
                (*toggle_mines, *toggled_mines)
            )
            proj.set_deadly_mines(
                [float(m.xpos) for m in deadly],
                [float(m.ypos) for m in deadly],
            )
            self._ghost_arc_mine_sig = sig
        # v16 (feature C): point the arc->goal features at the ACTIVE objective —
        # the exit switch pre-activation, the exit door after. Refreshed every
        # call (cheap: stores one tuple) so the phase flip is tracked mid-episode.
        # set_goal predates a native rebuild guard: skip on an older build.
        if hasattr(proj, "set_goal"):
            try:
                if bool(self.nplay_headless.exit_switch_activated()):
                    gx, gy = self.nplay_headless.exit_door_position()
                else:
                    gx, gy = self.nplay_headless.exit_switch_position()
                proj.set_goal(float(gx), float(gy))
            except Exception:
                proj.clear_goal()
        return proj

    def _fill_ghost_arc_bundle(self, _ninja, vx: float, vy: float) -> None:
        """Write the ghost-arc blocks into the game_state buffer.

        Committed block (27, GAME_STATE_GHOST_ARC_IDX) — airborne: three native
        arc projections (u = -1/0/+1) under the current jump-hold gravity
        (reduced gravity persists for the remaining jump frames while held,
        then falls — mirroring ninja.think's air branch). Grounded (or
        projector unavailable): fast-path self-contact fill — zeros with
        ny = -1 per arc so "standing on the contact" is readable.

        Jump-now block (27, GAME_STATE_JUMP_ARC_IDX, v13) — grounded: the
        counterfactual full-hold takeoff arcs. Flat-floor jump impulse exactly
        as ``Ninja.floor_jump`` applies it on flat ground (yspeed clamped to
        <= 0 then += JUMP_FLOOR_Y; position displaced by the impulse), then
        jf = MAX_JUMP_DURATION frames of reduced gravity. Zeros when airborne
        (the committed block covers the air).

        Tap-jump block (27, GAME_STATE_TAP_ARC_IDX, v14 B1) — grounded: the
        same takeoff impulse with jf = 0 (jump released immediately -> full
        gravity after the impulse). Under a low ceiling every full-hold arc
        slams the ceiling; this row makes the short hop visible. Zeros when
        airborne.

        Wall-jump-now block (27, GAME_STATE_WALL_ARC_IDX, v14) — airborne AND
        walled: the counterfactual regular wall-jump kick this frame
        (Ninja.wall_jump: xspeed zeroed if into the wall, downward yspeed
        zeroed, +1*wall_normal / -1.4 impulse, position displaced), full hold.
        Zeros when grounded or un-walled.

        Arc-phi block (6, GAME_STATE_ARC_PHI_IDX, v13) — per arc,
        tanh((Phi(ninja) - Phi(landing)) / ARC_PHI_NORM_COST) from the
        committed free-space/jump cost-to-completion source: positive = the
        arc's landing makes route progress. Mine/none arcs and non-finite
        costs emit 0.
        """
        i0 = GAME_STATE_GHOST_ARC_IDX
        j0 = GAME_STATE_JUMP_ARC_IDX
        t0 = GAME_STATE_TAP_ARC_IDX
        w0 = GAME_STATE_WALL_ARC_IDX
        p0 = GAME_STATE_ARC_PHI_IDX
        airborn = bool(getattr(_ninja, "airborn", False))
        proj = self._get_trajectory_projector()
        px, py = self.nplay_headless.ninja_position()

        bundle = None
        jump_bundle = None
        tap_bundle = None
        wall_bundle = None
        if proj is not None:
            try:
                if airborn:
                    # Reduced (jump) gravity remains only while state==3 with
                    # the jump input held; release or the 45-frame cap ends it.
                    jf = 0
                    if (
                        getattr(_ninja, "state", 4) == 3
                        and getattr(_ninja, "jump_input_old", 0) != 0
                    ):
                        jf = max(
                            0,
                            MAX_JUMP_DURATION
                            - int(getattr(_ninja, "jump_duration", 0)),
                        )
                    bundle = proj.project_bundle(
                        float(px), float(py), float(vx), float(vy),
                        int(jf), GHOST_ARC_MAX_FRAMES,
                    )
                    # Wall-jump-now row (v14): the counterfactual regular
                    # wall-jump kick, only while wall contact exists.
                    wn = float(getattr(_ninja, "wall_normal", 0.0) or 0.0)
                    if getattr(_ninja, "walled", False) and wn != 0.0:
                        vxw = 0.0 if float(vx) * wn < 0 else float(vx)
                        vyw = min(float(vy), 0.0)
                        vxw += JUMP_WALL_REGULAR_X_MULTIPLIER * wn
                        vyw += JUMP_WALL_REGULAR_Y
                        wall_bundle = proj.project_bundle(
                            float(px) + JUMP_WALL_REGULAR_X_MULTIPLIER * wn,
                            float(py) + JUMP_WALL_REGULAR_Y,
                            vxw, vyw,
                            int(MAX_JUMP_DURATION), GHOST_ARC_MAX_FRAMES,
                        )
                else:
                    # Counterfactual takeoff: flat-floor impulse (Ninja.floor_jump
                    # flat branch), full hold. Slope takeoffs use the same flat
                    # model — an approximation documented in constants.py.
                    vyj = min(float(vy), 0.0) + JUMP_FLOOR_Y
                    jump_bundle = proj.project_bundle(
                        float(px), float(py) + JUMP_FLOOR_Y, float(vx), vyj,
                        int(MAX_JUMP_DURATION), GHOST_ARC_MAX_FRAMES,
                    )
                    # Tap row (v14 B1): identical impulse, jump released
                    # immediately (jf=0) -> the low-ceiling hop affordance.
                    tap_bundle = proj.project_bundle(
                        float(px), float(py) + JUMP_FLOOR_Y, float(vx), vyj,
                        0, GHOST_ARC_MAX_FRAMES,
                    )
            except Exception:
                bundle = jump_bundle = tap_bundle = wall_bundle = None

        # Doomed-arc flag (v13 arc-risk PBRS + FiGAR interrupt): airborne with
        # EVERY steering arc ending in a deadly mine or a lethal impact
        # (impact_ratio norm is ratio/2, so > 0.5 means speed > threshold) —
        # no constant input saves the flight. Stashed for the same-frame
        # obs["_arc_doomed"] / info["arc_doomed"] emits. A no-contact arc
        # (none_flag) is NOT doomed — the horizon just ran out.
        self._arc_doomed = False
        if airborn and bundle is not None and len(bundle) == GAME_STATE_GHOST_ARC_DIM:
            self._arc_doomed = all(
                bundle[b + 4] > 0.5
                or (bundle[b + 5] <= 0.5 and bundle[b + 7] > 0.5)
                for b in range(
                    0,
                    GAME_STATE_GHOST_ARC_DIM,
                    GAME_STATE_GHOST_ARC_PER_ARC,
                )
            )

        if bundle is not None and len(bundle) == GAME_STATE_GHOST_ARC_DIM:
            self._game_state_buffer[i0:i0 + GAME_STATE_GHOST_ARC_DIM] = bundle
        else:
            # Grounded / degraded fast-path: zeros + floor-normal self-contact.
            self._game_state_buffer[i0:i0 + GAME_STATE_GHOST_ARC_DIM] = (
                [0.0] * GAME_STATE_GHOST_ARC_DIM
            )
            for a in range(GAME_STATE_GHOST_ARC_ARCS):
                b = i0 + a * GAME_STATE_GHOST_ARC_PER_ARC
                self._game_state_buffer[b + 3] = -1.0  # ny = -1 (floor at 0 offset)
                # v15 surface_class (offset +11): +1 = floor, consistent with the
                # ny=-1 self-contact so grounded frames don't read as a wall.
                # vx/vy contact (offsets +9/+10) stay 0 = standing still.
                self._game_state_buffer[b + 11] = 1.0

        if jump_bundle is not None and len(jump_bundle) == GAME_STATE_JUMP_ARC_DIM:
            self._game_state_buffer[j0:j0 + GAME_STATE_JUMP_ARC_DIM] = jump_bundle
        else:
            self._game_state_buffer[j0:j0 + GAME_STATE_JUMP_ARC_DIM] = (
                [0.0] * GAME_STATE_JUMP_ARC_DIM
            )

        if tap_bundle is not None and len(tap_bundle) == GAME_STATE_TAP_ARC_DIM:
            self._game_state_buffer[t0:t0 + GAME_STATE_TAP_ARC_DIM] = tap_bundle
        else:
            self._game_state_buffer[t0:t0 + GAME_STATE_TAP_ARC_DIM] = (
                [0.0] * GAME_STATE_TAP_ARC_DIM
            )

        if wall_bundle is not None and len(wall_bundle) == GAME_STATE_WALL_ARC_DIM:
            self._game_state_buffer[w0:w0 + GAME_STATE_WALL_ARC_DIM] = wall_bundle
        else:
            self._game_state_buffer[w0:w0 + GAME_STATE_WALL_ARC_DIM] = (
                [0.0] * GAME_STATE_WALL_ARC_DIM
            )

        # Passive no-input arc (v16): the "if I release everything NOW" fall —
        # jf=0 (jump released) + u=0 (no steer), projected every airborne frame
        # regardless of the current jump-hold. Unlike the committed u=0 arc (which
        # inherits jf>0 while the jump is held), this is always the do-nothing
        # outcome, with an EXPLICIT fatal flag. Zeros when grounded. One extra
        # single-arc projection; reuses the native projector (no rebuild).
        pa0 = GAME_STATE_PASSIVE_ARC_IDX
        self._game_state_buffer[pa0:pa0 + GAME_STATE_PASSIVE_ARC_DIM] = (
            [0.0] * GAME_STATE_PASSIVE_ARC_DIM
        )
        if airborn and proj is not None:
            try:
                c = proj.project(
                    float(px), float(py), float(vx), float(vy),
                    0, 0, GHOST_ARC_MAX_FRAMES,
                )
                # (class, cx, cy, nx, ny, tau, impact, threshold, cw, vx, vy, mine_clr)
                cls = int(c[0])
                impact, thr = float(c[6]), float(c[7])
                cvx, cvy = float(c[9]), float(c[10])
                is_none = cls == 4
                # Fatal: deadly-mine contact, OR a surface landing whose speed
                # into the surface exceeds that surface's survivable threshold.
                fatal = (
                    cls == 3
                    or (cls in (0, 1, 2) and thr > 1e-9 and impact >= thr)
                )
                self._game_state_buffer[pa0 + 0] = max(
                    -1.0, min(1.0, (float(c[1]) - float(px)) / 500.0)
                )
                self._game_state_buffer[pa0 + 1] = max(
                    -1.0, min(1.0, (float(c[2]) - float(py)) / 500.0)
                )
                self._game_state_buffer[pa0 + 2] = max(
                    -1.0, min(1.0, cvx / MAX_HOR_SPEED)
                )
                self._game_state_buffer[pa0 + 3] = max(-1.0, min(1.0, cvy / 10.0))
                self._game_state_buffer[pa0 + 4] = 1.0 if fatal else 0.0
                self._game_state_buffer[pa0 + 5] = 1.0 if is_none else 0.0
            except Exception:
                pass

        # Arc route-progress deltas: committed arcs (airborne) fill slots 0-2,
        # jump-now arcs (grounded) fill slots 3-5; inactive slots stay 0.
        self._game_state_buffer[p0:p0 + GAME_STATE_ARC_PHI_DIM] = (
            [0.0] * GAME_STATE_ARC_PHI_DIM
        )
        active = ((bundle, 0) if bundle is not None else
                  (jump_bundle, GAME_STATE_GHOST_ARC_ARCS)
                  if jump_bundle is not None else None)
        if active is not None:
            arc_bundle, slot0 = active
            try:
                src = self._get_free_space_nav_source()
                if src is not None and getattr(src, "available", False):
                    sw = bool(self.nplay_headless.exit_switch_activated())
                    c_here = src.cost_to_completion(float(px), float(py), sw)
                    if math.isfinite(c_here):
                        for a in range(GAME_STATE_GHOST_ARC_ARCS):
                            b = a * GAME_STATE_GHOST_ARC_PER_ARC
                            if arc_bundle[b + 4] > 0.5 or arc_bundle[b + 5] > 0.5:
                                continue  # mine / no-contact arc: no progress
                            lx = float(px) + float(arc_bundle[b]) * 500.0
                            ly = float(py) + float(arc_bundle[b + 1]) * 500.0
                            c_land = src.cost_to_completion(lx, ly, sw)
                            if math.isfinite(c_land):
                                self._game_state_buffer[p0 + slot0 + a] = math.tanh(
                                    (c_here - c_land) / ARC_PHI_NORM_COST
                                )
            except Exception:
                pass

    def _compute_cost_to_go_map(self, switch_activated: bool) -> np.ndarray:
        """Phase-2 (O2): the free-space cost-to-go FIELD as a (3, W, H) spatial map.

        Channels: 0 = ACTIVE-phase cost-to-go (to switch pre-activation, else exit),
        1 = OTHER-phase cost-to-go (full switch->exit route visible at once), 2 =
        reachable mask. Reuses the per-level FreeSpaceDistanceSource already built
        for the nav hint (line ~943) — a cheap slice, no extra Dijkstra.

        Two deliberate deviations from a raw 6px clip:
          * SOFT normalization ``c = m/(m+K)`` instead of ``clip(m/K,0,1)``. The
            linear clip flat-lined every cell past K cost-steps at 1.0, so long
            routes (the FAR levels this channel exists for) lost all gradient at
            the far end. The rational form asymptotes to 1.0 but keeps a monotone
            gradient at any distance, and matches ``m/K`` for small m (short-route
            behavior unchanged). K stays a FIXED constant → cross-map comparability
            is preserved (unlike a per-episode D0, which skews the scale per level).
          * DOWNSAMPLE to a 12px grid (COST_TO_GO_MAP_DOWNSAMPLE). The encoder pools
            to a 3x3 readout anyway, so this is ~lossless for routing while cutting
            this (largest) observation's payload ~4x. Cost channels min-pool
            (keep the shortest/optimistic route through each block, ignoring the
            1.0 unreachable filler); the reachable mask max-pools (block is
            reachable if any sub-cell is).

        Audit II P0-5d: the output is a PURE function of (level, switch phase) —
        the free-space fields are per-level constants — yet this was rebuilt
        every frame at ~645 us (the dominant env-side obs cost). Cache the two
        phase outputs keyed on the SOURCE OBJECT IDENTITY (a new level builds a
        new FreeSpaceDistanceSource, so identity is an exact per-level key; same
        pattern as the level-keyed graph caches). Zero behavior change."""
        W, H = HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT
        K = float(getattr(self, "_cost_to_go_map_k", 1000.0))
        # v18 Component 3: normalization mode. "fixed_k" = legacy m/(m+K);
        # "per_level_dmax" = m/D_max (per level & phase field) so compact mazes
        # span [0,1] instead of collapsing to a near-flat gradient under K=1000;
        # "log" = log1p(m)/log1p(K), fixed-scale compressive middle ground.
        norm_mode = str(getattr(self, "_cost_map_norm", "fixed_k"))
        ds = COST_TO_GO_MAP_DOWNSAMPLE

        src_for_key = getattr(self, "_free_space_nav_source", None)
        cache = getattr(self, "_cost_to_go_map_cache", None)
        if cache is None:
            cache = self._cost_to_go_map_cache = {}
        cached = cache.get(bool(switch_activated))
        if cached is not None and cached[0] is src_for_key:
            return cached[1]

        def _downsample(full: np.ndarray) -> np.ndarray:
            if ds <= 1:
                return full
            W2, H2 = W // ds, H // ds
            r = full[:, : W2 * ds, : H2 * ds].reshape(3, W2, ds, H2, ds)
            pooled = np.empty((3, W2, H2), dtype=np.float32)
            pooled[0] = r[0].min(axis=(1, 3))  # cost: shortest route in block
            pooled[1] = r[1].min(axis=(1, 3))
            pooled[2] = r[2].max(axis=(1, 3))  # reachable: any sub-cell reachable
            return pooled

        src = src_for_key
        if src is None or not getattr(src, "available", False):
            # Lever 4a: the free-space flood could not build (the ~8% OPEN official
            # maps). Fall back to a straight-line (Euclidean) cost-to-go field so
            # the channel carries a monotone gradient toward the goal instead of a
            # FLAT all-unreachable map (zero routing signal). Registered identically
            # to the free-space map. NOT cached (the None source key is not a valid
            # per-level identity — different levels share it), so each level gets its
            # own straight-line field. Falls through to the flat map only when the
            # goal positions are unavailable.
            eucl = self._euclidean_cost_to_go_map(bool(switch_activated))
            if eucl is not None:
                return eucl
            out = np.ones((3, W, H), dtype=np.float32)
            out[2] = 0.0  # reachable mask: nothing reachable when no source
            result = _downsample(out)
            cache[bool(switch_activated)] = (src, result)
            return result
        active = getattr(src, "_d_exit" if switch_activated else "_d_switch", None)
        other = getattr(src, "_d_switch" if switch_activated else "_d_exit", None)
        out = np.ones((3, W, H), dtype=np.float32)
        out[2] = 0.0

        # v19 R4b: ego-residual normalization. per_level_dmax divides by the whole-
        # level cost max, which crushes the local step-to-step gradient near the
        # ninja on big mazes (the greedy-descent probe confirms the local field is
        # already hard to read). Instead re-reference every cell to the ninja's
        # CURRENT cost and scale by a FIXED window W_c: channel =
        # clip((cost(cell) - cost(ninja))/W_c, -1, 1). Cells farther from the goal
        # than the ninja read positive, closer read negative, contrast is
        # level-size-invariant. Because this depends on the ninja's position it
        # changes every frame — so we cache only the per-level RAW downsampled
        # fields (source-identity key, the expensive part) and apply the cheap
        # residual per frame, preserving the per-level cache throughput win.
        if norm_mode == "ego_residual":
            _off_e = int(TILE_PIXEL_SIZE // 6)
            raw_cache = getattr(self, "_cost_to_go_raw_cache", None)
            if raw_cache is None:
                raw_cache = self._cost_to_go_raw_cache = {}
            rc = raw_cache.get(bool(switch_activated))
            if rc is None or rc[0] is not src:
                raw = np.full((3, W, H), np.inf, dtype=np.float32)
                raw[2] = 0.0
                if active is not None:
                    m = np.asarray(active, dtype=np.float32)[
                        _off_e : _off_e + W, _off_e : _off_e + H
                    ]
                    fin = np.isfinite(m)
                    raw[0] = np.where(fin, m, np.inf)
                    raw[2] = fin.astype(np.float32)
                if other is not None:
                    m = np.asarray(other, dtype=np.float32)[
                        _off_e : _off_e + W, _off_e : _off_e + H
                    ]
                    fin = np.isfinite(m)
                    raw[1] = np.where(fin, m, np.inf)
                # Cost channels min-pool (shortest route through the block, inf
                # preserved only when the whole block is unreachable); reachable
                # max-pools — identical to _downsample.
                raw_ds = _downsample(raw)
                raw_cache[bool(switch_activated)] = (src, raw_ds)
                rc = raw_cache[bool(switch_activated)]
            raw_ds = rc[1]
            W2, H2 = raw_ds.shape[1], raw_ds.shape[2]
            win = float(getattr(self, "_cost_map_ego_window", 8.0))
            win = win if win > 1e-6 else 8.0
            npos = None
            try:
                npos = self.nplay_headless.ninja_position()
            except Exception:  # noqa: BLE001 - fall back to neutral map
                npos = None
            res = np.ones((3, W2, H2), dtype=np.float32)
            res[2] = raw_ds[2]
            if npos is not None:
                nx, ny = float(npos[0]), float(npos[1])
                ref_a = src.cost_at_nearest_free(nx, ny, bool(switch_activated))
                ref_o = src.cost_at_nearest_free(nx, ny, not bool(switch_activated))
                if np.isfinite(ref_a):
                    fa = np.isfinite(raw_ds[0])
                    res[0] = np.where(
                        fa, np.clip((raw_ds[0] - ref_a) / win, -1.0, 1.0), 1.0
                    )
                if np.isfinite(ref_o):
                    fo = np.isfinite(raw_ds[1])
                    res[1] = np.where(
                        fo, np.clip((raw_ds[1] - ref_o) / win, -1.0, 1.0), 1.0
                    )
            return res

        # The free-space field is laid over RAW sim px (cell i covers
        # [i*6, (i+1)*6)), which includes the sim's 1-tile solid border. Skip
        # the border's 4 cells (24px / 6px) so the slice covers exactly the
        # playable region — co-registered with hazard_map / tile_grid, which
        # both use the border-stripped playable frame.
        _off = int(TILE_PIXEL_SIZE // 6)

        def _slice(field):
            m = np.asarray(field, dtype=np.float32)[
                _off : _off + W, _off : _off + H
            ]  # (W,H) playable-frame 6px
            finite = np.isfinite(m)
            # Unreachable / non-finite cells stay at 1.0 in every mode (== the
            # m->inf asymptote of the fixed-K form, so consistent).
            mf = np.where(finite, m, 0.0)
            if norm_mode == "per_level_dmax":
                # Per-level, per-phase-field linear span: the farthest reachable
                # cell maps to 1.0 whatever the level scale, so a 60-step maze
                # gets the same gradient budget as a 3000-step FAR route. The
                # reachable-mask channel disambiguates the 1.0 boundary cell
                # from unreachable filler.
                dmax = float(mf.max()) if bool(finite.any()) else 0.0
                if dmax > 1e-6:
                    c = np.where(finite, mf / dmax, 1.0).astype(np.float32)
                else:
                    c = np.where(finite, 0.0, 1.0).astype(np.float32)
            elif norm_mode == "log":
                c = np.where(
                    finite,
                    np.minimum(np.log1p(mf) / math.log1p(K), 1.0),
                    1.0,
                ).astype(np.float32)
            else:  # "fixed_k" (legacy soft-saturating)
                c = np.where(finite, mf / (mf + K), 1.0).astype(np.float32)
            return c, finite

        if active is not None:
            c, finite = _slice(active)
            out[0] = c
            out[2] = finite.astype(np.float32)  # reachable = active goal reachable
        if other is not None:
            out[1] = _slice(other)[0]
        result = _downsample(out)
        cache[bool(switch_activated)] = (src, result)
        return result

    def _euclidean_cost_to_go_map(self, switch_activated: bool):
        """Lever 4a: a straight-line (Euclidean) cost-to-go FIELD for
        ``cost_to_go_map`` when the free-space source is unavailable (the OPEN
        official maps where the flood cannot build a finite field).

        Registered IDENTICALLY to the free-space map — same 6px grid over raw sim
        px, same 24px border strip (``_off``), same 12px downsample — so the channel
        stays co-registered with ``tile_grid`` / ``visitation_map``. Channels: 0 =
        active-phase straight-line cost, 1 = other-phase straight-line cost, 2 =
        reachable mask (1 everywhere: a straight-line estimate exists at every
        cell). Uses the fixed-K soft-saturating norm (``m/(m+K)``) so the gradient
        is monotone-decreasing toward the goal at any distance. Returns ``None``
        when the goal positions are unavailable so the caller keeps the flat map."""
        try:
            nh = self.nplay_headless
            sw = nh.exit_switch_position()
            ex = nh.exit_door_position()
        except Exception:
            return None
        if sw is None or ex is None:
            return None
        from ..value_function.free_space_distance import (
            _CELL as FS_CELL, _W as FS_W, _H as FS_H,
        )

        W, H = HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT
        K = float(getattr(self, "_cost_to_go_map_k", 1000.0))
        ds = COST_TO_GO_MAP_DOWNSAMPLE
        _off = int(TILE_PIXEL_SIZE // 6)
        # Free-space grid cell centers in raw sim px along each axis.
        xs = np.arange(FS_W, dtype=np.float32) * FS_CELL + FS_CELL / 2.0
        ys = np.arange(FS_H, dtype=np.float32) * FS_CELL + FS_CELL / 2.0
        active_goal = ex if switch_activated else sw
        other_goal = sw if switch_activated else ex

        def _field(goal):
            gx, gy = float(goal[0]), float(goal[1])
            dxs = xs[:, None] - gx
            dys = ys[None, :] - gy
            m = np.sqrt(dxs * dxs + dys * dys)  # (FS_W, FS_H) straight-line px
            m = m[_off : _off + W, _off : _off + H]
            return (m / (m + K)).astype(np.float32)

        out = np.ones((3, W, H), dtype=np.float32)
        out[0] = _field(active_goal)
        out[1] = _field(other_goal)
        out[2] = 1.0  # best-effort: a straight-line estimate exists everywhere
        if ds <= 1:
            return out
        W2, H2 = W // ds, H // ds
        r = out[:, : W2 * ds, : H2 * ds].reshape(3, W2, ds, H2, ds)
        pooled = np.empty((3, W2, H2), dtype=np.float32)
        pooled[0] = r[0].min(axis=(1, 3))  # cost: shortest straight-line in block
        pooled[1] = r[1].min(axis=(1, 3))
        pooled[2] = r[2].max(axis=(1, 3))
        return pooled

    # ------------------------------------------------------------------
    # v17 episodic visitation (count-based exploration + obs channel)
    # ------------------------------------------------------------------
    def _reset_visit_counts(self):
        """Zero the per-episode visit-count grid. Called every reset (both fast
        and slow paths) so the visitation signal is strictly episodic. Allocates
        lazily only when the feature is enabled (zero cost otherwise)."""
        self._visit_novelty = 0.0
        self._visit_new_cell = False
        self._visit_cells = 0
        # v26 W4: room discovery is strictly episodic — the whole point is that a
        # room is novel again next episode even though the level's partition is a
        # per-level constant.
        self._regions_visited = set()
        self._region_new = False
        self._region_bonus = 0.0
        self._region_coverage = 0.0
        self._region_count = 0
        self._visit_coverage = 0.0
        self._visitation_bonus = 0.0
        self._visit_directed_wt = 1.0
        # v18 C1: frontier shaping state is strictly episodic too.
        self._frontier_bonus = 0.0
        self._frontier_dist = -1.0
        self._frontier_hwm = float("inf")
        self._frontier_cells = 0
        self._frontier_frac = 0.0
        # v19 R4a: per-episode step counter (frames processed this episode) that
        # timestamps each cell's last visit for the recency channel. Reset here so
        # recency is strictly episodic like the counts.
        self._visit_step = 0
        if not getattr(self, "_visit_map_enabled", False):
            self._visit_counts = None
            self._visit_last = None
            return
        if self._visit_counts is None:
            self._visit_counts = np.zeros(
                (VISITATION_MAP_WIDTH, VISITATION_MAP_HEIGHT), dtype=np.int32
            )
        else:
            self._visit_counts.fill(0)
        # Last-visit step grid (0 = never visited this episode). Allocated only
        # when the recency channel is active to keep the legacy path zero-cost.
        if self._visit_recency_tau > 0.0:
            if getattr(self, "_visit_last", None) is None:
                self._visit_last = np.zeros(
                    (VISITATION_MAP_WIDTH, VISITATION_MAP_HEIGHT), dtype=np.int32
                )
            else:
                self._visit_last.fill(0)
        else:
            self._visit_last = None

    def _visit_cell_of(self, nx: float, ny: float):
        """Map raw sim px (x, y) to the (cx, cy) visitation cell, co-registered
        with cost_to_go_map (12px cells past the 24px border), clipped to grid."""
        cx = _cell_x(
            nx - VISITATION_ORIGIN_PX, VISITATION_CELL_PX, self._mirror_odd_dir
        )
        cy = int((ny - VISITATION_ORIGIN_PX) // VISITATION_CELL_PX)
        cx = 0 if cx < 0 else (VISITATION_MAP_WIDTH - 1 if cx >= VISITATION_MAP_WIDTH else cx)
        cy = 0 if cy < 0 else (VISITATION_MAP_HEIGHT - 1 if cy >= VISITATION_MAP_HEIGHT else cy)
        return cx, cy

    def _update_visit_counts(self, nx: float, ny: float):
        """Advance the per-episode visit grid for the ninja's current cell and
        recompute this step's novelty / coverage. Called once per real step from
        _pre_reward_hook, BEFORE the reward hook consumes _visit_novelty."""
        if not getattr(self, "_visit_map_enabled", False):
            return
        if self._visit_counts is None:
            self._reset_visit_counts()
        # v19 R4a: advance the episodic frame clock and stamp this cell's last
        # visit (1-based so 0 stays the "never visited" sentinel).
        self._visit_step += 1
        cx, cy = self._visit_cell_of(nx, ny)
        if getattr(self, "_visit_last", None) is not None:
            self._visit_last[cx, cy] = self._visit_step
        count_before = int(self._visit_counts[cx, cy])
        self._visit_counts[cx, cy] = count_before + 1
        count_after = count_before + 1
        # Count-based novelty: 1.0 on first visit, decaying on revisits.
        self._visit_novelty = 1.0 / (count_after ** 0.5)
        self._visit_new_cell = count_before == 0
        if self._visit_new_cell:
            self._visit_cells += 1
        reachable = self._visit_reachable_cell_count()
        self._visit_coverage = self._visit_cells / float(max(1, reachable))
        # P2: optional d*-directedness — scale novelty toward the active goal.
        weight = self._visit_directed_weight(nx, ny)
        self._visit_directed_wt = weight
        # The reward term for this step (added post-clawback in _modify_reward_hook).
        self._visitation_bonus = (
            float(self._visitation_bonus_coef) * self._visit_novelty * weight
        )

    def _visit_directed_weight(self, nx: float, ny: float) -> float:
        """P2 directed-exploration weight in [FLOOR, 1]: high near the active goal
        (low d*), low deep in dead-ends (high d*). Per-level normalized by the
        field's finite max (NOT the map's fixed K — maze d* costs are tiny vs K).
        Returns 1.0 when directedness is off or the source is unavailable."""
        if not getattr(self, "_visitation_directed", False):
            return 1.0
        src = getattr(self, "_free_space_nav_source", None)
        if src is None or not getattr(src, "available", False):
            return 1.0
        sw = bool(self.nplay_headless.exit_switch_activated())
        c = src.cost_at_nearest_free(nx, ny, sw)
        cmax = getattr(src, "_d_max_exit" if sw else "_d_max_switch", 0.0)
        if not np.isfinite(c) or cmax <= 1e-6:
            return 1.0
        norm = min(1.0, float(c) / float(cmax))
        return max(VISIT_DIRECTED_FLOOR, 1.0 - norm)

    def _visit_reachable_cell_count(self) -> int:
        """Number of reachable 12px cells this level (denominator for coverage).
        Derived from the free-space nav source's reachable field (same source
        cost_to_go_map slices), cached on source identity. Falls back to the full
        playable cell count when no source is available."""
        self._visit_reachable_mask()  # refresh the (src, count, mask) cache
        return getattr(self, "_visit_reachable_cache", (None, 0, None))[1]

    # ------------------------------------------------------------------
    # v26 W4 topological region novelty
    # ------------------------------------------------------------------
    def _region_labels(self) -> Tuple[Optional[np.ndarray], int]:
        """Room partition of the reachable space, computed once per level.

        Cached on the free-space nav source's identity, exactly like
        ``_visit_reachable_mask`` — a new level builds a new source.
        """
        src = getattr(self, "_free_space_nav_source", None)
        cached = getattr(self, "_region_label_cache", None)
        if cached is not None and cached[0] is src:
            return cached[1], cached[2]
        labels, n = label_regions(self._visit_reachable_mask())
        self._region_label_cache = (src, labels, n)
        return labels, n

    def _update_region_novelty(self, nx: float, ny: float) -> None:
        """Reward the FIRST entry to each room this episode.

        Cell-count novelty saturates in a maze — once every corridor has been
        walked the bonus is uniformly near zero, yet the agent still does not
        know which room to try next. A room stays novel until it is entered no
        matter how stale the corridor leading to it is, so this signal survives
        precisely where the cell-level one dies.

        Runs from ``_pre_reward_hook`` alongside ``_update_visit_counts`` so
        ``_region_bonus`` is ready before the reward hook consumes it.
        """
        self._region_new = False
        self._region_bonus = 0.0
        if self._region_novelty_coef <= 0.0:
            return
        labels, n_regions = self._region_labels()
        self._region_count = n_regions
        if labels is None or n_regions == 0:
            self._region_coverage = 0.0
            return
        cx, cy = self._visit_cell_of(nx, ny)
        rid = int(labels[cx, cy])
        # rid 0 means the ninja is on a cell the reachability field calls
        # unreachable (it disagrees with the sim near geometry edges). Not a
        # room, so not a discovery.
        if rid > 0 and rid not in self._regions_visited:
            self._regions_visited.add(rid)
            self._region_new = True
            self._region_bonus = float(self._region_novelty_coef)
        self._region_coverage = len(self._regions_visited) / float(n_regions)

    def _visit_reachable_mask(self) -> Optional[np.ndarray]:
        """Pooled (84, 46) bool reachable mask on the SAME 12px grid as the
        visit-count grid / cost_to_go_map (identical slice + max-pool as the
        map's reachable channel — co-registration by construction). Cached on
        free-space-source identity (a new level builds a new source). Returns
        None when no source is available (then no frontier can be computed and
        coverage falls back to the full playable cell count)."""
        src = getattr(self, "_free_space_nav_source", None)
        cached = getattr(self, "_visit_reachable_cache", None)
        if cached is not None and cached[0] is src and cached[1] > 0:
            return cached[2]
        n = VISITATION_MAP_WIDTH * VISITATION_MAP_HEIGHT  # fallback: all cells
        pooled = None
        if src is not None and getattr(src, "available", False):
            field = getattr(src, "_d_switch", None)
            if field is None:
                field = getattr(src, "_d_exit", None)
            if field is not None:
                _off = int(TILE_PIXEL_SIZE // 6)
                W6, H6 = HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT
                m = np.asarray(field, dtype=np.float32)[
                    _off : _off + W6, _off : _off + H6
                ]
                finite = np.isfinite(m)
                ds = COST_TO_GO_MAP_DOWNSAMPLE
                W2, H2 = W6 // ds, H6 // ds
                pooled = (
                    finite[: W2 * ds, : H2 * ds]
                    .reshape(W2, ds, H2, ds)
                    .any(axis=(1, 3))
                )
                n = max(1, int(pooled.sum()))
        self._visit_reachable_cache = (src, n, pooled)
        return pooled

    def _pbrs_spawn_cost_finite(self) -> bool:
        """Whether the active PBRS/cost-to-go source has a FINITE cost at spawn.

        With ``use_jump_potential`` the nav source is the physics-faithful jump
        field, which is inf at spawn on a minority of CHAMBER_MAZE levels (the
        free-space field is finite there). This surfaces that slice for the
        pbrs/spawn_cost_finite_frac tripwire. Cached on source identity (a
        per-level constant). Returns True when no source / no spawn is available
        so an unknown never falsely trips the wire."""
        src = getattr(self, "_free_space_nav_source", None)
        if src is None or not getattr(src, "available", False):
            return True
        cached = getattr(self, "_spawn_finite_cache", None)
        if cached is not None and cached[0] is src:
            return cached[1]
        val = True
        try:
            ld = getattr(self, "level_data", None)
            spawn = getattr(ld, "start_position", None) if ld is not None else None
            if spawn is None:
                spawn = self.nplay_headless.ninja_position()
            if spawn is not None and hasattr(src, "spawn_finite"):
                val = bool(src.spawn_finite(float(spawn[0]), float(spawn[1])))
        except Exception:  # noqa: BLE001 - telemetry must never break stepping
            val = True
        self._spawn_finite_cache = (src, val)
        return val

    def _compute_visitation_map(self) -> np.ndarray:
        """(1, W, H) visitation obs channel from the per-episode visit grid.

        Two modes (v19 R4a):
        * ``visitation_recency_tau == 0`` (legacy): COUNT-decay 1/sqrt(count) —
          value 1.0 on first visit, decaying with revisits, 0 where never visited.
          Encodes *how often* but not *when*.
        * ``visitation_recency_tau > 0``: TIME-decay exp(-(t - t_last_visit)/tau)
          — value ~1.0 at a just-visited cell, decaying toward 0 as the visit
          recedes into the past, 0 where never visited. Encodes corridor
          ORDER/recency so a memoryless feedforward policy can represent
          "try corridors in the order I last searched them".
        """
        out = np.zeros(
            (VISITATION_MAP_CHANNELS, VISITATION_MAP_WIDTH, VISITATION_MAP_HEIGHT),
            dtype=np.float32,
        )
        counts = self._visit_counts
        if counts is None:
            return out
        tau = getattr(self, "_visit_recency_tau", 0.0)
        last = getattr(self, "_visit_last", None)
        if tau > 0.0 and last is not None:
            visited = last > 0
            age = (self._visit_step - last[visited]).astype(np.float32)
            out[0][visited] = np.exp(-age / float(tau))
        else:
            visited = counts > 0
            out[0][visited] = 1.0 / np.sqrt(counts[visited].astype(np.float32))
        return out

    # ------------------------------------------------------------------
    # v18 C1 spatial frontier map (reachable AND unvisited) + shaping
    # ------------------------------------------------------------------
    def _frontier_grid(self) -> Optional[np.ndarray]:
        """(84, 46) bool frontier = reachable AND (episodic visit count == 0).

        Both operands live on the SAME 12px grid (the reachable mask is pooled
        with the exact slice cost_to_go_map uses; the visit grid is keyed by
        _visit_cell_of, whose co-registration is pinned by the v17 tests), so
        the conjunction is co-registered by construction. Returns None when no
        reachability source is available (no frontier can be defined)."""
        if self._visit_counts is None:
            return None
        mask = self._visit_reachable_mask()
        if mask is None:
            return None
        return mask & (self._visit_counts == 0)

    def _compute_frontier_map(self) -> np.ndarray:
        """(1, 84, 46) float32 frontier obs channel. Binary by default; with
        frontier_map_weighted each frontier cell is scaled by the local density
        of unexplored reachable space (7x7 box mean, floored at 0.25) so
        frontier bordering a LARGE unexplored region reads brighter than a
        1-cell nook. All-zero when no reachability source is available."""
        out = np.zeros(
            (FRONTIER_MAP_CHANNELS, FRONTIER_MAP_WIDTH, FRONTIER_MAP_HEIGHT),
            dtype=np.float32,
        )
        frontier = self._frontier_grid()
        if frontier is None or not frontier.any():
            return out
        if not self._frontier_map_weighted:
            out[0] = frontier.astype(np.float32)
            return out
        # Local unexplored-region density via an integral image (O(WH)).
        unexplored = frontier.astype(np.float32)  # frontier == unvisited∧reachable
        W, H = unexplored.shape
        r = 3  # 7x7 window
        ii = np.zeros((W + 1, H + 1), dtype=np.float32)
        ii[1:, 1:] = unexplored.cumsum(axis=0).cumsum(axis=1)
        xs = np.arange(W)
        ys = np.arange(H)
        x0 = np.clip(xs - r, 0, W)[:, None]
        x1 = np.clip(xs + r + 1, 0, W)[:, None]
        y0 = np.clip(ys - r, 0, H)[None, :]
        y1 = np.clip(ys + r + 1, 0, H)[None, :]
        area = ((x1 - x0) * (y1 - y0)).astype(np.float32)
        box = ii[x1, y1] - ii[x0, y1] - ii[x1, y0] + ii[x0, y0]
        density = box / np.maximum(area, 1.0)
        out[0] = np.where(
            frontier, np.maximum(0.25, density), 0.0
        ).astype(np.float32)
        return out

    def _update_frontier_bonus(self, nx: float, ny: float):
        """Advance the go-to-frontier HWM shaping. Called once per real step
        from _pre_reward_hook, AFTER _update_visit_counts (the frontier set and
        _visit_new_cell must reflect THIS step).

        Pays ``coef * (hwm - d)`` when the distance-to-nearest-frontier d (in
        12px cells, Euclidean) makes a new episode-epoch minimum, then ratchets
        hwm = d. The hwm RE-ARMS to the current d whenever a new cell was first
        visited this step (the frontier set changed), so consuming frontier
        never blocks the next approach and oscillation pays nothing (hwm only
        ratchets down within an epoch). Bounded per episode by
        coef * O(total approach path length)."""
        self._frontier_bonus = 0.0
        frontier = self._frontier_grid()
        reach_n = getattr(self, "_visit_reachable_cache", (None, 0, None))[1]
        if frontier is None:
            self._frontier_dist = -1.0
            self._frontier_cells = 0
            self._frontier_frac = 0.0
            self._frontier_hwm = float("inf")
            return
        fx, fy = np.nonzero(frontier)
        self._frontier_cells = int(fx.size)
        self._frontier_frac = self._frontier_cells / float(max(1, reach_n))
        if fx.size == 0:
            self._frontier_dist = -1.0
            self._frontier_hwm = float("inf")
            return
        cx, cy = self._visit_cell_of(nx, ny)
        d = float(
            np.sqrt(
                np.min(
                    (fx.astype(np.float64) - cx) ** 2
                    + (fy.astype(np.float64) - cy) ** 2
                )
            )
        )
        self._frontier_dist = d
        if self._frontier_approach_coef <= 0.0:
            return
        if self._visit_new_cell or not math.isfinite(self._frontier_hwm):
            # Frontier set changed (or fresh epoch): re-arm, no payment.
            self._frontier_hwm = d
            return
        if d < self._frontier_hwm - 1e-9:
            self._frontier_bonus = self._frontier_approach_coef * (
                self._frontier_hwm - d
            )
            self._frontier_hwm = d

    def _pre_reward_hook(self, curr_obs: Dict[str, Any], player_won: bool):
        """Advance the episodic visit grid once per real step, before the reward
        hook reads _visit_novelty. Position from the sim (same source as the info
        hook), so it is consistent with the obs channel filled this same step."""
        super()._pre_reward_hook(curr_obs, player_won)
        if getattr(self, "_visit_map_enabled", False):
            nx, ny = self.nplay_headless.ninja_position()
            self._update_visit_counts(nx, ny)
            if getattr(self, "_frontier_enabled", False):
                self._update_frontier_bonus(nx, ny)
        # v26 W4: room-level novelty. Deliberately NOT gated on
        # _visit_map_enabled — it needs only the reachable mask, not the visit
        # grid, and must be usable on its own for a clean A/B against the
        # cell-count bonus it is meant to outlive.
        if self._region_novelty_coef > 0.0:
            nx, ny = self.nplay_headless.ninja_position()
            self._update_region_novelty(nx, ny)

    def _modify_reward_hook(
        self,
        reward: float,
        curr_obs: Dict[str, Any],
        player_won: bool,
        terminated: bool,
    ) -> float:
        """Add the count-based visitation exploration bonus AFTER the base reward
        (and any base shaping), so it is not subject to PBRS death-clawback. Lands
        in the single-head reward stream that feeds GAE — the same path the RND
        intrinsic used, without RND's beta-schedule / goal-decay zeroing."""
        reward = super()._modify_reward_hook(reward, curr_obs, player_won, terminated)
        if self._visitation_bonus_coef > 0.0:
            reward = reward + float(getattr(self, "_visitation_bonus", 0.0))
        # v18 C1 go-to-frontier shaping (same post-clawback slot).
        if getattr(self, "_frontier_approach_coef", 0.0) > 0.0:
            reward = reward + float(getattr(self, "_frontier_bonus", 0.0))
        # v26 W4 room-discovery bonus (same post-clawback slot).
        if self._region_novelty_coef > 0.0:
            reward = reward + float(getattr(self, "_region_bonus", 0.0))
        return reward

    def _compute_ego_crop(self, nx: float, ny: float) -> np.ndarray:
        """Square (size, size) window of raw tile IDs centered on the ninja's tile.

        Sliced from the full-resolution ``tile_grid`` ((MAP_TILE_WIDTH,
        MAP_TILE_HEIGHT) = (W, H), indexed [tx, ty]) with the ninja tile at the
        exact center cell. The ninja px position maps to the playfield tile grid
        the same way the entity overlays do — ``tx = int(px/24) - 1`` (the sim's
        +1 tile solid border is stripped from tile_grid) — so the crop stays
        co-registered with tile_grid / entity_overlays / hazard_map. Cells that
        overhang a level edge are filled with EGO_CROP_PAD_TILE (solid), since the
        border is a wall, not free space.

        Axis order matches tile_grid exactly: output[cx, cy] with cx along W (x)
        and cy along H (y); NO transpose (the crop is a direct slice)."""
        size = int(getattr(self.config.reachability, "ego_crop_size",
                           EGO_CROP_SIZE_DEFAULT))
        half = size // 2

        grid = self._cached_tile_grid
        if grid is None:
            grid = np.full((MAP_TILE_WIDTH, MAP_TILE_HEIGHT), EGO_CROP_PAD_TILE,
                           dtype=np.int8)

        # Ninja tile in tile_grid coordinates (strip the sim's +1 border).
        tx = _cell_x(nx, TILE_PIXEL_SIZE, self._mirror_odd_dir) - 1
        ty = int(ny / TILE_PIXEL_SIZE) - 1

        out = np.full((size, size), EGO_CROP_PAD_TILE, dtype=np.int8)
        # Source window (may extend past the grid); clip to valid region and copy
        # into the matching sub-region of the (pre-padded) output.
        sx0, sy0 = tx - half, ty - half
        gx0, gy0 = max(0, sx0), max(0, sy0)
        gx1 = min(MAP_TILE_WIDTH, sx0 + size)
        gy1 = min(MAP_TILE_HEIGHT, sy0 + size)
        if gx1 > gx0 and gy1 > gy0:
            out[gx0 - sx0:gx1 - sx0, gy0 - sy0:gy1 - sy0] = grid[gx0:gx1, gy0:gy1]
        return out

    def _compute_ego_crop_mines(
        self, nx: float, ny: float, entity_overlays: np.ndarray
    ) -> np.ndarray:
        """(2, size, size) mine-density crop co-registered 1:1 with the tile crop.

        Channel 0 = deadly (toggled) mine density, channel 1 = safe (untoggled)
        mine density, sliced from ``entity_overlays`` ch3/ch4 (same (W, H) tile
        grid as the tile ego_crop) with the ninja tile at the exact center cell.
        Off-level cells are filled 0.0 (no mine there — the border is a wall, so a
        zero mine reading is correct; solidity is carried by the tile crop). This
        gives the policy a local, per-decision view of mine placement that the
        tile-only crop is blind to — the #1 cause of compositional mine deaths."""
        size = int(getattr(self.config.reachability, "ego_crop_size",
                           EGO_CROP_SIZE_DEFAULT))
        half = size // 2

        out = np.zeros((2, size, size), dtype=np.float32)
        if entity_overlays is None:
            return out
        deadly = entity_overlays[ENTITY_OVERLAY_CH_MINE_DEADLY]
        safe = entity_overlays[ENTITY_OVERLAY_CH_MINE_SAFE]

        # Same window / coordinate mapping as _compute_ego_crop (strip +1 border).
        tx = _cell_x(nx, TILE_PIXEL_SIZE, self._mirror_odd_dir) - 1
        ty = int(ny / TILE_PIXEL_SIZE) - 1
        sx0, sy0 = tx - half, ty - half
        gx0, gy0 = max(0, sx0), max(0, sy0)
        gx1 = min(MAP_TILE_WIDTH, sx0 + size)
        gy1 = min(MAP_TILE_HEIGHT, sy0 + size)
        if gx1 > gx0 and gy1 > gy0:
            dx0, dy0 = gx0 - sx0, gy0 - sy0
            dx1, dy1 = gx1 - sx0, gy1 - sy0
            out[0, dx0:dx1, dy0:dy1] = deadly[gx0:gx1, gy0:gy1]
            out[1, dx0:dx1, dy0:dy1] = safe[gx0:gx1, gy0:gy1]
        return out

    def _get_tile_route_gradient(self, switch_activated: bool) -> np.ndarray:
        """(2, MAP_TILE_WIDTH, MAP_TILE_HEIGHT) steepest-descent UNIT vectors of
        the ACTIVE-phase cost-to-go at tile resolution (v14 B2).

        Per tile, the field is the MIN of the nav source's 6px cost cells inside
        the tile (optimistic, matching cost_to_go_map's min-pool), and the
        gradient is the unit vector toward the lowest-cost finite 8-neighbor
        tile — (0, 0) where the tile is unreachable, all neighbors are >= (the
        goal tile), or no source exists. A PURE function of (level, phase), so
        cached on the source object identity exactly like _compute_cost_to_go_map
        (a new level builds a new nav source -> exact per-level key)."""
        src = getattr(self, "_free_space_nav_source", None)
        cache = getattr(self, "_tile_route_grad_cache", None)
        if cache is None:
            cache = self._tile_route_grad_cache = {}
        key = bool(switch_activated)
        hit = cache.get(key)
        if hit is not None and hit[0] is src:
            return hit[1]

        W, H = MAP_TILE_WIDTH, MAP_TILE_HEIGHT
        out = np.zeros((2, W, H), dtype=np.float32)
        if src is not None and getattr(src, "available", False):
            try:
                from ..value_function.free_space_distance import _CELL

                field = np.asarray(
                    src._d_switch if not key else src._d_exit, dtype=np.float64
                )
                # Field cells are indexed int(px / _CELL) in SIM px (border
                # included): playfield tile (ix, iy) spans sim px
                # [(ix+1)*24, (ix+2)*24) -> a (24/_CELL)^2 cell block.
                per = int(round(TILE_PIXEL_SIZE / _CELL))
                fx0, fy0 = per, per  # +1 border tile offset
                blk = field[fx0:fx0 + W * per, fy0:fy0 + H * per]
                blk = np.where(np.isfinite(blk), blk, np.inf)
                tile_cost = blk.reshape(W, per, H, per).min(axis=(1, 3))

                pad = np.full((W + 2, H + 2), np.inf)
                pad[1:-1, 1:-1] = tile_cost
                best = np.full((W, H), np.inf)
                bdx = np.zeros((W, H), dtype=np.float64)
                bdy = np.zeros((W, H), dtype=np.float64)
                with np.errstate(invalid="ignore"):
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            if dx == 0 and dy == 0:
                                continue
                            nb = pad[1 + dx:1 + dx + W, 1 + dy:1 + dy + H]
                            # diagonal steps cost sqrt(2): compare per unit length
                            step = math.hypot(dx, dy)
                            # inf - inf -> nan for unreachable cells; nan < best
                            # is False, so those cells are never taken.
                            score = (nb - tile_cost) / step
                            take = score < best
                            best = np.where(take, score, best)
                            bdx = np.where(take, dx / step, bdx)
                            bdy = np.where(take, dy / step, bdy)
                live = np.isfinite(tile_cost) & (best < -1e-9)
                out[0] = np.where(live, bdx, 0.0).astype(np.float32)
                out[1] = np.where(live, bdy, 0.0).astype(np.float32)
            except Exception:
                out[:] = 0.0

        cache[key] = (src, out)
        return out

    def _compute_ego_crop_route(
        self, nx: float, ny: float, switch_activated: bool
    ) -> np.ndarray:
        """(2, size, size) route-gradient crop co-registered 1:1 with ego_crop.

        Channels are the (ux, uy) steepest-descent unit vector of the
        active-phase cost-to-go at each tile in the ninja-centered window
        (v14 B2 — the local, self-localized read of the global cost map).
        Off-level cells are 0 (no route through the border wall)."""
        size = int(getattr(self.config.reachability, "ego_crop_size",
                           EGO_CROP_SIZE_DEFAULT))
        half = size // 2

        out = np.zeros((EGO_CROP_ROUTE_CHANNELS, size, size), dtype=np.float32)
        grad = self._get_tile_route_gradient(switch_activated)

        # Same window / coordinate mapping as _compute_ego_crop (strip +1 border).
        tx = _cell_x(nx, TILE_PIXEL_SIZE, self._mirror_odd_dir) - 1
        ty = int(ny / TILE_PIXEL_SIZE) - 1
        sx0, sy0 = tx - half, ty - half
        gx0, gy0 = max(0, sx0), max(0, sy0)
        gx1 = min(MAP_TILE_WIDTH, sx0 + size)
        gy1 = min(MAP_TILE_HEIGHT, sy0 + size)
        if gx1 > gx0 and gy1 > gy0:
            dx0, dy0 = gx0 - sx0, gy0 - sy0
            dx1, dy1 = gx1 - sx0, gy1 - sy0
            out[:, dx0:dx1, dy0:dy1] = grad[:, gx0:gx1, gy0:gy1]
        return out

    def _current_cost_to_go_norm(self):
        """The active-phase normalized cost-to-go [0,1] this step (0 == at goal),
        or None when unavailable. Reads the free-space nav hint refreshed earlier
        in _get_observation (index 2 of _cached_free_space_nav)."""
        nav = getattr(self, "_cached_free_space_nav", None)
        if nav is None or len(nav) < 3:
            return None
        c = float(nav[2])
        return c if c >= 0.0 else None

    def _compute_blocked_toward_goal(self, nx, ny, switch_activated, ninja) -> float:
        """v22 §3-B: scalar in [0,1]. 1 when the ninja is WALLED, the routed
        direction (grad of the cost-to-go field) points INTO that wall, AND
        cost-to-go is not improving -- the observable twin of the near-switch wall
        oscillation. 0 otherwise. Always advances the cost-to-go trend memory."""
        cur = self._current_cost_to_go_norm()
        prev = getattr(self, "_prev_cost_to_go", None)
        self._prev_cost_to_go = cur

        walled = bool(getattr(ninja, "walled", False))
        wn = float(getattr(ninja, "wall_normal", 0.0) or 0.0)
        if not walled or wn == 0.0:
            return 0.0

        grad = self._get_tile_route_gradient(switch_activated)
        tx = _cell_x(nx, TILE_PIXEL_SIZE, self._mirror_odd_dir) - 1
        ty = int(ny / TILE_PIXEL_SIZE) - 1
        rdx = 0.0
        if 0 <= tx < grad.shape[1] and 0 <= ty < grad.shape[2]:
            rdx = float(grad[0, tx, ty])
        # wall_normal is the OUTWARD (into-free-space) horizontal normal (+-1); the
        # route points INTO the wall when rdx has the opposite sign -> rdx*wn < 0.
        into_wall = max(0.0, -(rdx * wn))  # [0,1]
        if into_wall <= 0.0:
            return 0.0
        # Not improving cost-to-go this step (dithering, not progressing).
        stalled = 1.0
        if cur is not None and prev is not None:
            stalled = 1.0 if cur >= prev - 1e-4 else 0.0
        return float(np.clip(into_wall * stalled, 0.0, 1.0))

    # Horizontal lookahead for the wall-contact tau probe, in frames. 12 frames
    # at MAX_HOR_SPEED is ~40px -- long enough to telegraph an approaching wall a
    # few frames early (the walljump window), short enough that the scalar is
    # saturated at 1.0 almost everywhere and so carries little noise.
    WALL_TAU_LOOKAHEAD_FRAMES = 12.0

    def _compute_wall_contact_tau(self, nx, ny, vx, vy, ninja) -> float:
        """Normalized frames until this ninja runs into a WALL, in [0, 1].

        1.0 = no wall within the lookahead (the common case); 0.0 = in contact.

        Why this exists: ``time_to_contact_frames`` is computed from the LANDING
        surface normal for an AIRBORNE ninja, so a GROUNDED ninja running
        horizontally at a wall has no contact-time signal at all. That is exactly
        the state a corner walljump is entered from -- the maneuver requires
        pressing RIGHT+JUMP ~2 frames BEFORE contact while still moving left at
        MAX_HOR_SPEED. Without a tau that is a frame-perfect guess; with it, it is
        the far more learnable "when tau is small, jump".

        Uses the physics sweep so the answer respects real level geometry
        (slopes, thin walls) rather than a tile-grid approximation.
        """
        if abs(vx) < 0.5:
            # Not meaningfully moving horizontally -> nothing approaching.
            return 1.0
        # Tile march rather than the physics segment sweep: the NATIVE backend's
        # _SimProxy carries tile_dic but no segment_dic, and sweep_circle_vs_tiles
        # needs segments. Training runs on native, so a segment-based probe would
        # be dead exactly where it matters. tile_dic exists on both backends.
        #
        # Approximation: any non-empty tile blocks. Exact for full tiles (type 1),
        # conservative for slopes/half-tiles, which can read "wall" slightly early.
        # Acceptable because this scalar is a TIMING HINT, not physics -- the sim
        # remains the authority on what actually collides.
        from ..constants.physics_constants import NINJA_RADIUS

        sim = getattr(self.nplay_headless, "sim", None)
        tile_dic = getattr(sim, "tile_dic", None)
        if not tile_dic:
            return 1.0

        max_px = abs(float(vx)) * self.WALL_TAU_LOOKAHEAD_FRAMES
        if max_px <= 1e-6:
            return 1.0
        step = 1 if vx > 0 else -1
        tx = _cell_x(float(nx), TILE_PIXEL_SIZE, self._mirror_odd_dir)
        ty = int(float(ny) // TILE_PIXEL_SIZE)

        # Only need to march as far as the lookahead can reach.
        max_tiles = int(max_px // TILE_PIXEL_SIZE) + 2
        for k in range(1, max_tiles + 1):
            cx = tx + step * k
            # Missing key = outside the map = solid border.
            if tile_dic.get((cx, ty), 1) != 0:
                edge = cx * TILE_PIXEL_SIZE if step > 0 else (cx + 1) * TILE_PIXEL_SIZE
                gap = abs(edge - float(nx)) - NINJA_RADIUS
                if gap <= 0.0:
                    return 0.0
                if gap >= max_px:
                    return 1.0
                return float(np.clip(gap / max_px, 0.0, 1.0))
        return 1.0

    # Frames used to normalize the dwell/recency counters below. 30 frames is
    # half a second at 60fps -- long enough to cover any contact dwell that
    # matters for a maneuver, short enough that the common case uses the full range.
    PRECISION_STATE_DWELL_CAP = 30.0
    # Number of scalars appended by _compute_precision_state. Keep in sync with
    # the vector built there AND with the observation-space growth.
    PRECISION_STATE_DIM = 10

    # EXECUTION_PRECISION_PLAN §6 time/phase block. Keep in sync with
    # _compute_time_phase AND with the observation-space growth.
    TIME_PHASE_DIM = 6
    # Periods (frames) of the sinusoidal clock components. A bare t/T ramp moves
    # ~0.002 per frame over a ~485-frame route, which a network cannot key
    # frame-precise timing on; the sinusoids give the same clock at usable
    # resolution. 64 separates route SEGMENTS, 16 separates individual inputs
    # within a maneuver. Coprime-ish periods so (ramp, sin/cos pairs) stay a
    # unique code over the episode rather than repeating.
    TIME_PHASE_PERIODS = (64.0, 16.0)

    def _compute_time_phase(self) -> np.ndarray:
        """Frame index and switch phase, as a fixed-length feature block.

        Layout (TIME_PHASE_DIM = 6):
            0   t / time_phase_norm_frames, clipped to [0, 1]
            1   switch phase: 0.0 before the exit switch is hit, 1.0 after
            2-5 sin/cos of t at TIME_PHASE_PERIODS

        The switch phase is the disambiguator that mattered for the pace table:
        the route passes through the same cells outbound and returning, and only
        the switch bit tells those two visits apart.
        """
        try:
            t = float(self.nplay_headless.sim.frame)
        except Exception:  # noqa: BLE001
            t = 0.0
        norm = float(
            getattr(
                getattr(self.config, "reachability", None),
                "time_phase_norm_frames",
                600,
            )
        ) or 600.0
        feats = [min(1.0, max(0.0, t / norm))]
        # NOTE: read the live switch state, NOT self._switch_activation_frame --
        # that attribute is only ever assigned None, so it would pin this bit to
        # 0.0 forever and silently give us a clock with no phase.
        try:
            sw = bool(self.nplay_headless.exit_switch_activated())
        except Exception:  # noqa: BLE001
            sw = False
        feats.append(1.0 if sw else 0.0)
        for period in self.TIME_PHASE_PERIODS:
            ang = 2.0 * np.pi * t / period
            feats.append(float(np.sin(ang)))
            feats.append(float(np.cos(ang)))
        return np.array(feats, dtype=np.float32)

    # v26 W6c. Keep in sync with _compute_commitment_state.
    COMMITMENT_STATE_DIM = 6

    @staticmethod
    def _arc_fatal_and_margin(arc) -> Tuple[bool, float]:
        """Classify one 15-dim ghost arc as fatal, with a survivability margin.

        Mirrors the sim's two kill paths exactly, using the same fields
        ``arc_doomed`` keys on: ``[4]`` deadly-mine landing, ``[5]`` no-contact
        flag, ``[7]`` impact_ratio/2 (so > 0.5 means the normal impact speed
        exceeds that surface's survivable threshold).

        The margin is ``1 - impact_ratio`` in ``[0, 1]``: 1.0 = arrives at rest,
        0.0 = arrives exactly at the kill threshold. An arc that never contacts
        within the horizon is not a death, so it scores a full margin.
        """
        if arc[4] >= 0.5:
            return True, 0.0
        if arc[5] >= 0.5:
            return False, 1.0
        ratio = 2.0 * float(arc[7])
        return ratio > 1.0, float(np.clip(1.0 - ratio, 0.0, 1.0))

    def _compute_commitment_state(self, ninja) -> np.ndarray:
        """v26 W6c: "is every option from here fatal?" — the point of no return.

        Impact deaths are committed at TAKEOFF (V26 §1.8), and the per-arc
        fatality bits the policy needs to answer this are already in game_state —
        spread across five 3-arc bundles (committed / jump-now / tap / wall-jump)
        plus the passive fall, 45 dims each. Asking the policy to learn the
        CONJUNCTION across those rows, and to learn which rows are even active in
        the current contact regime, is asking it to rediscover the level's
        irreversibility structure from scratch. This block computes it directly.

        Only the rows valid for the current regime are counted: the others are
        zero-filled by ``_fill_ghost_arc_bundle`` and a zero arc reads as a
        perfectly safe landing, which would make ``no_escape`` unreachable.

        All six dims are polarized so that ZERO IS SAFE, matching
        ``mine_proximity``: an all-zero block is the genuine "nothing is
        committing me" reading, so the fallback for a missing block is correct
        rather than merely conventional.

          [0] air_trapped_frac     fraction of COMMITTED arcs that are fatal
          [1] takeoff_trapped_frac fraction of (full-hold + tap) jump arcs fatal
          [2] wall_trapped_frac    fraction of WALL-JUMP arcs that are fatal
          [3] passive_fatal        1.0 = the do-nothing fall kills
          [4] no_escape            1.0 = EVERY applicable arc is fatal
          [5] escape_deficit       1 - best margin over applicable arcs
        """
        out = np.zeros(self.COMMITMENT_STATE_DIM, dtype=np.float32)
        buf = self._game_state_buffer
        airborne = bool(getattr(ninja, "airborn", False))
        walled = bool(getattr(ninja, "walled", False))
        pa = GAME_STATE_GHOST_ARC_PER_ARC

        def _row(idx0):
            return [
                buf[idx0 + a * pa : idx0 + (a + 1) * pa]
                for a in range(GAME_STATE_GHOST_ARC_ARCS)
            ]

        # (slot, arcs) for every row that is actually live this frame.
        groups = []
        if airborne:
            groups.append((0, _row(GAME_STATE_GHOST_ARC_IDX)))
            if walled:
                groups.append((2, _row(GAME_STATE_WALL_ARC_IDX)))
        else:
            groups.append(
                (
                    1,
                    _row(GAME_STATE_JUMP_ARC_IDX) + _row(GAME_STATE_TAP_ARC_IDX),
                )
            )

        any_applicable = False
        any_safe = False
        best_margin = 0.0
        for slot, arcs in groups:
            if not arcs:
                continue
            fatals = 0
            for arc in arcs:
                fatal, margin = self._arc_fatal_and_margin(arc)
                fatals += int(fatal)
                if not fatal:
                    any_safe = True
                    best_margin = max(best_margin, margin)
            out[slot] = fatals / len(arcs)
            any_applicable = True

        # The passive fall is its own 6-dim block with an explicit fatal flag —
        # the one arc that is always available while airborne, since it is what
        # happens if the policy simply stops acting.
        if airborne:
            p0 = GAME_STATE_PASSIVE_ARC_IDX
            passive_fatal = buf[p0 + 4] >= 0.5
            out[3] = 1.0 if passive_fatal else 0.0
            any_applicable = True
            if not passive_fatal:
                any_safe = True
                best_margin = max(best_margin, 1.0 if buf[p0 + 5] >= 0.5 else 0.5)

        if any_applicable and not any_safe:
            out[4] = 1.0
        out[5] = float(np.clip(1.0 - best_margin, 0.0, 1.0)) if any_applicable else 0.0
        return out

    def _compute_precision_state(self, ninja) -> np.ndarray:
        """Ninja state a feed-forward policy needs for FRAME-PRECISE maneuvers.

        Ten scalars in [-1, 1], all read straight off the ninja (python + native
        both expose them). Each entry closes a specific, verified hole:

        0  wall_normal_buffered  The shipped ``wall_normal`` (game_state idx 16) is
           gated on ``walled``, so it reads 0 during the wall_buffer COYOTE window
           -- exactly the frames when a walljump is still legal after leaving the
           wall. The policy could see "a walljump is available" (idx 9) without
           seeing WHICH WAY it would launch. Retained here across the window.
        1  is_jumping           The state one-hot (idx 2-5) collapses Jumping(3)
           and Falling(4) into one bucket, so "still rising, can extend" is
           indistinguishable from "falling, jump over".
        2  jump_duration_norm   Shipped idx 10 is masked to 0 unless state==3, and
           state 3 is not itself observable -- so 0 is ambiguous. Unmasked here.
        3  airborn_old          Landing/takeoff transition edge.
        4  floor_dwell          consecutive_floor_frames (grounded dwell).
        5  wall_dwell           consecutive_wall_frames (wall-slide dwell).
        6  state_recency        frames since the last state change.
        7  state_norm           FULL-resolution state (0-9), not the 4 buckets.
        8  prev_state_norm      previous frame's state at full resolution.
        9  last_jump_buffered   whether the last jump consumed a buffer.
           (``facing`` was considered here and REJECTED: the native _NinjaProxy
           does not carry it, so exposing it would make the observation differ
           between the python and native backends -- a silent parity break worse
           than the missing feature.)

        Entries 3-6 and 8 are the quantities otherwise only recoverable by frame
        stacking; exposing them directly removes that dependency.
        """
        cap = self.PRECISION_STATE_DWELL_CAP

        # A walljump stays legal while walled OR while the wall coyote buffer is
        # live; ninja.wall_normal is never cleared, only overwritten on contact,
        # so it still holds the correct direction throughout that window.
        walled = bool(getattr(ninja, "walled", False))
        wall_buf_active = -1 < getattr(ninja, "wall_buffer", -1) < 5
        wn = float(getattr(ninja, "wall_normal", 0.0) or 0.0)
        wall_normal_buffered = wn if (walled or wall_buf_active) else 0.0

        state = int(getattr(ninja, "state", 0) or 0)
        prev_state = int(getattr(ninja, "previous_state", state) or 0)
        jump_duration = float(getattr(ninja, "jump_duration", 0.0) or 0.0)

        def _dwell(v):
            return float(np.clip(v / cap, 0.0, 1.0)) * 2.0 - 1.0

        vec = np.array([
            float(np.clip(wall_normal_buffered, -1.0, 1.0)),
            1.0 if state == 3 else -1.0,
            float(np.clip(jump_duration / max(MAX_JUMP_DURATION, 1), 0.0, 1.0)) * 2.0 - 1.0,
            1.0 if bool(getattr(ninja, "airborn_old", False)) else -1.0,
            _dwell(float(getattr(ninja, "consecutive_floor_frames", 0) or 0)),
            _dwell(float(getattr(ninja, "consecutive_wall_frames", 0) or 0)),
            _dwell(float(getattr(ninja, "state_change_frame", 0) or 0)),
            (state / 9.0) * 2.0 - 1.0,
            (prev_state / 9.0) * 2.0 - 1.0,
            1.0 if bool(getattr(ninja, "last_jump_was_buffered", False)) else -1.0,
        ], dtype=np.float32)
        return vec

    def _compute_ego_crop_route_fine(
        self, nx: float, ny: float, switch_activated: bool
    ) -> np.ndarray:
        """v22 §3-C: (2, S, S) steepest-descent unit vectors of the ACTIVE-phase
        6px UN-POOLED free-space cost-to-go field in a ninja-centered window
        (S = EGO_CROP_ROUTE_FINE_SIZE at 6px/cell). Unlike _compute_ego_crop_route
        (24px tile min-pool), this keeps the go-around vector crisp one 6px cell
        behind a thin wall. Zeros where solid / unreachable / at the goal / off-grid."""
        size = EGO_CROP_ROUTE_FINE_SIZE
        half = size // 2
        out = np.zeros(
            (EGO_CROP_ROUTE_FINE_CHANNELS, size, size), dtype=np.float32
        )
        src = getattr(self, "_free_space_nav_source", None)
        if src is None or not getattr(src, "available", False):
            return out
        try:
            from ..value_function.free_space_distance import _CELL

            field = np.asarray(
                src._d_exit if switch_activated else src._d_switch, dtype=np.float64
            )
            W, H = field.shape
            cx = _cell_x(nx, _CELL, self._mirror_odd_dir)
            cy = int(ny / _CELL)
            # Padded window (1-cell border for the 8-neighbor diff), inf outside.
            win = np.full((size + 2, size + 2), np.inf)
            gx0, gy0 = cx - half - 1, cy - half - 1
            sx0, sy0 = max(0, gx0), max(0, gy0)
            sx1, sy1 = min(W, gx0 + size + 2), min(H, gy0 + size + 2)
            if sx1 > sx0 and sy1 > sy0:
                sub = field[sx0:sx1, sy0:sy1]
                win[sx0 - gx0:sx1 - gx0, sy0 - gy0:sy1 - gy0] = np.where(
                    np.isfinite(sub), sub, np.inf
                )
            center = win[1:-1, 1:-1]
            # Vectorized over the 8 neighbours. The scalar loop this replaces was
            # the single hottest function in env.step() -- 6.2% of self time for
            # 0.9% of the observation's bytes -- because each of its 8 iterations
            # allocated six (size, size) float64 temporaries via np.where.
            #
            # Bit-exact with that loop. What actually makes it so:
            #  * _EGO_ROUTE_NEIGHBOUR_OFFSETS is in the loop's original (dx, dy)
            #    order, and argmin returns the FIRST minimum, matching the loop's
            #    strict `score < best` (first-wins on ties). This one IS
            #    load-bearing: reordering the offsets changes the emitted descent
            #    direction on 60/60 random tied fields.
            #  * The nan -> +inf map is DEFENSIVE, not required. inf - inf = nan
            #    arises only where `center` itself is infinite, and `live` already
            #    masks those cells to zero, so argmin selecting a nan cannot reach
            #    the output today. It is kept because the loop's `score < best`
            #    was False for nan and argmin's is not, so without it the two
            #    forms agree only by way of the `live` mask -- a coupling that a
            #    later change to `live` would break silently.
            offs = _EGO_ROUTE_NEIGHBOUR_OFFSETS
            steps = _EGO_ROUTE_NEIGHBOUR_STEPS
            with np.errstate(invalid="ignore"):
                nb = np.stack(
                    [win[1 + dx:1 + dx + size, 1 + dy:1 + dy + size]
                     for dx, dy in offs]
                )
                score = (nb - center) / steps
            score = np.where(np.isnan(score), np.inf, score)
            k = np.argmin(score, axis=0)
            best = np.take_along_axis(score, k[None], axis=0)[0]
            live = np.isfinite(center) & (best < -1e-9)
            out[0] = np.where(live, _EGO_ROUTE_UNIT_DX[k], 0.0).astype(np.float32)
            out[1] = np.where(live, _EGO_ROUTE_UNIT_DY[k], 0.0).astype(np.float32)
        except Exception:
            out[:] = 0.0
        return out

    def _compute_ego_crop_mines_fine(self, nx: float, ny: float) -> np.ndarray:
        """v24: (2, S, S) fine ego MINE-SDF crop (S = EGO_CROP_MINES_FINE_SIZE at
        6px/cell). Slices the 6px hazard_map (ch0 deadly, ch1 safe; signed distance
        clip((d-14)/24,-1,1)) in a ninja-centered window so the policy sees sub-tile
        mine geometry locally — the global hazard_map is average-pooled to a ~7-tile
        embedding before the policy sees it, so the lethal 14px disc is otherwise
        unresolvable. Off-grid cells fill +1.0 (max "safe": the border is a wall, no
        mine there). Co-registered with ego_crop_route_fine (same window, same 6px
        playable frame as tile_grid/entity_overlays/hazard_map)."""
        size = EGO_CROP_MINES_FINE_SIZE
        half = size // 2
        # Off-grid = far from any mine = +1.0 (the SDF "safe" asymptote).
        out = np.ones((EGO_CROP_MINES_FINE_CHANNELS, size, size), dtype=np.float32)
        try:
            hazard = self._compute_hazard_map()  # (2, HW, HH) 6px, playable frame
        except Exception:
            return out
        if hazard is None:
            return out
        C, W, H = hazard.shape
        # World px -> 6px playable-frame cell: strip the 1-tile (24px = 4 cells) sim
        # border, matching hazard_map's frame (co-registered with tile_grid).
        cell6 = TILE_PIXEL_SIZE / 4.0  # 6px
        border_cells = int(TILE_PIXEL_SIZE / cell6)  # 4
        hx = _cell_x(nx, cell6, self._mirror_odd_dir) - border_cells
        hy = int(ny / cell6) - border_cells
        sx0, sy0 = hx - half, hy - half
        gx0, gy0 = max(0, sx0), max(0, sy0)
        gx1, gy1 = min(W, sx0 + size), min(H, sy0 + size)
        if gx1 > gx0 and gy1 > gy0:
            dx0, dy0 = gx0 - sx0, gy0 - sy0
            dx1, dy1 = gx1 - sx0, gy1 - sy0
            nchan = min(EGO_CROP_MINES_FINE_CHANNELS, C)
            out[:nchan, dx0:dx1, dy0:dy1] = hazard[:nchan, gx0:gx1, gy0:gy1]
        return out

    def _get_fall_lethality_map(self) -> np.ndarray:
        """v26 W6a: the current level's fall-lethality field, computed once.

        Pure tile geometry, so it is cached for the lifetime of the level and
        invalidated on map change — same contract as ``_get_cached_tile_grid``.
        The cache is keyed on the map NAME alone (not on the array being
        non-None) so a level whose surface extraction fails is not retried every
        single step.

        On failure the field degrades to "solid ground everywhere, no drop"
        rather than raising: a missing hazard signal costs the agent a lever, a
        raised exception costs the rollout.
        """
        current_map_name = getattr(self.map_loader, "current_map_name", None)
        if current_map_name == self._fall_lethality_map_name:
            return self._fall_lethality_map

        try:
            tile_dict = self.nplay_headless.get_tile_data()
            # No mine_positions: a mine does not stop a fall, and splitting the
            # floor at one would punch a phantom bottomless hole in the field.
            field = compute_fall_lethality_map(
                tile_dict,
                HAZARD_MAP_WIDTH,
                HAZARD_MAP_HEIGHT,
                center_registered=self._mirror_equivariant_obs,
                canonical=self._mirror_equivariant_obs,
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "fall-lethality field could not be built for %s; the fall obs "
                "streams will read as 'no drop anywhere' this level.",
                current_map_name,
                exc_info=True,
            )
            field = np.empty(
                (FALL_LETHALITY_CHANNELS, HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT),
                dtype=np.float32,
            )
            field[0].fill(1.0)
            field[1].fill(0.0)
        self._fall_lethality_map = field
        self._fall_lethality_map_name = current_map_name
        return field

    def _threat_imminence(self, e, type_id: int) -> float:
        """Normalised telegraph countdown in [0, 1] for one shooter.

        Delegates to `threat_proximity_reference.threat_imminence`, which is the
        single Python source of truth for the mapping (the Rust engine is the
        production one, pinned against it by the parity test). Kept as a method
        because `test_threat_proximity.py` pins all 30 state-machine cases
        through it.
        """
        from .threat_proximity_reference import threat_imminence

        return threat_imminence(e, type_id)

    def _compute_threat_proximity(self, nx: float, ny: float) -> np.ndarray:
        """(THREAT_PROXIMITY_DIM,) per-super-class bearing, motion and telegraph.

        See constants.py for the layout and why this exists: the threat field
        cannot represent a hitscan beam early enough to be learnable, so the
        wind-up is exposed directly, and it cannot carry a moving hazard's
        velocity at all, so that is exposed here too.

        Within a class the NEAREST in-range member is selected and all eight dims
        describe that one entity. Selecting instead by highest imminence was
        considered and rejected: it makes the geometry and the countdown describe
        different entities on different frames, which is a signal that flickers
        between objects.

        **Computed in Rust when the native backend is present.** It must be: the
        native `_build_entity_dic` whitelists {1,2,3,4,6,7,11,21} and carries no
        threat type at all, so the pure-Python path — which reads `entity_dic` —
        returned an all-zero vector on the backend training actually uses. The
        Python twin below is the dev-tool/Python-backend path and is pinned
        against the Rust one by `test_threat_proximity_parity.py`.
        """
        native = getattr(self.nplay_headless, "_native_sim", None)
        if native is not None and hasattr(native, "threat_proximity"):
            raw = native.threat_proximity()
            # The engine reuses its buffer across calls, so this must be a copy:
            # handing the rollout buffer a view of memory the next tick
            # overwrites corrupts observations in a way that cannot be
            # attributed later.
            return np.array(
                np.frombuffer(raw, dtype=np.float32), dtype=np.float32, copy=True
            )
        from .threat_proximity_reference import compute_threat_proximity_reference

        return compute_threat_proximity_reference(self.nplay_headless, nx, ny)

    def _compute_mine_proximity(self, nx: float, ny: float) -> np.ndarray:
        """v25: (MINE_PROXIMITY_DIM,) bearing + closing-speed vector for the K
        nearest deadly mines.

        The v24 fine mine crop restored sub-tile mine *geometry*, but it is a
        still frame: it cannot distinguish drifting harmlessly past a mine from
        falling into one. That gap shows up exactly where it hurts — mine-death
        is ~2% on the easy procgen buckets but ~0.51 on OFFICIAL layouts, with
        only 0.52 of official episodes reaching the switch at all.

        Per mine, ninja-centered and all bounded to [-1, 1]:
          [0,1] dx_hat, dy_hat  unit bearing to the mine
          [2]   proximity       1.0 at the 14px kill disc -> 0.0 at/past range
          [3]   closing         v . bearing / speed norm; > 0 = closing
          [4]   ttc             1/(1 + t/TAU) while closing, else 0.0

        Mines are ordered nearest-first. Unused slots (fewer than K deadly mines,
        or mines past range) stay all-zero, which is also the no-threat reading,
        so a mine-free level is indistinguishable from a mine-far-away level —
        the correct semantics for a *threat* signal.
        """
        out = np.zeros(MINE_PROXIMITY_DIM, dtype=np.float32)
        arr = self._get_deadly_mine_xy()
        if arr.shape[0] == 0:
            return out
        dx = arr[:, 0] - float(nx)
        dy = arr[:, 1] - float(ny)
        dist = np.hypot(dx, dy)
        # Only mines inside the reaction window are threats; the rest stay zero
        # rather than filling the slots with far-away noise.
        in_range = dist <= MINE_PROXIMITY_RANGE_PX
        if not np.any(in_range):
            return out
        idx = np.flatnonzero(in_range)
        idx = idx[np.argsort(dist[idx])[:MINE_PROXIMITY_K]]

        vx, vy = self.nplay_headless.ninja_velocity()
        for slot, i in enumerate(idx):
            d = float(dist[i])
            # At d == 0 the bearing is undefined; the ninja is already inside the
            # kill disc, so report maximum threat with a zero bearing.
            if d > 1e-6:
                ux, uy = float(dx[i]) / d, float(dy[i]) / d
            else:
                ux = uy = 0.0
            # Surface distance to the kill disc, not center distance: 0 = contact.
            gap = max(0.0, d - MINE_PROXIMITY_KILL_RADIUS_PX)
            span = max(1e-6, MINE_PROXIMITY_RANGE_PX - MINE_PROXIMITY_KILL_RADIUS_PX)
            proximity = float(np.clip(1.0 - gap / span, 0.0, 1.0))
            # Positive = the velocity component points AT the mine.
            closing_px = float(vx) * ux + float(vy) * uy
            closing = float(
                np.clip(closing_px / MINE_PROXIMITY_SPEED_NORM_PX, -1.0, 1.0)
            )
            if closing_px > 1e-6:
                t_frames = gap / closing_px
                ttc = float(1.0 / (1.0 + t_frames / MINE_PROXIMITY_TTC_TAU_FRAMES))
            else:
                # Receding or moving tangentially: no contact on the current
                # heading, so no time-to-contact pressure.
                ttc = 0.0
            base = slot * MINE_PROXIMITY_FEATURES
            out[base : base + MINE_PROXIMITY_FEATURES] = (
                ux,
                uy,
                proximity,
                closing,
                ttc,
            )
        return out

    def _get_coarse_solid_grid(self) -> np.ndarray:
        """Per-level (21, 12) coarse solid-fraction grid (2x2-tile mean pool).

        A tile counts as solid when its type is non-empty (density feature —
        half/slope tiles count fully; fine geometry lives in the fine crop and
        the ghost arcs). Cached on the tile-grid object identity (rebuilt per
        level, like the tile-grid cache itself)."""
        grid = self._get_cached_tile_grid()
        cached = getattr(self, "_coarse_solid_cache", None)
        if cached is not None and cached[0] is grid:
            return cached[1]
        P = EGO_CROP_COARSE_POOL_TILES
        solid = (np.asarray(grid) != 0).astype(np.float32)  # (42, 23)
        W, H = solid.shape
        CW, CH = (W + P - 1) // P, (H + P - 1) // P
        padded = np.ones((CW * P, CH * P), dtype=np.float32)  # border = solid
        padded[:W, :H] = solid
        coarse = padded.reshape(CW, P, CH, P).mean(axis=(1, 3))
        self._coarse_solid_cache = (grid, coarse)
        return coarse

    def _get_coarse_route_grid(self, switch_activated: bool) -> np.ndarray:
        """P1: per-level (2, 21, 12) coarse grad(d*) route grid — the 2x2-tile
        mean-pool of the tile route gradient (a pure function of level+phase, so
        cached on the nav-source identity like _get_tile_route_gradient itself).
        Border pools to 0 (no route through the wall)."""
        src = getattr(self, "_free_space_nav_source", None)
        cache = getattr(self, "_coarse_route_cache", None)
        if cache is None:
            cache = self._coarse_route_cache = {}
        key = bool(switch_activated)
        hit = cache.get(key)
        if hit is not None and hit[0] is src:
            return hit[1]
        P = EGO_CROP_COARSE_POOL_TILES
        grad = self._get_tile_route_gradient(switch_activated)  # (2, 42, 23)
        _, W, H = grad.shape
        CW, CH = (W + P - 1) // P, (H + P - 1) // P
        out = np.zeros((EGO_CROP_COARSE_ROUTE_CHANNELS, CW, CH), dtype=np.float32)
        padded = np.zeros((EGO_CROP_COARSE_ROUTE_CHANNELS, CW * P, CH * P),
                          dtype=np.float32)
        padded[:, :W, :H] = grad[:EGO_CROP_COARSE_ROUTE_CHANNELS]
        out = padded.reshape(
            EGO_CROP_COARSE_ROUTE_CHANNELS, CW, P, CH, P
        ).mean(axis=(2, 4)).astype(np.float32)
        cache[key] = (src, out)
        return out

    def _compute_ego_crop_coarse(
        self, nx: float, ny: float, entity_overlays, switch_activated: bool,
        switch_x, switch_y, exit_x, exit_y,
    ) -> np.ndarray:
        """Audit II P1-6: (4, 21, 21) coarse far window centered on the ninja.

        Each cell pools a 2x2-tile block; the window spans +-21 tiles Chebyshev
        (full level height / half its width from anywhere) — localizing a
        1-2-tile target to +-1 tile at 20 tiles, ~7x better than the global
        readouts. Channels: 0 = solid fraction (out-of-bounds = 1.0, the
        border is a wall), 1/2 = deadly/safe mine density (mean-pooled from
        entity_overlays, so mid-episode toggles propagate), 3 = objective
        marker (1.0 = the ACTIVE objective's cell, 0.5 = the other's).
        Axis order matches ego_crop/tile_grid: out[c, cx, cy], no transpose."""
        size = EGO_CROP_COARSE_SIZE
        half = size // 2
        P = EGO_CROP_COARSE_POOL_TILES
        with_route = bool(getattr(self.config.reachability,
                                  "ego_crop_coarse_route", False))
        n_ch = EGO_CROP_COARSE_CHANNELS + (
            EGO_CROP_COARSE_ROUTE_CHANNELS if with_route else 0
        )

        out = np.zeros((n_ch, size, size), dtype=np.float32)
        out[0] = 1.0  # OOB default: solid (route channels default 0 = no route)

        coarse_solid = self._get_coarse_solid_grid()
        CW, CH = coarse_solid.shape

        tx = _cell_x(nx, TILE_PIXEL_SIZE, self._mirror_odd_dir) - 1
        ty = int(ny / TILE_PIXEL_SIZE) - 1
        ccx, ccy = tx // P, ty // P
        sx0, sy0 = ccx - half, ccy - half
        gx0, gy0 = max(0, sx0), max(0, sy0)
        gx1 = min(CW, sx0 + size)
        gy1 = min(CH, sy0 + size)
        if gx1 > gx0 and gy1 > gy0:
            dx0, dy0 = gx0 - sx0, gy0 - sy0
            dx1, dy1 = gx1 - sx0, gy1 - sy0
            out[0, dx0:dx1, dy0:dy1] = coarse_solid[gx0:gx1, gy0:gy1]
            if entity_overlays is not None:
                deadly = entity_overlays[ENTITY_OVERLAY_CH_MINE_DEADLY]
                safe = entity_overlays[ENTITY_OVERLAY_CH_MINE_SAFE]
                W, H = deadly.shape
                pad = np.zeros((CW * P, CH * P), dtype=np.float32)
                for ch_out, plane in ((1, deadly), (2, safe)):
                    pad[:] = 0.0
                    pad[:W, :H] = plane
                    pooled = pad.reshape(CW, P, CH, P).mean(axis=(1, 3))
                    out[ch_out, dx0:dx1, dy0:dy1] = pooled[gx0:gx1, gy0:gy1]
            # P1: coarse grad(d*) route channels (4, 5) windowed into the crop.
            if with_route:
                route = self._get_coarse_route_grid(switch_activated)  # (2, CW, CH)
                base = EGO_CROP_COARSE_CHANNELS
                out[base, dx0:dx1, dy0:dy1] = route[0, gx0:gx1, gy0:gy1]
                out[base + 1, dx0:dx1, dy0:dy1] = route[1, gx0:gx1, gy0:gy1]

        # Objective markers (active = 1.0, other = 0.5) in window coords.
        for (ox, oy), val in (
            ((switch_x, switch_y), 0.5 if switch_activated else 1.0),
            ((exit_x, exit_y), 1.0 if switch_activated else 0.5),
        ):
            if ox is None or oy is None:
                continue
            otx = int(float(ox) / TILE_PIXEL_SIZE) - 1
            oty = int(float(oy) / TILE_PIXEL_SIZE) - 1
            wx = otx // P - sx0
            wy = oty // P - sy0
            if 0 <= wx < size and 0 <= wy < size:
                out[3, wx, wy] = max(out[3, wx, wy], val)
        return out

    def _compute_hazard_map(self) -> np.ndarray:
        """NPP-46: 2-channel mine SDF at 6px (hazard_map key).

        Each channel is `sdf = clip((d - 14) / 24, -1, 1)` where `d` is the
        Euclidean distance from each 6px cell center to the nearest mine of that
        class. Positive outside the 14px kill zone, zero at the boundary, negative
        inside. Channels: ch0 = deadly (toggled) mines, ch1 = safe (untoggled)
        mines. Shape: (2, W, H) — same axis order as tile_grid / entity_overlays.

        Recomputes a channel only when its mine set changes vs the last call.
        Classification uses the shared ``classify_mines_by_state`` helper so the
        overlay and the hazard map never disagree after a mid-episode toggle.
        """
        toggle_mines, toggled_mines = self.nplay_headless.get_mine_entities()
        # Fast path: a cheap (#deadly, #safe) signature. type-21 (toggled) mines
        # are always deadly; type-1 (toggle) mines are deadly when state==0. On a
        # field that never toggles this is constant after step 1, so we skip the
        # full rebuild entirely. (A simultaneous opposite-toggle that keeps both
        # counts stable would leave a 1-step-stale hazard OBS — harmless, self-heals
        # next step.) MINE_STATE_DEADLY == 0.
        n_deadly = len(toggled_mines) + sum(
            1 for m in toggle_mines if getattr(m, "state", 1) == 0
        )
        n_safe = len(toggle_mines) + len(toggled_mines) - n_deadly
        sig = (len(toggle_mines), len(toggled_mines), n_deadly, n_safe)
        if sig == self._hazard_sig and self._hazard_map_cache is not None:
            return self._hazard_map_cache
        self._hazard_sig = sig

        deadly_mines, safe_mines = classify_mines_by_state(
            (*toggle_mines, *toggled_mines)
        )
        # Blind the hazard SDF to mines sealed away from reachable space.
        deadly_mines = self._filter_relevant_mines(deadly_mines)
        safe_mines = self._filter_relevant_mines(safe_mines)
        deadly = [(float(m.xpos), float(m.ypos)) for m in deadly_mines]
        safe = [(float(m.xpos), float(m.ypos)) for m in safe_mines]

        # Each Rust engine emits a row-major (H, W) flat buffer (or None when its
        # mine set is unchanged); transpose to (W, H) so both channels co-register
        # with tile_grid / entity_overlays, then stack.
        deadly_flat = self._mine_threat_map_deadly_rs.compute_hazard_map_if_changed(
            deadly
        )
        safe_flat = self._mine_threat_map_safe_rs.compute_hazard_map_if_changed(safe)
        if deadly_flat is not None:
            self._hazard_deadly_chan = (
                np.asarray(deadly_flat, dtype=np.float32)
                .reshape(HAZARD_MAP_HEIGHT, HAZARD_MAP_WIDTH)
                .T.copy()
            )
        if safe_flat is not None:
            self._hazard_safe_chan = (
                np.asarray(safe_flat, dtype=np.float32)
                .reshape(HAZARD_MAP_HEIGHT, HAZARD_MAP_WIDTH)
                .T.copy()
            )
        self._hazard_map_cache = np.stack(
            (self._hazard_deadly_chan, self._hazard_safe_chan)
        )
        return self._hazard_map_cache

    def _drone_sdf_field(self, zap, mini) -> Optional[np.ndarray]:
        """Merged (W, H) drone SDF for one set of positions, or None if empty.

        `sdf = clip((d - kill_radius) / 24, -1, 1)` where `d` is the distance
        from each 6px cell center to the nearest lethal drone — the same sign
        convention and normalization as `hazard_map`, so the two co-register and
        an encoder reads both in the same units.

        Zap (type 14) and mini (type 26) drones are evaluated by SEPARATE engines
        because their contact-death radii differ (17.5 vs 14 px), then merged by
        element-wise minimum: for a signed distance where lower means nearer
        death, `min` keeps the worst case and can never average a threat away.
        The same reasoning drives the min-pooling on the way to the policy.
        """
        field = None
        for positions, engine in (
            (zap, self._drone_sdf_zap_rs),
            (mini, self._drone_sdf_mini_rs),
        ):
            if not positions:
                continue
            # Rust emits row-major (H, W) as raw f32 bytes (a Python list would
            # cost ~20x the SDF itself); transpose to (W, H) so the channel
            # co-registers with tile_grid / entity_overlays / hazard_map.
            # frombuffer gives a read-only view — fine, it is only ever read or
            # copied into the writable cache below.
            chan = (
                np.frombuffer(engine.compute(positions), dtype=np.float32)
                .reshape(HAZARD_MAP_HEIGHT, HAZARD_MAP_WIDTH)
                .T
            )
            field = chan if field is None else np.minimum(field, chan)
        return field

    def _read_native_field(self, name: str, channels: int, *args) -> np.ndarray:
        """Read a native `(C, H, W)` f32 field and return it as `(C, W, H)`.

        The Rust engines emit row-major `(row, col)` per channel; every spatial
        observation in this env is `(W, H)` so they co-register with `tile_grid`
        and `entity_overlays` without a transpose at the encoder.

        The buffer is REUSED by the engine across calls, so the transpose result
        is copied -- handing the rollout buffer a view of memory the next tick
        overwrites would corrupt observations in a way that is nearly impossible
        to attribute later.

        On the pure-Python backend `threat_ttl` falls back to the numpy reference
        (`threat_ttl_reference`), which the parity test pins against the Rust
        engine; `solid_map` has no reference and returns zeros, which is the
        correct neutral value for an OCCUPANCY channel (no bodies) unlike the
        threat field's 1.0 (no danger).
        """
        native = getattr(self.nplay_headless, "_native_sim", None)
        if native is not None and hasattr(native, name):
            raw = getattr(native, name)(*args)
            flat = np.frombuffer(raw, dtype=np.float32)
            return np.ascontiguousarray(
                flat.reshape(channels, HAZARD_MAP_OBS_HEIGHT, HAZARD_MAP_OBS_WIDTH)
                .transpose(0, 2, 1)
            )
        if name == "threat_ttl":
            from .threat_ttl_reference import compute_threat_ttl_reference

            field = compute_threat_ttl_reference(self.nplay_headless, *args)
            return np.ascontiguousarray(field.transpose(0, 2, 1))
        # Unreachable: `solid_map` is rejected at construction on a non-native
        # backend precisely so this never silently returns an all-clear field.
        raise RuntimeError(
            f"no fallback for native field {name!r} on this backend"
        )

    def _compute_drone_hazard_map(self) -> np.ndarray:
        """Patrol-drone SDF now and at each forecast horizon (``drone_hazard_map``).

        Shape (1 + len(DRONE_HAZARD_HORIZONS), W, H): channel 0 is the present
        frame, channel `i+1` is the same field as it will be
        `DRONE_HAZARD_HORIZONS[i]` sim frames from now. Encoding is identical
        across channels — see :meth:`_drone_sdf_field`.

        The forecast channels are EXACT, not estimated. Zap and mini drones
        patrol a fixed grid with no ninja dependence, so their future positions
        are a lookup rather than a prediction (the roll-out shares the
        simulator's own motion code; see ``Drone::forecast``). This is what a
        present-frame-only hazard map cannot express: whether the drone in the
        corridor ahead is walking towards the ninja or away from it, which is the
        difference between "go now" and "wait", from an identical frame.

        Unlike the mine map there is NO change-gate. Mines only enter or leave
        the deadly set when a `state` toggles, so their SDF is near-static;
        drones move every frame, so a gate would miss every time and only cost a
        position compare. An engine is skipped entirely when its class is absent
        from the level, which is the common case (most maps carry one or neither).
        """
        zap, mini = self.nplay_headless.get_drone_positions()

        # No drones at all: the all-1.0 initial cache is already correct
        # ("no hazard anywhere") for every channel, and matches the mine map's
        # saturation value. Also skips the forecast entirely, which is the
        # common case — most levels carry no patrol drones.
        if not zap and not mini:
            self._drone_hazard_cache.fill(1.0)
            return self._drone_hazard_cache

        self._drone_hazard_cache[0] = self._drone_sdf_field(zap, mini)

        forecast = self.nplay_headless.get_predicted_drone_positions(
            DRONE_HAZARD_HORIZONS
        )
        for i, (fzap, fmini) in enumerate(forecast):
            field = self._drone_sdf_field(fzap, fmini)
            # A drone can only vanish from the forecast by deactivating, in which
            # case "no hazard" (1.0) is the right answer for that horizon.
            self._drone_hazard_cache[i + 1] = 1.0 if field is None else field
        return self._drone_hazard_cache

    # Normalizer for the scalar nearest-deadly-mine feature: 7 tiles. Near-mine
    # resolution is fine; threats beyond this saturate to 1.0 (effectively "safe").
    _NEAREST_MINE_NORM_PX = 168.0

    def _mine_lists_and_signature(self):
        """``(toggle_mines, toggled_mines, sig)`` for the current sim frame.

        ``sig`` is ``(n_toggle, n_toggled, n_deadly)`` — the cache key several
        observation features use to decide whether the deadly set has changed.
        Counting the deadly type-1 mines is O(mines) and three call sites wanted
        the same number every step, so it is computed once per frame here.

        Memoized on the sim frame counter, which advances on every tick, so a
        toggle can never be missed: a new frame recomputes.
        """
        frame = getattr(self.nplay_headless.sim, "frame", None)
        cached = self._mine_sig_cache
        if cached is not None and frame is not None and frame == self._mine_sig_frame:
            return cached
        toggle_mines, toggled_mines = self.nplay_headless.get_mine_entities()
        n_deadly = len(toggled_mines) + sum(
            1 for m in toggle_mines if getattr(m, "state", 1) == 0
        )
        result = (
            toggle_mines,
            toggled_mines,
            (len(toggle_mines), len(toggled_mines), n_deadly),
        )
        self._mine_sig_frame = frame
        self._mine_sig_cache = result
        return result

    def _get_deadly_mine_xy(self) -> np.ndarray:
        """(N, 2) array of deadly-mine pixel centers, cached by mine signature.

        Recomputed only when the deadly set changes (a toggle / level change);
        otherwise the per-step nearest-mine read is just a vectorized min."""
        toggle_mines, toggled_mines, sig = self._mine_lists_and_signature()
        if sig == self._deadly_xy_sig and self._deadly_mine_xy is not None:
            return self._deadly_mine_xy
        self._deadly_xy_sig = sig
        pts = [
            (float(m.xpos), float(m.ypos))
            for m in self._filter_relevant_mines([*toggle_mines, *toggled_mines])
            if getattr(m, "state", 1) == 0  # STATE_TOGGLED = 0, deadly
        ]
        self._deadly_mine_xy = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return self._deadly_mine_xy

    def _compute_nearest_deadly_mine_distance_norm(self) -> float:
        """Normalized [0, 1] distance to the nearest deadly mine (1.0 = none /
        beyond the local normalizer, 0.0 = at the ninja center)."""
        arr = self._get_deadly_mine_xy()
        if arr.shape[0] == 0:
            return 1.0
        ninja = self.nplay_headless.sim.ninja
        d2 = ((arr[:, 0] - ninja.xpos) ** 2 + (arr[:, 1] - ninja.ypos) ** 2).min()
        return float(min((d2 ** 0.5) / self._NEAREST_MINE_NORM_PX, 1.0))

    def _compute_nearest_deadly_mine_distance_px(self) -> float:
        """Ungated Euclidean px distance to the nearest deadly mine (inf = none).

        Velocity-independent companion to the approaching-only distance; used by
        the FiGAR airborne mine interrupt. Vectorized over the cached deadly set."""
        arr = self._get_deadly_mine_xy()
        if arr.shape[0] == 0:
            return float("inf")
        ninja = self.nplay_headless.sim.ninja
        d2 = ((arr[:, 0] - ninja.xpos) ** 2 + (arr[:, 1] - ninja.ypos) ** 2).min()
        return float(d2 ** 0.5)

    def _compute_nearest_approaching_deadly_mine_distance(self) -> float:
        """Distance in pixels to the nearest deadly mine the ninja is approaching.

        Uses dot(velocity, mine_direction) > 0 as the approach filter — identical
        to the velocity_dot_direction feature in spatial_context.py. Only mines in
        STATE_TOGGLED (state == 0) are considered deadly.

        Returns:
            Euclidean distance to the nearest approaching deadly mine, or
            float('inf') if no deadly mine is currently being approached.
        """
        ninja = self.nplay_headless.sim.ninja
        ninja_x, ninja_y = ninja.xpos, ninja.ypos
        vx, vy = self.nplay_headless.ninja_velocity()
        toggle_mines, toggled_mines = self.nplay_headless.get_mine_entities()

        min_dist = float("inf")
        for mine in self._filter_relevant_mines([*toggle_mines, *toggled_mines]):
            if getattr(mine, "state", 1) != 0:  # STATE_TOGGLED = 0, deadly
                continue
            dx = mine.xpos - ninja_x
            dy = mine.ypos - ninja_y
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1e-12:
                continue
            # Positive dot product means velocity has a component toward the mine
            if (vx * dx + vy * dy) > 0:
                dist = dist_sq**0.5
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    # Grounded takeoff-arc (bundle) index -> jump action id. project_bundle
    # emits arcs in u order [-1, 0, +1]; _START_ACT maps (h, jump=1):
    # u=-1 -> action 5 (jump+left), u=0 -> action 3 (jump), u=+1 -> action 4.
    _JUMP_ARC_ACTION = {0: 5, 1: 3, 2: 4}

    # Half-width (px) of the fall-field window the Layer-3 impact gate inspects.
    # The longest duration bin is 64 frames; free fall covers ~138 px in that
    # time and the horizontal reach at MAX_HOR_SPEED is comparable, so a ~7-tile
    # box around the ninja bounds anywhere a single macro can put it.
    _FALL_RISK_PROBE_PX = 168.0

    def _fall_risk_nearby(self) -> bool:
        """Is there any fatal drop within reach of one macro from here?

        Layer-3 impact masking removes the "no deadly mines -> nothing to check"
        early-out, so without a gate every decision point on a mine-free level
        would pay a full forward simulation. The W6a fall field is already
        cached and static, so answering this is a slice and a comparison.

        Fails OPEN (returns True) whenever the field is unavailable — a missed
        early-out costs compute, a missed mask costs a death.
        """
        if not getattr(self.config, "layer3_mask_lethal_impact", False):
            return False
        try:
            field = self._get_fall_lethality_map()
            nx, ny = self.nplay_headless.ninja_position()
        except Exception:
            return True
        if field is None:
            return True
        half = int(self._FALL_RISK_PROBE_PX / FALL_CELL_PX)
        cx = int(nx / FALL_CELL_PX) - FALL_BORDER_CELLS
        cy = int(ny / FALL_CELL_PX) - FALL_BORDER_CELLS
        _, W, H = field.shape
        x0, x1 = max(0, cx - half), min(W, cx + half + 1)
        y0, y1 = max(0, cy - half), min(H, cy + half + 1)
        if x1 <= x0 or y1 <= y0:
            return True
        return bool((field[0, x0:x1, y0:y1] < 0.0).any())

    def _apply_takeoff_arc_gap_mask(self, action_mask, duration_mask):
        """v16 PRECISION: veto a jump whose takeoff arc falls into a mine.

        When GROUNDED, mask a jump direction iff BOTH its full-hold and tap
        counterfactual takeoff arcs land on a deadly mine — i.e. NO jump hold
        clears the gap at the current speed, so the correct move is to keep
        running and build speed. Requiring both hold extremes fatal cleanly
        separates the SPEED problem (mask -> build speed) from the hold-TIMING
        problem (leave jumping available so the duration head can pick the hold
        that reaches the far platform). Never masks the last action or the last
        grounded (non-jump) option, so the agent is never trapped."""
        if not getattr(self.config, "takeoff_arc_gap_mask", False):
            return action_mask, duration_mask
        ninja = getattr(getattr(self.nplay_headless, "sim", None), "ninja", None)
        if ninja is None or bool(getattr(ninja, "airborn", False)):
            return action_mask, duration_mask  # takeoff decisions only
        proj = self._get_trajectory_projector()
        if proj is None:
            return action_mask, duration_mask
        try:
            px, py = self.nplay_headless.ninja_position()
            vx, vy = self.nplay_headless.ninja_velocity()
            # SAME flat-floor takeoff impulse as the grounded ghost-arc bundle.
            vyj = min(float(vy), 0.0) + JUMP_FLOOR_Y
            full = proj.project_bundle(
                float(px), float(py) + JUMP_FLOOR_Y, float(vx), vyj,
                int(MAX_JUMP_DURATION), GHOST_ARC_MAX_FRAMES,
            )
            tap = proj.project_bundle(
                float(px), float(py) + JUMP_FLOOR_Y, float(vx), vyj,
                0, GHOST_ARC_MAX_FRAMES,
            )
        except Exception:
            return action_mask, duration_mask
        pa = GAME_STATE_GHOST_ARC_PER_ARC
        # v26 W6b: the arc's fatality test. Historically mine-landing only
        # ([4] == 1) — the takeoff-commitment machinery existed and was keyed on
        # the wrong hazard, so a jump that commits to a fatal FALL was never
        # vetoed even though ~40% of losses are exactly that. With
        # ``takeoff_arc_lethal_impact`` an arc also counts as fatal when it makes
        # contact ([5] none_flag == 0) at a lethal normal speed ([7] carries
        # impact_ratio/2, so > 0.5 means the impact exceeds the surface's
        # survivable threshold — the same predicate `arc_doomed` already uses).
        _impact = bool(getattr(self.config, "takeoff_arc_lethal_impact", False))

        def _arc_fatal(bundle, arc_i: int) -> bool:
            b = arc_i * pa
            if (b + 4) >= len(bundle):
                return False
            if bundle[b + 4] >= 0.5:
                return True
            return (
                _impact
                and (b + 7) < len(bundle)
                and bundle[b + 5] <= 0.5
                and bundle[b + 7] > 0.5
            )

        # Per-direction: fatal in BOTH the full-hold and tap extremes, i.e. no
        # hold clears it, so the answer is to build speed rather than re-time.
        fatal_actions = [
            act
            for arc_i, act in self._JUMP_ARC_ACTION.items()
            if _arc_fatal(full, arc_i) and _arc_fatal(tap, arc_i)
        ]
        if not fatal_actions:
            return action_mask, duration_mask
        am = np.asarray(action_mask).copy()
        # Guard: only veto when a grounded (non-jump) option survives, so the
        # agent can build speed rather than being forced into a fatal jump.
        if not any(am[a] for a in (0, 1, 2) if a < len(am)):
            return action_mask, duration_mask  # fail open
        for act in fatal_actions:
            if act < len(am):
                am[act] = 0
        if not am.any():
            return action_mask, duration_mask  # never all-masked
        if duration_mask is not None:
            dm = np.asarray(duration_mask).copy()
            for act in fatal_actions:
                if act < dm.shape[0]:
                    dm[act, :] = 0
            duration_mask = dm
        self._takeoff_gap_mask_count = (
            getattr(self, "_takeoff_gap_mask_count", 0) + len(fatal_actions)
        )
        return am, duration_mask

    def _get_action_and_duration_mask(
        self,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compose the legacy action mask with the Layer 3 mine veto."""
        action_mask, duration_mask = super()._get_action_and_duration_mask()
        # v16 takeoff-arc gap veto (applied before Layer 3 so it sees the reduced
        # set). No-op unless config.takeoff_arc_gap_mask.
        action_mask, duration_mask = self._apply_takeoff_arc_gap_mask(
            action_mask, duration_mask
        )
        if not self.enable_layer3_macro_action_mask or self.layer3_masker is None:
            return action_mask, duration_mask

        layer3_result = self.layer3_masker.compute(
            headless=self.nplay_headless,
            base_action_mask=action_mask,
            actions_to_execute=self._actions_to_execute,
            fall_risk_nearby=self._fall_risk_nearby(),
        )

        self._layer3_decision_points += 1
        self._layer3_masked_actions_total += layer3_result.masked_actions
        self._layer3_masked_candidates_total += layer3_result.masked_candidates
        self._layer3_mask_override_count += layer3_result.mask_overrides
        self._layer3_masked_by_mine_total += layer3_result.masked_by_mine
        self._layer3_masked_by_impact_total += layer3_result.masked_by_impact
        if layer3_result.all_masked_event:
            self._layer3_all_masked_events += 1
            if layer3_result.fallback_fatal:
                self._layer3_fallback_fatal_events += 1

        return layer3_result.action_mask, layer3_result.duration_mask

    def _get_adjacency_for_rewards(self) -> Optional[Dict]:
        """Get adjacency graph for reward calculation.

        Returns adjacency graph from current_graph_data if available.
        This is used by PBRS path-aware reward shaping.

        Returns:
            Adjacency dictionary, or None if not available
        """
        if self.current_graph_data is None:
            return None
        return self.current_graph_data.get("adjacency")

    def _extract_locked_door_positions(self, locked_door):
        """Extract switch and door positions from a locked door entity.

        Args:
            locked_door: Locked door entity

        Returns:
            Tuple of (switch_x, switch_y, door_x, door_y, switch_collected)
        """
        # Door position from segment (actual door location)
        door_segment = getattr(locked_door, "segment", None)
        if door_segment and hasattr(door_segment, "p1"):
            door_x = (door_segment.p1[0] + door_segment.p2[0]) / 2.0
            door_y = (door_segment.p1[1] + door_segment.p2[1]) / 2.0
        else:
            door_x = getattr(locked_door, "xpos", 0.0)
            door_y = getattr(locked_door, "ypos", 0.0)

        # Switch position (stored in sw_xpos, sw_ypos OR in xpos, ypos)
        switch_x = getattr(locked_door, "sw_xpos", getattr(locked_door, "xpos", 0.0))
        switch_y = getattr(locked_door, "sw_ypos", getattr(locked_door, "ypos", 0.0))

        # Switch collected state (active=True means switch not collected, active=False means collected)
        switch_collected = 0.0 if getattr(locked_door, "active", True) else 1.0

        return switch_x, switch_y, door_x, door_y, switch_collected

    def _build_switch_states_array(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Build switch states array with detailed locked door information.

        Format per door (5 features):
        - switch_x_norm: Normalized X position of switch (0-1)
        - switch_y_norm: Normalized Y position of switch (0-1)
        - door_x_norm: Normalized X position of door (0-1)
        - door_y_norm: Normalized Y position of door (0-1)
        - collected: 1.0 if switch collected (door open), 0.0 otherwise (door closed)

        Returns:
            Array of shape (SWITCH_STATES_DIM,) for up to MAX_LOCKED_DOORS
        """
        switch_states_array = np.zeros(SWITCH_STATES_DIM, dtype=np.float32)
        locked_doors = obs.get("locked_doors", [])

        for i, locked_door in enumerate(locked_doors[:MAX_LOCKED_DOORS]):
            switch_x, switch_y, door_x, door_y, switch_collected = (
                self._extract_locked_door_positions(locked_door)
            )

            # Normalize positions to [0, 1]
            base_idx = i * FEATURES_PER_DOOR
            switch_states_array[base_idx + 0] = np.clip(
                switch_x / LEVEL_WIDTH_PX, 0.0, 1.0
            )
            switch_states_array[base_idx + 1] = np.clip(
                switch_y / LEVEL_HEIGHT_PX, 0.0, 1.0
            )
            switch_states_array[base_idx + 2] = np.clip(
                door_x / LEVEL_WIDTH_PX, 0.0, 1.0
            )
            switch_states_array[base_idx + 3] = np.clip(
                door_y / LEVEL_HEIGHT_PX, 0.0, 1.0
            )
            switch_states_array[base_idx + 4] = switch_collected

        return switch_states_array

    def _hash_switch_states(self, switch_states: Dict[int, bool]) -> str:
        """
        Create hashable key from switch states.

        Args:
            switch_states: Dictionary mapping switch IDs to their states

        Returns:
            String hash of switch states for caching
        """
        if not switch_states:
            return "empty"
        return ",".join(f"{k}:{int(v)}" for k, v in sorted(switch_states.items()))

    def invalidate_switch_cache(self):
        """
        Invalidate switch-dependent caches.

        Called by entity collision methods when switch states change.
        This is more efficient than computing hash keys every frame.

        Clears:
        - Pathfinding visualization cache (debug overlay renderer)
        - Path distance calculator cache (forces rebuild with new goals)
        - Cached level_data (entities may be stale after switch activation)
        - Exit features cache (switch activation changes objectives)
        """
        self._switch_states_changed = True

        # Clear cached level_data so it gets refreshed with fresh entities
        # CRITICAL: level_data.entities may be stale after switch activation
        if hasattr(self, "_cached_level_data"):
            self._cached_level_data = None

        # Clear pathfinding visualization cache in debug overlay renderer
        # This ensures path visualizations update when switch state changes
        if (
            hasattr(self, "nplay_headless")
            and hasattr(self.nplay_headless, "sim_renderer")
            and hasattr(self.nplay_headless.sim_renderer, "debug_overlay_renderer")
        ):
            debug_renderer = self.nplay_headless.sim_renderer.debug_overlay_renderer
            if hasattr(debug_renderer, "clear_pathfinding_cache"):
                debug_renderer.clear_pathfinding_cache()

        # Clear path distance calculator cache
        # This forces rebuild with updated goals (exit switch removed, exit door becomes primary)
        if hasattr(self, "_path_calculator") and self._path_calculator is not None:
            self._path_calculator.clear_cache()

        # Also clear the reward calculator's (separate) path_calculator cache.
        if hasattr(self, "reward_calculator") and hasattr(
            self.reward_calculator, "path_calculator"
        ):
            self.reward_calculator.path_calculator.clear_cache()

    def _extract_mine_features(
        self, obs: Dict[str, Any], buffer: np.ndarray
    ) -> np.ndarray:
        """Extract enhanced mine features (8 dims).

        Features:
        [0-1] nearest_mine_rel_x, rel_y (normalized -1 to 1)
        [2] nearest_mine_state (0=deadly, 0.5=toggling, 1=safe, -1=none)
        [3] nearest_mine_path_distance (normalized 0-1)
        [4] deadly_mines_nearby_count (normalized 0-1)
        [5] mine_state_certainty (0=unknown to 1=recently seen)
        [6] safe_mines_nearby_count (normalized 0-1)
        [7] mine_avoidance_difficulty (0-1: spatial complexity)
        """
        NEARBY_RADIUS = 100.0
        MAX_NEARBY = 10.0
        CERTAINTY_RADIUS = 150.0

        features = buffer
        features.fill(0.0)

        # Early exit checks
        if (
            not hasattr(self, "_path_calculator")
            or self._path_calculator is None
            or self.current_graph_data is None
        ):
            features[2] = -1.0
            features[5] = 1.0  # High certainty when no mines
            return features

        adjacency = self.current_graph_data.get("adjacency")
        if not adjacency:
            features[2] = -1.0
            features[5] = 1.0
            return features

        # Extract ninja position
        ninja_x = obs.get("player_x", 0.0)
        ninja_y = obs.get("player_y", 0.0)

        ninja_pos = (int(ninja_x), int(ninja_y))

        # Collect mines
        entities = obs.get("entities", [])
        mines = []
        deadly_nearby = 0
        safe_nearby = 0

        for entity in entities:
            entity_type = getattr(entity, "entity_type", None)
            if entity_type in (EntityType.TOGGLE_MINE, EntityType.TOGGLE_MINE_TOGGLED):
                mx = getattr(entity, "xpos", 0.0)
                my = getattr(entity, "ypos", 0.0)
                mstate = getattr(entity, "state", 1)

                if entity_type == EntityType.TOGGLE_MINE_TOGGLED:
                    mstate = 0

                dist = np.sqrt((mx - ninja_x) ** 2 + (my - ninja_y) ** 2)

                mines.append({"x": mx, "y": my, "state": mstate, "dist": dist})

                if dist < NEARBY_RADIUS:
                    if mstate == 0:
                        deadly_nearby += 1
                    elif mstate == 1:
                        safe_nearby += 1

        if not mines:
            features[2] = -1.0
            features[5] = 1.0
            return features

        # Find nearest mine
        nearest = min(mines, key=lambda m: m["dist"])

        # Features 0-1: Relative position
        area_scale = self._get_reachable_area_scale()
        features[0] = np.clip((nearest["x"] - ninja_x) / area_scale, -1.0, 1.0)
        features[1] = np.clip((nearest["y"] - ninja_y) / area_scale, -1.0, 1.0)

        # Feature 2: State
        if nearest["state"] == 0:
            features[2] = 0.0  # Deadly
        elif nearest["state"] == 2:
            features[2] = 0.5  # Toggling
        else:
            features[2] = 1.0  # Safe

        # Feature 3: Path distance
        try:
            from ..constants.physics_constants import TOGGLE_MINE_RADII

            mine_radius = TOGGLE_MINE_RADII.get(nearest["state"], 4.0)
            mine_pos = (int(nearest["x"]), int(nearest["y"]))
            path_dist = self._safe_path_distance(
                ninja_pos,
                mine_pos,
                adjacency,
                "mine",
                level_data=self.level_data,
                graph_data=self.current_graph_data,
                entity_radius=mine_radius,
            )
            features[3] = min(path_dist / area_scale, 1.0)
        except Exception:
            features[3] = min(nearest["dist"] / area_scale, 1.0)

        # Feature 4: Deadly mines nearby
        features[4] = min(deadly_nearby / MAX_NEARBY, 1.0)

        # Feature 5: State certainty (based on distance)
        features[5] = 1.0 - min(nearest["dist"] / CERTAINTY_RADIUS, 1.0)

        # Feature 6: Safe mines nearby
        features[6] = min(safe_nearby / MAX_NEARBY, 1.0)

        # Feature 7: Avoidance difficulty
        total_nearby = deadly_nearby + safe_nearby
        if total_nearby > 0:
            danger_ratio = deadly_nearby / total_nearby
            density = total_nearby / MAX_NEARBY
            features[7] = min(0.7 * danger_ratio + 0.3 * density, 1.0)

        return features

    def _compute_progress_features(
        self, obs: Dict[str, Any], buffer: np.ndarray
    ) -> np.ndarray:
        """
        Compute progress tracking features.

        Returns PROGRESS_FEATURES_DIM (3) features:
        - current_objective_type (0=switch, 0.33=door, 0.67=exit, normalized 0-1)
        - objectives_completed_ratio (0 to 1)
        - total_path_distance_remaining (normalized)
        """
        MAX_OBJECTIVE_PATHS = 3.0  # switch + door + exit

        features = buffer
        features.fill(0.0)

        if not hasattr(self, "_path_calculator") or self._path_calculator is None:
            return features

        if self.current_graph_data is None:
            return features

        adjacency = self.current_graph_data.get("adjacency")
        if adjacency is None or len(adjacency) == 0:
            return features

        # Extract positions directly from raw observation (not dependent on processing order)
        ninja_x = obs.get("player_x", 0.0)
        ninja_y = obs.get("player_y", 0.0)
        switch_x = obs.get("switch_x", 0.0)
        switch_y = obs.get("switch_y", 0.0)
        exit_door_x = obs.get("exit_door_x", 0.0)
        exit_door_y = obs.get("exit_door_y", 0.0)

        ninja_pos = (
            int(ninja_x),
            int(ninja_y),
        )

        exit_switch_collected = obs.get("switch_activated", False)
        locked_doors = obs.get("locked_doors", [])

        completed = 0
        total = 1

        for door in locked_doors:
            total += 1
            if not getattr(door, "active", True):
                completed += 1

        if not exit_switch_collected:
            current_obj_type = 0.0
        elif any(getattr(door, "active", True) for door in locked_doors):
            current_obj_type = 0.33
        else:
            current_obj_type = 0.67

        total_path_dist = 0.0

        if not exit_switch_collected:
            from ..constants.physics_constants import EXIT_SWITCH_RADIUS

            exit_switch_pos = (
                int(switch_x),
                int(switch_y),
            )
            switch_dist = self._safe_path_distance(
                ninja_pos,
                exit_switch_pos,
                adjacency,
                "progress_exit_switch",
                level_data=self.level_data,
                graph_data=self.current_graph_data,
                entity_radius=EXIT_SWITCH_RADIUS,
            )
            total_path_dist += switch_dist

        for door in locked_doors:
            if getattr(door, "active", True):
                from ..constants.physics_constants import LOCKED_DOOR_SWITCH_RADIUS

                switch_x = getattr(door, "sw_xpos", getattr(door, "xpos", 0.0))
                switch_y = getattr(door, "sw_ypos", getattr(door, "ypos", 0.0))
                switch_pos = (int(switch_x), int(switch_y))

                door_dist = self._safe_path_distance(
                    ninja_pos,
                    switch_pos,
                    adjacency,
                    "progress_locked_door_switch",
                    level_data=self.level_data,
                    graph_data=self.current_graph_data,
                    entity_radius=LOCKED_DOOR_SWITCH_RADIUS,
                )
                total_path_dist += door_dist

        from ..constants.physics_constants import EXIT_DOOR_RADIUS

        exit_door_pos = (
            int(exit_door_x),
            int(exit_door_y),
        )
        exit_dist = self._safe_path_distance(
            ninja_pos,
            exit_door_pos,
            adjacency,
            "progress_exit_door",
            level_data=self.level_data,
            graph_data=self.current_graph_data,
            entity_radius=EXIT_DOOR_RADIUS,
        )
        total_path_dist += exit_dist

        features[0] = current_obj_type
        features[1] = completed / max(total, 1)
        area_scale = self._get_reachable_area_scale()
        features[2] = np.clip(
            total_path_dist / (area_scale * MAX_OBJECTIVE_PATHS), 0.0, 1.0
        )

        return features

    def _process_observation(self, obs):
        """Reduce the raw observation dict to the Layer 1 policy contract.

        Raw obs carries reward-side metadata (`_adjacency_graph`, `_graph_data`,
        `level_data`, `switch_states_dict`, cached path distances). That
        metadata is consumed by the reward calculator BEFORE this call and is
        stripped here so the rollout buffer only holds the four policy
        tensors plus control masks.
        """
        processed_obs = super()._process_observation(obs)

        # Gold-fraction-remaining scalar (docs/GOLD_AND_SCORE_OPTIMIZATION.md part b),
        # appended UNCONDITIONALLY right after the base game_state truncation and
        # BEFORE the optional expose_* blocks, so the policy game_state length
        # matches the (+1) Box in _build_extended_observation_space. Global "how much
        # gold is left" signal (in [0, 1]; 0.0 on no-gold levels), complementing the
        # spatial gold-density overlay channel. Read from the raw obs dict, which
        # base_environment populates on both the python and native backends.
        _gf = float(obs.get("gold_fraction_remaining", 0.0))
        processed_obs["game_state"] = np.append(
            np.asarray(processed_obs["game_state"], dtype=np.float32),
            np.float32(_gf),
        ).astype(np.float32)

        # v22 §3-B: re-attach the blocked-toward-goal scalar AFTER the base
        # process_game_state truncates game_state to GAME_STATE_CHANNELS. The
        # policy game_state length then matches the (+1) Box in
        # _build_extended_observation_space.
        if getattr(getattr(self.config, "reachability", None),
                   "expose_blocked_goal", False):
            feat = getattr(self, "_blocked_goal_feat", None)
            if feat is None:
                feat = np.zeros(1, dtype=np.float32)
            processed_obs["game_state"] = np.append(
                np.asarray(processed_obs["game_state"], dtype=np.float32), feat
            ).astype(np.float32)

        # Wall-contact tau re-attached AFTER blocked_goal so the game_state layout
        # is a stable, documented order: [... base ..., blocked_goal?, wall_tau?].
        if getattr(getattr(self.config, "reachability", None),
                   "expose_wall_contact_tau", False):
            feat = getattr(self, "_wall_tau_feat", None)
            if feat is None:
                feat = np.ones(1, dtype=np.float32)
            processed_obs["game_state"] = np.append(
                np.asarray(processed_obs["game_state"], dtype=np.float32), feat
            ).astype(np.float32)

        # Precision block re-attached LAST, so the documented game_state layout is
        # [... base ..., blocked_goal?, wall_tau?, precision_state?].
        if getattr(getattr(self.config, "reachability", None),
                   "expose_precision_state", False):
            feat = getattr(self, "_precision_state_feat", None)
            if feat is None:
                feat = np.zeros(self.PRECISION_STATE_DIM, dtype=np.float32)
            processed_obs["game_state"] = np.append(
                np.asarray(processed_obs["game_state"], dtype=np.float32), feat
            ).astype(np.float32)

        # Time/phase block re-attached after precision_state, matching the growth
        # order in _build_extended_observation_space. Full layout:
        # [... base ..., blocked_goal?, wall_tau?, precision_state?, time_phase?].
        if getattr(getattr(self.config, "reachability", None),
                   "expose_time_phase", False):
            feat = getattr(self, "_time_phase_feat", None)
            if feat is None:
                feat = np.zeros(self.TIME_PHASE_DIM, dtype=np.float32)
            processed_obs["game_state"] = np.append(
                np.asarray(processed_obs["game_state"], dtype=np.float32), feat
            ).astype(np.float32)

        # v26 W6c commitment block re-attached LAST. Full layout:
        # [... base ..., blocked_goal?, wall_tau?, precision_state?, time_phase?,
        #  commitment_state?]. Zeros are the genuine "nothing is committing me"
        # reading (every dim is polarized so zero is safe), so the fallback is
        # correct and not merely conventional.
        if getattr(getattr(self.config, "reachability", None),
                   "expose_commitment_state", False):
            feat = getattr(self, "_commitment_state_feat", None)
            if feat is None:
                feat = np.zeros(self.COMMITMENT_STATE_DIM, dtype=np.float32)
            processed_obs["game_state"] = np.append(
                np.asarray(processed_obs["game_state"], dtype=np.float32), feat
            ).astype(np.float32)

        if "hazard_map" not in processed_obs:
            processed_obs["hazard_map"] = obs.get(
                "hazard_map",
                np.zeros(
                    (
                        HAZARD_MAP_CHANNELS,
                        HAZARD_MAP_OBS_WIDTH,
                        HAZARD_MAP_OBS_HEIGHT,
                    ),
                    dtype=np.float32,
                ),
            )

        # Strip reward-only / legacy metadata so rollout buffer holds just
        # tile_grid, entity_overlays, hazard_map, game_state (+ masks).
        internal_only_keys = [
            "_adjacency_graph",
            "_graph_data",
            "_cached_switch_distance",
            "_cached_exit_distance",
            "_potential_substituted",
            "waypoint_leg_distance",
            "waypoint_index",
            "necessary_doors_open",
            "necessary_doors_total",
            "switch_states_dict",
            "level_data",
            "entity_positions",
        ]
        for key in internal_only_keys:
            processed_obs.pop(key, None)

        # Final safety net: the policy/value network must NEVER see a non-finite
        # input. A graph-OOB query makes path_distance_calculator return inf,
        # which can reach a normalized obs feature (e.g. the reachability path-
        # distance dims) and produce NaN logits / a NaN critic output -> NaN SIL
        # priorities -> crash. Distances are sanitized at the reward boundary, but
        # the OBS path is independent, so clamp every float feature array here.
        for key, val in processed_obs.items():
            if isinstance(val, np.ndarray) and val.dtype.kind == "f":
                if not np.isfinite(val).all():
                    # Capture the level BEFORE sanitizing (bad procgen or a
                    # graph/grid logic error); best-effort, deduped, gated.
                    try:
                        from .invalid_value_capture import capture_invalid_value

                        capture_invalid_value(
                            f"nonfinite_obs_{key}",
                            getattr(self.nplay_headless, "current_map_data", None),
                            {
                                "obs_key": key,
                                "shape": list(val.shape),
                                "n_nonfinite": int((~np.isfinite(val)).sum()),
                                "step": int(getattr(self, "_episode_step", -1)),
                            },
                        )
                    except Exception:
                        pass
                    np.nan_to_num(
                        val, copy=False, nan=0.0, posinf=1.0, neginf=0.0
                    )

        return processed_obs

    def _get_cached_tile_grid(self) -> np.ndarray:
        """Return the tile grid for the current level, computing it once per level load.

        The tile grid is static for the lifetime of a level episode, so we cache it
        on first access and invalidate whenever the map name changes.

        Returns:
            np.ndarray: (42, 23) int8 tile type IDs, values in [0, 37]
        """
        current_map_name = getattr(self.map_loader, "current_map_name", None)
        # Invalidate cache when level changes (compare against stored name)
        if self._cached_tile_grid is None or current_map_name != getattr(
            self, "_tile_grid_map_name", None
        ):
            tile_data = self.nplay_headless.sim.get_tile_grid()
            self._cached_tile_grid = (
                np.array(tile_data, dtype=np.int8).reshape(23, MAP_TILE_WIDTH).T
            )  # -> (42, 23)
            self._tile_grid_map_name = current_map_name
            # Recompute stable integer level ID when tile grid changes
            self._cached_level_id_int = None

        return self._cached_tile_grid

    def set_feature_extractor_callback(self, callback):
        """Set callback for feature extractor cache precomputation.

        Called by EnvironmentFactory to wire hybrid_cached_graph extractor.
        The callback is triggered at reset() to precompute static graph embeddings.

        Args:
            callback: Function(level_id, graph_data) to call at level load
        """
        self._feature_extractor_cache_callback = callback

    def clear_caches(self, verbose: bool = False):
        """
        Clear all caches manually (for testing/debugging).

        This method provides external access to cache clearing,
        useful for manual cache busting in test environments.

        Args:
            verbose: If True, print cache clearing operations
        """
        from nclone.cache_management import clear_all_caches_for_new_level

        clear_all_caches_for_new_level(self, verbose=verbose)

    def __getstate__(self):
        """Custom pickle method to handle non-picklable pygame objects and support vectorization."""
        state = super().__getstate__()

        # Remove non-picklable objects that will be recreated by mixins
        non_picklable_attrs = [
            "graph_builder",
            "_reachability_system",
            "_reachability_extractor",
            "logger",
        ]

        for attr in non_picklable_attrs:
            if attr in state:
                del state[attr]

        return state

    def __setstate__(self, state):
        """Custom unpickle method to restore the environment and support vectorization."""
        super().__setstate__(state)

        self._reinit_graph_system_after_unpickling(self.config.graph.debug)
        self._reinit_reachability_system_after_unpickling()

        if getattr(self, "_path_calculator", None) is not None and hasattr(
            self, "reward_calculator"
        ):
            self.reward_calculator.path_calculator = self._path_calculator

        # Mark that we need to reinitialize on next reset
        self._needs_reinit = True
