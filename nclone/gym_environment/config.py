"""
Configuration classes for NPP-RL environment.

This module provides structured configuration management for the NppEnvironment
and its various components, replacing the previous parameter explosion pattern.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging

# Import RewardConfig for reward system integration
from .constants import EGO_CROP_SIZE_DEFAULT
from .reward_calculation.reward_config import RewardConfig


@dataclass
class RenderConfig:
    """Configuration for rendering and visualization.

    Rendering is only used for human playback, video capture, and debug overlays.
    It is decoupled from the policy observation path (which is structured tensors
    only — see Layer 1 observation contract).
    """

    render_mode: str = "grayscale_array"
    enable_animation: bool = False
    enable_debug_overlay: bool = False
    # Width (px) of a reserved UI strip on the right of the human window. When
    # > 0 the game frame is left-aligned and this many pixels are kept free for
    # an external panel (the test_environment dev panel). 0 = no panel (default),
    # so video/training render paths are unaffected.
    panel_width: int = 0

    def __post_init__(self):
        """Validate render configuration."""
        valid_modes = ["human", "grayscale_array", "rgb_array"]
        if self.render_mode not in valid_modes:
            raise ValueError(f"render_mode must be one of {valid_modes}")


@dataclass
class PBRSConfig:
    """Configuration for Potential-Based Reward Shaping.

    For heuristic potential functions (path distance), γ=1.0 eliminates negative bias.
    Policy invariance holds for ANY γ, but γ=1.0 ensures clean telescoping in episodic tasks.
    """

    pbrs_gamma: float = 1.0  # Retained for BSRS γ matching on the trainer side

    def __post_init__(self):
        """Validate PBRS configuration."""
        if self.pbrs_gamma < 0 or self.pbrs_gamma > 1:
            raise ValueError("pbrs_gamma must be between 0 and 1")


@dataclass
class GraphConfig:
    """Configuration for internal graph infrastructure used by reward shaping.

    The policy observation does NOT include graph tensors. Graph state is only
    retained as reward-side infrastructure (A* cache, path distance calculator,
    PBRS). See Linear I.2 (Retain Surface Graph as Offline-Only Curriculum Tool).
    """

    debug: bool = False

    def __post_init__(self):
        """Validate graph configuration."""
        if self.debug:
            logging.info("Graph debug mode enabled")


@dataclass
class ReachabilityConfig:
    """Configuration for reachability analysis (reward-side only)."""

    debug: bool = False
    # When True, compute full next-hop + directional-platform features (3–6) in
    # ``compute_reachability_features_from_graph``. Layer 1 policy does not use
    # them; default False saves several ms per step on reachability cache miss.
    compute_navigation_features: bool = False

    # --- Lattice d* reward potential (substitute + geometric fallback) ------
    # When True, the HWM distance signal (_cached_switch/exit_distance) is taken
    # from the physics-aware lattice d* cost-to-go where the spawn has a finite
    # plan, falling back to the geometric A* otherwise (robust to no-route
    # levels).  Default False: opt-in per rollout so existing training dynamics
    # are unchanged until validated.  See value_function/lattice_distance.py.
    use_lattice_potential: bool = False
    # When True, use the CHEAP mine-aware free-space grid cost-to-go as the
    # reward potential DIRECTLY, skipping the (often infeasible) surface-lattice
    # build entirely -- the surface lattice can take seconds-to-minutes/level on
    # open/platform geometry, whereas the free-space grid is ~0.1-0.5s and gives
    # a dense corridor-following signal well suited to airborne/platform layouts.
    # This is the scalable PBRS potential for procgen curricula (every reset is a
    # new level). Independent of use_lattice_potential; ignored when lattice obs
    # or locked-door rewards are on (those need the full lattice/door plan).
    use_free_space_potential: bool = False
    # When True, use the physics-faithful jump-aware cost-to-go
    # (``JumpDistanceSource``) instead of the geometric free-space flood, for BOTH
    # the PBRS reward potential AND the ``cost_to_go_map`` observation. Unlike the
    # 8-connected free-space flood (which leaks through the air between platforms
    # and points the gradient into lethal gaps), the jump field connects standable
    # support cells with validated ballistic JUMP/FALL edges, so the gradient
    # points toward reachable platforms/launch points. It is a drop-in subclass of
    # the free-space source (same scalar API), so it transparently replaces it at
    # both construction points in ReachabilityMixin. The build is SLOW (6-11 s/
    # level vs ~0.1-0.5 s free-space), so it has its own two-tier (mem LRU + shared
    # on-disk) cache and an offline precompute (scripts/precompute_jump_fields.py)
    # to make it viable under multi-level training. Default False keeps the
    # geometric free-space path everywhere; when True it takes precedence over
    # use_free_space_potential.
    use_jump_potential: bool = False
    # Audit II P0-2 (--ballistic-air-potential): while the ninja is AIRBORNE,
    # evaluate the committed PBRS potential at the landing point of the CURRENT
    # no-input ballistic arc (native TrajectoryProjector, real airborne
    # recurrence) instead of at the ninja's air position. The default airborne
    # value is a straight-down, velocity-blind column projection, so mid-air
    # trim toward a landing is reward-invisible without this. Grounded states
    # are exactly unchanged; still a true state potential (telescopes). Most
    # useful with use_jump_potential (jump-aware landing values) but valid on
    # any substituted free-space/threading potential.
    ballistic_air_potential: bool = False
    # v13 ghost-arc extension of ballistic_air_potential: instead of evaluating
    # the airborne potential at the u=0 (no-input) landing only, project all
    # three steering arcs u in {-1, 0, +1} and take the BEST (min finite
    # cost-to-go) surviving landing. The u=0 arc alone undervalues states where
    # a live steering correction still reaches a good landing (and overvalues
    # states where only steering away saves the flight). Still a pure state
    # potential — it's a fixed function of (pos, vel, remaining jump frames) —
    # so PBRS telescoping/policy-invariance is preserved. No-op unless
    # ballistic_air_potential is also True.
    ballistic_air_best_arc: bool = False
    # When True (and use_lattice_potential is on), the substituted lattice
    # distance includes an orienteering gold tail-cost (NN tour over only the
    # WORTHWHILE remaining gold, i.e. gold whose detour costs < lattice_gold_value
    # frames) so the PBRS potential guides toward gold worth collecting.  The tail
    # depends on the remaining-gold SET, so it's a valid (policy-invariant)
    # potential — see LatticeDistanceSource.gold_cost_to_go.  Default off (parity).
    lattice_gold_potential: bool = False
    lattice_gold_weight: float = 1.0
    # Per-gold time-saving exchange rate (frames) used by the orienteering filter;
    # mirror of reward_constants.GOLD_FRAME_VALUE. A gold enters the potential only
    # if its detour costs fewer than this many frames.
    lattice_gold_value: float = 120.0
    # Planner objective knobs threaded to the per-level precompute
    # (completion>loops>speed).  None => edge_risk module defaults.
    lattice_w_risk: Optional[float] = None
    lattice_w_loop: Optional[float] = None
    lattice_risk_cap: Optional[float] = None
    lattice_use_native: bool = True
    # When True, expose a small physics-aware navigation hint to the POLICY obs
    # under the ``lattice_nav`` key ([dx_hop, dy_hop, cost_to_go_norm, phase]).
    # Default False keeps the strict Layer-1 contract; enabling it requires the
    # feature extractor to accept the extra key.  Builds the lattice source even
    # if use_lattice_potential is False.
    expose_lattice_features: bool = False

    # --- Lattice-backed waypoint subgoals (long-horizon distance fix) -------
    # When True, build a LatticeWaypointPlanner per map and drive a WaypointTracker
    # each step. It feeds the reward calculator's per-leg waypoint PBRS
    # (RewardConfig.enable_waypoint_pbrs) via the obs keys ``waypoint_leg_distance``
    # + ``waypoint_index`` (reward-only, stripped from the policy obs). Decomposes a
    # far goal into a chain of near subgoals so the per-step gradient is
    # route-length-invariant. Building the lattice is slow (seconds-minutes/level),
    # so this is for fixed/offline or pool-precomputed levels; at procgen scale,
    # inject precomputed waypoints via ``set_level_waypoints`` instead of building on
    # reset. See docs/LONG_HORIZON_DISTANCE_HANDOFF.md.
    enable_waypoint_tracker: bool = False
    # Build waypoints on each reset when none were injected. With
    # ``waypoint_use_free_space`` (default) this uses the CHEAP disk-cached
    # free-space cost-to-go (steepest-descent, ~ms) so it scales to procgen
    # without the lattice's seconds-per-level cost. Set False + inject via
    # ``set_level_waypoints`` to disable on-reset building entirely.
    waypoint_build_on_reset: bool = True
    # Use the free-space waypoint planner (cheap) over the lattice planner
    # (physics-aware but precompute-heavy). Free-space falls back to lattice when
    # unavailable. Recommended True for procgen-scale on-reset building.
    waypoint_use_free_space: bool = True
    # Densify: split route legs longer than this many px into sub-waypoints (keeps
    # every leg short). None = use the raw lattice arc landings.
    waypoint_max_leg_px: Optional[float] = None
    # When True, also expose the 4-dim waypoint nav hint to the POLICY obs under the
    # ``waypoint_nav`` key ([dx_hat, dy_hat, cost_to_go_norm, phase]). Default False
    # keeps the Layer-1 contract; enabling it requires the feature extractor to
    # accept the extra key. Implies enable_waypoint_tracker.
    expose_waypoint_features: bool = False

    # Phase-2 (O2): expose the free-space cost-to-go FIELD as low-res spatial obs
    # channels (``cost_to_go_map``, 3x84x46: active-phase cost-to-go, other-phase
    # cost-to-go, reachable mask; projected at 12px, half the 6px hazard grid).
    # Gives the policy the GLOBAL route (the basin is high-cost, the way out goes
    # up-and-around) instead of only the local next-hop hint — a planning shortcut
    # fully compatible with a Markov policy. Default False keeps the Layer-1 contract.
    expose_cost_to_go_map: bool = False

    # v24: route the POLICY-OBSERVED free-space nav hint, cost_to_go_map, and the
    # ego route crops through the MINE-FAITHFUL free-space field — deadly-mine
    # cells are cut from the traversable mask, so the gradient the policy follows
    # routes AROUND mines instead of through them (the legacy mine-SOFT field made
    # mine cells traversable at ~9x cost, so steepest-descent could point straight
    # into a kill disc on the near-optimal line — the root cause of the ~60%
    # mine-death rate on official maps). Reward potentials are UNAFFECTED (they
    # already cascade lattice -> mine-threading -> free-space). On a mine-severed
    # level the mine-hard field is unavailable and the obs falls back to the
    # Euclidean cost-to-go (same as any failed free-space build). Default False.
    mine_faithful_obs: bool = False

    # v18 (Component 3): normalization mode for the cost_to_go_map cost channels.
    #   "fixed_k"        — c = m/(m+K), K=1000 (legacy; tuned for long FAR routes,
    #                      but crushes compact-maze costs to a near-flat gradient:
    #                      a 60-step maze spans only [0, 0.057]).
    #   "per_level_dmax" — c = m/D_max with D_max = the field's finite max over
    #                      the playable slice, per level & per phase field. Costs
    #                      span the full [0, 1] whatever the level scale, so the
    #                      switch-direction gradient survives in compact mazes.
    #                      Trades away cross-map comparability (the fixed-K
    #                      argument) for within-map gradient — the right trade
    #                      for maze search per V18_PLAN.md §4 Component 3.
    #   "log"            — c = log1p(m)/log1p(K) clipped to 1.0: fixed scale
    #                      (cross-map comparable) with compressive gradient
    #                      boost at small m. Middle-ground option.
    #   "ego_residual"   — v19 R4b: c = clip((m - m_ninja)/W_c, -1, 1). Every cell
    #                      is re-referenced to the NINJA's current cost and scaled
    #                      by a fixed window (cost_map_ego_window). Level-size-
    #                      invariant local contrast at the junction where the
    #                      decision happens (per_level_dmax crushes exactly this on
    #                      big mazes). Position-dependent (changes every frame); the
    #                      per-level RAW field is cached and the residual applied
    #                      per frame, so the throughput cost is negligible.
    # Same key/shape either way (values shift) → resume-safe in principle, but a
    # warm start sees shifted inputs; V18/V19 ship it inside a fresh run.
    cost_map_norm: str = "fixed_k"
    # v19 R4b: window (in field cost units) for the ego_residual mode. ~8 ≈ a few
    # tiles of free-space cost, so a 1-cost step reads ~0.12 (visible) while cells
    # beyond the window saturate at +/-1. Only used when cost_map_norm ==
    # "ego_residual".
    cost_map_ego_window: float = 8.0

    # v17 episodic visitation map (``visitation_map``, 1x84x46): a per-episode
    # "where have I been" grid co-registered with cost_to_go_map. Carries a
    # recency-decayed visit signal per 12px cell so a Markov policy can tell
    # visited junctions/dead-ends from unexplored ones in tree-structured mazes
    # (CHAMBER_MAZE state aliasing). Shares the per-episode visit-count grid that
    # also drives the count-based visitation reward bonus (visitation_bonus_coef).
    # Default False keeps the Layer-1 contract; enabling it grows a visitation
    # stream in the structured_tile_v1 extractor.
    expose_visitation_map: bool = False

    # v19 R4a: recency (time-decay) mode for the visitation_map channel. When > 0
    # the channel emits exp(-(t - t_last_visit)/tau) (tau in env frames) instead of
    # the legacy 1/sqrt(count) count-decay, so corridor ORDER/recency (not just
    # frequency) is visible to the memoryless feedforward policy — the mechanism
    # behind "try corridors in the order I last searched them" (Neural Map / SRU
    # spatial-memory literature). 0 = legacy count-decay. ~600 recommended.
    # BREAKING obs semantics (same shape) → fresh run + BC cache rebuild.
    visitation_recency_tau: float = 0.0

    # v18 C1 spatial frontier map (``frontier_map``, 1x84x46): the reachable-but-
    # unvisited set as a spatial obs channel, co-registered with cost_to_go_map /
    # visitation_map. frontier = reachable AND (episodic visit count == 0) — the
    # directed "where have I NOT searched yet" signal a memoryless feedforward
    # policy cannot reconstruct from the visitation map alone (it would need the
    # complement against reachability). Enabling it forces the per-episode visit
    # grid on (same grid the visitation map / bonus use). Default False keeps the
    # Layer-1 contract; enabling grows a frontier stream in the structured_tile_v1
    # extractor. BREAKING obs change → fresh run + BC cache rebuild.
    expose_frontier_map: bool = False
    # v18 C1 option: weight each frontier cell by the local density of
    # unexplored reachable space (7x7 box mean), so frontier bordering a LARGE
    # unexplored region reads brighter than a 1-cell nook. Values stay in
    # [0.25, 1] on frontier cells (0 elsewhere). Default False = binary mask.
    frontier_map_weighted: bool = False

    # Ego-centric high-res crop obs (``ego_crop``): a square window of raw tile IDs
    # sliced from the full-resolution tile_grid and re-centered on the ninja's tile
    # each step. Pure geometry (no reachability), grouped here alongside the other
    # spatial obs toggles. Precision-platforming fix — the global tile_grid is
    # CNN-downsampled and the cost_to_go_map is 3x3-pooled, so neither localizes a
    # single-tile (24px) landing platform; the ninja-centered crop keeps per-tile
    # geometry at native resolution. Default False keeps the Layer-1 contract; the
    # structured_tile_v1 extractor grows an ego-crop stream when the key is present.
    enable_ego_crop: bool = False
    # Crop side length in tiles (odd => ninja on the exact center cell). See
    # EGO_CROP_SIZE_DEFAULT.
    ego_crop_size: int = EGO_CROP_SIZE_DEFAULT
    # Audit II P1-6 (``ego_crop_coarse``): a SECOND ninja-centered 21x21 window
    # at 2-tile pooling — +-21 tiles Chebyshev (full level height / half its
    # width from anywhere), with channels [solid-fraction, deadly-mine density,
    # safe-mine density, objective marker]. Fills the gap between the fine
    # +-10-tile ego crop (which by physics coincidence ends exactly at the max
    # survivable free-fall) and the 168-336 px/cell global readouts: localizes
    # a 1-2-tile target to +-1 tile at 20 tiles. Direction-agnostic (a
    # velocity-biased crop was rejected: frame semantics change per step and
    # destabilize BC re-simulation). Default False keeps the contract.
    enable_ego_crop_coarse: bool = False
    # v17 P1: append the 2-tile-pooled grad(d*) route vector (2 channels) to the
    # coarse crop, so the +-21-tile maze-scale view is route-DIRECTED (which
    # corridor leads to the switch) and not just a target bearing. Requires
    # enable_ego_crop_coarse. Grows ego_crop_coarse 4->6 channels. Default False.
    ego_crop_coarse_route: bool = False
    # v22 §3-B: append a derived BLOCKED-TOWARD-GOAL scalar to game_state (+1 dim).
    # 1.0 when the ninja is walled AND the routed direction (grad of the cost-to-go
    # field) points INTO that wall AND cost-to-go is not decreasing -- the perceptual
    # twin of the near-switch wall oscillation. Makes the dither state observable so
    # the policy can learn to escape it. Default False (BREAKING obs-dim change).
    expose_blocked_goal: bool = False
    # Append a FRAMES-TO-WALL-CONTACT scalar to game_state (+1 dim), in [0,1]
    # (1 = no wall within the ~12-frame horizontal lookahead, 0 = in contact).
    # ``time_to_contact_frames`` only covers the LANDING surface of an AIRBORNE
    # ninja, so a GROUNDED ninja running horizontally at a wall currently has no
    # contact-time signal -- precisely the state a corner walljump is entered
    # from, where the input must be pressed ~2 frames BEFORE contact. Turns a
    # frame-perfect guess into "when tau is small, jump".
    # Default False (BREAKING obs-dim change).
    expose_wall_contact_tau: bool = False
    # Append the PRECISION-STATE block to game_state (+10 dims). Closes the
    # verified holes a feed-forward policy hits on frame-precise maneuvers:
    #   * wall_normal is zeroed during the wall_buffer coyote window, so the
    #     policy knows a walljump is legal but not which way it launches;
    #   * the state one-hot merges Jumping(3) with Falling(4), and jump_duration
    #     is masked outside state 3, so variable-height jump control is blind;
    #   * airborn_old / consecutive_floor_frames / consecutive_wall_frames /
    #     state_change_frame / previous_state are absent entirely and were
    #     recoverable only via frame stacking.
    # See NppEnvironment._compute_precision_state for the exact layout.
    # Default False (BREAKING obs-dim change).
    expose_precision_state: bool = False
    # v22 §3-C: expose a FINE ego route-gradient crop (``ego_crop_route_fine``, 2ch)
    # -- the steepest-descent unit vector of the 6px UN-POOLED free-space cost-to-go
    # field in a small ninja-centered window. Unlike ego_crop_route (tile/24px
    # min-pooled, which goes degenerate at thin walls) this keeps the go-around
    # vector crisp one 6px cell behind a thin obstacle. Default False (BREAKING
    # obs-dim change; adds a spatial stream to the extractor).
    expose_ego_crop_route_fine: bool = False
    # v24: expose a FINE ego MINE crop (``ego_crop_mines_fine``, 2ch) -- the 6px
    # signed-distance-to-mine field (deadly + safe) sliced in a small ninja-centered
    # window. The policy's only other fine (6px) mine input, the global hazard_map,
    # is average-pooled to a ~7-tile embedding before the policy sees it, so the
    # lethal 14px disc is unresolvable; this restores sub-tile mine geometry where
    # deaths happen (around the ninja). Default False (BREAKING obs-dim change; adds
    # a spatial stream to the extractor). Pairs with mine_faithful_obs.
    expose_ego_crop_mines_fine: bool = False
    # Drone work, step 1: expose ``drone_hazard_map`` (1ch) -- the 6px signed
    # distance to the nearest lethal patrol drone (zap type 14 + mini type 26),
    # same grid/sign convention as ``hazard_map``. Without it the policy sees a
    # patrolling drone as EMPTY SPACE: the whole hazard pipeline is mine-shaped,
    # and drones are absent from the observation entirely. Default False
    # (BREAKING obs-dim change; adds a spatial stream to the extractor).
    #
    # This channel is PRESENT-FRAME only. Drones are deterministic and
    # ninja-independent, so their future positions are computable; exposing a
    # forward roll is the follow-up step, and is what actually makes the timing
    # learnable from a single frame.
    enable_drone_hazard: bool = False
    # Phase 1: the time-to-lethal threat field (`threat_ttl`, 4 channels) and the
    # solid-entity surface map (`solid_map`, 5 channels). Both are computed
    # entirely inside the native simulator -- no per-entity data crosses FFI --
    # and both are OFF by default so existing checkpoints and the strict Layer 1
    # contract are unaffected.
    #
    # They answer different questions and are separately gated on purpose:
    # `threat_ttl` is "when does this cell kill me", `solid_map` is "what can I
    # stand on". A thwump is both at once, on different faces, which is exactly
    # why one tensor cannot carry both.
    enable_threat_ttl: bool = False
    #: Window the TTL channels look ahead over, in sim frames.
    threat_ttl_max_frames: int = 64
    #: Tier-C sampling interval, in sim frames. The reactive tier is closed-form
    #: and ignores this.
    threat_ttl_stride: int = 4
    enable_solid_map: bool = False
    # v25: expose the directional nearest-mine bearing + CLOSING-SPEED vector
    # (``mine_proximity``, MINE_PROXIMITY_DIM floats). The v24 fine mine crop
    # above is a still frame — it says where the mines are but not whether the
    # ninja is closing on one. Measured after v24: mine-death fell to ~2% on the
    # easy procgen buckets but stayed ~0.51 on OFFICIAL layouts, with
    # switch_reached only 0.52 (half of official episodes die before the switch).
    # Adds K x 5 motion-relative floats (bearing, proximity, closing speed,
    # time-to-contact) as their OWN key rather than mutating the
    # truncation-sensitive game_state vector. Default False (adds an extractor
    # stream). Pairs with expose_ego_crop_mines_fine + mine_faithful_obs.
    expose_mine_proximity: bool = False
    # v26 W6a: expose the static FALL-LETHALITY field — per 6px cell, the signed
    # margin between the survivable-impact threshold at the floor below and the
    # speed a fall from that cell would arrive with, plus the normalized drop
    # distance. Impact deaths are ~40% of v25's losses and most are committed
    # unrecoverable falls: the lethal decision is made at TAKEOFF, so every
    # existing impact signal (lethal_landing_tau, the landing PBRS/aux terms)
    # fires far too late. Both thresholds are normal-dependent and pull opposite
    # ways, which makes a flat floor the LEAST forgiving surface (~6.8 tiles
    # kills) — a "drop > K tiles" heuristic is wrong in both directions, hence a
    # precomputed field rather than a scalar.
    #
    # ``expose_fall_lethality_map`` adds the global (2, 168, 92) field;
    # ``expose_ego_crop_fall_fine`` adds the ninja-centered fine crop. They are
    # independent flags but share one cached field, so enabling both costs
    # nothing extra to compute. Default False (BREAKING obs-dim change).
    expose_fall_lethality_map: bool = False
    expose_ego_crop_fall_fine: bool = False
    # v26 W6c: append the COMMITMENT / irreversibility block to game_state
    # (+6 dims). The per-arc fatality bits already exist, but spread over five
    # 3-arc bundles of 45 dims each, and which bundles are even valid depends on
    # the contact regime — so "is EVERY option from here fatal" (the actual
    # point-of-no-return question the impact-death failure mode is about) is a
    # conjunction the policy currently has to rediscover from scratch. Every dim
    # is polarized so zero = safe. Default False (BREAKING obs-dim change).
    expose_commitment_state: bool = False
    # EXECUTION_PRECISION_PLAN §6: append a TIME/PHASE block to game_state
    # (+TIME_PHASE_DIM dims). On a fixed-start deterministic level the optimal
    # policy is OPEN-LOOP -- a fixed input sequence -- but a feed-forward policy
    # conditioned only on observations must disambiguate states that LOOK alike
    # yet need different inputs. SI-A-00-00's route doubles back through the same
    # cells, and that exact aliasing already corrupted the position-keyed pace
    # table (fixed only by keying on (cell, switch_phase)). If the reward signal
    # was aliased by the return leg, the policy input plausibly is too.
    #
    # WARNING: this makes the policy MORE level-specific by design. It is a
    # single-level tool; do not fold it into the generalist contract.
    # Default False (BREAKING obs-dim change).
    expose_time_phase: bool = False
    # Normalisation horizon for the time ramp, in frames. The ramp is
    # t/time_phase_norm_frames clipped to [0,1], so this should sit a little
    # above the target route length -- normalising by MAX_TIME_IN_FRAMES (4500)
    # would squash a ~485-frame route into [0, 0.108] and waste the range.
    time_phase_norm_frames: int = 600
    # HORIZONTAL_AUGMENTATION_PLAN §2/§6: make the observation exactly
    # equivariant under a left-right reflection of the level, so a mirrored
    # sample synthesised by ``nclone.map_augmentation.obs_mirror`` is a FAITHFUL
    # observation of the mirrored state rather than an approximate one.
    #
    # The commutation harness (``map_augmentation/mirror_verify.py``) found the
    # simulator itself to be exactly symmetric — positions and velocities agree
    # to float precision — and located every divergence in two *derived* obs
    # conventions, both of which are half-cell registration errors that this
    # flag corrects:
    #
    #   1. The entity-overlay bilinear splat anchors on a tile's LEFT EDGE
    #      (``px/24 - 1``), so a tile's sample point is half a tile off its own
    #      centre. Under reflection the anchor set is not closed and the ninja /
    #      switch / exit channels land up to a quarter of a unit mass in the
    #      wrong cell. ``px/24 - 1.5`` anchors on the tile centre, which is both
    #      correctly registered against ``tile_grid`` and exactly symmetric.
    #   2. The forward-biased scandots pick their travel direction with
    #      ``sign(vx) or 1``, which resolves to +1 in BOTH orientations while the
    #      ninja is at rest (a common state: every spawn frame, every stationary
    #      frame). The block is direction-relative, so an even tie-break makes it
    #      sample the mirror-image columns. With this flag the tie falls back to
    #      the bearing to the active objective, which IS reflection-odd.
    #
    # Both change observation VALUES, so this is opt-in: a run resumed from a
    # checkpoint trained without it would see two subtly different inputs. Turn
    # it on for fresh trains that use mirror augmentation (Tier 2/3).
    mirror_equivariant_obs: bool = False

    def __post_init__(self):
        """Validate reachability configuration."""
        if self.debug:
            logging.info("Reachability debug mode enabled")
        if self.cost_map_norm not in {"fixed_k", "per_level_dmax", "log", "ego_residual"}:
            raise ValueError(
                "cost_map_norm must be one of {'fixed_k', 'per_level_dmax', 'log', 'ego_residual'}"
            )


@dataclass
class EnvironmentConfig:
    """Main configuration class for NppEnvironment.

    The policy observation dict is fixed to the Layer 1 contract
    (`tile_grid`, `entity_overlays`, `hazard_map`, `game_state`) plus
    `action_mask` / `duration_mask`. There are no visual, graph, reachability,
    or spatial_context observations exposed to the policy.
    """

    # Basic environment settings
    seed: Optional[int] = None
    eval_mode: bool = False
    custom_map_path: Optional[str] = None
    test_dataset_path: Optional[str] = None  # Path to test dataset for evaluation
    # When True, skip BaseNppEnvironment's initial ``load_map()``; the first level is
    # supplied via ``reset(options={"map_data": ...})`` (benchmark / payload-driven).
    defer_initial_map_load: bool = False
    enable_logging: bool = False
    # When True, the env populates per-EPISODE-END route-visualization data into
    # the step ``info`` dict (full tile grid as a dict, mine/locked-door lists, the
    # episode trajectory, and a next-hop planner path). This is ONLY consumed by
    # the route-visualization callback. It is EXPENSIVE — the tile dict + path get
    # pickled through every worker pipe on every episode end, which (with short
    # early-training episodes across many envs) dominates the SubprocVecEnv
    # round-trip and starves throughput. Default False so a run that doesn't render
    # routes pays nothing; the trainer sets it True only when route viz is enabled.
    enable_route_visualization: bool = False
    enable_profiling: bool = (
        False  # Enable detailed performance profiling (~5% overhead)
    )
    # Seconds between "ENVIRONMENT PROFILING REPORT" blocks on stdout; 0 disables
    # printing (timings still recorded for get_profiling_summary). Use 0 for
    # multi-worker runs to avoid interleaved garbage on the console.
    profiling_console_interval: float = 10.0

    # Component configurations
    render: RenderConfig = field(default_factory=RenderConfig)
    pbrs: PBRSConfig = field(default_factory=PBRSConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    reachability: ReachabilityConfig = field(default_factory=ReachabilityConfig)

    # Reward system configuration
    reward_config: Optional[RewardConfig] = None

    # Shared level cache (for zero-copy multi-worker training on same level)
    shared_level_cache: Optional[Any] = None

    # v25 (game-faithful): frames added to the episode's truncation budget per
    # gold collected. In real N++ each gold adds 1 second to the countdown; the
    # sim's truncation limit was STATIC per level (surface area + mines + route
    # cost, fixed at reset) and did not grow with gold, so a gold detour was
    # punished twice — once by the frames it cost, and again by a premature
    # timeout. Measured consequence (v24 scratch 18M -> 21M): timeouts nearly
    # doubled (29 -> 49) as gold/solve rose.
    #
    # With this set, the effective limit is re-evaluated every step as
    #   base_limit * curriculum_multiplier + gold_time_bonus_frames * gold_collected
    # and the same value feeds the ``time_remaining`` obs scalar, so the policy
    # SEES the budget it just earned. 60 = 1 second = game-faithful; 0 disables
    # (and costs nothing — no gold scan is performed).
    #
    # This changes ONLY the timeout. The SCORE budget stays a fixed 90 s, so
    # score semantics are unchanged. It deliberately does NOT weaken the
    # completion-primary reward: pair it with
    # ``RewardConfig.gold_bank_until_completion`` (so gold still cannot outrank
    # the exit) and keep the stall penalty on (so the extra time is not spent
    # dawdling).
    gold_time_bonus_frames: int = 0

    # Physics backend: "python" (NPlayHeadless) or "native" (Rust NativePlayHeadless).
    # Native is a drop-in for the training hot path (much faster, no rendering); use
    # "python" for human render / video capture. See base_environment for the gaps.
    sim_backend: str = "python"

    # Post-switch start-state reverse curriculum (FAR_OBJECTIVE loop levels).
    # When enabled the env lazily builds a per-level POST-SWITCH start-state pool
    # the first time it sees a level whose bucket is in
    # ``post_switch_curriculum_buckets`` (disk-cached, keyed by map_data hash),
    # and seeds resets from a switch-already-activated state so the agent practises
    # the under-trained return-to-exit leg. The reverse-curriculum difficulty is
    # driven by the trainer's ReverseCurriculumCallback (anneals start at-exit ->
    # at-switch). Off by default; does nothing when disabled.
    post_switch_curriculum: bool = False
    # v21d/L5: LOOP_CORRIDOR added — its switch->exit RETURN leg is the wall
    # (pre-switch seeds get switch_reached ~0.6 but true_completion stays 0 because
    # the return is never practised). LOOP is a round-trip like FAR_OBJECTIVE, so it
    # needs post-switch (at-exit -> at-switch) seeding too. The env alternates
    # pre/post per reset for levels gated by both (see npp_environment reset).
    post_switch_curriculum_buckets: tuple[str, ...] = ("FAR_OBJECTIVE", "13", "LOOP_CORRIDOR")
    post_switch_curriculum_n_states: int = 300
    post_switch_curriculum_max_iter: int = 120000

    # PRE-switch start-state reverse curriculum (v16; PRECISION execution wall).
    # The pre-switch sibling of the above: when enabled the env lazily builds a
    # per-level PRE-SWITCH start-state pool (mid-route states on the way to the
    # exit SWITCH, disk-cached) for levels whose reset ``bucket`` is in
    # ``start_state_curriculum_buckets`` and seeds resets from a mid-route state.
    # Targets PRECISION (p=0.06, 0.81 mine-death): seeding the hard final approach
    # lets the agent practise the mine-thread directly instead of dying long
    # before reaching it. Shares the reverse-curriculum difficulty knob
    # (``_ssc_difficulty`` via set_start_state_difficulty) with the post-switch
    # curriculum. Off by default; does nothing when disabled.
    start_state_curriculum: bool = False
    start_state_curriculum_buckets: tuple[str, ...] = ("PRECISION",)
    start_state_curriculum_n_states: int = 400
    start_state_curriculum_max_iter: int = 200000
    # Fraction of gated-bucket resets that skip seeding and start from the NATURAL
    # spawn instead (both pre- and post-switch). A principled reverse curriculum
    # never seeds 100% of episodes: the natural rollouts (a) measure genuine
    # full-task progress and (b) accumulate the natural-spawn pool the phase gate
    # needs (per-bucket episode floor). Without this, a heavily-seeded-but-solved
    # bucket (e.g. CHAMBER_MAZE) starves the pool and stalls phase advancement even
    # though it completes 100%. 0.0 = seed every gated reset (legacy behaviour).
    start_state_natural_fraction: float = 0.0

    # Goal-distance reverse-curriculum SCOPING. The goal-distance reverse
    # curriculum (``set_goal_distance_fraction``: succeed at an intermediate
    # cost-to-go threshold, annealed outward) is a global env knob. When this
    # tuple is non-empty, that curriculum is applied ONLY on levels whose reset
    # ``bucket`` is in the set; on every other level the fraction is forced to 1.0
    # (off, true-goal-only). Empty = apply globally (legacy behavior). Scoping the
    # p^6 -> p^1 reverse-spawn to PLATFORM_STAIRS keeps it from shortening episodes
    # on buckets that already complete end-to-end. See
    # docs/PLATFORM_STAIRS_CURRICULUM_FOLLOWUP.md §4.2/§6.4.
    goal_distance_curriculum_buckets: tuple[str, ...] = ()

    # Frame skip
    frame_skip: int = 4

    # Layer 3 FiGAR macro-action safety masking
    enable_layer3_macro_action_mask: bool = False
    layer3_duration_bins: tuple[int, ...] = (
        1,
        2,
        3,
        4,
        5,
        6,
        8,
        10,
        12,
        16,
        20,
        24,
        32,
        48,
        64,
    )
    layer3_all_masked_fallback: str = "noop"

    # v26 W6b: extend the Layer-3 fatality predicate from "hits a deadly mine" to
    # "ends in ANY physics death", which brings unsurvivable impacts under the
    # same hard guarantee. The split is ~60% mine / ~40% impact deaths in v25,
    # and before this every hard safety guarantee the agent had was mine-only —
    # so the interrupt layer was shortening macros near mines while leaving
    # fatal-gap macros uncapped, exactly backwards.
    layer3_mask_lethal_impact: bool = False
    # Frames of NEUTRAL-input simulation past the macro's own duration. Impact
    # deaths are committed at TAKEOFF: a 12-frame macro that steps off a ledge is
    # survivable for all 12 of its frames and fatal ~30 frames later, so the
    # macro-length horizon alone can never see it. ~40 frames covers a ~7-tile
    # fall. 0 = macro horizon only.
    layer3_lethal_impact_lookahead: int = 0
    # Also extend the TAKEOFF-ARC gap mask below from mine-landing to
    # lethal-impact landing. Same defect, same fix: the commitment machinery
    # exists and is keyed on the wrong hazard.
    takeoff_arc_lethal_impact: bool = False

    # v16 takeoff-arc gap mask (PRECISION): when GROUNDED, veto a jump action
    # whose counterfactual takeoff arc lands on a deadly mine at BOTH the
    # full-hold and tap extremes — i.e. jumping now (at any hold) falls into the
    # mine gap, so the correct move is to keep running and build speed. Masks
    # only the fatal jump direction(s), and only when a grounded (non-jump)
    # option survives, so the agent is never trapped. Targets the PRECISION
    # failure mode: missed platforming jumps across gaps that end in mine death.
    # Off by default; does nothing when disabled.
    takeoff_arc_gap_mask: bool = False

    # v17 count-based episodic visitation reward bonus (CHAMBER_MAZE exploration).
    # Each step the env maps the ninja to a 12px cell on a per-episode visit-count
    # grid and adds ``visitation_bonus_coef / sqrt(count_after_visit)`` to the step
    # reward — a robust count-based novelty signal (first visit -> full coef, decays
    # on revisits) that replaces the dead RND intrinsic for maze exploration. The
    # bonus is added AFTER the reward calculator (so it is NOT subject to PBRS
    # death-clawback) and lands in the same single-head reward stream that feeds
    # GAE, exactly like the (now-defunct) RND intrinsic. 0.0 = off (default).
    visitation_bonus_coef: float = 0.0

    # v26 W4 topological ROOM novelty. CHAMBER_MAZE T3 sits at 0.71 over
    # 3027-frame episodes and is the only bucket that times out. Every intrinsic
    # available today (RND / NovelD / E3B / visitation / frontier) is STATE
    # novelty, which saturates in a maze: by the time it matters the agent has
    # stood in every corridor, so every cell is stale and the bonus is uniformly
    # near zero. What it does not know is the ORDER — which room to enter next.
    # This partitions the reachable space into rooms (see region_novelty.py) and
    # pays a flat bonus for the FIRST entry to each room per episode, so a room
    # stays novel until entered no matter how stale the corridor to it is.
    # Added post-clawback in the same slot as the visitation bonus. 0.0 = off.
    region_novelty_coef: float = 0.0

    # v17 P2: make the visitation bonus d*-DIRECTED. When True, each cell's
    # count-based novelty is scaled by max(VISIT_DIRECTED_FLOOR, 1 - d*_norm),
    # where d*_norm is the per-level-normalized free-space cost-to-go toward the
    # ACTIVE objective (switch pre / exit post). Amplifies exploration along the
    # route to the goal and damps it in dead-ends (which raise d*) — countering the
    # reward pipeline's tendency to make dead-end probing net-negative, without
    # rewarding aimless dead-end wandering. No effect unless visitation_bonus_coef>0.
    visitation_directed: bool = False

    # v18 C1 go-to-frontier shaping. Each step the env tracks the distance (in
    # 12px cells) from the ninja to the NEAREST frontier cell (reachable and
    # unvisited this episode) and pays
    # ``coef * (hwm - d)`` whenever d drops below the episode's running
    # high-water-mark, then ratchets hwm down to d. The hwm RE-ARMS to the
    # current distance whenever a new cell is first visited (the frontier set
    # changed), so the bonus pays exactly for purposeful travel TOWARD
    # unexplored space (e.g. backtracking out of an exhausted dead-end to the
    # nearest open junction) and cannot be farmed by oscillation. Added
    # post-clawback into the single-head reward stream, like the visitation
    # bonus. 0.0 = off (default). Suggested 0.03.
    frontier_approach_bonus: float = 0.0

    def __post_init__(self):
        """Validate environment configuration."""
        if self.enable_logging:
            logging.basicConfig(level=logging.DEBUG)
            logging.info("Environment logging enabled")

        if self.layer3_all_masked_fallback not in {"noop"}:
            raise ValueError(
                "layer3_all_masked_fallback must currently be 'noop'"
            )

    @classmethod
    def for_training(cls, **kwargs) -> "EnvironmentConfig":
        """Create configuration optimized for training."""
        return cls(**kwargs)

    @classmethod
    def for_evaluation(cls, **kwargs) -> "EnvironmentConfig":
        """Create configuration optimized for evaluation."""
        return cls(eval_mode=True, **kwargs)

    @classmethod
    def for_visual_testing(cls, **kwargs) -> "EnvironmentConfig":
        """Create configuration for human playback / video capture.

        The policy observation is identical; this only enables the pygame window
        for interactive debugging.
        """
        return cls(
            render=RenderConfig(render_mode="human"),
            frame_skip=1,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for backward compatibility."""
        return {
            "seed": self.seed,
            "eval_mode": self.eval_mode,
            "custom_map_path": self.custom_map_path,
            "enable_logging": self.enable_logging,
            "render_mode": self.render.render_mode,
            "enable_animation": self.render.enable_animation,
            "enable_debug_overlay": self.render.enable_debug_overlay,
            "pbrs_gamma": self.pbrs.pbrs_gamma,
            "reward_config_enabled": self.reward_config is not None,
            "reward_phase": self.reward_config.training_phase
            if self.reward_config
            else "N/A",
            "pbrs_objective_weight": self.reward_config.pbrs_objective_weight
            if self.reward_config
            else "N/A",
            "time_penalty_per_step": self.reward_config.time_penalty_per_step
            if self.reward_config
            else "N/A",
            "debug": self.graph.debug or self.reachability.debug,
            "frame_skip": self.frame_skip,
            "enable_layer3_macro_action_mask": self.enable_layer3_macro_action_mask,
            "layer3_duration_bins": self.layer3_duration_bins,
            "layer3_all_masked_fallback": self.layer3_all_masked_fallback,
        }


def validate_config(config: EnvironmentConfig) -> bool:
    """Validate environment configuration for common issues."""
    try:
        if config.render.render_mode == "human" and not config.render.enable_animation:
            logging.info(
                "Human render mode without animation - consider enabling animation for better visualization"
            )

        if config.render.enable_debug_overlay and not config.enable_logging:
            logging.warning(
                "Debug overlay enabled without logging - consider enabling logging for debug info"
            )

        return True

    except Exception as e:
        logging.error(f"Configuration validation failed: {e}")
        return False
