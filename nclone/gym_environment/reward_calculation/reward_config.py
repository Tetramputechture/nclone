"""Centralized reward configuration for HWM A* reward pipeline.

Simplified from the original curriculum-adaptive PBRS configuration.
Fixed sparse reward magnitudes (no curriculum scaling). The only
adaptive components are entropy coefficient and RND beta, which
follow a 3-phase schedule based on training progress.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

from .reward_constants import (
    R_SWITCH,
    R_EXIT,
    R_LOCKED_DOOR_SWITCH,
    R_DEATH,
    R_DEATH_MINE,
    R_GOLD,
    R_ALL_GOLD,
    GOLD_FRAME_VALUE,
    HWM_ALPHA,
    LATTICE_PBRS_WEIGHT,
    LATTICE_PBRS_GAMMA,
    TIME_PENALTY_PER_STEP,
    TRUNCATION_PENALTY,
)


@dataclass
class RewardConfig:
    """Single source of truth for reward component lifecycle.

    HWM pipeline uses fixed reward magnitudes:
    - R_SWITCH = 0.5 (switch activation / phase bonus)
    - R_EXIT = 5.0 (level completion)
    - R_DEATH = -0.5 (any death)
    - HWM_ALPHA = 1.0 (shaping weight, constant)

    Adaptive components (3-phase schedule):
    - entropy_coefficient: 0.05 -> 0.02 -> 0.01
    - rnd_beta: managed externally by ScheduleCallback
    """

    total_timesteps: int = 10_000_000
    current_timesteps: int = 0
    recent_success_rate: float = 0.0

    _curriculum_stage: int = 0
    _num_curriculum_stages: int = 1

    # --- NPP-14 / NPP-45: BSRS blend with HWM (A* distance surrogate) ---
    # When enable_bsrs_blend is True, HWM dense shaping is scaled by beta_blend(timesteps)
    # and the trainer adds (1-beta) * eta_bsrs * (gamma * V(s') - V(s)) on each step.
    enable_bsrs_blend: bool = False
    eta_bsrs: float = 0.005  # Theorem-compatible with gamma=0.99: (1-gamma)/(1+gamma) ≈ 0.005025
    bsrs_gamma: float = 0.99  # Must match PPO gamma; used in BSRS term only
    bsrs_blend_horizon_fraction: float = 0.7  # beta goes 1→0 linearly over this fraction of total_timesteps

    # --- Gold collection (docs/GOLD_COLLECTION_HANDOFF.md) ---
    # Off by default for a clean A/B against the completion-only pipeline. When
    # True, the calculator pays a per-gold pickup reward calibrated so each gold
    # is worth GOLD_FRAME_VALUE (120) frames of saved time — the exchange rate is
    # tied to TIME_PENALTY_PER_STEP (see reward_constants). Gold is a soft
    # objective: it never gates completion. No all-gold bonus by default (uniform
    # valuation); set _all_gold_bonus_override for the collect-everything variant.
    enable_gold_rewards: bool = False
    _gold_reward_per_piece_override: Optional[float] = None
    _all_gold_bonus_override: Optional[float] = None

    # --- Locked-door switches (docs/ENTITY_AWARE_LATTICE_PLANNER_HANDOFF.md) ---
    # When True, activating a NECESSARY locked-door switch (one the planner's
    # door-dependency solver marks as required for completion) pays a one-shot
    # milestone bonus, teaching the agent to route through gating switches.
    # Opening a non-necessary door pays nothing (no detour incentive).
    enable_locked_door_rewards: bool = False
    _locked_door_switch_reward_override: Optional[float] = None

    # --- Lattice / free-space PBRS (reward_constants LATTICE_PBRS_* block) ---
    # When True, a true policy-invariant potential-based shaping term
    # F = pbrs_weight * (pbrs_gamma * Phi(s') - Phi(s)), with
    # Phi(s) = -normalized cost-to-completion from the env's reachability source,
    # REPLACES the HWM ratchet + backtrack penalty as the sole dense extrinsic
    # signal. Requires the env's ReachabilityConfig.use_lattice_potential so the
    # substituted (lattice/free-space) cost-to-completion reaches the obs
    # distances; the trainer wires that automatically when this is set.
    enable_lattice_pbrs: bool = False
    _pbrs_weight_override: Optional[float] = None
    _pbrs_gamma_override: Optional[float] = None
    # Fixed reference for the PBRS potential normalization (D0 = cost-to-completion
    # denominator), in the same units as the active cost-to-go source. Default None
    # = per-EPISODE D0 (the spawn cost-to-completion), which makes the per-step
    # shaping magnitude scale as 1/route-length: on a long route (e.g. the tight
    # mine-morph rungs, D0~4438) a forward step earns ~6x less than on a short one
    # (D0~763), so the dense gradient fades exactly where it is needed most. Set a
    # FIXED reference (e.g. the hardest rung's D0) so per-step magnitude is
    # ROUTE-LENGTH-INVARIANT across a geometry-morphing curriculum — each pixel of
    # progress pays the same reward on every rung (the total return then scales with
    # route length, which the value head absorbs). Still policy-invariant within an
    # episode (a constant rescale of a state-only potential). See the
    # geometry-morphing curriculum docs.
    _pbrs_d0_ref_override: Optional[float] = None
    # Fraction of the terminal PBRS potential paid on DEATH (a true terminal).
    #   1.0 = strict policy-invariant: death pays -w*Phi(s_last) so shaping
    #         telescopes to a constant +w for every terminal (no approach gradient
    #         survives death -> on a hard exploration level the agent collapses to
    #         a fast-death local optimum, empirically observed).
    #   0.0 = exploration-friendly (DEFAULT): death pays NO terminal potential, so
    #         the dense approach reward w*(Phi(s_last)+1) the agent earned on the
    #         way IS BANKED -> dying nearer the goal is strictly better than dying
    #         at spawn, giving a per-episode gradient that rewards reaching deeper.
    #         Not policy-invariant, but a desired exploration bias when the agent
    #         cannot otherwise leave spawn. (Win always pays the FULL terminal
    #         potential, so completion's +w bonus is unaffected.)
    pbrs_death_clawback: float = 0.0

    # Fraction of the gold reward BANKED THIS EPISODE that is clawed back on DEATH.
    #   0.0 = disabled / legacy (DEFAULT): gold collected en route stays banked even
    #         if the ninja then dies, so a "grab-gold-then-mine-die" trajectory nets
    #         +gold - death_penalty, making risky gold detours look cheaper than they
    #         are (the agent chases gold into hazards).
    #   1.0 = claw back 100%: on death, SUBTRACT the full episode-cumulative gold
    #         reward from the terminal reward so the episode return contains no net
    #         gold contribution -> dying erases the gold's value and the policy stops
    #         treating hazardous gold as free. Fractional values (e.g. 0.5) claw back
    #         a proportion. Applies ONLY on death (has_died) — a WIN or a truncation
    #         always keeps the full gold, so completing the level with gold is
    #         rewarded exactly as before.
    gold_death_clawback: float = 0.0

    # Completion-primary gold BANKING (V25). Strictly stronger than the death
    # clawback above, which only covers deaths.
    #   False = legacy (DEFAULT): gold pays the moment it is picked up, so a
    #         gold detour that ends in a TIMEOUT still banks the gold. Measured
    #         consequence (v24 scratch, 18M -> 21M): det-eval completion fell
    #         18.8% -> 16.1% while gold/solve rose 7.0 -> 17.4 and timeouts
    #         nearly doubled — gold was outcompeting completion even at
    #         gold_death_clawback=1.0.
    #   True  = bank the per-gold pickup reward for the whole episode and pay it
    #         out ONLY on a completing episode (real exit, or the
    #         reverse-curriculum intermediate goal). A death or a truncation
    #         pays exactly 0 gold. Gold therefore can never trade against
    #         completion: an unfinished gold grab is worth nothing, while a win
    #         still collects the full gold value.
    # When True, ``gold_death_clawback`` becomes a no-op (nothing was paid to
    # claw back) and is ignored.
    gold_bank_until_completion: bool = False

    # --- Goal-distance reverse curriculum ---
    # Move the GOAL closer to spawn early, then outward. The "near goal" is a
    # cost-to-go THRESHOLD on the (reachable) free-space path: the episode SUCCEEDS
    # (pays the exit reward + terminates) as soon as the agent's cost-to-completion
    # drops to <= C_spawn*(1-goal_distance_fraction), where C_spawn is the spawn
    # cost-to-completion. goal_distance_fraction=1.0 -> true goal only (off);
    # f=0.3 -> succeed after covering 30% of the path. Annealed low->1.0 as the
    # agent learns. Bootstraps long-horizon (FAR_OBJECTIVE) + makes the sparse
    # terminal reachable + shortens early episodes (helps value stability).
    enable_goal_distance_curriculum: bool = False
    goal_distance_fraction: float = 1.0

    # --- Per-leg waypoint PBRS (docs/LONG_HORIZON_DISTANCE_HANDOFF.md) ---
    # When True, a per-leg potential term F = pbrs_weight*(pbrs_gamma*Phi(s') - Phi(s))
    # with Phi(s) = -clamp(waypoint_leg_distance / waypoint_leg_scale, 0, 1) shapes
    # progress toward the CURRENT local waypoint (from a WaypointTracker, fed into obs
    # as ``waypoint_leg_distance`` + ``waypoint_index``). The potential basis is REBASED
    # every time ``waypoint_index`` advances, so each leg is short and the per-step
    # gradient is ROUTE-LENGTH-INVARIANT -- the long-horizon distance-dropoff fix
    # (decompose one far goal into a chain of near subgoals). Mutually exclusive with the
    # HWM ratchet and the lattice/free-space PBRS: ``hwm_alpha`` is forced to 0 and the
    # backtrack penalty is skipped (the per-leg term already penalizes regress
    # symmetrically). Reuses ``pbrs_weight`` / ``pbrs_gamma``.
    enable_waypoint_pbrs: bool = False
    waypoint_leg_scale: float = 96.0  # px; denominator for the per-leg potential
    # Audit II P0-5c: DECOUPLED weight for the per-leg waypoint PBRS term. The leg
    # term historically reused ``pbrs_weight`` (2.0), banking ~+0.93/leg =>
    # ~+19.5 per 1000 px of route (~2x R_EXIT on long routes, 29x every other
    # dense term, kept in full on death since the clawback touches only the main
    # ``_prev_potential``) — drowning any precision-sized signal and unaccounted
    # in the completion-dominance calibration. None = legacy (reuse pbrs_weight);
    # set ~0.5 to bound the bank.
    waypoint_pbrs_weight: Optional[float] = None
    # Explicit per-waypoint REACH BONUS: paid each time the WaypointTracker cursor
    # advances forward (monotonic; the tracker never retreats so it cannot be
    # farmed). Unlike the policy-invariant per-leg PBRS, this is a concrete dense
    # progress reward on the long return leg -- exactly where the agent times out.
    # 0.0 = off (pure PBRS). A densified route is ~48px/leg, so e.g. 0.05 pays
    # ~0.05 every 48px of forward route progress, route-length-invariant.
    waypoint_reach_bonus: float = 0.0

    # --- Frontier watermark bonus (long-horizon "loop"-level commitment) ---
    # On long loop levels the agent times out wandering a safe basin instead of
    # committing to the long climb to the goal: the dense free-space cost-to-go
    # PBRS is correct but its per-step gradient over 600-1100px is too weak to
    # overcome the discount. When True, a small one-sided bonus pays the agent
    # ONLY when it reaches a strictly new closest-approach (a new minimum
    # free-space cost-to-go) to the current live objective this episode -- the
    # switch pre-activation, the exit post-activation. The bonus is
    # ``frontier_bonus_weight * (best_prev - now)`` (positive, proportional to the
    # improvement); a non-improving step pays nothing, so basin-wandering earns 0
    # while monotonic progress around the loop is rewarded. It REUSES the same
    # phase-aware cost-to-go (``current_goal_dist``) the PBRS potential is built
    # from -- no new distance source -- so the two signals stay consistent. The
    # episode watermark is reset on episode reset and REBASED (no spurious bonus)
    # when the live objective changes switch->exit, exactly like the waypoint
    # rebase. Off by default; additive (does not change behavior when off).
    # ``frontier_bonus_weight`` is in the same reward units as the per-step PBRS
    # term and is intentionally MUCH smaller than the sparse switch/exit rewards.
    enable_frontier_bonus: bool = False
    frontier_bonus_weight: float = 0.02

    # --- Dense mine-hazard avoidance PBRS (T3/T4 mine-death fix) ---
    # On long mine levels the agent dies threading the mines (~65% of T3/T4 FAR
    # episodes) because the ONLY mine signal in the reward is the TERMINAL death
    # penalty (-3.5) — there is no per-step gradient to steer AROUND a mine, and
    # with pbrs_death_clawback=0 a deep death even BANKS the approach reward, so
    # "die deeper" out-scores "die at spawn". This adds a policy-invariant PBRS
    # term on the normalized nearest-deadly-mine distance d in [0,1] (1=safe,
    # 0=touching), read from game_state: Phi_haz = d, F = w*(Phi' - Phi) with
    # gamma=1 (pure telescoping -> exactly 0 when no mine is near, so no survival
    # drift). Moving away from a mine pays +, moving toward pays - — a dense
    # avoidance gradient the agent can act on BEFORE dying. Deliberately a SHAPING
    # gradient, NOT a bigger terminal penalty (run-220/221 showed a larger death
    # penalty causes passive risk-aversion). Requires the policy to SEE mines
    # locally (HazardEncoder spatial readout) to be exploitable. Off by default.
    enable_hazard_pbrs: bool = False
    hazard_pbrs_weight: float = 0.5
    # v11: make the hazard gradient (a) ONE-SIDED and (b) SHORT-RANGE. The original
    # symmetric telescoping paid the agent to back AWAY from mines as much as it
    # penalized approaching, and its 168px range made the penalty field so wide the
    # agent stayed timid across whole regions -> it under-commits mined jumps.
    #   one_sided: keep only the approach PENALTY (min(0, dPhi)); dropping the
    #     retreat reward removes the "farm by backing off" incentive.
    #   range_px: rescale the potential to saturate at this distance so the
    #     gradient is LOCAL (near-contact only), not a wide timidity field.
    # The obs feature (game_state nearest-mine, /168px) is unchanged; this only
    # reshapes the reward. Defaults preserve legacy behavior when unset.
    hazard_pbrs_one_sided: bool = False
    hazard_pbrs_range_px: float = 168.0
    # Audit II P0-5b: suppress the hazard approach PENALTY on steps that strictly
    # DECREASED the goal distance (cost-to-go). The one-sided tax otherwise re-taxes
    # every NECESSARY mine-adjacent landing (~-0.35/attempt, 2.5-7x a stair rung's
    # progress credit, re-paid on every retry) — a pure per-attempt tax exactly on
    # the frontier's required maneuvers, with zero steering value when the target
    # lies BETWEEN mines. Penalty-only term, so gating cannot create farming; the
    # mine death penalty still polices contact. Off by default (legacy behavior).
    hazard_pbrs_progress_gated: bool = False
    # The norm the obs nearest-mine distance was divided by (NppEnvironment
    # _NEAREST_MINE_NORM_PX); used to recover raw px for the short-range rescale.
    hazard_pbrs_obs_norm_px: float = 168.0

    # --- Impact-survivability PBRS (target the dominant failure mode) ---
    # ~95% of deaths are impact/fall, not hazards. A dense PBRS on the SAFETY
    # potential Phi = 1 - impact_risk, where impact_risk in [0,1] measures the
    # ninja's speed INTO the look-ahead landing surface relative to that surface's
    # survivable-impact threshold (computed in npp_environment, exposed as
    # obs["_impact_risk"]). The agent is rewarded for REDUCING impact risk
    # (slowing / re-orienting before a fast landing) and penalized for building
    # it. gamma=1 telescoping -> policy-invariant within the episode; risk is 0 at
    # safe speeds and with no surface ahead, so it is a no-op except near fast
    # impacts. Helps slopes (steep-slope landings) AND all buckets (impact deaths).
    enable_impact_pbrs: bool = False
    impact_pbrs_weight: float = 0.5

    # --- Doomed-arc risk PBRS (v13 ghost-arc reward) ---
    # Phi_risk = -1 while AIRBORNE with ALL THREE committed ghost arcs ending in
    # a deadly-mine contact or a lethal impact (no steering input can save the
    # flight), else 0 (computed in npp_environment from the P0-1 bundle, exposed
    # as obs["_arc_doomed"]). PBRS-shaped (gamma=1 telescoping -> policy-
    # invariant): the penalty lands on the STEP THE STATE BECAME UNRECOVERABLE
    # — typically the bad takeoff/steer frame 30-60 frames before the death the
    # terminal penalty would otherwise be the first signal of. This is the
    # credit-assignment fix for landing-miss mine deaths (PLATFORMING T2).
    enable_arc_risk_pbrs: bool = False
    arc_risk_pbrs_weight: float = 0.5

    # --- Completion SPEED bonus (encourage direct + fast successful routes) ---
    # A terminal bonus paid ONLY on a real WIN, scaling with the normalized time
    # remaining at completion (game_state time_remaining in [0,1]): finishing with
    # more time left -> bigger bonus -> the agent is rewarded for the FASTER, more
    # DIRECT route among winning routes. Theoretically sound: it is a pure terminal
    # reward, so it does NOT touch the per-step PBRS telescoping / policy-invariance;
    # it never pays on death or timeout, so completion always dominates dying; and
    # kept BELOW R_EXIT so winning slowly still beats dying (no speed-vs-survival
    # inversion). 0.0 = off. Complements the per-step time penalty (which discourages
    # dawdling) with a positive pull toward fast wins.
    completion_speed_bonus: float = 0.0
    # v15: reference the speed bonus to a PER-LEVEL PAR (optimal route length)
    # instead of the generous truncation budget. The budget form saturates near
    # 1.0 on short levels (any competent win leaves ~full budget) -> no
    # fast-vs-fastest gradient there; it is only learnable on long levels where
    # the budget is tight. The par form gives a learnable gradient on EVERY level:
    #   par_frames  = optimal_path_length(px) / completion_speed_par_speed_px
    #   speed_bonus = completion_speed_bonus
    #                 * clip(par_frames / max(frames_used, par_frames), 0, 1)
    # Efficiency-ratio form => in (0, 1], = 1.0 at/under par, decaying as slower.
    # Every safety property of the base bonus is preserved: terminal + win-gated
    # (never farmed, never paid on death/timeout), bounded < R_EXIT, and the ratio
    # CLAMPS at 1.0 so "beating par" cannot inflate it past completion_speed_bonus
    # (no incentive for risky sub-par shortcuts). Falls back to the legacy
    # budget/time_remaining form when optimal_path_length is unavailable. Off by
    # default (legacy budget form) so it is A/B-able against the current behavior.
    completion_speed_par_norm: bool = False
    # Effective route speed (px/frame) converting optimal route px -> par frames.
    # Conservative (< MAX_HOR_SPEED = 3.33) to account for jumps/turns/climbs;
    # because the bonus is a RATIO this only sets where the gradient is steepest,
    # not the ceiling, so mild misspecification is harmless.
    completion_speed_par_speed_px: float = 2.5
    # SINGLE-LEVEL SPEEDRUN: reference the speed bonus to the fastest win this
    # env has ever recorded on the level (a personal-best RECORD) instead of a
    # static par. Motivation: both the par and budget forms clamp at 1.0, so once
    # the policy is at/under par the time gradient VANISHES -- fine for generalist
    # training (it removes any pull toward risky sub-par shortcuts) but fatal when
    # the whole objective is to grind toward the theoretical minimum frame count.
    #   frac = best_frames_so_far / frames_used     (best = record BEFORE this ep)
    #   bonus = completion_speed_bonus * clip(frac, 0, completion_speed_record_cap)
    # Tying the record pays 1.0x; BEATING it pays >1.0x, and the record then
    # ratchets down, so there is always a live gradient no matter how fast the
    # policy already is. The first win pays 1.0x and seeds the record.
    # Safety: still terminal-only and win-gated (never farmed, never paid on
    # death/timeout), so PBRS telescoping/policy-invariance is untouched. The cap
    # bounds it: keep completion_speed_bonus * cap < R_EXIT (10.0, see
    # reward_constants.R_EXIT) so a slow win still strictly beats dying
    # (3.0 * 1.5 = 4.5 << 10.0). NOTE: the module docstring above says
    # "R_EXIT = 5.0"; that is stale -- the shipped constant is 10.0.
    completion_speed_record_norm: bool = False
    completion_speed_record_cap: float = 1.5
    # DISCRETE new-record bonus, paid once when a genuine spawn-to-exit run beats
    # the standing record. Motivation (measured on SI-000 at a 479-frame record):
    # the record-RATIO form is almost flat near the record, because 479/465 =
    # 1.03 -- so closing the ENTIRE remaining 14-frame gap to optimal was worth
    # only +0.26 on a ~14.0 episode return (1.86%; 0.13% per frame). That is
    # below what PPO can resolve against episode-to-episode variance, which is
    # why every configuration converged to ~479-486 and stopped. The ratio form
    # tells the agent how good it is; it does not pay it enough to improve.
    #
    # Safety: terminal-only, win-gated, and SPAWN-gated (a checkpoint-restored
    # episode cannot claim it). Unfarmable by construction -- claiming it raises
    # the bar, so each payment requires a strictly new best, and the total number
    # of payments over a run is bounded by the number of genuine improvements.
    # It is NOT potential-based, so it does shift the optimum: deliberately, and
    # toward faster completion. Watch success_rate for risk-taking regressions.
    completion_speed_record_bonus: float = 0.0

    # --- PACE (split-time) PBRS ------------------------------------------
    # Dense per-frame answer to "am I ahead of or behind the reference run's
    # pace at my current progress?", using a table of cost-to-go -> reference
    # frame built by scripts/build_pace_table.py.
    #
    #   Phi(s) = -(frames_elapsed - reference_frame_at(cost_to_go)) / pace_scale
    #
    # Why this and not just a bigger terminal bonus: the completion-time
    # objective is TERMINAL, so the agent learns whether it was fast ~480 frames
    # after the maneuver that decided it. Measured on SI-000, closing the entire
    # remaining 479->465 gap was worth 1.86% of episode return (0.13%/frame) --
    # below what PPO resolves against episode variance, which is what pinned
    # every configuration at 479-507.
    #
    # It telescopes to Phi_T - Phi_0 = -(total_frames - reference_total), i.e.
    # the dense signal sums to EXACTLY the time saved versus the reference, so
    # it is policy-invariant and cannot be farmed: standing still grows elapsed
    # while progress does not, so loitering is strictly penalised. Unlike the
    # flat time penalty it pays AT the frames where time is actually lost.
    enable_pace_pbrs: bool = False
    pace_pbrs_weight: float = 1.0
    pace_scale: float = 60.0     # frames; normalizes the potential to ~O(1)
    pace_table_path: Optional[str] = None

    # --- MOMENTUM (closing-speed) PBRS -------------------------------------
    # Credit-assignment aid for momentum-conversion maneuvers whose payoff is
    # hundreds of frames away. Motivating case: a corner walljump that flips
    # xspeed -3.19 -> +1.08, doing a turnaround, a climb and a re-acceleration in
    # one move. Under terminal-only speed reward the agent gets no signal for
    # ~340 frames, so it never links the 3-frame input pattern to the payoff.
    #
    # The potential is the CLOSING SPEED on the goal (px/frame, normalized), NOT
    # |velocity|. Two reasons |v| would be wrong:
    #   * the walljump momentarily REDUCES |vx| (3.19 -> 1.08), so a |v| potential
    #     would actively penalize exactly the maneuver we want;
    #   * speed that is not directed at the goal is worthless here.
    #
    # Honest limitation: PBRS telescopes to Phi_T - Phi_0 regardless of how many
    # frames were taken, so this term CANNOT by itself reward "finish sooner" --
    # only the per-step time penalty does that. What it buys is dense, immediate
    # credit for recovering closing speed, which is the part the agent currently
    # has to discover blind. Policy-invariant by construction (Ng et al.), so it
    # cannot introduce a reward hack; worst case it is inert.
    enable_momentum_pbrs: bool = False
    momentum_pbrs_weight: float = 0.3

    # --- Grounded-landing high-water-mark bonus (safe precision-jump shaping) ---
    # Rewards the ninja for reaching a NEW closest-ever *grounded* contact with the
    # current sub-goal: safely landing (or standing) on a surface closer to the goal
    # than any surface it has touched this episode. Motivation: the per-step
    # free-space/jump PBRS rewards proximity to the goal even while AIRBORNE and
    # drifting toward a gap -- it weights a mid-air drift the same as a completed
    # jump. On stair/precision buckets (single-tile platforms, mines under every
    # gap, zero recovery) that under-rewards the thing that actually matters:
    # sticking the landing. This term pays ONLY on solid ground and ONLY for strict
    # new-closest progress (a high-water-mark, never re-paid), so it CANNOT be
    # farmed by oscillating (jump forward, fall back, jump forward again) -- each
    # payment requires a landing strictly closer than EVERY prior landing this
    # episode. Since landing on a mine is DEATH (not a surface contact), it
    # exclusively rewards SAFE landings. It is a one-shot ratchet (gamma-free),
    # bounded by the weight (telescopes to w*(D0 - min_landing_cost)/D0 <= w), so it
    # does NOT touch the per-step PBRS telescoping / policy-invariance of
    # enable_lattice_pbrs. Off by default; intended for the precision buckets
    # (PLATFORM_STAIRS, PRECISION). Reuses the PBRS D0 spawn cost-to-completion as
    # the normalizer so the magnitude is route-length-invariant.
    #
    # Fires ONLY on the airborne->contact TRANSITION (an actual landing / wall-catch)
    # AND only when that contact is strictly closer to the goal than any prior safe
    # contact -- continuous ground contact (flat running) earns nothing. Each landing
    # is clamped to <= landing_pbrs_weight. Weight kept small so it is a subordinate
    # precision nudge, never a term that can rival the sparse completion reward.
    enable_landing_pbrs: bool = False
    landing_pbrs_weight: float = 0.1

    # Progress-stall (dither) penalty [v22, §3-A]. Under the active PBRS pipeline
    # (waypoint/lattice) the backtrack penalty above is disabled and PBRS (gamma=1)
    # telescopes to ~0 on net-zero oscillation, so DITHERING near an obstacle (the
    # near-switch wall pocket) is effectively FREE. This term makes it cost
    # something: it keeps a high-water-mark of the best (min) cost-to-go and, when
    # cost-to-go has not improved for >= ``stall_patience_frames`` sim frames WHILE
    # the ninja stays within ``stall_radius_px`` of where it was (low position
    # variance -- so a real jump/traverse, which moves a lot, never triggers it),
    # applies a small ESCALATING penalty that grows the longer the stall persists,
    # capped at ``stall_penalty_cap`` per step. Rebased on the switch->exit flip
    # (like the landing HWM) so the basis change is not counted as a stall.
    # Off unless ``enable_stall_penalty``.
    enable_stall_penalty: bool = False
    stall_penalty_weight: float = 0.01
    stall_patience_frames: int = 30
    stall_radius_px: float = 24.0
    stall_penalty_cap: float = 0.05

    # HAZARD EXEMPTION for the stall penalty. Holding position until a patrol
    # clears is the CORRECT play against a moving threat, and it is
    # indistinguishable from dithering by the test above: cost-to-go does not
    # improve and the ninja does not move. Measured on the gamma-1 DRONE bucket,
    # patrol drones run at 1.143 px/frame over circuits from 31 px (period ~124
    # frames) to 855 px (period > 1500), so a worst-case wait for a window is
    # ~600-700 frames. Un-exempted that costs ~-9.3 against R_EXIT = 10 -- i.e.
    # waiting for a safe window costs almost exactly as much as completing the
    # level, and past ~2,100 frames of episode length the per-step penalty
    # exceeds the whole DISCOUNTED completion reward. The policy's best response
    # is to rush the drone or die trying, which is the opposite of the skill the
    # hazard curriculum exists to teach.
    #
    # While a threat is live within ``stall_hazard_exempt_radius_px`` the clock
    # FREEZES rather than resets: a genuine ditherer parked next to a hazard
    # resumes accruing the moment the threat clears, and
    # ``stall_hazard_exempt_max_frames`` bounds the total exemption per episode
    # so "camp beside a drone forever" is not a way to buy immunity. The budget
    # is set above the measured worst-case wait, not above any wait.
    stall_hazard_exempt: bool = True
    stall_hazard_exempt_radius_px: float = 48.0
    stall_hazard_exempt_max_frames: int = 900

    SUCCESS_THRESHOLD_MID: float = 0.15
    SUCCESS_THRESHOLD_LATE: float = 0.40

    # Optuna overrides (kept for hyperparameter search compatibility)
    _death_penalty_override: Optional[float] = None
    _pbrs_objective_weight_override: Optional[float] = None
    _time_penalty_per_step_override: Optional[float] = None
    _level_completion_reward_override: Optional[float] = None
    _velocity_alignment_weight_override: Optional[float] = None
    _mine_hazard_cost_multiplier_override: Optional[float] = None
    _revisit_penalty_weight_override: Optional[float] = None
    _entropy_coefficient_override: Optional[float] = None
    _hwm_alpha_override: Optional[float] = None

    @property
    def training_phase(self) -> str:
        """Current training phase based on success rate.

        Returns:
            'early': <15% success
            'mid': 15-40% success
            'late': >40% success
        """
        if self.recent_success_rate < self.SUCCESS_THRESHOLD_MID:
            return "early"
        if self.recent_success_rate < self.SUCCESS_THRESHOLD_LATE:
            return "mid"
        return "late"

    @property
    def death_penalty(self) -> float:
        """Fixed death penalty per spec."""
        if self._death_penalty_override is not None:
            return self._death_penalty_override
        return R_DEATH

    @property
    def mine_death_penalty(self) -> float:
        """Extra penalty applied on top of death_penalty when death_cause == 'mine'."""
        return R_DEATH_MINE

    @property
    def truncation_penalty(self) -> float:
        """Terminal penalty when episode ends by timeout (truncation)."""
        return TRUNCATION_PENALTY

    @property
    def level_completion_reward(self) -> float:
        """Fixed level completion reward per spec."""
        if self._level_completion_reward_override is not None:
            return self._level_completion_reward_override
        return R_EXIT

    @property
    def switch_activation_reward(self) -> float:
        """Fixed switch activation reward per spec."""
        return R_SWITCH

    @property
    def locked_door_switch_reward(self) -> float:
        """Per-necessary-locked-door-switch activation bonus (only used when
        enable_locked_door_rewards)."""
        if self._locked_door_switch_reward_override is not None:
            return self._locked_door_switch_reward_override
        return R_LOCKED_DOOR_SWITCH

    @property
    def gold_reward_per_piece(self) -> float:
        """Per-gold sparse pickup reward (only used when enable_gold_rewards)."""
        if self._gold_reward_per_piece_override is not None:
            return self._gold_reward_per_piece_override
        return R_GOLD

    @property
    def all_gold_bonus(self) -> float:
        """One-shot bonus for collecting all gold (only used when enable_gold_rewards).

        Default 0.0 under the uniform 'each gold = GOLD_FRAME_VALUE frames'
        objective. Set ``_all_gold_bonus_override`` for a collect-everything A/B.
        """
        if self._all_gold_bonus_override is not None:
            return self._all_gold_bonus_override
        return R_ALL_GOLD

    @property
    def gold_frame_value(self) -> float:
        """Frames of saved time each collected gold is worth (the exchange rate).

        Single source of truth shared with the planner and the lattice gold
        potential: the orienteering rule is 'collect a gold iff its detour costs
        fewer than this many frames'. ``R_GOLD`` is calibrated to it.
        """
        return GOLD_FRAME_VALUE

    @property
    def hwm_alpha(self) -> float:
        """HWM shaping weight. Zero when HWM is disabled (e.g. value-function-only mode).

        Forced to 0 when ``enable_lattice_pbrs`` or ``enable_waypoint_pbrs`` is
        set: those dense PBRS terms REPLACE the HWM ratchet, never stack with it.
        """
        if self.enable_lattice_pbrs or self.enable_waypoint_pbrs:
            return 0.0
        if self._hwm_alpha_override is not None:
            return self._hwm_alpha_override
        return HWM_ALPHA

    @property
    def hwm_fallback_alpha(self) -> float:
        """HWM shaping weight WITHOUT the PBRS force-to-zero.

        Used only for the per-episode degraded fallback in the reward
        calculator: when PBRS is enabled but the potential could not be
        substituted OR composed into a continuous basis, that episode
        explicitly re-enables the HWM ratchet at this weight (otherwise it
        would have no dense progress signal at all).
        """
        if self._hwm_alpha_override is not None:
            return self._hwm_alpha_override
        return HWM_ALPHA

    @property
    def pbrs_weight(self) -> float:
        """Weight on the lattice/free-space PBRS term (only used when
        ``enable_lattice_pbrs``)."""
        if self._pbrs_weight_override is not None:
            return self._pbrs_weight_override
        return LATTICE_PBRS_WEIGHT

    @property
    def pbrs_gamma(self) -> float:
        """Discount inside the PBRS shaping term (only used when
        ``enable_lattice_pbrs``). Match the PPO gamma for strict invariance."""
        if self._pbrs_gamma_override is not None:
            return self._pbrs_gamma_override
        return LATTICE_PBRS_GAMMA

    @property
    def entropy_coefficient(self) -> float:
        """3-phase entropy schedule per spec Section 7.

        Schedule:
        - Phase 1 (early, <15% success): 0.05 (explore)
        - Phase 2 (mid, 15-40% success): 0.02 (refine)
        - Phase 3 (late, >40% success): 0.01 (exploit)

        Returns:
            Entropy coefficient for PPO.
        """
        if self._entropy_coefficient_override is not None:
            return self._entropy_coefficient_override

        phase = self.training_phase
        if phase == "early":
            return 0.05
        elif phase == "mid":
            return 0.02
        return 0.01

    # ---- Deprecated properties (kept for backward compatibility) ----

    @property
    def pbrs_objective_weight(self) -> float:
        """DEPRECATED: HWM uses alpha instead of PBRS weight."""
        if self._pbrs_objective_weight_override is not None:
            return self._pbrs_objective_weight_override
        return HWM_ALPHA

    @property
    def time_penalty_per_step(self) -> float:
        """Per-step time penalty to discourage backtracking/aimless wandering.

        Uses TIME_PENALTY_PER_STEP from constants (-0.001). At 500 steps this
        costs -0.5 (same as R_DEATH), making inaction as costly as dying.
        Set via _time_penalty_per_step_override for Optuna/ablation studies.
        """
        if self._time_penalty_per_step_override is not None:
            return self._time_penalty_per_step_override
        return TIME_PENALTY_PER_STEP

    @property
    def exploration_bonus(self) -> float:
        """DEPRECATED: RND handles exploration."""
        return 0.0

    @property
    def revisit_penalty_weight(self) -> float:
        """DEPRECATED: HWM handles backtracking via watermark."""
        if self._revisit_penalty_weight_override is not None:
            return self._revisit_penalty_weight_override
        return 0.0

    @property
    def pbrs_normalization_scale(self) -> float:
        """DEPRECATED."""
        return 1.0

    def get_curriculum_pbrs_scale(self, curriculum_stage: int, num_stages: int) -> float:
        """DEPRECATED."""
        return 1.0

    @property
    def waypoint_action_diversity_bonus(self) -> float:
        """DEPRECATED."""
        return 0.0

    @property
    def completion_approach_bonus_weight(self) -> float:
        """DEPRECATED."""
        return 0.0

    @property
    def velocity_alignment_weight(self) -> float:
        """DEPRECATED."""
        if self._velocity_alignment_weight_override is not None:
            return self._velocity_alignment_weight_override
        return 0.0

    @property
    def mine_hazard_cost_multiplier(self) -> float:
        """Mine hazard cost for A* pathfinding (not a reward parameter)."""
        if self._mine_hazard_cost_multiplier_override is not None:
            return self._mine_hazard_cost_multiplier_override
        return 2.0

    @property
    def survival_bonus_per_100_frames(self) -> float:
        """DEPRECATED."""
        return 0.0

    @property
    def action_masking_mode(self) -> str:
        """Curriculum-aware action masking mode (preserved from original)."""
        curriculum_stage = getattr(self, "_curriculum_stage", 0)
        num_stages = getattr(self, "_num_curriculum_stages", 1)
        curriculum_progress = curriculum_stage / max(num_stages - 1, 1)

        if curriculum_progress < 0.3 or self.recent_success_rate < 0.15:
            return "hard"
        return "none"

    @property
    def logit_bias_strength(self) -> float:
        """Logit bias strength for soft masking mode."""
        if self.action_masking_mode != "soft":
            return 0.0
        t = (self.recent_success_rate - 0.15) / 0.25
        t = min(max(t, 0.0), 1.0)
        return 2.0 - 1.5 * t

    def update(self, timesteps: int, success_rate: float) -> None:
        """Update configuration with current training metrics."""
        self.current_timesteps = timesteps
        self.recent_success_rate = success_rate

    def sync_training_timesteps(self, timesteps: int) -> None:
        """Sync global training timestep counter (for beta_blend) without success-rate churn."""
        self.current_timesteps = int(timesteps)

    def beta_blend(self) -> float:
        """Anneal A* (HWM) weight vs BSRS: 1.0 early → 0.0 after horizon fraction of training.

        Returns 1.0 when BSRS blend is disabled (full HWM scale).
        """
        if not self.enable_bsrs_blend:
            return 1.0
        horizon = max(int(self.total_timesteps * self.bsrs_blend_horizon_fraction), 1)
        t = int(self.current_timesteps)
        if t >= horizon:
            return 0.0
        return 1.0 - (float(t) / float(horizon))

    def get_active_components(self) -> Dict[str, Any]:
        """Get current state of all reward components for logging."""
        return {
            "phase": self.training_phase,
            "timesteps": self.current_timesteps,
            "success_rate": self.recent_success_rate,
            "death_penalty": self.death_penalty,
            "level_completion_reward": self.level_completion_reward,
            "switch_activation_reward": self.switch_activation_reward,
            # PBRS shaping knobs (consumed by npp-rl PBRSLoggingCallback for the
            # curriculum/* TB panel). These are @property values, surfaced here.
            "pbrs_objective_weight": self.pbrs_objective_weight,
            "pbrs_normalization_scale": self.pbrs_normalization_scale,
            "time_penalty_per_step": self.time_penalty_per_step,
            "hwm_alpha": self.hwm_alpha,
            "entropy_coefficient": self.entropy_coefficient,
            "action_masking_mode": self.action_masking_mode,
            "logit_bias_strength": self.logit_bias_strength,
            "enable_bsrs_blend": self.enable_bsrs_blend,
            "eta_bsrs": self.eta_bsrs,
            "bsrs_gamma": self.bsrs_gamma,
            "beta_blend": self.beta_blend(),
            "enable_lattice_pbrs": self.enable_lattice_pbrs,
            "enable_waypoint_pbrs": self.enable_waypoint_pbrs,
            "waypoint_leg_scale": self.waypoint_leg_scale,
            "waypoint_pbrs_weight": self.waypoint_pbrs_weight,
            "hazard_pbrs_progress_gated": self.hazard_pbrs_progress_gated,
            "pbrs_weight": self.pbrs_weight,
            "pbrs_gamma": self.pbrs_gamma,
            # Fixed PBRS D0 reference (None = per-episode spawn D0). Surfaced so the
            # geometry-morphing curriculum's route-length-invariant calibration is
            # visible in TB.
            "pbrs_d0_ref": self._pbrs_d0_ref_override,
        }

    def __str__(self) -> str:
        return (
            f"RewardConfig(phase={self.training_phase}, "
            f"timesteps={self.current_timesteps:,}, "
            f"success={self.recent_success_rate:.1%}, "
            f"hwm_alpha={self.hwm_alpha:.1f}, "
            f"death={self.death_penalty:.2f}, "
            f"entropy={self.entropy_coefficient:.3f})"
        )
