"""Dynamic truncation limit calculator based on level complexity.

The episode budget is, in practice, ``BASE_TIME_PER_NODE * TRUNCATION_MULTIPLIER
* sqrt(free_space)``, floored by the level's own route cost and clipped to
``[MIN_TRUNCATION_FRAMES, HARD_TRUNCATION_CEILING]``. Two things measured in
2026-07 (``npp-rl/scripts/probe_truncation_binding_term.py`` and
``npp-rl/scripts/measure_route_budget.py``) should be kept in mind before
touching any constant here:

1. **The surface-area term is the only one that ever binds.** The mine term is
   identically zero (see ``TIME_PER_MINE``) and the ``cost_to_completion`` floor
   measures 50-689 frames against surface terms of 2,591-24,826 -- two orders of
   magnitude smaller. So this file's effective behaviour is one term, and any
   retune is a retune of that term.
2. **The budget bore no relation to how long a win actually takes.** Planning and
   validating real winning routes gave median solves of ~49 frames on WALL_JUMP/T2
   and ~146 frames on PLATFORMING/T2, against budgets of 2,771 and 17,338 -- 41x
   and 116x headroom respectively. The budget rank order matched the det-eval
   FAILURE rank order across the whole benchmark, and the two cells that had never
   cleared the phase-4 gate were the two longest.

Why that matters: a non-progressing episode that is allowed to run 17,000-24,000
frames makes ~96-98% of a blocking cell's collected frames one immobile absorbing
state, and lets a -0.05/frame shaping term integrate to 80-115x the completion
reward. Bounding the episode addresses both at once. See
``npp-rl/docs/RESEARCH_RESPONSE_generalization.md`` §1 and P1, and
[Pardo et al., *Time Limits in RL*, arXiv:1712.00378] for why a time limit far
outside the discount horizon causes state-aliasing (gamma 0.997 is a ~333-frame
horizon; 24,000 frames is 72x that).
"""

import numpy as np

# Frames per sqrt(reachable node), before TRUNCATION_MULTIPLIER.
#
# 20.0 -> 5.0 (2026-07-30). MEASURED CALIBRATION, not a guess: at 20.0 every cell
# in the benchmark sat between 41x and 116x the frames a validated planner route
# needs to win it, and the most over-budgeted cells were exactly the cells that
# had never cleared the curriculum gate. 5.0 keeps every measured cell well above
# its p90 winning-route length while cutting the worst offenders ~4x.
#
# This scales ONLY the surface-area term. The ``cost_to_completion`` floor below
# is untouched, so the "long OFFICIAL/FAR round-trip" levels that motivated
# COST_TIME_FACTOR 3.0 -> 3.5 keep exactly the protection that change bought them.
# Re-run ``npp-rl/scripts/measure_route_budget.py`` after any further change: it
# reports, per cell, the validated win length against the budget and flags any
# cell a candidate value would STARVE.
BASE_TIME_PER_NODE = 5.0

# DEAD PARAMETER -- kept only so the signature stays stable. 0.0, deliberately.
#
# ``NppEnvironment`` has always called ``set_level_truncation_limit(surface_area,
# 0, ...)`` with a hardcoded literal zero, so this term has never contributed a
# single frame on any level in any run, and the "increased for toggle mine state
# complexity" tuning history attached to its former value of 75.0 describes an
# effect that never existed.
#
# It is NOT being revived. Plumbing the real reachable-mine count would ADD
# ~53 mines * 75 * 10 = ~39,750 frames to a PLATFORMING episode -- the exact
# opposite of what the measurement above calls for. Recorded as zero rather than
# deleted so that a future attempt to "fix the plumbing" finds this note first.
TIME_PER_MINE = 0.0

# Tightened 2026-06: ~66% of episodes were timeouts at ~6.7k frames, so failed
# episodes burned a lot of wall-clock wandering. A typical small-level solve is
# only ~200-800 frames, so we still leave generous headroom while cycling failed
# episodes ~2.5x faster (multiplier 25->10, min 1200->600).
TRUNCATION_MULTIPLIER = 10  # was 25
MIN_TRUNCATION_FRAMES = 600  # was 1200 (~10s floor for tiny levels)
# Frames of budget per pixel of free-space cost-to-completion. Optimal traversal
# is ~cost/MAX_HOR_SPEED (~0.3 frame/px); ~6.5x exploration headroom -> 2.0.
# Raised 2.0 -> 3.0 (2026-06-23): FAR_OBJECTIVE is a DOUBLE round-trip (climb to
# the switch, then climb back to the exit) — cost_to_completion already covers the
# full round-trip, but the agent climbs slower than optimal, so it was reaching the
# switch (~40%) yet timing out before the exit (exit_given_switch=0, 100% timeout).
# The earlier 4.0 fed a high-gamma value divergence; that is now retired by PopArt +
# gamma=0.997, so 3.0 buys the round-trip headroom safely.
# v17 (2026-07-09): 3.0 -> 3.5. The longest OFFICIAL/FAR levels still timed out
# pre-completion (v16 solved only 3/45 held-out OFFICIAL, several hit the budget);
# this floor is a per-level FLOOR keyed on the level's OWN route cost, so it only
# lengthens genuinely long episodes and leaves short/medium levels on their
# (smaller) formula limit untouched.
#
# 3.5 -> 20.0 (2026-07-30). THIS IS THE OTHER HALF OF THE SURFACE-TERM CUT, and it
# is not optional -- measured, cutting the surface term alone STARVES the one cell
# the floor was supposed to protect.
#
# The floor never bound on any cell at 3.5: it landed 2 orders of magnitude below
# the surface term everywhere, so the belief that it "protects long-route levels"
# was untested. Cutting BASE_TIME_PER_NODE 4x did not hand protection over to it,
# it just left FAR_OBJECTIVE/T2 short. Validated planner routes, against the
# budget each setting grants:
#
#   cell                 win_p90   old    B=5,C=3.5      B=5,C=20
#   FAR_OBJECTIVE/T2       477    6928   1732 ( 3.6x)   3934 ( 8.2x)  <- the fix
#   OFFICIAL/T1            367   18335   4583 (12.5x)   4583 (12.5x)
#   PLATFORMING/T2         147   17190   4297 (29.2x)   4297 (29.2x)
#   PLATFORM_STAIRS/T1     348   24000   6206 (17.8x)   6206 (17.8x)
#   PLATFORM_STAIRS/T2     111   24000   6198 (55.8x)   6198 (55.8x)
#   WALL_JUMP/T2            68    2771    692 (10.2x)    692 (10.2x)
#
# FAR_OBJECTIVE/T2 has the SMALLEST surface area of any measured cell (1,200) and
# the LONGEST route (477 frames) -- it is a double round-trip through a small
# space. That is precisely the case a surface-area budget mis-ranks and a
# route-cost floor exists to catch, and `cost_to_completion` does rank it
# correctly (196.7, the largest of any cell); the factor was simply too small for
# the floor to ever take over.
#
# Only CHAMBER_MAZE/T2 else moves (647 -> 982, a maze, where more time is
# defensible). Every other cell is byte-identical, because their surface term
# still dominates.
COST_TIME_FACTOR = 20.0
# Absolute ceiling regardless of level size (prevents a pathological estimate
# from producing an unbounded episode). v17: 18000 -> 24000 (~6.7 min @ 60 fps).
# This is the ONLY hard clamp on the longest levels; at 18000 a long round-trip
# needing ~24k frames for a non-optimal-but-valid solve was truncated before it
# could ever complete. Raising ONLY the ceiling targets exactly those long levels
# (anything under 18000 is unaffected), so throughput on the common short/medium
# levels — governed by the smaller formula limit — is unchanged.
HARD_TRUNCATION_CEILING = 24000


def calculate_truncation_limit(
    surface_area: float,
    reachable_mine_count: int,
    cost_to_completion: float | None = None,
) -> int:
    """
    Calculate dynamic truncation limit based on level complexity.

    Formula:
        base_time   = sqrt(surface_area) * BASE_TIME_PER_NODE
        truncation  = base_time * TRUNCATION_MULTIPLIER
        truncation  = max(truncation, cost_to_completion * COST_TIME_FACTOR)
        truncation  = clip(truncation, MIN_TRUNCATION_FRAMES,
                           HARD_TRUNCATION_CEILING)

    Args:
        surface_area: Number of reachable nodes (from PBRS)
        reachable_mine_count: **Ignored.** See ``TIME_PER_MINE`` -- the caller has
            always passed 0 and the term is deliberately not revived. Retained so
            the signature stays stable for existing call sites and tests.
        cost_to_completion: Free-space cost-to-completion at spawn (~pixels);
            acts as a per-level FLOOR so long, winding levels get proportional
            time. Optional.

    Returns:
        Truncation limit in frames, clamped to
        ``[MIN_TRUNCATION_FRAMES, HARD_TRUNCATION_CEILING]``.

    Examples (at BASE_TIME_PER_NODE = 5.0):
        WALL_JUMP/T2   (~192 nodes):   sqrt(192)*5*10  =   692 frames
        PLATFORMING/T2 (~7500 nodes):  sqrt(7500)*5*10 = ~4330 frames
    """
    # Base time from traversal complexity (surface area). This is the only term
    # that binds in practice; see the module docstring.
    base_time = np.sqrt(surface_area) * BASE_TIME_PER_NODE

    truncation_limit = base_time * TRUNCATION_MULTIPLIER

    # Path-length-aware floor: the free-space cost-to-completion (~pixels) is a
    # direct measure of how far the goal is, so a LONG, winding level gets enough
    # frames to be solved/explored even if its surface area is modest. Without
    # this a 100-1000 env-step level would be truncated before the agent could
    # ever reach the goal (a hard ceiling on learnable difficulty).
    if cost_to_completion is not None and cost_to_completion > 0:
        truncation_limit = max(truncation_limit, cost_to_completion * COST_TIME_FACTOR)

    # Previously this read
    #     cap = min(HARD_TRUNCATION_CEILING, max(MAX_TIME_IN_FRAMES, limit))
    #     return int(np.clip(limit, MIN_TRUNCATION_FRAMES, cap))
    # which is EXACTLY equivalent to the clip below -- verified over every integer
    # input in [0, 40000], zero differences. MAX_TIME_IN_FRAMES (4500) had no
    # effect on the result at all, despite its constant being documented as "Cap
    # on the dynamic truncation limit": whenever the formula exceeded 4500 the
    # max() returned the formula and the cap collapsed to HARD_TRUNCATION_CEILING,
    # and whenever it was below 4500 the value was already under the cap. Written
    # directly so the real bound is readable.
    return int(
        np.clip(truncation_limit, MIN_TRUNCATION_FRAMES, HARD_TRUNCATION_CEILING)
    )
