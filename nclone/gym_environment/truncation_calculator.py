"""Dynamic truncation limit calculator based on level complexity."""

import numpy as np
from .constants import MAX_TIME_IN_FRAMES

# Heuristic parameters (tunable)
BASE_TIME_PER_NODE = (
    20.0  # frames per sqrt(node) (was 15.0, increased for more generous limits)
)
TIME_PER_MINE = (
    75.0  # frames per reachable mine (increased for toggle mine state complexity)
)
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
COST_TIME_FACTOR = 3.5
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
        base_time = sqrt(surface_area) * BASE_TIME_PER_NODE
        mine_time = reachable_mine_count * TIME_PER_MINE
        truncation = (base_time + mine_time) * TRUNCATION_MULTIPLIER

    Args:
        surface_area: Number of reachable nodes (from PBRS)
        reachable_mine_count: Number of toggle mines reachable from spawn

    Returns:
        Truncation limit in frames, clamped to [MIN, MAX]

    Examples:
        Small level (100 nodes, 0 mines):
            (sqrt(100)*20.0 + 0*75) * 25 = 5000 frames (~83 seconds)
        Medium level (400 nodes, 5 mines):
            (sqrt(400)*20.0 + 5*75) * 25 = 19375 frames → clamped to 10000 (~2.8 minutes)
        Large level (900 nodes, 15 mines):
            (sqrt(900)*20.0 + 15*75) * 25 = 43125 frames → clamped to 10000 (~2.8 minutes)
    """
    # Base time from traversal complexity (surface area)
    base_time = np.sqrt(surface_area) * BASE_TIME_PER_NODE

    # Additional time for mine navigation
    mine_time = reachable_mine_count * TIME_PER_MINE

    # Apply generous multiplier for learning phase
    truncation_limit = (base_time + mine_time) * TRUNCATION_MULTIPLIER

    # Path-length-aware floor: the free-space cost-to-completion (~pixels) is a
    # direct measure of how far the goal is, so a LONG, winding level gets enough
    # frames to be solved/explored even if its surface area is modest. Without
    # this a 100-1000 env-step level would be truncated before the agent could
    # ever reach the goal (a hard ceiling on learnable difficulty).
    if cost_to_completion is not None and cost_to_completion > 0:
        truncation_limit = max(truncation_limit, cost_to_completion * COST_TIME_FACTOR)

    # Let the cap SCALE with the level's own need rather than hard-clamping every
    # level at MAX_TIME_IN_FRAMES (which would starve genuinely long levels);
    # short levels keep their small formula limit, long levels get proportional
    # time, all bounded by an absolute ceiling.
    cap = min(HARD_TRUNCATION_CEILING, max(MAX_TIME_IN_FRAMES, truncation_limit))

    return int(np.clip(truncation_limit, MIN_TRUNCATION_FRAMES, cap))
