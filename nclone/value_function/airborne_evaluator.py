"""Airborne state evaluation via forward simulation of representative strategies.

When the agent is airborne and the arc cache has no hit, this module
forward-simulates a small set of representative control strategies
(~45 total) to find the best achievable cost-to-go:

    cost = min over strategies { frames_to_landing + d*[landing_node] }

Strategy categories (from spec Section 7):
  - Jump-holding: 5 release times x 3 h_inputs = 15 strategies
  - Falling only: 3 h_inputs = 3 strategies
  - Extended mid-change: 3 switch times x 3x3 input combos (different) ~27 strategies
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from ..constants.physics_constants import MAX_JUMP_DURATION
from .surface_graph import (
    SurfaceGraph,
    find_nearest_point,
    velocity_to_bin,
    node_id,
)
from .surface_extraction import SURFACE_FLOOR, SURFACE_WALL
from .arc_cache import ArcCache

# Ninja state codes
_STATE_GROUNDED = 0
_STATE_JUMPING = 1
_STATE_FALLING = 2
_STATE_WALL_SLIDING = 5
_STATE_DEAD = 6
_STATE_AWAITING_DEATH = 7

_DEATH_STATES = {_STATE_DEAD, _STATE_AWAITING_DEATH}
_SURFACE_STATES = {_STATE_GROUNDED, _STATE_WALL_SLIDING}

# Max frames for forward simulation
_MAX_SIM_FRAMES = 300

# Horizontal inputs
_H_INPUTS = [-1, 0, 1]


# ---------------------------------------------------------------------------
# Strategy generation
# ---------------------------------------------------------------------------
def _generate_strategies(
    jump_held: bool,
    jump_frames_remaining: int,
) -> List[Tuple]:
    """Generate representative airborne control strategies.

    Each strategy is a tuple describing control phases:
      ("constant", h_input, jump_input_per_frame_or_release_frame)
      ("two_phase", h1, h2, switch_frame)
      ("jump_release", h_input, release_frame, h_after)

    Returns a list of strategy descriptors.
    """
    strategies = []

    if jump_held and jump_frames_remaining > 0:
        # Jump-holding strategies: vary release timing and horizontal input
        remaining = jump_frames_remaining
        release_times = set()
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            release_times.add(max(0, int(frac * remaining)))
        release_times = sorted(release_times)

        for release_time in release_times:
            for h_input in _H_INPUTS:
                strategies.append(("jump_release", h_input, release_time, h_input))

        # Jump-release with different post-release input
        for release_time in [0, remaining // 2, remaining]:
            for h_during in _H_INPUTS:
                for h_after in _H_INPUTS:
                    if h_after != h_during:
                        strategies.append(
                            ("jump_release", h_during, release_time, h_after))

    else:
        # Falling only: 3 constant horizontal inputs
        for h_input in _H_INPUTS:
            strategies.append(("constant", h_input))

    # Extended mid-change strategies: switch horizontal input partway
    for switch_frac in [0.25, 0.5, 0.75]:
        for h_before in _H_INPUTS:
            for h_after in _H_INPUTS:
                if h_after != h_before:
                    strategies.append(
                        ("two_phase", h_before, h_after, switch_frac))

    return strategies


# ---------------------------------------------------------------------------
# Strategy simulation
# ---------------------------------------------------------------------------
def _simulate_strategy(
    sim,
    strategy,
    max_frames: int = _MAX_SIM_FRAMES,
) -> Tuple[Optional[Tuple[float, float, float, float, int]], int]:
    """Simulate one strategy from the current simulator state until landing.

    Args:
        sim: nclone_native.Simulator with state already injected.
        strategy: Strategy descriptor tuple.
        max_frames: Maximum frames.

    Returns:
        (landing_state, frame_count)
        - landing_state: (x, y, vx, vy, state_code) or None if died/timeout.
        - frame_count: Frames simulated.
    """
    kind = strategy[0]

    if kind == "constant":
        h_input = strategy[1]
        return _sim_constant(sim, h_input, 0, max_frames)

    elif kind == "jump_release":
        h_during, release_frame, h_after = strategy[1], strategy[2], strategy[3]
        return _sim_jump_release(sim, h_during, release_frame, h_after, max_frames)

    elif kind == "two_phase":
        h_before, h_after, switch_frac = strategy[1], strategy[2], strategy[3]
        # Estimate total time as fraction of max
        switch_frame = int(switch_frac * min(100, max_frames))
        return _sim_two_phase(sim, h_before, h_after, switch_frame, max_frames)

    return None, 0


def _sim_constant(
    sim, h_input: int, jump_input: int, max_frames: int,
) -> Tuple[Optional[Tuple[float, float, float, float, int]], int]:
    """Simulate with constant inputs until landing or death."""
    was_airborne = True  # We start airborne
    for frame in range(1, max_frames + 1):
        sim.tick_physics_only(h_input, jump_input)
        sc = sim.get_ninja_state_code()
        if sc in _DEATH_STATES:
            return None, frame
        if sc in _SURFACE_STATES:
            pos = sim.get_ninja_pos()
            vel = sim.get_ninja_velocity()
            return (pos[0], pos[1], vel[0], vel[1], sc), frame
    return None, max_frames


def _sim_jump_release(
    sim,
    h_during: int,
    release_frame: int,
    h_after: int,
    max_frames: int,
) -> Tuple[Optional[Tuple[float, float, float, float, int]], int]:
    """Simulate: hold jump for release_frame, then fall with h_after."""
    total = 0

    # Phase 1: hold jump
    for _ in range(release_frame):
        total += 1
        sim.tick_physics_only(h_during, 1)
        sc = sim.get_ninja_state_code()
        if sc in _DEATH_STATES:
            return None, total
        if sc in _SURFACE_STATES:
            pos = sim.get_ninja_pos()
            vel = sim.get_ninja_velocity()
            return (pos[0], pos[1], vel[0], vel[1], sc), total

    # Phase 2: fall
    remaining = max_frames - total
    for _ in range(max(0, remaining)):
        total += 1
        sim.tick_physics_only(h_after, 0)
        sc = sim.get_ninja_state_code()
        if sc in _DEATH_STATES:
            return None, total
        if sc in _SURFACE_STATES:
            pos = sim.get_ninja_pos()
            vel = sim.get_ninja_velocity()
            return (pos[0], pos[1], vel[0], vel[1], sc), total

    return None, total


def _sim_two_phase(
    sim,
    h_before: int,
    h_after: int,
    switch_frame: int,
    max_frames: int,
) -> Tuple[Optional[Tuple[float, float, float, float, int]], int]:
    """Simulate: h_before until switch_frame, then h_after until landing."""
    total = 0

    # Phase 1
    for _ in range(switch_frame):
        total += 1
        sim.tick_physics_only(h_before, 0)
        sc = sim.get_ninja_state_code()
        if sc in _DEATH_STATES:
            return None, total
        if sc in _SURFACE_STATES:
            pos = sim.get_ninja_pos()
            vel = sim.get_ninja_velocity()
            return (pos[0], pos[1], vel[0], vel[1], sc), total

    # Phase 2
    remaining = max_frames - total
    for _ in range(max(0, remaining)):
        total += 1
        sim.tick_physics_only(h_after, 0)
        sc = sim.get_ninja_state_code()
        if sc in _DEATH_STATES:
            return None, total
        if sc in _SURFACE_STATES:
            pos = sim.get_ninja_pos()
            vel = sim.get_ninja_velocity()
            return (pos[0], pos[1], vel[0], vel[1], sc), total

    return None, total


# ---------------------------------------------------------------------------
# Landing resolution
# ---------------------------------------------------------------------------
def _landing_to_cost(
    graph: SurfaceGraph,
    d_star: np.ndarray,
    landing: Tuple[float, float, float, float, int],
    frames: int,
) -> float:
    """Convert a landing result to total cost = frames + d*[landing_node]."""
    x, y, vx, vy, sc = landing

    if sc == _STATE_WALL_SLIDING:
        stype = SURFACE_WALL
        vel = vy
    else:
        stype = SURFACE_FLOOR
        vel = vx

    pt_idx = find_nearest_point(graph, x, y, stype)
    if pt_idx is None:
        pt_idx = find_nearest_point(graph, x, y, surface_type=None)
        if pt_idx is None:
            return float("inf")
        sp = graph.surface_points[pt_idx]
        stype = sp.surface_type
        vel = vy if stype == SURFACE_WALL else vx

    vel_bin = velocity_to_bin(vel, stype)
    nid = node_id(pt_idx, vel_bin, graph.max_vel_bins)
    if nid >= len(d_star):
        return float("inf")

    d = float(d_star[nid])
    if not np.isfinite(d):
        return float("inf")

    return frames + d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class AirborneEvaluator:
    """Evaluate airborne states by forward-simulating representative strategies.

    Requires a simulator instance for physics and a SurfaceGraph + d_star
    for landing cost lookup.

    Typical usage at runtime::

        evaluator = AirborneEvaluator(sim, graph, d_star)
        cost = evaluator.evaluate(x, y, vx, vy, jump_held, jump_remaining)
    """

    def __init__(
        self,
        sim,
        graph: SurfaceGraph,
        d_star: np.ndarray,
        arc_cache: Optional[ArcCache] = None,
    ) -> None:
        self._sim = sim
        self._graph = graph
        self._d_star = d_star
        self._arc_cache = arc_cache

    def evaluate(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        jump_held: bool = False,
        jump_frames_remaining: int = 0,
    ) -> float:
        """Compute cost-to-go for an airborne state.

        Tries the arc cache first (O(1)), falls back to forward simulation
        of ~45 strategies.

        Args:
            x, y: Ninja center position (pixels).
            vx, vy: Ninja velocity (pixels/frame).
            jump_held: Whether jump is currently held.
            jump_frames_remaining: Frames of jump hold remaining.

        Returns:
            Estimated minimum cost-to-go (frames).  ``inf`` if no viable
            landing is found by any strategy.
        """
        # Try arc cache first
        if self._arc_cache is not None:
            cached = self._arc_cache.lookup_neighborhood(x, y, vx, vy)
            if cached is not None:
                return cached

        # Generate strategies
        strategies = _generate_strategies(jump_held, jump_frames_remaining)

        snapshot = self._sim.save_ninja_state()
        best_cost = float("inf")

        for strategy in strategies:
            # Inject airborne state
            self._sim.restore_ninja_state(snapshot)
            jd = jump_frames_remaining if jump_held else 0
            self._sim.inject_ninja_for_value_iteration(x, y, vx, vy, jd)

            landing, frames = _simulate_strategy(self._sim, strategy)
            if landing is not None:
                cost = _landing_to_cost(
                    self._graph, self._d_star, landing, frames)
                if cost < best_cost:
                    best_cost = cost

        self._sim.restore_ninja_state(snapshot)
        return best_cost
