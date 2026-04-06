"""Analytical (closed-form) physics for N++ ground traversal and jump arcs.

Provides two main capabilities:

1. **Flat-floor ground traversal** (Part 1 fix):
   Replaces the broken simulation-based running edges for flat floors
   with exact arithmetic.  The root cause of Bug 1 was that
   inject_ninja_for_value_iteration() always set airborn=True, so the
   first simulate_arc frame triggered the early-return condition and
   every running action returned after exactly 1 frame.  Analytical
   computation bypasses this entirely.

2. **Jump arc tables** (Part 3):
   Level-independent precomputed trajectories for all combinations of
   (launch_vx_bin, hold_frames, h_during, h_after).  Computed once at
   import using the same arithmetic as the physics engine.  Used with
   arc_collision_validator.py to validate arcs against level geometry.

Physics frame order (single tick):
  1. integrate():  vx *= DRAG_REGULAR; vy *= DRAG_REGULAR; vy += gravity
                   x += vx; y += vy
  2. collision resolution (tiles)
  3. think():      grounded: vx += GROUND_ACCEL * h (uncapped if |result| < MAX)
                   airborne: vx += AIR_ACCEL * h (uncapped if |result| < MAX)

Note: position update uses vx AFTER drag and gravity but BEFORE think-accel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..constants.physics_constants import (
    DRAG_REGULAR,
    GROUND_ACCEL,
    AIR_ACCEL,
    MAX_HOR_SPEED,
    GRAVITY_FALL,
    GRAVITY_JUMP,
    MAX_JUMP_DURATION,
)
from .surface_graph import _FLOOR_BIN_CENTERS

logger = logging.getLogger(__name__)

# Default sweep parameters — keep in sync with edge_enumeration.py.
# Defined here to avoid circular imports (edge_enumeration imports this module).
_HOLD_DURATIONS: List[int] = [1, 3, 5, 8, 12, 16, 20, 25, 30, 35, 40, 45]
_H_INPUTS: List[int] = [-1, 0, 1]
_POST_INPUTS: List[int] = [-1, 0, 1]

# Flat-floor jump impulse constants (from constants.rs)
JUMP_FLOOR_Y: float = -2.0
JUMP_FLOOR_X: float = 0.0

# Maximum frames for free-space arc computation (arcs almost always land < 300)
_MAX_ARC_FRAMES: int = 300


# ---------------------------------------------------------------------------
# Ground traversal helpers
# ---------------------------------------------------------------------------

def ground_step_flat(vx: float, h_input: int) -> Tuple[float, float]:
    """Single frame of flat-floor physics.

    Mirrors the Rust physics engine for state=1 (Running) on a horizontal
    surface (floor_normalized_x == 0):
      1. integrate(): vx_drag = vx * DRAG_REGULAR; dx = vx_drag
      2. think(): vx_new = clip(vx_drag + GROUND_ACCEL * h, MAX_HOR_SPEED)

    Args:
        vx: Velocity at the START of the frame (after previous think()).
        h_input: Horizontal input (-1, 0, or +1).

    Returns:
        (dx, vx_new):
            dx: displacement for this frame (= vx_drag, the dragged velocity).
            vx_new: velocity at END of frame (after drag + ground accel).
    """
    vx_drag = vx * DRAG_REGULAR
    dx = vx_drag
    vx_proposed = vx_drag + GROUND_ACCEL * h_input
    if abs(vx_proposed) < MAX_HOR_SPEED:
        vx_new = vx_proposed
    else:
        vx_new = vx_drag  # cap: accel rejected, velocity stays at drag value
    return dx, vx_new


def simulate_ground_flat(
    vx_start: float,
    h_input: int,
    n_frames: int,
) -> Tuple[float, float]:
    """Simulate flat-floor running for exactly n_frames.

    Identical arithmetic to the Rust engine; no simulator call needed.

    Args:
        vx_start: Initial velocity.
        h_input:  Horizontal input (-1, 0, or +1) held constant.
        n_frames: Number of frames to simulate.

    Returns:
        (total_dx, vx_final): cumulative x-displacement and final velocity.
    """
    vx = vx_start
    total_dx = 0.0
    for _ in range(n_frames):
        dx, vx = ground_step_flat(vx, h_input)
        total_dx += dx
    return total_dx, vx


# ---------------------------------------------------------------------------
# Jump arc trajectory (free-space, level-independent)
# ---------------------------------------------------------------------------

@dataclass
class JumpArcFrame:
    """A single frame in a precomputed jump arc trajectory."""
    rel_x: float  # x position relative to launch point
    rel_y: float  # y position relative to launch point (negative = up)
    vx: float     # x velocity at END of frame (after think-accel)
    vy: float     # y velocity at END of frame


def compute_jump_arc_relative(
    launch_vx: float,
    launch_vy: float,
    hold_frames: int,
    h_during: int,
    h_after: int,
    max_frames: int = _MAX_ARC_FRAMES,
) -> List[JumpArcFrame]:
    """Compute a free-space jump arc analytically, relative to launch position.

    The launch state represents the ninja's velocity immediately AFTER
    floor_jump() or wall_jump() fires (inside think() of the initiation tick).
    Frame 1 in the returned list is the NEXT tick's integrate().

    Gravity during hold: GRAVITY_JUMP.
    Gravity after release (hold_frames+1 transition frame and beyond):
    - Frame hold_frames+1: integrate() still uses GRAVITY_JUMP (the previous
      think set it), then think() switches to GRAVITY_FALL.
    - Frame hold_frames+2 onwards: GRAVITY_FALL.

    Args:
        launch_vx: x velocity at launch (after impulse, before first air frame).
        launch_vy: y velocity at launch (e.g. -2.0 for flat floor).
        hold_frames: Number of frames with jump_input=1 in the simulated arc.
        h_during: Horizontal input during hold phase.
        h_after: Horizontal input after release.
        max_frames: Hard cap on arc length.

    Returns:
        List of JumpArcFrame, one per frame from frame 1 up to landing or
        max_frames.  Indices are 0-based in the list (list[0] = frame 1).
    """
    vx = launch_vx
    vy = launch_vy
    rel_x = 0.0
    rel_y = 0.0
    effective_hold = min(hold_frames, MAX_JUMP_DURATION)
    frames: List[JumpArcFrame] = []

    for frame in range(1, max_frames + 1):
        # Gravity selection mirrors Rust think() state machine:
        #   Frames 1..effective_hold: GRAVITY_JUMP (jump held)
        #   Frame effective_hold+1:   GRAVITY_JUMP (last transition frame —
        #     integrate() uses gravity set by previous think, which was JUMP)
        #   Frames effective_hold+2..: GRAVITY_FALL
        if frame <= effective_hold + 1:
            gravity = GRAVITY_JUMP
        else:
            gravity = GRAVITY_FALL

        # integrate(): drag and gravity applied before position update
        vx_drag = vx * DRAG_REGULAR
        vy_drag = vy * DRAG_REGULAR + gravity

        # Position update uses post-drag, post-gravity velocity
        rel_x += vx_drag
        rel_y += vy_drag

        # think(): air acceleration (same formula for h_during and h_after)
        h = h_during if frame <= effective_hold else h_after
        vx_proposed = vx_drag + AIR_ACCEL * h
        if abs(vx_proposed) < MAX_HOR_SPEED:
            vx = vx_proposed
        else:
            vx = vx_drag  # cap applied

        vy = vy_drag  # y velocity unchanged by think() during normal flight

        frames.append(JumpArcFrame(rel_x, rel_y, vx, vy))

    return frames


# ---------------------------------------------------------------------------
# Jump arc table (precomputed, level-independent)
# ---------------------------------------------------------------------------

@dataclass
class _ArcKey:
    """Immutable key for a precomputed arc."""
    launch_vx_bin: int  # index into _FLOOR_BIN_CENTERS
    hold_frames: int
    h_during: int       # -1, 0, or +1
    h_after: int        # -1, 0, or +1


@dataclass
class PrecomputedArc:
    """A precomputed free-space jump arc from a flat floor node."""
    launch_vx_bin: int
    hold_frames: int
    h_during: int
    h_after: int
    # Launch velocity AFTER one drag frame (initiation frame effect)
    launch_vx: float
    launch_vy: float
    # Per-frame trajectory (relative to the arc launch position)
    frames: List[JumpArcFrame] = field(default_factory=list)


class JumpArcTable:
    """Precomputed level-independent jump arc trajectories for flat floors.

    For each combination of (launch_vx_bin, hold_frames, h_during, h_after)
    stores the full free-space trajectory.  Built once at module level.

    The launch state for flat-floor jumps:
        - Before initiation tick: ninja grounded at (sp.x, sp.y) with vx = vx0
        - Initiation tick integrate(): vx_drag = vx0 * DRAG_REGULAR (one drag)
        - Initiation tick think(): floor_jump fires:
          vx += JUMP_FLOOR_X = 0;  vy += JUMP_FLOOR_Y = -2.0
          ypos += JUMP_FLOOR_Y = -2.0
        - Launch state: vx = vx0 * DRAG_REGULAR, vy = -2.0

    Relative arc positions are offset from launch pos (sp.x + vx0*DRAG, sp.y-2).
    To get absolute arc position:
        abs_x = launch_x + arc_frame.rel_x
        abs_y = launch_y + arc_frame.rel_y
    """

    def __init__(self) -> None:
        # keyed by (launch_vx_bin, hold_frames, h_during, h_after)
        self._table: dict = {}
        self._built = False

    def build(
        self,
        hold_durations: Optional[List[int]] = None,
        h_inputs: Optional[List[int]] = None,
        post_inputs: Optional[List[int]] = None,
    ) -> None:
        """Precompute all flat-floor jump arcs.

        Args:
            hold_durations: Hold frame counts to enumerate.
            h_inputs: Horizontal inputs during hold.
            post_inputs: Horizontal inputs after release.
        """
        if self._built:
            return

        if hold_durations is None:
            hold_durations = _HOLD_DURATIONS
        if h_inputs is None:
            h_inputs = _H_INPUTS
        if post_inputs is None:
            post_inputs = _POST_INPUTS

        n_arcs = 0
        for bin_i, vx0 in enumerate(_FLOOR_BIN_CENTERS):
            # After one drag frame (initiation tick), launch_vx is dragged
            launch_vx = float(vx0) * DRAG_REGULAR
            launch_vy = JUMP_FLOOR_Y  # flat floor impulse

            for hold in hold_durations:
                for h_d in h_inputs:
                    for h_a in post_inputs:
                        key = (bin_i, hold, h_d, h_a)
                        arc = PrecomputedArc(
                            launch_vx_bin=bin_i,
                            hold_frames=hold,
                            h_during=h_d,
                            h_after=h_a,
                            launch_vx=launch_vx,
                            launch_vy=launch_vy,
                            frames=compute_jump_arc_relative(
                                launch_vx, launch_vy, hold - 1, h_d, h_a
                            ),
                        )
                        self._table[key] = arc
                        n_arcs += 1

        self._built = True
        logger.debug(f"JumpArcTable: built {n_arcs} arcs")

    def get(
        self,
        launch_vx_bin: int,
        hold_frames: int,
        h_during: int,
        h_after: int,
    ) -> Optional[PrecomputedArc]:
        """Look up a precomputed arc by exact key.

        Returns None if the key was not precomputed (e.g., out-of-range bin).
        """
        if not self._built:
            self.build()
        return self._table.get((launch_vx_bin, hold_frames, h_during, h_after))

    @property
    def n_arcs(self) -> int:
        return len(self._table)


# Module-level singleton built on first use
_jump_arc_table: Optional[JumpArcTable] = None


def get_jump_arc_table() -> JumpArcTable:
    """Return the module-level precomputed jump arc table, building it if needed."""
    global _jump_arc_table
    if _jump_arc_table is None:
        _jump_arc_table = JumpArcTable()
        _jump_arc_table.build()
    return _jump_arc_table
