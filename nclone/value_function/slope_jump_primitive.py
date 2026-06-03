"""Slope-jump (a.k.a. slope-boost / "wave") composite primitive for the planner.

N's ``Ninja.floor_jump`` (see ``nclone/ninja.py`` ~L554-580) launches differently
depending on the floor's slope angle and whether the ninja is moving up- or
down-hill relative to the held input.  The resolution-complete lattice and the
macro forward search only emit a SINGLE continuous-hold jump from a node — they
cannot express the **chained re-contact** slope bounce a human uses to convert a
slope into vertical height or slope-direction momentum.  Like the CDJ
(``cdj_primitive``), the maneuver must therefore be authored as an explicit
composite primitive and handed to the planners as a routable trick edge.

Mechanism (verified against ``nclone/ninja.py`` ``floor_jump`` and the slope
constants in ``constants/physics_constants.py``)
-------------------------------------------------------------------------------
With ``floor_normalized = (dx, dy)`` the outward floor normal (``dx==0`` on flat
ground), the jump impulse is::

    flat            : jy = JUMP_FLOOR_Y (-2)                                  # dx==0
    downhill jump   : jx = ⅔·dx, jy = 2·dy        # xspeed·dx>=0 and xspeed·hor>=0
    uphill "forward": jy = JUMP_SLOPE_UPHILL_DEFAULT_Y (-1.4)  # xspeed·hor>0 uphill
    perp jump       : xspeed:=0; jx = ⅔·dx, jy = 2·dy   # uphill, input opposite/neutral

The two levers a human chains (the user's description):

  * **jump + downhill direction** — the *downhill* branch ADDS ``jx=⅔·dx`` to the
    ninja's existing (already downhill) ``xspeed`` without zeroing it, so each
    re-contact bounce compounds **momentum in the slope direction**.
  * **jump + opposite (perp) direction** — the *perp* branch zeroes ``xspeed`` and
    launches ``(⅔·dx, 2·dy)`` straight off the surface normal: **vertical height**
    converted from the approach run, the steepest single-jump launch off a slope.

Because the impulse depends on the *live* ``xspeed`` / ``hor_input`` at the jump
frame (and the chain depends on re-contact timing), this is authored as an
ADAPTIVE closed-loop maneuver driven by sim feedback — the climb length / hop
count is detected, not hard-coded — exactly like ``cdj_primitive.execute_cdj``.

This is the planner trick action; an enumerated edge stores the recorded
``tape`` (replayed verbatim) plus the measured climb / landing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .cdj_primitive import _as_sim_view  # unified Python-nsim / native view

# A per-tick input is (h_input, jump_input): h_input in {-1, 0, +1}, jump in {0, 1}.
Input = Tuple[int, int]


@dataclass
class SlopeJumpResult:
    """Outcome of simulating a slope-jump chain from a given start state."""

    won: bool
    died: bool
    peak_y: float          # min y reached (smaller y = higher in screen space)
    climb_px: float        # start_y - peak_y (height the maneuver gained)
    frames: int            # ticks consumed before win/death or end of sequence
    n_hops: int            # number of slope jumps that actually fired
    landing: Optional[Tuple[float, float, float, float]]  # (x,y,vx,vy) or None


@dataclass
class SlopeJumpExecution:
    """Outcome of the adaptive slope-jump chain, with the tape it actually ran."""

    tape: List[Input]      # per-tick (h, j) executed (replayable on native)
    result: SlopeJumpResult
    fired: bool            # whether at least one slope jump left the ground
    hop_dirs: List[int] = field(default_factory=list)  # the dir of each fired hop


def execute_slope_jump(
    sim,
    *,
    approach_dir: int = 1,
    jump_dirs: Sequence[int] = (1,),
    hold_frames: int = 8,
    approach_frames: Optional[int] = None,
    max_approach: int = 24,
    max_air: int = 90,
    settle_between: int = 0,
    post_inputs: Optional[Sequence[Input]] = None,
    bail_if_no_fire: bool = False,
) -> SlopeJumpExecution:
    """Adaptive slope-jump chain, driven by per-frame sim feedback.

    Executes from the ninja's CURRENT state on either the Python ``nsim``
    (``NPlayHeadless``) or the native ``nclone_native.Simulator``.  Phases:

      1. **approach** — run toward the slope (``approach_dir``) to build along-slope
         velocity.  With ``approach_frames`` given, run exactly that many; else
         settle adaptively until grounded and moving (capped at ``max_approach``).
      2. **chain** — for each direction ``jd`` in ``jump_dirs``: press ``(jd, 1)``
         and hold for ``hold_frames`` (the jump sustains GRAVITY_JUMP for height),
         then release ``(jd, 0)`` and fall until re-grounded (capped at
         ``max_air``).  The release is mandatory: the next press is a fresh rising
         edge that re-fires the floor jump (two held ticks would not re-fire).
         ``jd>0``/``jd<0`` selects the downhill (momentum) vs perp (height) branch
         depending on the slope orientation and the approach direction.

    ``post_inputs`` (e.g. the map traverse to the switch) is run after the chain.
    Returns the executed :class:`SlopeJumpExecution` (mutates ``sim``); snapshot
    first for non-destructive use.  ``fired=False`` means no slope jump ever left
    the ground (the caller should discard the attempt).  ``bail_if_no_fire``
    short-circuits the chain the moment the first hop fails to leave the ground,
    a large speedup when sweeping many candidate slope nodes.
    """
    v = _as_sim_view(sim)
    a = int(approach_dir)
    tape: List[Input] = []
    start_y = float(v.pos()[1])
    peak_y = start_y
    n_hops = 0
    hop_dirs: List[int] = []

    def step(hi: int, ji: int) -> bool:
        """Tick once; record; update peak. Returns True to stop (win/death)."""
        nonlocal peak_y
        v.tick(hi, ji)
        tape.append((hi, ji))
        peak_y = min(peak_y, float(v.pos()[1]))
        return v.won() or v.died()

    def _finish() -> SlopeJumpExecution:
        won = v.won()
        died = v.died()
        pos = v.pos()
        vel = v.vel()
        landing = None if (won or died) else (
            float(pos[0]), float(pos[1]), float(vel[0]), float(vel[1]))
        res = SlopeJumpResult(won, died, peak_y, start_y - peak_y,
                              len(tape), n_hops, landing)
        return SlopeJumpExecution(tape, res, n_hops > 0, hop_dirs)

    # 1. approach / settle onto the slope, building along-surface velocity.
    if approach_frames is not None:
        for _ in range(int(approach_frames)):
            if step(a, 0):
                return _finish()
    else:
        for _ in range(max_approach):
            if step(a, 0):
                return _finish()
            if not v.airborn():
                break

    # 2. chain — one adaptive hop per requested direction.
    for jd in jump_dirs:
        jd = int(jd)
        # Optional inter-hop settle (lets the ninja re-seat on the slope).
        for _ in range(int(settle_between)):
            if v.airborn():
                break
            if step(jd, 0):
                return _finish()
        # Must be grounded to fire a floor jump.
        if v.airborn():
            if bail_if_no_fire and n_hops == 0:
                return _finish()
            continue
        y_before = float(v.pos()[1])
        # Press + hold the jump (sustains the climb up to MAX_JUMP_DURATION).
        for _ in range(int(hold_frames)):
            if step(jd, 1):
                return _finish()
        # The hop "fired" if the press lifted the ninja off the ground.
        if v.airborn() and float(v.pos()[1]) < y_before - 0.5:
            n_hops += 1
            hop_dirs.append(jd)
        elif bail_if_no_fire and n_hops == 0:
            return _finish()
        # Release and fall until re-grounded, so the next press re-fires.
        for _ in range(int(max_air)):
            if not v.airborn():
                break
            if step(jd, 0):
                return _finish()

    if post_inputs:
        for hi, ji in post_inputs:
            if step(hi, ji):
                break

    return _finish()
