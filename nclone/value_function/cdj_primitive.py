"""Corner Double Jump (CDJ) composite primitive for the route planner.

Distilled from the validated human replay
``datasets/human_replays/20260525_130015_di-required.replay`` (replays bit-for-bit
to a WIN on both the pure-Python ``nsim`` and the native sim; see
``docs/CDJ_REPLAY_HANDOFF.md``).

The CDJ (taxonomy ``WCu`` = **W**all-jump then **C**orner/floor-jump, 2nd held)
is a frame-perfect *buffer-intersection* trick. The resolution-complete lattice
and the macro forward-search **cannot discover it** — they only emit single
continuous-hold jumps from a node, with no facility for a press / release /
re-press within one edge. It must therefore be authored as an explicit composite
primitive and handed to the planners.

Mechanism (verified against ``nclone/ninja.py`` and the per-frame replay trace)
-------------------------------------------------------------------------------
At a wall-floor **corner** the ninja can briefly hold both ``wall_buffer`` and
``floor_buffer`` live at the same time. ``Ninja.think()`` gives the wall jump
priority over the floor jump when airborne (``ninja.py`` ~L1037-1044)::

    if in_jump_buffer or new_jump_check:
        if self.walled or in_wall_buffer:
            self.wall_jump(...)   # priority
            return
        if in_floor_buffer:
            self.floor_jump(...)
            return

The trick exploits that ``wall_jump()`` resets ``jump_buffer`` / ``wall_buffer`` /
``launch_pad_buffer`` **but NOT ``floor_buffer``** (``ninja.py`` ~L607-609), while
``floor_jump()`` resets ``jump_buffer`` / ``floor_buffer`` / ``launch_pad_buffer``
(``ninja.py`` ~L547-549). So the per-frame input core is::

    1. press jump   -> buffers the jump (jump_buffer set; ninja just went airborne)
    2. release      -> the buffered jump fires as a WALL jump (wall_buffer live,
                       priority). floor_buffer SURVIVES the wall jump.
    3. press + hold -> the fresh rising edge fires a FLOOR jump off the surviving
                       floor_buffer; its impulse STACKS on the wall jump's upward
                       velocity, and holding sustains GRAVITY_JUMP for max height.

The release frame (2) is mandatory: it makes (3) a ``new_jump_check`` rising edge
(two jumps on adjacent held ticks would not re-fire). In the di-required replay
this stacked ``yspeed`` from -2.75 (wall jump) to -4.12 (corner jump) and, held
for 74 frames, climbed +212px from a single CDJ — enough to reach the switch.
This is the user's "wall-jump, release, jump when on a corner".

Reaching the corner is geometry-dependent. In di-required the approach is a
floor jump straight up the right wall: run into the corner, floor-jump, ride the
wall up ~7 frames until the ledge re-arms ``floor_buffer`` while ``wall_buffer``
is still live. ``cdj_approach_inputs`` captures that specific approach; the
geometry-independent trick is ``cdj_core_inputs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# A per-tick input is (h_input, jump_input): h_input in {-1, 0, +1}, jump in {0, 1}.
Input = Tuple[int, int]


@dataclass(frozen=True)
class CDJParams:
    """Parameters of a Corner Double Jump composite.

    Attributes:
        approach_dir: +1 to drive into a wall on the right, -1 for a left wall.
            This is the horizontal input held throughout the trick (toward the
            wall on the approach, and the wall jump pushes off in the opposite
            direction regardless).
        approach_frames: frames of plain approach (run/fall) into the corner
            before the launching floor jump. di-required: 8.
        climb_frames: frames riding up the wall after the launching floor jump,
            until the corner re-arms ``floor_buffer``. di-required: 7.
        hold_frames: frames the stacked corner jump is held; longer = higher,
            capped by ``MAX_JUMP_DURATION`` in the sim. di-required: 74.
        with_approach: include the di-required-style floor-jump approach. Set
            False to emit only the geometry-independent trick core (use when the
            executor has already delivered the ninja to a live corner).
    """

    approach_dir: int = 1
    approach_frames: int = 8
    climb_frames: int = 7
    hold_frames: int = 74
    with_approach: bool = True


#: Canonical CDJ distilled from the validated di-required replay. Reproduces the
#: replay inputs through the apex (ticks 0..91) exactly and climbs ~212px.
DI_REQUIRED_CDJ = CDJParams(
    approach_dir=1, approach_frames=8, climb_frames=7, hold_frames=74
)


def cdj_core_inputs(approach_dir: int = 1, hold_frames: int = 74) -> List[Input]:
    """The geometry-independent CDJ trick core.

    Executed *at* a wall-floor corner while ``wall_buffer`` and ``floor_buffer``
    are both live: press (buffer the jump) -> release (the buffered WALL jump
    fires) -> press + hold (the FLOOR jump stacks and sustains the climb).

    Returns ``hold_frames + 2`` per-tick inputs.
    """
    h = int(approach_dir)
    return [(h, 1), (h, 0)] + [(h, 1)] * int(hold_frames)


def cdj_approach_inputs(
    approach_dir: int = 1, approach_frames: int = 8, climb_frames: int = 7
) -> List[Input]:
    """di-required-style approach that delivers the ninja to a live corner.

    Run into the wall corner, floor-jump straight up it, then ride the wall up
    until the corner ledge re-arms ``floor_buffer`` with ``wall_buffer`` still
    live. The single ``(h, 1)`` between the two runs is the launching floor jump.
    """
    h = int(approach_dir)
    return (
        [(h, 0)] * int(approach_frames)
        + [(h, 1)]
        + [(h, 0)] * int(climb_frames)
    )


def cdj_input_sequence(params: CDJParams = DI_REQUIRED_CDJ) -> List[Input]:
    """Full per-tick ``(h_input, jump_input)`` sequence for a CDJ composite.

    With ``params.with_approach`` (default), this is approach + core. Without it,
    only the trick core (caller is responsible for the approach/corner state).
    """
    seq: List[Input] = []
    if params.with_approach:
        seq += cdj_approach_inputs(
            params.approach_dir, params.approach_frames, params.climb_frames
        )
    seq += cdj_core_inputs(params.approach_dir, params.hold_frames)
    return seq


# ---------------------------------------------------------------------------
# Sim views — let the CDJ run on either the Python nsim or the native sim
# ---------------------------------------------------------------------------
# Corner detection needs airborn / walled / wall_buffer. The Python `nsim`
# exposes these on ``sim.ninja``; the native `nclone_native.Simulator` exposes
# them via ``get_ninja_airborn`` / ``get_ninja_walled`` / ``get_ninja_wall_buffer``
# (added so the trick runs natively, ~bit-identical to Python). A thin view
# unifies the two so ``execute_cdj`` works on either engine.

class _PyNsimView:
    """Adapter for ``NPlayHeadless`` (Python ``nsim``)."""

    def __init__(self, nph):
        self._nph = nph
        self._n = nph.sim.ninja

    def tick(self, h, j):
        self._nph.tick(h, j)

    def pos(self):
        return self._nph.ninja_position()

    def vel(self):
        return self._nph.ninja_velocity()

    def won(self):
        return bool(self._nph.ninja_has_won())

    def died(self):
        return bool(self._nph.ninja_has_died())

    def airborn(self):
        return bool(self._n.airborn)

    def walled(self):
        return bool(self._n.walled)

    def wall_buffer(self):
        return int(self._n.wall_buffer)

    def floor_buffer(self):
        return int(self._n.floor_buffer)

    def yspeed(self):
        return float(self._n.yspeed)


class _NativeView:
    """Adapter for ``nclone_native.Simulator``."""

    def __init__(self, sim):
        self._s = sim

    def tick(self, h, j):
        self._s.tick(h, j)

    def pos(self):
        return self._s.get_ninja_pos()

    def vel(self):
        return self._s.get_ninja_velocity()

    def won(self):
        return bool(self._s.has_won())

    def died(self):
        return bool(self._s.has_died())

    def airborn(self):
        return bool(self._s.get_ninja_airborn())

    def walled(self):
        return bool(self._s.get_ninja_walled())

    def wall_buffer(self):
        return int(self._s.get_ninja_wall_buffer())

    def floor_buffer(self):
        return int(self._s.get_ninja_floor_buffer())

    def yspeed(self):
        return float(self._s.get_ninja_velocity()[1])


def _as_sim_view(sim):
    """Wrap a Python ``NPlayHeadless`` or a native ``Simulator`` in a view."""
    if hasattr(sim, "ninja_position") and hasattr(sim, "sim"):
        return _PyNsimView(sim)
    if hasattr(sim, "get_ninja_airborn") and hasattr(sim, "get_ninja_pos"):
        return _NativeView(sim)
    raise TypeError(
        "execute_cdj/simulate_cdj need an NPlayHeadless or nclone_native.Simulator "
        "(native must expose get_ninja_airborn/walled/wall_buffer — rebuild the "
        "native module)."
    )


@dataclass
class CDJResult:
    """Outcome of simulating a CDJ composite from a given start state."""

    won: bool
    died: bool
    peak_y: float          # min y reached (smaller y = higher in screen space)
    climb_px: float        # start_y - peak_y (how much height the trick gained)
    frames: int            # ticks consumed before win/death or end of sequence
    landing: Optional[Tuple[float, float, float, float]]  # (x,y,vx,vy) or None


def simulate_cdj(
    sim,
    params: CDJParams = DI_REQUIRED_CDJ,
    extra_inputs: Optional[Sequence[Input]] = None,
) -> CDJResult:
    """Run the CDJ composite on a sim (``NPlayHeadless`` or native) from its state.

    ``extra_inputs`` (e.g. the map-specific traverse from the apex to the switch)
    is appended after the trick. Measurement (``peak_y`` / ``climb_px``) covers
    the whole executed sequence. This is the validation/measurement path; the
    planners consume :func:`cdj_input_sequence` / :func:`cdj_core_inputs` directly.
    """
    v = _as_sim_view(sim)
    seq: List[Input] = list(cdj_input_sequence(params))
    if extra_inputs:
        seq += list(extra_inputs)

    start_y = float(v.pos()[1])
    peak_y = start_y
    for i, (h, j) in enumerate(seq):
        v.tick(h, j)
        peak_y = min(peak_y, float(v.pos()[1]))
        if v.won():
            return CDJResult(True, False, peak_y, start_y - peak_y, i + 1, None)
        if v.died():
            return CDJResult(False, True, peak_y, start_y - peak_y, i + 1, None)

    pos = v.pos()
    vel = v.vel()
    landing = (float(pos[0]), float(pos[1]), float(vel[0]), float(vel[1]))
    return CDJResult(False, False, peak_y, start_y - peak_y, len(seq), landing)


# ---------------------------------------------------------------------------
# Adaptive closed-loop CDJ (sim-feedback) — the planner trick action
# ---------------------------------------------------------------------------

@dataclass
class CDJExecution:
    """Outcome of the adaptive closed-loop CDJ, with the tape it actually ran."""

    tape: List[Input]      # per-tick (h, j) inputs executed (replayable on native)
    result: CDJResult      # peak height / climb / win-death / landing
    corner_found: bool     # whether the adaptive climb detected a live corner
    climb_frames: int      # frames the climb phase ran before firing the core


def execute_cdj(
    sim,
    approach_dir: int = 1,
    hold_frames: int = 74,
    *,
    approach_frames: Optional[int] = None,
    max_approach: int = 24,
    max_climb: int = 32,
    post_inputs: Optional[Sequence[Input]] = None,
    bail_if_no_corner: bool = False,
    launch_mode: str = "floor_jump",
    fire_on_buffer_intersect: bool = False,
) -> CDJExecution:
    """Adaptive closed-loop Corner Double Jump, driven by per-frame sim feedback.

    Executes from the ninja's CURRENT state on either the Python ``nsim``
    (``NPlayHeadless``) or the native ``nclone_native.Simulator`` — both expose the
    ``airborn`` / ``walled`` / ``wall_buffer`` signals corner detection needs (the
    native accessors were added for this; native↔Python are ~bit-identical, so the
    tape produced on one replays on the other). Unlike the fixed
    :func:`cdj_input_sequence`, the **climb length is detected**, not hard-coded,
    so the trick generalizes across wall-height-to-corner. Phases:

      1. **approach** — drive toward the wall (``approach_dir``). If
         ``approach_frames`` is given, run exactly that many; otherwise settle
         adaptively until grounded (capped at ``max_approach``).
      2. **launch** — one floor jump (``(h, 1)``) up the wall.
      3. **climb** — ride up (``(h, 0)``) until the corner re-arms floor contact
         while wall contact is live: stop on the first re-grounded frame after
         having been airborne *and* in wall contact (capped at ``max_climb``).
      4. **core** — press ``(h, 1)`` / release ``(h, 0)`` (the buffered WALL jump
         fires) / press+hold ``(h, 1)`` × ``hold_frames`` (the FLOOR jump stacks).

    ``post_inputs`` (e.g. the map traverse to the switch) is run after the trick.
    Returns the executed :class:`CDJExecution` (mutates ``sim``). For
    non-destructive use, snapshot first (``save_ninja_state``) or run on a scratch
    copy. ``corner_found=False`` means no live corner was reached (the caller
    should discard the attempt).

    ``bail_if_no_corner`` short-circuits: if the adaptive climb finds no live
    corner, return immediately WITHOUT running the (expensive) ~76-frame trick
    core.  Enumeration discards non-corner launches anyway, so this is a large
    speedup when sweeping many candidate floor nodes (no behavioural change to
    the kept, corner-found results).

    ``launch_mode`` generalizes where the upward velocity that carries the ninja
    up the wall to the corner comes from:

      * ``"floor_jump"`` (default) — the di-required path above: settle on the
        floor, then floor-jump up the wall.  A floor jump from rest is only ONE
        way to reach the corner rising.
      * ``"carry"`` — skip the settle + launch floor jump and run the corner
        detection / core directly from the ninja's CURRENT state, which is
        assumed to be airborne, rising and wall-contacting (delivered by the
        caller — e.g. injected into a wall node with upward velocity, or arriving
        off a prior wall jump / CDJ).  If the ninja is not airborne on entry the
        attempt is abandoned (``corner_found=False``) so a grounded delivery
        cannot spuriously fire detection on tick 0.  The default path is
        byte-identical to before (``became_airborne`` still starts ``False``).
    """
    h = int(approach_dir)
    v = _as_sim_view(sim)
    carry = (launch_mode == "carry")
    tape: List[Input] = []
    start_y = float(v.pos()[1])
    peak_y = start_y

    def step(hi: int, ji: int) -> bool:
        """Tick once; record; update peak. Returns True to stop (win/death)."""
        nonlocal peak_y
        v.tick(hi, ji)
        tape.append((hi, ji))
        peak_y = min(peak_y, float(v.pos()[1]))
        return v.won() or v.died()

    def _finish(corner_found: bool, climb_frames: int) -> CDJExecution:
        won = v.won()
        died = v.died()
        pos = v.pos()
        vel = v.vel()
        landing = None if (won or died) else (
            float(pos[0]), float(pos[1]), float(vel[0]), float(vel[1])
        )
        res = CDJResult(won, died, peak_y, start_y - peak_y, len(tape), landing)
        return CDJExecution(tape, res, corner_found, climb_frames)

    if not carry:
        # 1. approach / settle onto the launch floor. Step-first so the spawn-settle
        #    artifact (ninja reports grounded but sits ~2px above the surface) does
        #    not abort the settle at frame 0; stop once grounded with ~zero vy.
        if approach_frames is not None:
            for _ in range(int(approach_frames)):
                if step(h, 0):
                    return _finish(False, 0)
        else:
            for _ in range(max_approach):
                if step(h, 0):
                    return _finish(False, 0)
                if (not v.airborn()) and abs(v.yspeed()) < 0.2:
                    break

        # 2. launch (floor jump up the wall)
        if step(h, 1):
            return _finish(False, 0)
    else:
        # carry: start from the ninja's current rising wall-contact state — no
        # settle, no launching floor jump.  Require an airborne arrival so a
        # grounded delivery can't make detection fire on the first frame.
        if not v.airborn():
            return _finish(False, 0)

    # 3. climb — adaptive corner detection.  In carry mode the ninja arrives
    #    mid-air rising, so seed ``became_airborne`` true; the event logic
    #    (re-grounded after airborne + wall contact => corner) is otherwise
    #    identical, and ``carry`` is False on the default path (byte-identical).
    became_airborne = carry
    saw_wall = False
    climb_frames = 0
    corner_found = False
    for _ in range(max_climb):
        became_airborne = became_airborne or v.airborn()
        saw_wall = saw_wall or v.walled() or (v.wall_buffer() >= 0)
        if became_airborne and saw_wall and not v.airborn():
            corner_found = True
            break
        # SHORT-WALL / SHORT-FLOOR clincher: off a 1-tile block the ninja never
        # "re-grounds" after the wall — instead it grounds on the block TOP
        # (arming floor_buffer), slides off the edge into the side wall (arming
        # wall_buffer) and both buffers go live together for a few frames.  That
        # buffer INTERSECTION is the corner; firing the core there stacks the
        # wall+floor jumps the same way (validated: ~116 px off a 24 px block vs
        # ~29 px for the re-ground-only detector).  Gated so the canonical
        # di-required path (which hits both conditions on the same frame) stays
        # byte-identical when off.
        if fire_on_buffer_intersect and v.wall_buffer() >= 0 and v.floor_buffer() >= 0:
            corner_found = True
            break
        if step(h, 0):
            return _finish(False, climb_frames)
        climb_frames += 1

    # No live corner — skip the costly core when the caller only wants corners.
    if bail_if_no_corner and not corner_found:
        return _finish(False, climb_frames)

    # 4. core — wall jump / release / stacked corner jump (held)
    for hi, ji in cdj_core_inputs(h, hold_frames):
        if step(hi, ji):
            return _finish(corner_found, climb_frames)

    if post_inputs:
        for hi, ji in post_inputs:
            if step(hi, ji):
                break

    return _finish(corner_found, climb_frames)
