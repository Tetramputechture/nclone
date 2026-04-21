"""Pure-Python ninja physics model for closed-form / bounded-search planning.

This is a deliberately minimal re-implementation of the subset of
:class:`nclone.ninja.Ninja` behaviour that applies inside flat, mine-free
rectangular chambers. It preserves the real simulator's per-frame ordering
(``integrate`` -> ``collide`` -> ``think``) and its constants so that an
action sequence designed against this model replays identically (or
extremely close) in the full nclone simulator.

Scope
-----

* Supports: flat horizontal floor/ceiling, flat vertical left/right walls,
  floor jumps, wall jumps (regular and slide variants), wall sliding,
  airborne horizontal acceleration, ground acceleration / friction, drag,
  gravity regimes (jump vs fall), and the variable-hold jump.

* Out of scope: slopes, mines, one-way platforms, launch pads, bounce
  blocks, doors, ragdoll death, terminal-impact death. Any of these would
  invalidate the plan, which is why
  :func:`nclone.planning.optimal_route.geometry.extract_level_geometry`
  refuses to classify maps that contain them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

from ...constants.physics_constants import (
    AIR_ACCEL,
    DRAG_REGULAR,
    FRICTION_GROUND,
    FRICTION_WALL,
    GRAVITY_FALL,
    GRAVITY_JUMP,
    GROUND_ACCEL,
    JUMP_FLOOR_Y,
    JUMP_WALL_REGULAR_X_MULTIPLIER,
    JUMP_WALL_REGULAR_Y,
    JUMP_WALL_SLIDE_X_MULTIPLIER,
    JUMP_WALL_SLIDE_Y,
    MAX_HOR_SPEED,
    MAX_JUMP_DURATION,
    MAX_SURVIVABLE_IMPACT,
    NINJA_RADIUS,
)

from .geometry import LevelGeometry


STATE_IMMOBILE = 0
STATE_RUNNING = 1
STATE_SLIDING = 2
STATE_JUMPING = 3
STATE_FALLING = 4
STATE_WALL_SLIDING = 5


@dataclass
class NinjaState:
    """Mutable per-frame state of the planning-time ninja model."""

    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0

    applied_gravity: float = GRAVITY_FALL
    applied_drag: float = DRAG_REGULAR

    state: int = STATE_IMMOBILE
    airborn: bool = False
    walled: bool = False
    wall_normal: int = 0

    jump_duration: int = 0
    jump_buffer: int = -1
    floor_buffer: int = -1
    wall_buffer: int = -1

    jump_input_old: int = 0

    dead: bool = False
    death_cause: str = ""

    @classmethod
    def at_spawn(cls, geom: LevelGeometry) -> "NinjaState":
        """Initialize the state at the spawn position stored in ``geom``."""
        return cls(x=geom.ninja_spawn_px[0], y=geom.ninja_spawn_px[1])

    def copy(self) -> "NinjaState":
        return replace(self)


WALL_CONTACT_SLACK_PX = 0.1
"""Small contact slack so that sub-pixel horizontal jitter right after a
wall collision still registers as ``walled``. The real simulator performs
circle-vs-segment collision detection that is naturally tolerant to this
kind of jitter; the planner's axis-aligned clamp needs an explicit epsilon
so that a wall jump fired on the frame after contact picks up the correct
``wall_normal`` (and therefore applies a non-zero impulse)."""


CEILING_IMPACT_DEATH_THRESHOLD = MAX_SURVIVABLE_IMPACT - 4.0 / 3.0
"""Absolute value of ``yspeed`` at the frame of ceiling contact above which
the real simulator kills the ninja (``ninja.py`` lines 513-531). The ``4/3``
term reduces to this fixed value on flat horizontal ceilings where
``|ceiling_normalized_y| == 1``."""

FLOOR_IMPACT_DEATH_THRESHOLD = MAX_SURVIVABLE_IMPACT - 4.0 / 3.0
"""Same threshold applied on flat horizontal floors. Included for
completeness; in practice these planners rarely drop fast enough from a
closed chamber to trigger it, but the check keeps the planner honest."""


def _clamp_to_chamber(s: NinjaState, geom: LevelGeometry) -> None:
    """Resolve collisions against the chamber's four flat surfaces.

    Sets :attr:`airborn`, :attr:`walled`, and :attr:`wall_normal` based on
    which surfaces the ninja is touching after integration. When a surface
    is hit, the velocity component into that surface is zeroed out. The
    position is clamped so that the ninja center remains offset by
    ``NINJA_RADIUS`` from each surface.

    ``wall_normal`` persists across frames while the ninja remains within
    ``WALL_CONTACT_SLACK_PX`` of the most-recently-touched wall. This mirrors
    ``nclone.ninja.Ninja``, where ``wall_normal`` is updated only when a new
    wall contact is detected and retains its value across the few frames
    covered by ``wall_buffer``.
    """
    left = geom.left_wall_surface_x_px + NINJA_RADIUS
    right = geom.right_wall_surface_x_px - NINJA_RADIUS
    top = geom.ceiling_surface_y_px + NINJA_RADIUS
    bottom = geom.floor_surface_y_px - NINJA_RADIUS

    on_floor = False
    on_ceiling = False
    touching_left = False
    touching_right = False

    if s.x < left:
        s.x = left
        if s.vx < 0:
            s.vx = 0.0
        touching_left = True
    elif s.x > right:
        s.x = right
        if s.vx > 0:
            s.vx = 0.0
        touching_right = True

    if s.y > bottom:
        prev_vy = s.vy
        s.y = bottom
        if s.vy > 0:
            s.vy = 0.0
        on_floor = True
        if prev_vy > FLOOR_IMPACT_DEATH_THRESHOLD:
            s.dead = True
            s.death_cause = "floor_impact"
    elif s.y < top:
        prev_vy = s.vy
        s.y = top
        if s.vy < 0:
            s.vy = 0.0
        on_ceiling = True
        if -prev_vy > CEILING_IMPACT_DEATH_THRESHOLD:
            s.dead = True
            s.death_cause = "ceiling_impact"

    near_left = s.x <= left + WALL_CONTACT_SLACK_PX
    near_right = s.x >= right - WALL_CONTACT_SLACK_PX

    s.airborn = not on_floor
    s.walled = (touching_left or touching_right or near_left or near_right) and not on_floor

    if touching_left or near_left:
        s.wall_normal = +1
    elif touching_right or near_right:
        s.wall_normal = -1
    elif on_floor:
        s.wall_normal = 0
    _ = on_ceiling


def _integrate(s: NinjaState) -> None:
    """Apply drag / gravity and advance position by one frame.

    Mirrors :meth:`nclone.ninja.Ninja.integrate` (ninja.py lines 198-206).
    """
    s.vx *= s.applied_drag
    s.vy *= s.applied_drag
    s.vy += s.applied_gravity
    s.x += s.vx
    s.y += s.vy


def _perform_floor_jump(s: NinjaState) -> None:
    """Flat-floor jump: apply ``(0, JUMP_FLOOR_Y)`` to velocity and position."""
    if s.vy > 0:
        s.vy = 0.0
    s.vy += JUMP_FLOOR_Y
    s.y += JUMP_FLOOR_Y
    s.state = STATE_JUMPING
    s.applied_gravity = GRAVITY_JUMP
    s.jump_duration = 0
    s.jump_buffer = -1
    s.floor_buffer = -1


def _perform_wall_jump(s: NinjaState, hor_input: int) -> None:
    """Wall jump selecting slide vs regular based on input and state.

    Slide wall jump: ``hor_input * wall_normal < 0 and state == STATE_WALL_SLIDING``
    yields ``(jx, jy) = (2/3, -1)``. Otherwise the regular variant
    ``(jx, jy) = (1, -1.4)`` applies. Matches ``ninja.py`` lines 583-610.
    """
    if hor_input * s.wall_normal < 0 and s.state == STATE_WALL_SLIDING:
        jx = JUMP_WALL_SLIDE_X_MULTIPLIER
        jy = JUMP_WALL_SLIDE_Y
    else:
        jx = JUMP_WALL_REGULAR_X_MULTIPLIER
        jy = JUMP_WALL_REGULAR_Y
    s.state = STATE_JUMPING
    s.applied_gravity = GRAVITY_JUMP
    if s.vx * s.wall_normal < 0:
        s.vx = 0.0
    if s.vy > 0:
        s.vy = 0.0
    impulse_x = jx * s.wall_normal
    s.vx += impulse_x
    s.vy += jy
    s.x += impulse_x
    s.y += jy
    s.jump_buffer = -1
    s.wall_buffer = -1
    s.jump_duration = 0


def _think(s: NinjaState, hor_input: int, jump_input: int) -> None:
    """Simplified :meth:`Ninja.think` for flat chambers.

    Buffers are ticked first (matching ``ninja.py`` lines 870-904), then
    state-specific logic runs: grounded ninja accelerates/frictions and may
    initiate a floor jump; airborne ninja accelerates in air and may wall
    jump; wall sliding applies wall friction when input is into the wall.
    """
    new_jump_check = jump_input != 0 and s.jump_input_old == 0
    s.jump_input_old = jump_input

    if -1 < s.jump_buffer < 5:
        s.jump_buffer += 1
    else:
        s.jump_buffer = -1
    in_jump_buffer = -1 < s.jump_buffer < 5

    if -1 < s.wall_buffer < 5:
        s.wall_buffer += 1
    else:
        s.wall_buffer = -1
    in_wall_buffer = -1 < s.wall_buffer < 5

    if -1 < s.floor_buffer < 5:
        s.floor_buffer += 1
    else:
        s.floor_buffer = -1
    in_floor_buffer = -1 < s.floor_buffer < 5

    if new_jump_check and s.airborn:
        s.jump_buffer = 0
    if s.walled:
        s.wall_buffer = 0
        in_wall_buffer = True
    if not s.airborn:
        s.floor_buffer = 0
        in_floor_buffer = True

    if not s.airborn:
        xspeed_new = s.vx + GROUND_ACCEL * hor_input
        if abs(xspeed_new) < MAX_HOR_SPEED:
            s.vx = xspeed_new

        if in_jump_buffer or new_jump_check:
            _perform_floor_jump(s)
            return

        if hor_input != 0:
            s.state = STATE_RUNNING
        else:
            s.vx *= FRICTION_GROUND
            s.state = STATE_IMMOBILE if abs(s.vx) < 0.01 else STATE_RUNNING
        s.applied_gravity = GRAVITY_FALL
        return

    xspeed_new = s.vx + AIR_ACCEL * hor_input
    if abs(xspeed_new) < MAX_HOR_SPEED:
        s.vx = xspeed_new

    if s.state < STATE_JUMPING:
        s.state = STATE_FALLING
    elif s.state == STATE_JUMPING:
        s.jump_duration += 1
        if jump_input == 0 or s.jump_duration > MAX_JUMP_DURATION:
            s.applied_gravity = GRAVITY_FALL
            s.state = STATE_FALLING

    if in_jump_buffer or new_jump_check:
        if s.walled or in_wall_buffer:
            _perform_wall_jump(s, hor_input)
            return

    if s.walled:
        if s.state == STATE_WALL_SLIDING:
            if hor_input * s.wall_normal <= 0:
                s.vy *= FRICTION_WALL
            else:
                s.state = STATE_FALLING
        else:
            if s.vy > 0 and hor_input * s.wall_normal < 0:
                s.state = STATE_WALL_SLIDING
                s.vy *= FRICTION_WALL
    else:
        if s.state == STATE_WALL_SLIDING:
            s.state = STATE_FALLING


def step(
    s: NinjaState, geom: LevelGeometry, hor_input: int, jump_input: int
) -> None:
    """Advance ``s`` by one simulator frame.

    Mirrors the real tick order from :meth:`Simulator.tick` (nsim.py lines
    278-350): :func:`_integrate` -> collision clamp -> :func:`_think`.
    """
    _integrate(s)
    _clamp_to_chamber(s, geom)
    _think(s, hor_input, jump_input)


def frames_to_reach_max_horizontal(initial_vx: float, direction: int) -> int:
    """Closed-form count of ground-accel frames to reach ``MAX_HOR_SPEED``.

    Drag is applied to ground motion each frame (``vx *= DRAG_REGULAR``)
    followed by the ground-accel increment. The fixed point of that
    iteration is ``v* = GROUND_ACCEL / (1 - DRAG_REGULAR)``, which equals
    ~9.98 px/frame; but the hard cap ``MAX_HOR_SPEED`` (10/3) is
    reached much earlier, so we just step the linear recurrence until
    ``|vx| >= MAX_HOR_SPEED``.

    ``direction`` is ``+1`` for running right, ``-1`` for running left.
    Returns the integer number of frames required.
    """
    if direction == 0:
        return 0
    vx = initial_vx
    frames = 0
    safety = 1000
    while abs(vx) < MAX_HOR_SPEED and frames < safety:
        vx = vx * DRAG_REGULAR + GROUND_ACCEL * direction
        frames += 1
    return frames


def simulate_until(
    geom: LevelGeometry,
    actions,
    initial: NinjaState = None,
    max_frames: int = 4000,
):
    """Replay ``actions`` in the planning-time sim and return the final state.

    ``actions`` is any iterable of ``(hor_input, jump_input)`` tuples.
    Iteration stops after the last action or after ``max_frames``, whichever
    comes first. Callers typically use this for internal verification of
    candidate action sequences before the full-sim replay.
    """
    s = initial.copy() if initial is not None else NinjaState.at_spawn(geom)
    count = 0
    for (h, j) in actions:
        if count >= max_frames:
            break
        step(s, geom, int(h), int(bool(j)))
        count += 1
    return s, count
