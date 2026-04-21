"""Optimal-route planner for the WALL_JUMP shaft topology.

The shaft is a single-tile-wide vertical rectangle (one empty map column)
with a flat floor, flat ceiling, and flat left/right walls. This matches
:class:`MapWallJump` after its width is fixed to one tile and
:func:`nclone.planning.optimal_route.geometry.classify` restricts
``wall_jump_shaft`` to ``chamber_tile_width == 1``. The ninja starts on one horizontal row (floor
or ceiling) and must reach an exit switch + exit door placed on the
opposite row. The optimal strategy alternates wall jumps between the two
walls until reaching the goal row.

Approach
--------

Rather than derive a closed-form for the non-trivial per-frame drag +
variable-jump-gravity recurrence, we forward-simulate a small number of
parameter-controlled candidate strategies in the pure-Python planner sim
(:mod:`nclone.planning.optimal_route.physics`) and pick the one that
reaches the exit door in the fewest frames. Each candidate is parameterised
by how long to hold the jump button after each wall contact (``hold_after_contact``)
and which wall to target first. Because the planner sim faithfully mirrors
the real simulator's tick order and constants, the resulting action
sequence replays essentially identically in the full nclone simulator.

State machine
-------------

Each upward-traversal rollout walks through four sub-phases per jump:

* ``FLOOR_JUMP``: grounded, press jump + hor toward first wall. Floor jump
  fires this frame (new_jump_check or in_floor_buffer).
* ``ASCEND``: airborne, keep jump held + hor toward current wall. Runs
  until the ninja becomes walled on the target wall.
* ``RELEASE``: release jump for exactly one frame so the next press is a
  fresh edge (``new_jump_check``).
* ``PRESS``: press jump + hor toward the *opposite* wall. Wall jump fires,
  ``wall_jumps_performed`` increments, target flips, return to ``ASCEND``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ...constants.physics_constants import NINJA_RADIUS, TILE_PIXEL_SIZE

from .geometry import (
    LevelGeometry,
    TOPOLOGY_WALL_JUMP_SHAFT,
    classify,
)
from .physics import NinjaState, step
from .types import ActionFrame, OptimalRoute


EXIT_SWITCH_RADIUS = 6.0
EXIT_DOOR_RADIUS = 12.0
CONTACT_SAFETY_PX = 0.5
"""Shrinks the analytical contact radius by 0.5px relative to the real sim.

The real simulator uses ``dist < r_switch + r_ninja`` in
:func:`nclone.physics.overlap_circle_vs_circle`. Due to tiny per-frame
numerical differences between the planner sim and the full simulator
(e.g. the clamp-vs-swept-circle collision model), a plan that is
"exactly" at the collision boundary can replay on the non-touching side
in the real sim. Subtracting a small safety margin guarantees that if
the planner believes the ninja is in contact, the real sim will too.
"""
MAX_PLAN_FRAMES = 3000


def _switch_reached(s: NinjaState, geom: LevelGeometry) -> bool:
    sx, sy = geom.switch_px
    dx = s.x - sx
    dy = s.y - sy
    reach = NINJA_RADIUS + EXIT_SWITCH_RADIUS - CONTACT_SAFETY_PX
    return (dx * dx + dy * dy) < reach * reach


def _door_reached(s: NinjaState, geom: LevelGeometry) -> bool:
    dx_px, dy_px = geom.exit_door_px
    dx = s.x - dx_px
    dy = s.y - dy_px
    reach = NINJA_RADIUS + EXIT_DOOR_RADIUS - CONTACT_SAFETY_PX
    return (dx * dx + dy * dy) < reach * reach


def _choose_first_wall(geom: LevelGeometry) -> int:
    """Return ``+1`` for right-first, ``-1`` for left-first."""
    spawn_x = geom.ninja_spawn_px[0]
    left = geom.left_wall_surface_x_px + NINJA_RADIUS
    right = geom.right_wall_surface_x_px - NINJA_RADIUS
    d_left = spawn_x - left
    d_right = right - spawn_x
    if d_left < d_right - 1e-6:
        return -1
    if d_right < d_left - 1e-6:
        return +1
    door_x, _ = geom.exit_door_px
    return -1 if door_x < spawn_x else +1


def _simulate_upward(
    geom: LevelGeometry,
    first_wall_dir: int,
    hold_after_contact: int,
    max_wall_jumps: int = 10_000,
) -> Tuple[Optional[int], List[ActionFrame], Dict[str, object]]:
    """Simulate one wall-jump-chain candidate heading upward.

    ``first_wall_dir`` is ``-1`` (go to left wall first) or ``+1`` (right
    wall first). ``hold_after_contact`` is the number of extra frames the
    jump button remains *held* after wall contact before releasing for one
    frame and pressing again. Setting it to ``0`` yields the tightest
    press-release-press sequence.

    ``max_wall_jumps`` caps the number of wall jumps before the planner
    transitions into a ``COAST`` phase where the ninja releases jump, aims
    horizontally at the exit-door x coordinate, and lets gravity / residual
    vertical velocity carry it toward the goal. This is needed in tall
    shafts to avoid slamming into the ceiling with ``|vy| > MAX_SURVIVABLE_IMPACT``.
    """
    s = NinjaState.at_spawn(geom)

    actions: List[ActionFrame] = []
    notes: Dict[str, object] = {
        "first_wall_dir": first_wall_dir,
        "hold_after_contact": hold_after_contact,
        "max_wall_jumps": max_wall_jumps,
    }

    target_dir = first_wall_dir
    phase = "WAIT_FOR_GROUND"
    hold_counter = 0
    switch_activated = False
    wall_jumps_performed = 0
    door_x = geom.exit_door_px[0]

    for frame in range(MAX_PLAN_FRAMES):
        if s.dead:
            notes["died_in_planner"] = True
            notes["death_cause"] = s.death_cause
            notes["wall_jumps_performed"] = wall_jumps_performed
            return None, actions, notes
        if _switch_reached(s, geom):
            switch_activated = True
        if switch_activated and _door_reached(s, geom):
            notes["wall_jumps_performed"] = wall_jumps_performed
            notes["final_state"] = s.state
            notes["final_phase"] = phase
            return frame, actions, notes

        if phase == "WAIT_FOR_GROUND":
            hor_input = 0
            jump_input = False
        elif phase == "FLOOR_JUMP_PRESS":
            hor_input = target_dir
            jump_input = True
        elif phase == "ASCEND":
            hor_input = target_dir
            jump_input = True
        elif phase == "HOLD_AFTER_CONTACT":
            hor_input = target_dir
            jump_input = True
        elif phase == "RELEASE_FOR_WJ":
            hor_input = -target_dir
            jump_input = False
        elif phase == "PRESS_FOR_WJ":
            hor_input = -target_dir
            jump_input = True
        elif phase == "COAST":
            if s.x < door_x - 1.0:
                hor_input = 1
            elif s.x > door_x + 1.0:
                hor_input = -1
            else:
                hor_input = 0
            jump_input = False
        else:
            hor_input = target_dir
            jump_input = True

        actions.append((hor_input, jump_input))
        step(s, geom, hor_input, 1 if jump_input else 0)

        if phase == "WAIT_FOR_GROUND":
            if not s.airborn:
                phase = "FLOOR_JUMP_PRESS"
        elif phase == "FLOOR_JUMP_PRESS":
            phase = "ASCEND"
        elif phase == "ASCEND":
            if s.walled and s.wall_normal == -target_dir:
                if wall_jumps_performed >= max_wall_jumps:
                    phase = "COAST"
                else:
                    hold_counter = hold_after_contact
                    phase = "HOLD_AFTER_CONTACT" if hold_counter > 0 else "RELEASE_FOR_WJ"
        elif phase == "HOLD_AFTER_CONTACT":
            hold_counter -= 1
            if hold_counter <= 0:
                phase = "RELEASE_FOR_WJ"
        elif phase == "RELEASE_FOR_WJ":
            phase = "PRESS_FOR_WJ"
        elif phase == "PRESS_FOR_WJ":
            target_dir = -target_dir
            wall_jumps_performed += 1
            phase = "ASCEND"

    notes["wall_jumps_performed"] = wall_jumps_performed
    notes["switch_activated"] = switch_activated
    notes["final_state"] = s.state
    notes["final_phase"] = phase
    return None, actions, notes


def _simulate_downward(
    geom: LevelGeometry,
) -> Tuple[Optional[int], List[ActionFrame], Dict[str, object]]:
    """Free-fall strategy for ``goal_position == 'bottom'`` shafts.

    When the exit is below the spawn, the ninja just needs gravity. We
    nudge horizontal input toward the exit's x to line up with the switch
    and door. Jump is never pressed.
    """
    s = NinjaState.at_spawn(geom)
    actions: List[ActionFrame] = []
    notes: Dict[str, object] = {"strategy_sub": "freefall"}
    switch_activated = False
    exit_x = geom.exit_door_px[0]
    for frame in range(MAX_PLAN_FRAMES):
        if _switch_reached(s, geom):
            switch_activated = True
        if switch_activated and _door_reached(s, geom):
            return frame, actions, notes
        hor_input = 0
        if s.x < exit_x - 1.0:
            hor_input = 1
        elif s.x > exit_x + 1.0:
            hor_input = -1
        actions.append((hor_input, False))
        step(s, geom, hor_input, 0)
    return None, actions, notes


class WallJumpShaftPlanner:
    """Planner for narrow vertical shafts (see :class:`MapWallJump`)."""

    name = "wall_jump_shaft"

    def supports(self, geom: LevelGeometry) -> bool:
        return classify(geom) == TOPOLOGY_WALL_JUMP_SHAFT

    def plan(self, geom: LevelGeometry) -> OptimalRoute:
        """Sweep the tiny parameter grid and pick the min-frame candidate."""
        spawn_x, spawn_y = geom.ninja_spawn_px
        door_x, door_y = geom.exit_door_px
        goal_upward = door_y < spawn_y - TILE_PIXEL_SIZE * 0.5

        best_frames: Optional[int] = None
        best_actions: List[ActionFrame] = []
        best_notes: Dict[str, object] = {}
        best_strategy = "no_solution"

        if not goal_upward:
            frames, actions, notes = _simulate_downward(geom)
            if frames is not None:
                best_frames = frames
                best_actions = actions
                best_notes = notes
                best_strategy = "downward_drop"
        else:
            first_wall = _choose_first_wall(geom)
            chamber_tile_height = geom.chamber_max_gy - geom.chamber_min_gy + 1
            max_wj_upper = max(2, chamber_tile_height + 2)
            candidates: List[Tuple[int, int, int]] = []
            for first_wall_dir in (first_wall, -first_wall):
                for hold_after_contact in (0, 1, 2):
                    for max_wj in range(1, max_wj_upper + 1):
                        candidates.append((first_wall_dir, hold_after_contact, max_wj))

            for first_wall_dir, hold_after_contact, max_wj in candidates:
                frames, actions, notes = _simulate_upward(
                    geom,
                    first_wall_dir=first_wall_dir,
                    hold_after_contact=hold_after_contact,
                    max_wall_jumps=max_wj,
                )
                if frames is None:
                    continue
                if best_frames is None or frames < best_frames:
                    best_frames = frames
                    best_actions = actions
                    best_notes = notes
                    best_strategy = "wj_ping_pong"

        if best_frames is None:
            best_frames = MAX_PLAN_FRAMES
            best_actions = []

        best_notes["chamber_tile_width"] = geom.chamber_max_gx - geom.chamber_min_gx + 1
        best_notes["chamber_tile_height"] = geom.chamber_max_gy - geom.chamber_min_gy + 1
        best_notes["goal_upward"] = goal_upward
        best_notes["spawn_px"] = geom.ninja_spawn_px
        best_notes["door_px"] = geom.exit_door_px

        return OptimalRoute(
            map_type=TOPOLOGY_WALL_JUMP_SHAFT,
            strategy=best_strategy,
            optimal_frames=int(best_frames),
            action_sequence=best_actions,
            verified_in_sim=False,
            verified_frames=None,
            notes=best_notes,
        )
