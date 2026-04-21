"""Optimal-route planner for the FLAT_RUN horizontal corridor topology.

The corridor is a short, wide rectangular chamber with a flat floor. The
ninja spawns on the floor at one end, the exit switch and exit door are
placed somewhere inside the chamber. The planner evaluates a family of
candidate strategies and returns the one that reaches the exit door in
the fewest simulation frames:

* **direct_run**: accelerate toward the switch on the ground, activate
  it, then accelerate toward the exit door on the ground. This is the
  correct answer whenever the switch and door are both within ``NINJA_RADIUS``
  of the floor line.

* **run_jump**: same horizontal inputs as ``direct_run`` but insert a
  single ground jump (release-frame then press-frame, to satisfy
  :attr:`ninja.Ninja.new_jump_check`) at a configurable trigger time.
  Sweeps the trigger frame over a coarse grid so a well-timed jump can
  intercept switches/doors that float a fraction of a tile above the floor.

* **wall_sling**: run to the wall *opposite* the goal, wall-jump off it
  to bounce toward the goal, then finish by running (or flying) to the
  door. The planner records the timing margin between this candidate and
  ``direct_run`` in ``notes['wall_sling_margin_frames']`` so downstream
  consumers can answer the "is the detour worth it?" question directly.

The returned action sequence is always suitable for sim-replay verification
via :mod:`nclone.planning.optimal_route.verifier`. The planner stops
exploring a candidate as soon as the planner-sim ninja dies (``s.dead``),
hits the ceiling while stuck, or exceeds :data:`MAX_PLAN_FRAMES`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ...constants.physics_constants import NINJA_RADIUS

from .geometry import (
    LevelGeometry,
    TOPOLOGY_FLAT_RUN_CORRIDOR,
    classify,
)
from .physics import NinjaState, step
from .types import ActionFrame, OptimalRoute


EXIT_SWITCH_RADIUS = 6.0
EXIT_DOOR_RADIUS = 12.0
CONTACT_SAFETY_PX = 0.5
"""Shrinks the planner's contact radius by 0.5 px relative to the real
simulator's strict ``dist < r1 + r2`` overlap check. See the matching
docstring in ``wall_jump_shaft.py`` for rationale."""

MAX_PLAN_FRAMES = 3000


def _switch_reached(s: NinjaState, geom: LevelGeometry) -> bool:
    sx, sy = geom.switch_px
    dx = s.x - sx
    dy = s.y - sy
    reach = NINJA_RADIUS + EXIT_SWITCH_RADIUS - CONTACT_SAFETY_PX
    return (dx * dx + dy * dy) < reach * reach


def _door_reached(s: NinjaState, geom: LevelGeometry) -> bool:
    ex, ey = geom.exit_door_px
    dx = s.x - ex
    dy = s.y - ey
    reach = NINJA_RADIUS + EXIT_DOOR_RADIUS - CONTACT_SAFETY_PX
    return (dx * dx + dy * dy) < reach * reach


def _target_direction(s: NinjaState, tx: float) -> int:
    if abs(s.x - tx) < 0.5:
        return 0
    return 1 if tx > s.x else -1


def _direct_run(
    geom: LevelGeometry, hold_jump: bool
) -> Tuple[Optional[int], List[ActionFrame], Dict[str, object]]:
    """Pure ground run (``hold_jump=False``) or held-jump run with no
    release.

    ``hold_jump=True`` only fires a floor jump on the *first* frame the
    ninja is grounded (because ``jump_input_old`` is 0 only on that frame);
    after the jump, input stays held so gravity brings the ninja back
    down and it continues running.
    """
    s = NinjaState.at_spawn(geom)
    actions: List[ActionFrame] = []
    switch_activated = False
    switch_frame: Optional[int] = None
    sx, _ = geom.switch_px
    ex, _ = geom.exit_door_px

    for frame in range(MAX_PLAN_FRAMES):
        if s.dead:
            return None, actions, {"died": True, "cause": s.death_cause}
        if not switch_activated and _switch_reached(s, geom):
            switch_activated = True
            switch_frame = frame
        if (
            switch_activated
            and switch_frame is not None
            and frame > switch_frame
            and _door_reached(s, geom)
        ):
            return frame, actions, {
                "switch_frame": switch_frame,
                "hold_jump": hold_jump,
            }

        target_x = sx if not switch_activated else ex
        hor = _target_direction(s, target_x)
        jump = bool(hold_jump)
        actions.append((hor, jump))
        step(s, geom, hor, 1 if jump else 0)

    return None, actions, {"hold_jump": hold_jump, "timed_out": True}


def _run_jump(
    geom: LevelGeometry,
    jump_trigger_frame: int,
    jump_hold_frames: int,
) -> Tuple[Optional[int], List[ActionFrame], Dict[str, object]]:
    """Ground run with one pressed floor jump at ``jump_trigger_frame``.

    The sequence is: run toward the switch; at the configured frame
    release jump for one frame (forces ``jump_input_old=0``); then press
    jump for ``jump_hold_frames`` frames (the first of those frames fires
    the floor jump because ``new_jump_check`` is true); then release and
    continue running. This is the minimum-overhead way to arc over a
    floor-mounted switch or hop up to a switch floating slightly above
    the floor.
    """
    s = NinjaState.at_spawn(geom)
    actions: List[ActionFrame] = []
    switch_activated = False
    switch_frame: Optional[int] = None
    sx, _ = geom.switch_px
    ex, _ = geom.exit_door_px

    for frame in range(MAX_PLAN_FRAMES):
        if s.dead:
            return None, actions, {"died": True, "cause": s.death_cause}
        if not switch_activated and _switch_reached(s, geom):
            switch_activated = True
            switch_frame = frame
        if (
            switch_activated
            and switch_frame is not None
            and frame > switch_frame
            and _door_reached(s, geom)
        ):
            return frame, actions, {
                "jump_trigger_frame": jump_trigger_frame,
                "jump_hold_frames": jump_hold_frames,
                "switch_frame": switch_frame,
            }

        target_x = sx if not switch_activated else ex
        hor = _target_direction(s, target_x)
        if frame == jump_trigger_frame:
            jump = False
        elif (
            jump_trigger_frame < frame <= jump_trigger_frame + jump_hold_frames
        ):
            jump = True
        else:
            jump = False
        actions.append((hor, jump))
        step(s, geom, hor, 1 if jump else 0)

    return None, actions, {"timed_out": True}


def _wall_sling(
    geom: LevelGeometry,
    wait_frames_after_wj: int = 0,
) -> Tuple[Optional[int], List[ActionFrame], Dict[str, object]]:
    """Detour-to-opposite-wall, wall-jump, fly/run to the goal.

    The ninja first runs toward the wall *away* from the switch. On wall
    contact we release jump for one frame (to clear ``jump_input_old``),
    then press jump on the next frame with horizontal input toward the
    goal, triggering a regular wall jump that imparts
    ``JUMP_WALL_REGULAR_X_MULTIPLIER * wall_normal`` horizontal velocity.
    The ninja is now airborne and aimed at the switch; from there the
    controller switches to switch-then-door targeting.

    ``wait_frames_after_wj`` delays the "run toward goal" phase by N
    frames after the wall jump; this is occasionally useful when the
    ninja would collide with the opposite wall again too early.
    """
    s = NinjaState.at_spawn(geom)
    actions: List[ActionFrame] = []
    sx, _ = geom.switch_px
    ex, _ = geom.exit_door_px

    detour_dir = -1 if sx > s.x else 1
    target_wall_normal = 1 if detour_dir == -1 else -1

    phase = "run_to_wall"
    wait_frames_remaining = 0
    switch_activated = False
    switch_frame: Optional[int] = None
    notes: Dict[str, object] = {
        "detour_dir": detour_dir,
        "wait_frames_after_wj": wait_frames_after_wj,
    }

    for frame in range(MAX_PLAN_FRAMES):
        if s.dead:
            notes.update({"died": True, "cause": s.death_cause})
            return None, actions, notes
        if not switch_activated and _switch_reached(s, geom):
            switch_activated = True
            switch_frame = frame
            notes["switch_frame"] = frame
        if (
            switch_activated
            and switch_frame is not None
            and frame > switch_frame
            and _door_reached(s, geom)
        ):
            notes["phase_final"] = phase
            return frame, actions, notes

        if phase == "run_to_wall":
            hor = detour_dir
            jump = False
            if s.walled and s.wall_normal == target_wall_normal:
                phase = "release"
        elif phase == "release":
            hor = -detour_dir
            jump = False
            phase = "press_wall_jump"
        elif phase == "press_wall_jump":
            hor = -detour_dir
            jump = True
            wait_frames_remaining = wait_frames_after_wj
            phase = "wait_or_fly" if wait_frames_remaining > 0 else "fly_to_goal"
        elif phase == "wait_or_fly":
            hor = 0
            jump = False
            wait_frames_remaining -= 1
            if wait_frames_remaining <= 0:
                phase = "fly_to_goal"
        else:
            target_x = sx if not switch_activated else ex
            hor = _target_direction(s, target_x)
            jump = False

        actions.append((hor, jump))
        step(s, geom, hor, 1 if jump else 0)

    notes["timed_out"] = True
    return None, actions, notes


class FlatRunCorridorPlanner:
    """Planner for flat horizontal corridors (see :class:`MapHorizontalCorridor`)."""

    name = "flat_run_corridor"

    def supports(self, geom: LevelGeometry) -> bool:
        return classify(geom) == TOPOLOGY_FLAT_RUN_CORRIDOR

    def plan(self, geom: LevelGeometry) -> OptimalRoute:
        """Evaluate all candidate strategies and pick the fastest one."""
        candidates: Dict[str, Tuple[Optional[int], List[ActionFrame], Dict[str, object]]] = {}
        candidates["direct_run"] = _direct_run(geom, hold_jump=False)
        candidates["running_jump_cruise"] = _direct_run(geom, hold_jump=True)

        best_run_jump: Tuple[Optional[int], List[ActionFrame], Dict[str, object]] = (
            None,
            [],
            {},
        )
        for trigger in range(8, 120, 2):
            for hold in (5, 15, 30, 45):
                frames, acts, notes = _run_jump(geom, trigger, hold)
                if frames is None:
                    continue
                if best_run_jump[0] is None or frames < best_run_jump[0]:
                    best_run_jump = (frames, acts, notes)
        candidates["run_jump"] = best_run_jump

        best_sling: Tuple[Optional[int], List[ActionFrame], Dict[str, object]] = (
            None,
            [],
            {},
        )
        for wait in (0, 2, 5, 10):
            frames, acts, notes = _wall_sling(geom, wait_frames_after_wj=wait)
            if frames is None:
                continue
            if best_sling[0] is None or frames < best_sling[0]:
                best_sling = (frames, acts, notes)
        candidates["wall_sling"] = best_sling

        best_name: Optional[str] = None
        best_frames: Optional[int] = None
        best_actions: List[ActionFrame] = []
        per_candidate: Dict[str, Optional[int]] = {}

        for name, (frames, _actions, _notes) in candidates.items():
            per_candidate[name] = frames
            if frames is None:
                continue
            if best_frames is None or frames < best_frames:
                best_frames = frames
                best_name = name
                best_actions = _actions

        direct_frames = per_candidate.get("direct_run")
        sling_frames = per_candidate.get("wall_sling")
        if direct_frames is not None and sling_frames is not None:
            margin = direct_frames - sling_frames
        else:
            margin = None

        notes: Dict[str, object] = {
            "candidates": per_candidate,
            "wall_sling_margin_frames": margin,
            "chamber_tile_width": geom.chamber_max_gx - geom.chamber_min_gx + 1,
            "chamber_tile_height": geom.chamber_max_gy - geom.chamber_min_gy + 1,
            "spawn_px": geom.ninja_spawn_px,
            "switch_px": geom.switch_px,
            "door_px": geom.exit_door_px,
        }
        if best_name is not None:
            notes.update(candidates[best_name][2])

        if best_frames is None:
            return OptimalRoute(
                map_type=TOPOLOGY_FLAT_RUN_CORRIDOR,
                strategy="no_solution",
                optimal_frames=MAX_PLAN_FRAMES,
                action_sequence=[],
                notes=notes,
            )

        return OptimalRoute(
            map_type=TOPOLOGY_FLAT_RUN_CORRIDOR,
            strategy=best_name,
            optimal_frames=int(best_frames),
            action_sequence=best_actions,
            verified_in_sim=False,
            verified_frames=None,
            notes=notes,
        )
