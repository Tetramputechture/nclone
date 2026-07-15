"""Level-INDEPENDENT jump-trajectory library at 6px (Approach A bedrock).

The ninja's in-air dynamics (drag, dual-phase gravity, capped air-accel) and the
floor-jump impulse are LEVEL-INDEPENDENT and DETERMINISTIC, so every ballistic
trajectory can be enumerated ONCE, offline, over a grid of launch velocities and
in-air control sequences. Per level, a jump edge is then validated by a cheap
bitboard collision test against that level's known 6px solid mask (no per-level
simulation) -- see ``jump_distance.py``.

The trajectories come from the REAL simulator (``nclone_native``), NOT from a
hand-reimplementation of the physics. The earlier version drove the
``analytical_physics.jump_hull`` reimplementation (``_simulate_trajectory`` +
``compute_slope_impulse``); calibration against ``nclone_native`` showed that
reimplementation OVER-PREDICTS the floor-jump reach -- its trajectories keep
falling far below launch height at near-max horizontal speed, inflating the
horizontal reach (~107 cells vs the real ~61) and the descent (~76 cells below
launch vs ~0). An over-confident library would mark un-jumpable gaps as
reachable -- the opposite of what the jump-aware cost-to-go is for. So every
trajectory is now enumerated by driving the real sim on an open floor: run to the
target launch speed, jump holding for ``hold`` frames with input ``h1``, release
and continue with ``h2``, and record the arc until the ninja descends back to
launch height. Flat-floor jump physics is left/right symmetric, so only the
non-negative launch-speed bins are simulated and the negative bins are produced
by mirroring (``dx -> -dx``); this also keeps every run-up away from the side
walls. Each trajectory keeps its ORDERED swept-cell path (a prefix-collision
check needs the ordering).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from nclone.constants.physics_constants import MAX_HOR_SPEED, MAX_JUMP_DURATION

CELL = 6.0  # px (the obs / free-space grid resolution)

# Launch horizontal-speed bins at the moment of a floor jump. The ninja can hold
# any speed in [-MAX_HOR_SPEED, +MAX_HOR_SPEED]; 9 bins covers the reach envelope.
N_VX_BINS = 9
VX_BINS: np.ndarray = np.linspace(-MAX_HOR_SPEED, MAX_HOR_SPEED, N_VX_BINS)

# In-air control sweep: jump-hold duration (apex height) x horizontal input during
# the hold (h1) and after release (h2). This is the steering that gives the ninja
# its reachable-landing envelope.
HOLD_DURATIONS: Tuple[int, ...] = (1, 3, 5, 8, 12, 16, 20, 25, 30, MAX_JUMP_DURATION)
H_INPUTS: Tuple[int, ...] = (-1, 0, 1)

# Open-level geometry for the offline rollouts: a solid floor across the bottom
# with clear space above and to the right. The ninja spawns near the LEFT so a
# rightward run-up to max speed plus the jump arc stays well inside the side
# walls (so no arc is truncated by a wall). LEVEL-INDEPENDENT constant.
_OPEN_MAP_FLOOR_ROW = 14
_OPEN_MAP_SPAWN_COL = 6
_MAX_RUNUP_FRAMES = 400  # cap on the run-to-target-speed phase
_MAX_ARC_FRAMES = 160  # safety cap on a single jump arc


@dataclass(frozen=True)
class Trajectory:
    """One ordered ballistic path, in CELL offsets relative to the launch cell.

    ``offsets`` : (K, 2) int, consecutive-deduped (dcx, dcy) the ninja passes through.
    ``descending`` : (K,) bool, True where vy > 0 (the ninja can LAND here if the
                    cell directly below is solid).
    ``frames`` : (K,) int, cumulative frame cost at each cell (the edge weight).
    """

    offsets: np.ndarray
    descending: np.ndarray
    frames: np.ndarray


def _cells_from_points(points) -> Trajectory:
    """Convert a frame-by-frame (frame, px, py, vx, vy) arc to an ordered,
    consecutive-deduped 6px cell path relative to its first point."""
    if not points:
        return Trajectory(np.zeros((0, 2), np.int32), np.zeros(0, bool), np.zeros(0, np.int32))
    _, px0, py0, _, _ = points[0]
    cx0, cy0 = round(px0 / CELL), round(py0 / CELL)
    offs: List[Tuple[int, int]] = []
    desc: List[bool] = []
    frs: List[int] = []
    last = None
    for frame, px, py, _vx, vy in points:
        c = (round(px / CELL) - cx0, round(py / CELL) - cy0)
        if c != last:
            offs.append(c)
            desc.append(bool(vy > 0.0))
            frs.append(int(frame))
            last = c
    return Trajectory(
        np.asarray(offs, np.int32),
        np.asarray(desc, bool),
        np.asarray(frs, np.int32),
    )


def _build_open_map_data() -> List[int]:
    """A solid-floor open level for the offline real-sim rollouts."""
    from nclone.map_generation.map import Map

    m = Map()
    fr = _OPEN_MAP_FLOOR_ROW
    for x in range(42):
        for r in range(fr, 23):
            m.set_tile(x, r, 1)  # full solid tile
    m.set_ninja_spawn(_OPEN_MAP_SPAWN_COL, fr - 1)  # standing on the floor
    return m.map_data()


def _establish_launch_state(sim, rest_blob, target_vx: float):
    """Restore the grounded-at-rest state, then run RIGHT until vx reaches
    ``target_vx`` (>= 0), and snapshot the launch state. Returns the blob."""
    sim.restore_ninja_state(rest_blob)
    if target_vx > 1e-6:
        prev = 0.0
        for _ in range(_MAX_RUNUP_FRAMES):
            sim.tick(1, 0)  # run right, no jump
            vx, _ = sim.get_ninja_velocity()
            # Stop at the target, or when the run-up plateaus (top speed bin).
            if vx >= target_vx - 1e-9 or vx <= prev:
                break
            prev = vx
    return sim.save_ninja_state()


def _rollout_arc(sim, launch_blob, hold: int, h1: int, h2: int):
    """Drive ONE real-sim floor jump and return the (frame, px, py, vx, vy) arc.

    From the launch state: jump holding for ``hold`` frames with horizontal input
    ``h1``, then release and continue with ``h2``, recording every frame until the
    ninja descends back to launch height (a landing-height crossing) or the safety
    frame cap is hit. The arc stays above the floor the whole time, so it is the
    collision-free ballistic path the per-level mask test in ``jump_distance``
    needs."""
    sim.restore_ninja_state(launch_blob)
    px0, py0 = sim.get_ninja_pos()
    vx0, vy0 = sim.get_ninja_velocity()
    points: List[Tuple[int, float, float, float, float]] = [(0, px0, py0, vx0, vy0)]
    for frame in range(1, _MAX_ARC_FRAMES + 1):
        jump = 1 if frame <= hold else 0
        hor = h1 if frame <= hold else h2
        sim.tick(hor, jump)
        px, py = sim.get_ninja_pos()
        vx, vy = sim.get_ninja_velocity()
        points.append((frame, px, py, vx, vy))
        if sim.has_died():
            break
        # py increases downward; once past the hold and back at/below launch
        # height, the arc has completed its up-and-over.
        if frame > hold and py >= py0 - 0.4:
            break
    return points


def _mirror(traj: Trajectory) -> Trajectory:
    """Horizontal mirror of a trajectory (dx -> -dx). Flat-floor jump physics is
    left/right symmetric, so the negative-vx bins are exactly the mirror of the
    positive ones (over the symmetric h-input sweep)."""
    offs = traj.offsets.copy()
    if offs.shape[0]:
        offs[:, 0] = -offs[:, 0]
    return Trajectory(offs, traj.descending.copy(), traj.frames.copy())


def build_floor_trajectories() -> Dict[int, List[Trajectory]]:
    """Enumerate all level-independent floor-jump trajectories, keyed by vx bin.

    Returns ``{vx_bin_index: [Trajectory, ...]}``. Trajectories are produced by
    rolling out the REAL simulator (``nclone_native``) on an open floor, so the
    reach envelope is FAITHFUL (no physics reimplementation). Level-independent +
    deterministic => compute once (cacheable to disk by the caller).
    """
    import nclone_native as Nnative

    sim = Nnative.Simulator.load(_build_open_map_data())
    sim.fast_reset()
    # Settle to a grounded, zero-velocity state so a clean floor jump can fire.
    for _ in range(30):
        sim.tick(0, 0)
    rest_blob = sim.save_ninja_state()

    def _bin_trajectories(target_vx: float) -> List[Trajectory]:
        launch_blob = _establish_launch_state(sim, rest_blob, target_vx)
        trajs: List[Trajectory] = []
        seen = set()
        for hold in HOLD_DURATIONS:
            for h1 in H_INPUTS:
                for h2 in H_INPUTS:
                    pts = _rollout_arc(sim, launch_blob, hold, h1, h2)
                    traj = _cells_from_points(pts)
                    if traj.offsets.shape[0] < 2:
                        continue
                    # Dedup identical paths (many (h1,h2,hold) combos coincide).
                    key = traj.offsets.tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    trajs.append(traj)
        return trajs

    # Flat-floor physics is left/right symmetric: simulate the non-negative bins
    # (run-up always rightward, away from the left wall) and mirror for negatives.
    pos_cache: Dict[float, List[Trajectory]] = {}
    out: Dict[int, List[Trajectory]] = {}
    for bi, vx in enumerate(VX_BINS):
        target = abs(float(vx))
        if target not in pos_cache:
            pos_cache[target] = _bin_trajectories(target)
        base = pos_cache[target]
        out[bi] = base if vx >= 0 else [_mirror(t) for t in base]
    return out


# ---------------------------------------------------------------------------
# WALL-JUMP trajectory library (level-INDEPENDENT, LEFT-wall orientation).
#
# Like the floor library, the wall-jump impulse + in-air dynamics are level-
# independent, so every wall-jump arc is enumerated ONCE offline by driving the
# REAL sim: inject a wall-slide state against a LEFT wall (wall_normal=+1, so the
# ninja launches up-and-RIGHT), fire the jump, and record the ballistic arc. The
# per-level cost is then just a cheap bitboard sweep of these arcs against the
# level's 6px solid mask (in jump_distance). Right-wall arcs are the horizontal
# mirror (``_mirror``) of these.
#
# TRICK JUMPS: ``wall_jump`` ADDS its vertical impulse WITHOUT zeroing a retained
# UPWARD velocity (nclone/ninja.py: ``if yspeed > 0: yspeed = 0`` then
# ``yspeed += jy``), so a wall-jump chained off a previous one keeps accumulating
# upward speed toward -MAX_VER_SPEED and reaches MUCH farther per jump. A single
# from-rest launch is just one point in a whole range of arrival vertical
# velocities, so we bin the LAUNCH vy over the full physically-reachable range
# (terminal downward slide down to full upward -MAX_VER_SPEED) exactly like the
# floor library bins horizontal launch speed, and key the library by vy-bin.
# ---------------------------------------------------------------------------

_TILE_PX = 24.0  # map tile size (the free-space CELL is 6px = a quarter tile)
_WALL_MAP_WALL_COLS = 2      # solid interior wall columns -> right face at 72px
# Launch LOW in the map so tall trick-jump arcs are not clipped by the ceiling
# (the reach for the most-upward vy bin spans nearly the full map height).
_WALL_MAP_LAUNCH_ROW = 20
_MAX_WALL_ARC_FRAMES = 140
_WALL_DESCEND_MARGIN_CELLS = 8  # terminate an arc once it falls this far below launch

# LAUNCH vertical-velocity bins: from terminal downward wall-slide (~+5.5) up to
# full upward -MAX_VER_SPEED (-12.0). Denser near the upward extreme where reach
# varies most (this is where trick jumps live). Mirrors floor VX_BINS. The
# downward/near-rest bins reach nearly identically (~7-8 cells), so only two are
# kept there; the upward regime is sampled finely (reach grows steeply with vy).
WALL_VY_BINS: np.ndarray = np.array(
    [5.5, 0.0, -1.4, -3.0, -5.0, -7.0, -9.0, -12.0], dtype=np.float64
)

# LEFT-wall launch geometry: solid interior cols 0..WALL_COLS-1 sit at grid tx
# 1..WALL_COLS (a 1-tile border shifts everything right), so the wall's right
# FACE is at pixel (WALL_COLS + 1) * 24.
_WALL_FACE_PX = (_WALL_MAP_WALL_COLS + 1) * _TILE_PX

_WALL_LIBRARY: "Dict[int, List[Trajectory]] | None" = None


def _build_wall_map_data() -> List[int]:
    """A full-height LEFT wall with clear space to the right and a bottom floor.

    The ninja clings to the wall's right face and wall-jumps up-and-right into the
    clear region; the bottom floor only terminates far downward arcs. The launch
    is near the bottom so a max-upward trick jump has the full column of headroom."""
    from nclone.map_generation.map import Map

    m = Map()
    for r in range(0, 23):
        for c in range(0, _WALL_MAP_WALL_COLS):
            m.set_tile(c, r, 1)  # solid left wall
    for x in range(42):
        m.set_tile(x, 22, 1)  # floor (below the launch)
    m.set_ninja_spawn(6, 21)
    return m.map_data()


def _find_wall_launch(sim, y: float) -> float:
    """Probe a launch x that reliably FIRES the wall-jump impulse.

    Wall contact is recomputed from real geometry during the collision phase, so
    the injected x must penetrate the wall face slightly (a couple px) for the sim
    to register wall contact and fire the jump; too deep ejects the ninja. Pick the
    first penetration depth whose launch reaches state 3 (Jumping) for BOTH the
    slide (h toward wall) and regular (h away from wall) launches."""
    from nclone.constants.physics_constants import NINJA_RADIUS

    for pen in (2.0, 1.5, 2.5, 1.0, 3.0):
        x0 = _WALL_FACE_PX + NINJA_RADIUS - pen
        sim.inject_ninja_for_value_iteration_wall_slide(x0, y, 0.0, 1)
        sim.tick_tiles_only(-1, 1)  # slide launch (toward wall)
        if sim.get_ninja_state_code() != 3:
            continue
        sim.inject_ninja_for_value_iteration_wall_slide(x0, y, 0.0, 1)
        sim.tick_tiles_only(1, 1)  # regular launch (away from wall)
        if sim.get_ninja_state_code() == 3:
            return x0
    return _WALL_FACE_PX + NINJA_RADIUS - 2.0


def _rollout_wall_arc(sim, x0, y, target_vy, jump_type, hold, h1, h2):
    """Drive ONE real-sim wall jump off a LEFT wall and return its (frame, px, py,
    vx, vy) arc.

    The ninja is injected into the wall-slide state with the ARRIVAL vertical
    velocity ``target_vy`` (negative = moving UP, the trick-jump regime); the
    ``inject_ninja_for_value_iteration_wall_slide`` injector takes vy directly and
    forces state 5 + wall contact, so the launch tick's ``wall_jump`` ADDS its
    impulse on top of the retained velocity. The launch frame fires the jump
    (input TOWARD the wall = -1 for the ``slide`` impulse; ``h1`` for the
    ``regular`` impulse). Subsequent hold frames steer with ``h1`` while jump is
    held, then ``h2`` after release. The arc ends when the ninja lands (airborne
    then grounded), descends a safe margin below launch, or hits the frame cap."""
    sim.inject_ninja_for_value_iteration_wall_slide(x0, y, float(target_vy), 1)
    px0, py0 = sim.get_ninja_pos()
    vx0, vy0 = sim.get_ninja_velocity()
    points: List[Tuple[int, float, float, float, float]] = [(0, px0, py0, vx0, vy0)]
    launch_h = -1 if jump_type == "slide" else h1
    airborne_seen = False
    descend_limit = py0 + _WALL_DESCEND_MARGIN_CELLS * CELL
    for frame in range(1, _MAX_WALL_ARC_FRAMES + 1):
        if frame == 1:
            hor, jump = launch_h, 1
        else:
            jump = 1 if frame <= hold else 0
            hor = h1 if frame <= hold else h2
        sim.tick_tiles_only(hor, jump)
        px, py = sim.get_ninja_pos()
        vx, vy = sim.get_ninja_velocity()
        points.append((frame, px, py, vx, vy))
        if sim.has_died():
            break
        if sim.get_ninja_airborn():
            airborne_seen = True
        elif airborne_seen:
            break  # landed on a surface
        if airborne_seen and py >= descend_limit:
            break  # descended below launch -> the up-and-over has completed
    return points


# Cap on distinct trajectories kept per vy-bin (guards the per-level sweep cost:
# the tall trick arcs are long, so the sweep is O(wall_cells x trajectories x
# arc_len)). When a bin exceeds the cap we keep the FARTHEST-reaching arcs (sorted
# by 2*up + horizontal) so the cap never sacrifices reach, and LOG it.
_WALL_MAX_TRAJ_PER_BIN = 28


def build_wall_trajectories() -> Dict[int, List[Trajectory]]:
    """Enumerate level-independent LEFT-wall wall-jump trajectories, keyed by
    vy-bin (arrival vertical velocity at the wall).

    Returns ``{vy_bin_index: [Trajectory, ...]}`` (LEFT-wall orientation: launch
    up-and-RIGHT). Right-wall arcs are ``_mirror`` of these. Rolled out from the
    REAL sim so the reach envelope is FAITHFUL, including chained trick jumps that
    arrive with retained upward velocity. De-duplicated ACROSS bins by
    ``offsets.tobytes()`` so the union the caller sweeps does not explode.
    Level-independent + deterministic => computed once and cached at module scope."""
    global _WALL_LIBRARY
    if _WALL_LIBRARY is not None:
        return _WALL_LIBRARY

    import logging

    import nclone_native as Nnative

    sim = Nnative.Simulator.load(_build_wall_map_data())
    sim.fast_reset()
    y = _WALL_MAP_LAUNCH_ROW * _TILE_PX + 12.0
    x0 = _find_wall_launch(sim, y)

    def _reach(t: Trajectory) -> int:
        up = -int(t.offsets[:, 1].min())
        dx = int(np.abs(t.offsets[:, 0]).max())
        return 2 * up + dx  # prioritise vertical reach (the trick-jump payoff)

    out: Dict[int, List[Trajectory]] = {}
    seen = set()  # dedup across ALL bins (the caller sweeps the union)
    for bi, vy in enumerate(WALL_VY_BINS):
        trajs: List[Trajectory] = []
        for jump_type in ("regular", "slide"):
            for hold in HOLD_DURATIONS:
                for h1 in H_INPUTS:
                    for h2 in H_INPUTS:
                        pts = _rollout_wall_arc(sim, x0, y, vy, jump_type, hold, h1, h2)
                        traj = _cells_from_points(pts)
                        if traj.offsets.shape[0] < 2:
                            continue
                        # The jump must fire and leave the wall (move right).
                        if not (traj.offsets[:, 0] >= 1).any():
                            continue
                        key = traj.offsets.tobytes()
                        if key in seen:
                            continue
                        seen.add(key)
                        trajs.append(traj)
        if len(trajs) > _WALL_MAX_TRAJ_PER_BIN:
            logging.getLogger(__name__).info(
                "build_wall_trajectories: vy-bin %d (vy=%.1f) capped %d -> %d "
                "(keeping farthest-reaching arcs)",
                bi, float(vy), len(trajs), _WALL_MAX_TRAJ_PER_BIN,
            )
            trajs.sort(key=_reach, reverse=True)
            trajs = trajs[:_WALL_MAX_TRAJ_PER_BIN]
        out[bi] = trajs

    _WALL_LIBRARY = out
    return out


def wall_reach_envelope(trajs) -> Dict[str, float]:
    """Summary of the wall-jump reach envelope (max up / horizontal, count).

    Accepts either the vy-binned dict or a flat list of trajectories."""
    if isinstance(trajs, dict):
        flat = [t for ts in trajs.values() for t in ts]
    else:
        flat = list(trajs)
    max_up = 0
    max_dx = 0
    for t in flat:
        if t.offsets.shape[0] == 0:
            continue
        max_up = max(max_up, -int(t.offsets[:, 1].min()))
        max_dx = max(max_dx, int(np.abs(t.offsets[:, 0]).max()))
    return {"max_up_cells": max_up, "max_horizontal_cells": max_dx, "count": len(flat)}


def reach_envelope(trajs: Dict[int, List[Trajectory]]) -> Dict[str, float]:
    """Summary of the reachable landing envelope (for sanity checks): max up / max
    horizontal reach in CELLs over all trajectories, considering descending cells
    as landing candidates."""
    max_up = 0
    max_dx = 0
    max_down = 0
    for trajs_b in trajs.values():
        for t in trajs_b:
            if t.offsets.shape[0] == 0:
                continue
            land = t.offsets[t.descending] if t.descending.any() else t.offsets
            up = -int(t.offsets[:, 1].min())          # cells above launch (y up = neg)
            down = int(t.offsets[:, 1].max())
            dx = int(np.abs(t.offsets[:, 0]).max())
            max_up = max(max_up, up)
            max_dx = max(max_dx, dx)
            max_down = max(max_down, down)
    return {"max_up_cells": max_up, "max_horizontal_cells": max_dx, "max_down_cells": max_down}
