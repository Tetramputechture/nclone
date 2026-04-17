"""Launch / landing cache builders.

For a given 3x3 signature, the launch builder sweeps every (surface_type,
start_subcell, vel_bin, input_class) combination on a tiny harness map and
records either:

* A full in-window edge (stayed_in_window = True) - the ninja landed on a
  surface inside the 3x3 signature window.  Equivalent to the legacy
  simulator's EdgeResult, with the landing expressed as an offset relative
  to the signature center cell.
* A window-exit handoff state (escaped = True) - the ninja's center cell
  stepped outside the 3x3 signature.  We record the state at that frame so
  the ballistic propagator can continue on level tiles.

The landing builder is the symmetric twin: the ninja is injected mid-air
entering the 3x3 window from a given direction / velocity and the
simulator resolves collision to produce a surface-state node.

Design decisions
----------------

* Harness cells outside the 3x3 signature are empty for one ring (the
  "buffer") and solid afterwards, so the ninja leaves the window cleanly.
* Window exit is detected by ninja CELL position.  If
  ``(cell_col - center_col, cell_row - center_row)`` is outside
  ``{-1, 0, +1}**2`` we treat this as an exit and record the pre-tick
  state (position, velocity, hold-remaining).
* We store relative coordinates (pixels, ints) so the same entries work
  anywhere the signature appears in the level grid.
* The builder is fully deterministic and can be parallelised by
  distributing distinct ``(sig_hash, surface_type)`` groups across
  workers - each worker owns its own ``nclone_native.Simulator`` instance.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from ...constants import (
    NINJA_RADIUS,
    TILE_PIXEL_SIZE,
    MAX_JUMP_DURATION,
    AIR_STATES,
    GROUND_STATES,
    WALL_STATES,
)
from ..surface_extraction import SURFACE_FLOOR, SURFACE_WALL
from .harness import (
    HARNESS_CENTER_COL,
    HARNESS_CENTER_ROW,
    HARNESS_WINDOW,
    build_landing_harness_map,
    build_launch_harness_map,
    launch_cell_origin_px,
)
from .signature import hash_signature
from .store import LandingEntry, LaunchEntry, TileCacheStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Input primitive sweep - kept in sync with edge_enumeration but owned here
# so the tile-cache package has no circular dependency on edge_enumeration.
# ---------------------------------------------------------------------------
RUN_DURATIONS: Tuple[int, ...] = (1, 2, 3, 5, 8, 12, 18, 24, 36, 48, 60)
JUMP_HOLD_DURATIONS: Tuple[int, ...] = (1, 3, 5, 8, 12, 16, 20, 25, 30, 35, 40, 45)
H_INPUTS: Tuple[int, ...] = (-1, 0, 1)
POST_RELEASE_INPUTS: Tuple[int, ...] = (-1, 0, 1)

# Per-arc frame budget inside the harness.  Arcs that never exit the
# window AND never land within this budget are discarded.
MAX_HARNESS_FRAMES: int = 240

# Quantisation: position -> int8 relative to launch (pixels, clipped to int16
# range). velocity -> int16 at 0.01 px/frame resolution.
_POS_SCALE: float = 1.0
_VEL_SCALE: float = 100.0


# ---------------------------------------------------------------------------
# Input-class enumeration
# ---------------------------------------------------------------------------
# Stable integer encoding used as one of the launch cache key components.
# The enumeration is chosen so all floor inputs precede wall inputs, which
# simplifies iteration for a given surface type.

_RUN_BASE: int = 0
_FLOOR_JUMP_BASE: int = _RUN_BASE + len(H_INPUTS) * len(RUN_DURATIONS)
_WALL_JUMP_BASE: int = (
    _FLOOR_JUMP_BASE
    + len(H_INPUTS) * len(JUMP_HOLD_DURATIONS) * len(POST_RELEASE_INPUTS)
)
_N_INPUT_CLASSES: int = _WALL_JUMP_BASE + len(H_INPUTS) * len(JUMP_HOLD_DURATIONS)


@dataclass(frozen=True)
class InputClass:
    """One launch primitive - decoded from an integer input_class id."""
    kind: str  # 'run' | 'floor_jump' | 'wall_jump'
    h_during: int
    hold: int = 0
    h_after: int = 0
    duration: int = 0  # for 'run'


def decode_input_class(idx: int) -> InputClass:
    """Decode an input_class integer id back into its components."""
    if idx < 0 or idx >= _N_INPUT_CLASSES:
        raise ValueError(f"input_class {idx} out of range")
    if idx < _FLOOR_JUMP_BASE:
        j = idx - _RUN_BASE
        h_i = j // len(RUN_DURATIONS)
        d_i = j % len(RUN_DURATIONS)
        return InputClass(
            kind="run",
            h_during=H_INPUTS[h_i],
            duration=RUN_DURATIONS[d_i],
        )
    if idx < _WALL_JUMP_BASE:
        j = idx - _FLOOR_JUMP_BASE
        n_after = len(POST_RELEASE_INPUTS)
        n_hold = len(JUMP_HOLD_DURATIONS)
        h_i = j // (n_hold * n_after)
        rest = j % (n_hold * n_after)
        hold_i = rest // n_after
        after_i = rest % n_after
        return InputClass(
            kind="floor_jump",
            h_during=H_INPUTS[h_i],
            hold=JUMP_HOLD_DURATIONS[hold_i],
            h_after=POST_RELEASE_INPUTS[after_i],
        )
    j = idx - _WALL_JUMP_BASE
    n_hold = len(JUMP_HOLD_DURATIONS)
    h_i = j // n_hold
    hold_i = j % n_hold
    return InputClass(
        kind="wall_jump",
        h_during=H_INPUTS[h_i],
        hold=JUMP_HOLD_DURATIONS[hold_i],
        h_after=H_INPUTS[h_i],
    )


def classes_for_surface(surface_type: int) -> Iterable[int]:
    """Enumerate the input_class ids relevant to a given surface type."""
    from ..surface_extraction import (
        SURFACE_SLIDE as _SURFACE_SLIDE,
        SURFACE_LAUNCH_PAD as _SURFACE_LAUNCH_PAD,
    )
    if surface_type == SURFACE_FLOOR:
        yield from range(_RUN_BASE, _WALL_JUMP_BASE)
    elif surface_type == SURFACE_WALL or surface_type == _SURFACE_SLIDE:
        # Wall / wall-slide surfaces only support wall-jump / wall-slide style arcs.
        yield from range(_WALL_JUMP_BASE, _N_INPUT_CLASSES)
        # Wall-slide falls can reuse the 'run' encoding (horizontal input,
        # duration) since their physics are a hold-free drop along the wall.
        yield from range(_RUN_BASE, _FLOOR_JUMP_BASE)
    elif surface_type == _SURFACE_LAUNCH_PAD:
        # Launch pads emit a single kick; the "run" class encoding is used to
        # sweep post-release horizontal inputs / arc durations.
        yield from range(_RUN_BASE, _FLOOR_JUMP_BASE)


# ---------------------------------------------------------------------------
# Subcell quantisation
# ---------------------------------------------------------------------------
SUBCELL_BINS: int = 4                  # 4x4 = 16 start positions per cell
_SUBCELL_WIDTH: float = TILE_PIXEL_SIZE / SUBCELL_BINS  # 6 px


def quantise_subcell(x_in_cell: float, y_in_cell: float) -> Tuple[int, int]:
    """Map an in-cell pixel offset to (sx, sy) bin indices 0..SUBCELL_BINS-1."""
    sx = int(x_in_cell // _SUBCELL_WIDTH)
    sy = int(y_in_cell // _SUBCELL_WIDTH)
    return (
        max(0, min(SUBCELL_BINS - 1, sx)),
        max(0, min(SUBCELL_BINS - 1, sy)),
    )


def subcell_centre(sx: int, sy: int) -> Tuple[float, float]:
    """Inverse of :func:`quantise_subcell` - returns the centre of the bin."""
    return (
        (sx + 0.5) * _SUBCELL_WIDTH,
        (sy + 0.5) * _SUBCELL_WIDTH,
    )


# ---------------------------------------------------------------------------
# Airborne velocity quantisation for the landing cache
# ---------------------------------------------------------------------------
# 2D air velocity bins: 15 along x (matches floor bin count), 25 along y
# (wider range so terminal velocity near 6 is covered).  This is a separate
# schedule from the on-surface velocity bins because landing velocity gets
# clamped and re-expressed as a surface velocity anyway.
AIR_VX_BIN_CENTERS = tuple(
    -6.0 + 12.0 * i / 14 for i in range(15)
)
AIR_VY_BIN_CENTERS = tuple(
    -8.0 + 16.0 * i / 24 for i in range(25)
)


def quantise_air_velocity(vx: float, vy: float) -> Tuple[int, int]:
    def _nearest(centers: Sequence[float], v: float) -> int:
        best_i, best_d = 0, float("inf")
        for i, c in enumerate(centers):
            d = abs(c - v)
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    return _nearest(AIR_VX_BIN_CENTERS, vx), _nearest(AIR_VY_BIN_CENTERS, vy)


# ---------------------------------------------------------------------------
# State codes for clarity
# ---------------------------------------------------------------------------
_STATE_DEAD: int = 6
_STATE_AWAITING_DEATH: int = 7
_DEATH_STATES = frozenset({_STATE_DEAD, _STATE_AWAITING_DEATH})
_STATE_WALL_SLIDING: int = 5


@dataclass
class _SimObs:
    """Per-tick observation captured during harness simulation."""
    x: float
    y: float
    vx: float
    vy: float
    state: int
    jump_duration: int = 0
    hold_left: int = 0


# ---------------------------------------------------------------------------
# Launch simulation core
# ---------------------------------------------------------------------------

def _cell_col_row(x: float, y: float) -> Tuple[int, int]:
    return int(math.floor(x / TILE_PIXEL_SIZE)), int(math.floor(y / TILE_PIXEL_SIZE))


def _inject_on_surface(
    sim,
    surface_type: int,
    abs_x: float,
    abs_y: float,
    vx: float,
    vy: float,
) -> None:
    """Inject the ninja into the simulator on the given surface type."""
    from ..surface_extraction import (
        SURFACE_SLIDE as _SURFACE_SLIDE,
        SURFACE_LAUNCH_PAD as _SURFACE_LAUNCH_PAD,
    )
    if surface_type == SURFACE_FLOOR:
        # Floor normal points up (nx=0, ny=-1 in screen coords).
        sim.inject_ninja_for_value_iteration_grounded(
            abs_x, abs_y, vx, vy, 0.0, -1.0,
        )
    elif surface_type == SURFACE_WALL or surface_type == _SURFACE_SLIDE:
        # Wall / wall-slide node: inject_ninja_for_value_iteration_wall_slide
        # takes a wall_normal sign (+1 = normal points right, ninja on left
        # wall).  Slide nodes carry the current vy as their bin coordinate.
        sim.inject_ninja_for_value_iteration_wall_slide(abs_x, abs_y, vy, 1.0)
    elif surface_type == _SURFACE_LAUNCH_PAD:
        # Launch-pad kick injects airborne at the pad with the boost velocity.
        sim.inject_ninja_for_value_iteration(abs_x, abs_y, vx, vy, 0)
    else:
        sim.inject_ninja_for_value_iteration(abs_x, abs_y, vx, vy, 0)


def _quantise_pos_rel(dx: float, dy: float) -> Tuple[int, int]:
    # int16 range sufficient for harness displacements (< 200 px).
    return int(round(dx * _POS_SCALE)), int(round(dy * _POS_SCALE))


def _quantise_vel(vx: float, vy: float) -> Tuple[int, int]:
    # +/- 6 px/frame typical; 0.01 px/frame resolution -> int16 safe.
    return int(round(vx * _VEL_SCALE)), int(round(vy * _VEL_SCALE))


@dataclass
class LaunchSimResult:
    """Outcome of one harness simulation (pre-packing)."""
    escaped: bool
    died: bool
    frames: int
    # Absolute end-of-run state (in harness coords)
    end_x: float = 0.0
    end_y: float = 0.0
    end_vx: float = 0.0
    end_vy: float = 0.0
    end_state: int = 0
    hold_remaining: int = 0


def _simulate_one_input_class(
    sim,
    input_class_id: int,
    launch_abs_x: float,
    launch_abs_y: float,
    launch_vx: float,
    launch_vy: float,
    surface_type: int,
    center_col: int,
    center_row: int,
    max_frames: int = MAX_HARNESS_FRAMES,
) -> Optional[LaunchSimResult]:
    """Run one harness sim for a single input-class / start / velocity triple.

    Returns ``None`` if the simulation could not start (initiation tick
    killed the ninja).  Returns a :class:`LaunchSimResult` otherwise.
    """
    ic = decode_input_class(input_class_id)

    snapshot = sim.save_ninja_state()
    sim.restore_ninja_state(snapshot)
    _inject_on_surface(sim, surface_type, launch_abs_x, launch_abs_y,
                       launch_vx, launch_vy)

    # Run sequence:
    # - 'run': tick_tiles_only(h_during, 0) for `duration` frames
    # - 'floor_jump' / 'wall_jump': one initiation tick with jump=1,
    #   then up to hold-1 more hold-ticks, then h_after ticks until
    #   exit/land/timeout.
    frames_used = 0
    hold_budget = ic.hold
    hold_remaining = 0

    def _step(h_input: int, jump_input: int) -> Optional[LaunchSimResult]:
        nonlocal frames_used
        # Capture pre-tick state (needed for the "exited last frame" case).
        prev_pos = sim.get_ninja_pos()
        prev_vel = sim.get_ninja_velocity()
        prev_jumpd = sim.get_ninja_jump_duration()
        sim.tick_tiles_only(h_input, jump_input)
        frames_used += 1

        sc = sim.get_ninja_state_code()
        if sc in _DEATH_STATES:
            return LaunchSimResult(
                escaped=False, died=True, frames=frames_used,
                end_x=prev_pos[0], end_y=prev_pos[1],
                end_vx=prev_vel[0], end_vy=prev_vel[1],
                end_state=sc,
            )

        pos = sim.get_ninja_pos()
        vel = sim.get_ninja_velocity()
        jumpd = sim.get_ninja_jump_duration()
        c, r = _cell_col_row(pos[0], pos[1])
        half = HARNESS_WINDOW // 2
        if abs(c - center_col) > half or abs(r - center_row) > half:
            # Exited window on this tick -> record post-tick handoff state.
            # We keep the post-tick state (not pre-tick) because the ninja
            # did already move through one integration frame this tick.
            return LaunchSimResult(
                escaped=True, died=False, frames=frames_used,
                end_x=pos[0], end_y=pos[1],
                end_vx=vel[0], end_vy=vel[1],
                end_state=sc,
                hold_remaining=max(0, MAX_JUMP_DURATION - jumpd)
                                if jumpd > 0 else 0,
            )

        # Landed inside the window?
        if sc in GROUND_STATES | WALL_STATES and frames_used > 0:
            # If we started on a surface and stayed on a surface (frame 1 for
            # 'run' class) that's still a valid "in-window" landing.
            if ic.kind == "run" and frames_used >= 1:
                return LaunchSimResult(
                    escaped=False, died=False, frames=frames_used,
                    end_x=pos[0], end_y=pos[1],
                    end_vx=vel[0], end_vy=vel[1],
                    end_state=sc,
                )
            # Jumps: landing after an airborne phase.
            if ic.kind != "run":
                return LaunchSimResult(
                    escaped=False, died=False, frames=frames_used,
                    end_x=pos[0], end_y=pos[1],
                    end_vx=vel[0], end_vy=vel[1],
                    end_state=sc,
                )
        return None

    if ic.kind == "run":
        for _ in range(ic.duration):
            res = _step(ic.h_during, 0)
            if res is not None:
                return res
        pos = sim.get_ninja_pos()
        vel = sim.get_ninja_velocity()
        sc = sim.get_ninja_state_code()
        return LaunchSimResult(
            escaped=False, died=False, frames=frames_used,
            end_x=pos[0], end_y=pos[1],
            end_vx=vel[0], end_vy=vel[1],
            end_state=sc,
        )

    # Jump kinds share initiation+hold+fall structure.
    hold_budget = min(ic.hold, MAX_JUMP_DURATION)
    for _ in range(hold_budget):
        res = _step(ic.h_during, 1)
        if res is not None:
            return res

    # Post-hold phase: the ninja has either already exited or is airborne.
    for _ in range(max_frames - frames_used):
        res = _step(ic.h_after, 0)
        if res is not None:
            return res

    pos = sim.get_ninja_pos()
    vel = sim.get_ninja_velocity()
    sc = sim.get_ninja_state_code()
    return LaunchSimResult(
        escaped=False, died=False, frames=frames_used,
        end_x=pos[0], end_y=pos[1],
        end_vx=vel[0], end_vy=vel[1],
        end_state=sc,
    )


def _sim_result_to_launch_entry(
    res: LaunchSimResult,
    launch_abs_x: float,
    launch_abs_y: float,
    center_col: int,
    center_row: int,
    surface_type_in_window: int = SURFACE_FLOOR,
    velocity_to_bin=None,
) -> LaunchEntry:
    """Convert a ``LaunchSimResult`` to the packed ``LaunchEntry`` format.

    ``velocity_to_bin`` is a callable ``(v: float, surface_type: int) -> int``
    that maps a surface velocity to the schedule index (injected from
    surface_graph to avoid a circular import).
    """
    if res.died:
        return LaunchEntry(
            escaped=0, died=1, frames=res.frames,
        )

    if res.escaped:
        dx = res.end_x - launch_abs_x
        dy = res.end_y - launch_abs_y
        pos_dx, pos_dy = _quantise_pos_rel(dx, dy)
        vx_q, vy_q = _quantise_vel(res.end_vx, res.end_vy)
        return LaunchEntry(
            escaped=1, died=0, frames=res.frames,
            exit_pos_dx=pos_dx, exit_pos_dy=pos_dy,
            exit_vx_q=vx_q, exit_vy_q=vy_q,
            hold_remaining=res.hold_remaining,
        )

    # Landed in-window.  Compute cell offset relative to the signature centre.
    cell_col, cell_row = _cell_col_row(res.end_x, res.end_y)
    land_cell_dx = cell_col - center_col
    land_cell_dy = cell_row - center_row
    # Subcell inside the landing cell (0..23)
    sub_x = int(round(res.end_x - cell_col * TILE_PIXEL_SIZE))
    sub_y = int(round(res.end_y - cell_row * TILE_PIXEL_SIZE))

    # Determine landing surface type from the terminal state code.
    if res.end_state == _STATE_WALL_SLIDING:
        land_stype = SURFACE_WALL
        vel_bin = 0
        if velocity_to_bin is not None:
            vel_bin = int(velocity_to_bin(res.end_vy, SURFACE_WALL))
    else:
        land_stype = SURFACE_FLOOR
        vel_bin = 0
        if velocity_to_bin is not None:
            vel_bin = int(velocity_to_bin(res.end_vx, SURFACE_FLOOR))

    return LaunchEntry(
        escaped=0, died=0, frames=res.frames,
        land_cell_dx=land_cell_dx, land_cell_dy=land_cell_dy,
        land_subcell_x=max(0, min(TILE_PIXEL_SIZE - 1, sub_x)),
        land_subcell_y=max(0, min(TILE_PIXEL_SIZE - 1, sub_y)),
        land_surface_type=land_stype,
        land_vel_bin=vel_bin,
    )


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------

def populate_launch_entry(
    store: TileCacheStore,
    signature: Sequence[int],
    surface_type: int,
    subcell: Tuple[int, int],
    vel_bin_in: int,
    input_class_id: int,
    vel_bin_centre: float,
    *,
    velocity_to_bin=None,
    sim=None,
) -> LaunchEntry:
    """Populate a single launch entry in the store, running the harness sim.

    Args:
        store: Persistent tile cache store.
        signature: 9 tile ids (NW..SE row-major).
        surface_type: SURFACE_FLOOR / SURFACE_WALL.
        subcell: (sx, sy) quantised in-cell start bin indices.
        vel_bin_in: Schedule index of the source velocity bin.
        input_class_id: Integer id of the input primitive.
        vel_bin_centre: Physical velocity corresponding to ``vel_bin_in``.
        velocity_to_bin: Callable that quantises landing velocity back to
            a bin index (passed in from ``surface_graph`` to avoid a
            circular import).
        sim: Optional pre-loaded ``nclone_native.Simulator`` instance.  When
            ``None``, a fresh simulator is constructed from the harness map.
            Reuse across many input classes with the same signature.

    Returns:
        The newly computed :class:`LaunchEntry` (also stored).
    """
    sig_hash = int(hash_signature(signature))
    key_extra = (
        int(surface_type),
        int(subcell[0]),
        int(subcell[1]),
        int(vel_bin_in),
        int(input_class_id),
    )

    # Harness sim - build or reuse provided simulator.
    if sim is None:
        from nclone_native import Simulator as NativeSimulator
        map_data = build_launch_harness_map(signature)
        sim = NativeSimulator.load(map_data)

    # Map subcell -> absolute pixel position in the harness center cell.
    cx_px, cy_px = launch_cell_origin_px()
    sub_cx, sub_cy = subcell_centre(subcell[0], subcell[1])
    if surface_type == SURFACE_FLOOR:
        # Ninja centre is NINJA_RADIUS above the floor; we assume the floor
        # is the bottom of the centre tile so y = cy_px + TILE - NINJA_RADIUS.
        abs_x = cx_px + sub_cx
        abs_y = cy_px + TILE_PIXEL_SIZE - NINJA_RADIUS
        vx0, vy0 = vel_bin_centre, 0.0
    else:
        # Wall node: ninja centre is NINJA_RADIUS inside the cell along x,
        # sub_cy controls vertical placement within the cell.
        abs_x = cx_px + NINJA_RADIUS
        abs_y = cy_px + sub_cy
        vx0, vy0 = 0.0, vel_bin_centre

    padded_col = HARNESS_CENTER_COL + 1
    padded_row = HARNESS_CENTER_ROW + 1
    result = _simulate_one_input_class(
        sim, input_class_id,
        launch_abs_x=abs_x, launch_abs_y=abs_y,
        launch_vx=vx0, launch_vy=vy0,
        surface_type=surface_type,
        center_col=padded_col, center_row=padded_row,
    )
    if result is None:
        entry = LaunchEntry(escaped=0, died=1, frames=0)
    else:
        entry = _sim_result_to_launch_entry(
            result, abs_x, abs_y, padded_col, padded_row,
            velocity_to_bin=velocity_to_bin,
        )
    store.put_launch(sig_hash, key_extra, entry)
    return entry


def populate_landing_entry(
    store: TileCacheStore,
    signature: Sequence[int],
    entry_subcell: Tuple[int, int],
    entry_vx_bin_air: int,
    entry_vy_bin_air: int,
    *,
    velocity_to_bin=None,
    sim=None,
) -> LandingEntry:
    """Populate a single landing entry via the landing harness.

    The ninja is injected mid-air at the specified entry subcell with the
    bin-centre velocity; one or more collision ticks resolve its final
    surface.  Returns the computed :class:`LandingEntry`.
    """
    sig_hash = int(hash_signature(signature))
    key_extra = (
        int(entry_subcell[0]),
        int(entry_subcell[1]),
        int(entry_vx_bin_air),
        int(entry_vy_bin_air),
    )

    if sim is None:
        from nclone_native import Simulator as NativeSimulator
        map_data = build_landing_harness_map(signature)
        sim = NativeSimulator.load(map_data)

    cx_px, cy_px = launch_cell_origin_px()
    sub_cx, sub_cy = subcell_centre(entry_subcell[0], entry_subcell[1])
    abs_x = cx_px + sub_cx
    abs_y = cy_px + sub_cy
    vx = float(AIR_VX_BIN_CENTERS[entry_vx_bin_air])
    vy = float(AIR_VY_BIN_CENTERS[entry_vy_bin_air])

    sim.inject_ninja_for_value_iteration(abs_x, abs_y, vx, vy, 0)

    padded_col = HARNESS_CENTER_COL + 1
    padded_row = HARNESS_CENTER_ROW + 1
    half = HARNESS_WINDOW // 2

    # Tick with h=0 until ninja reaches a surface state or exits window.
    contact_frame = 0
    end_pos = (abs_x, abs_y)
    end_vel = (vx, vy)
    end_state = 4  # Falling
    for _ in range(MAX_HARNESS_FRAMES):
        sim.tick_tiles_only(0, 0)
        contact_frame += 1
        sc = sim.get_ninja_state_code()
        pos = sim.get_ninja_pos()
        vel = sim.get_ninja_velocity()
        c, r = _cell_col_row(pos[0], pos[1])
        if abs(c - padded_col) > half or abs(r - padded_row) > half:
            # Ninja exited window without landing - cannot be resolved
            # in this 3x3 alone.  Record an empty entry so the caller can
            # skip, keyed by a surface_type of 0 and vel_bin 0.
            entry = LandingEntry(
                landing_cell_dx=0, landing_cell_dy=0,
                landing_subcell_x=0, landing_subcell_y=0,
                landing_surface_type=0, landing_vel_bin=0,
                contact_frame_offset=contact_frame,
            )
            store.put_landing(sig_hash, key_extra, entry)
            return entry
        if sc in GROUND_STATES | WALL_STATES:
            end_pos = pos
            end_vel = vel
            end_state = sc
            break

    cell_col, cell_row = _cell_col_row(end_pos[0], end_pos[1])
    sub_x = int(round(end_pos[0] - cell_col * TILE_PIXEL_SIZE))
    sub_y = int(round(end_pos[1] - cell_row * TILE_PIXEL_SIZE))
    if end_state == _STATE_WALL_SLIDING:
        land_stype = SURFACE_WALL
        vel_bin = 0 if velocity_to_bin is None else int(
            velocity_to_bin(end_vel[1], SURFACE_WALL)
        )
    else:
        land_stype = SURFACE_FLOOR
        vel_bin = 0 if velocity_to_bin is None else int(
            velocity_to_bin(end_vel[0], SURFACE_FLOOR)
        )

    entry = LandingEntry(
        landing_cell_dx=cell_col - padded_col,
        landing_cell_dy=cell_row - padded_row,
        landing_subcell_x=max(0, min(TILE_PIXEL_SIZE - 1, sub_x)),
        landing_subcell_y=max(0, min(TILE_PIXEL_SIZE - 1, sub_y)),
        landing_surface_type=land_stype,
        landing_vel_bin=vel_bin,
        contact_frame_offset=contact_frame,
    )
    store.put_landing(sig_hash, key_extra, entry)
    return entry


# ---------------------------------------------------------------------------
# Bulk-sweep helpers
# ---------------------------------------------------------------------------

def sweep_launch_signature(
    store: TileCacheStore,
    signature: Sequence[int],
    surface_types: Iterable[int],
    subcells: Iterable[Tuple[int, int]],
    vel_bins: Iterable[Tuple[int, float]],
    *,
    velocity_to_bin=None,
) -> int:
    """Populate all launch entries for one signature.

    Intended for prewarming corpora offline.  Returns the number of
    entries written.

    Args:
        store: Destination tile cache store.
        signature: 9-tile signature.
        surface_types: Iterable of SURFACE_FLOOR / SURFACE_WALL.
        subcells: Iterable of (sx, sy) bin indices.
        vel_bins: Iterable of (bin_index, bin_centre_velocity) tuples.
        velocity_to_bin: Callable for quantising landing velocities.
    """
    from nclone_native import Simulator as NativeSimulator

    map_data = build_launch_harness_map(signature)
    sim = NativeSimulator.load(map_data)

    n_written = 0
    for stype in surface_types:
        for sub in subcells:
            for (vb, vcenter) in vel_bins:
                for cls_id in classes_for_surface(stype):
                    populate_launch_entry(
                        store, signature,
                        surface_type=stype,
                        subcell=sub,
                        vel_bin_in=vb,
                        input_class_id=cls_id,
                        vel_bin_centre=vcenter,
                        velocity_to_bin=velocity_to_bin,
                        sim=sim,
                    )
                    n_written += 1
    return n_written


__all__ = [
    "AIR_VX_BIN_CENTERS",
    "AIR_VY_BIN_CENTERS",
    "H_INPUTS",
    "InputClass",
    "JUMP_HOLD_DURATIONS",
    "LaunchSimResult",
    "MAX_HARNESS_FRAMES",
    "POST_RELEASE_INPUTS",
    "RUN_DURATIONS",
    "SUBCELL_BINS",
    "classes_for_surface",
    "decode_input_class",
    "populate_landing_entry",
    "populate_launch_entry",
    "quantise_air_velocity",
    "quantise_subcell",
    "subcell_centre",
    "sweep_launch_signature",
]
