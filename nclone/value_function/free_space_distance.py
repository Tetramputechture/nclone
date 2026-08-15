"""Free-space grid cost-to-go — a mine-aware potential for AIRBORNE / mine-threading
levels where the surface lattice is severed.

The surface lattice (``LatticeDistanceSource``) places nodes only on walkable
surfaces and masks every node within a mine's kill-radius. On dense mine corridors
where the route is a controlled fall + corner-jumps through empty space, that leaves
the switch/exit with ``inf`` cost-to-go — no reward signal at all (the reward path
then falls back to a mine-blind geometric A*).

This source instead floods a 2D grid over the **tile-free corridor** with a
**mine-soft cost** (prefers the mine-sparse centreline), giving a finite, dense,
corridor-following cost-to-go everywhere the corridor reaches. It is:

* a valid, policy-invariant PBRS potential (any scalar potential is — the RL value
  function handles the non-monotonic "fall away then climb back" structure that
  defeats greedy planning);
* cheap (one Dijkstra per goal over a fixed grid) and N-invariant in the mine count,
  so it scales to procgen;
* explicitly NOT an executable plan — it is the *potential* the RL agent distils
  execution against (cf. docs/mine_demo/, the planner-is-execution-bound finding).

Used as a fallback by ``ReachabilityMixin`` when the surface lattice has no finite
cost-to-go at spawn this episode.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import os
import pickle
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..constants.physics_constants import (
    MAP_TILE_WIDTH, MAP_TILE_HEIGHT, TILE_PIXEL_SIZE,
    NINJA_RADIUS, EXIT_SWITCH_RADIUS, EXIT_DOOR_RADIUS,
)
from ..tile_definitions import (
    TILE_GRID_EDGE_MAP, TILE_EDGE_GRID_TERNARY,
    TILE_SEGMENT_DIAG_MAP, TILE_SEGMENT_CIRCULAR_MAP,
)

# ---------------------------------------------------------------------------
# Canonical per-tile COLLISION SEGMENTS (the same geometry the physics ninja
# collides against). The flood below blocks any cell-to-cell step that crosses
# one of these, so the cost-to-go / paths cannot leak THROUGH a solid segment —
# in particular the thin orthogonal grid-edge "walls" on the boundary/midline of
# slopes, steep slopes and quarter-pipes, which the center-sampled node-free mask
# alone cannot see (both adjacent cell centers are non-solid yet a solid wall
# sits between them). Derived from tile_definitions.py so ALL 38 tile types are
# handled by construction rather than a hand-maintained reimplementation.
# ---------------------------------------------------------------------------

# The 12 grid-edge half-segments in tile-local pixels (0..24), indexed exactly as
# TILE_GRID_EDGE_MAP: first 6 = horizontal half-edges (y in {0,12,24}, left then
# right), last 6 = vertical half-edges (x in {0,12,24}, top then bottom).
_GRID_EDGE_SEGS = (
    (0, 0, 12, 0), (12, 0, 24, 0),      # 0,1  horizontal y=0
    (0, 12, 12, 12), (12, 12, 24, 12),  # 2,3  horizontal y=12
    (0, 24, 12, 24), (12, 24, 24, 24),  # 4,5  horizontal y=24
    (0, 0, 0, 12), (0, 12, 0, 24),      # 6,7  vertical x=0
    (12, 0, 12, 12), (12, 12, 12, 24),  # 8,9  vertical x=12
    (24, 0, 24, 12), (24, 12, 24, 24),  # 10,11 vertical x=24
)
_ARC_CHORDS = 6  # polyline segments approximating each quarter-circle arc
_TILE_LOCAL_SEG_CACHE: dict = {}


def _tile_local_solid_segments(tile_type: int):
    """Solid collision segments for a tile in tile-local px, as (x1,y1,x2,y2).

    Union of the orthogonal grid-edge walls, the diagonal slope segment
    (TILE_SEGMENT_DIAG_MAP) and a chord approximation of the curved segment
    (TILE_SEGMENT_CIRCULAR_MAP). Empty tile (0) and the glitched tiles 34-37 have
    NO collision (matching the simulator, which treats 34-37 as empty).

    The straight walls come from ``TILE_EDGE_GRID_TERNARY`` and are emitted ONLY
    for a full flat face (``2``). ``TILE_GRID_EDGE_MAP`` collapses that table's
    ``1`` (a PARTIAL face -- where a slope or curve meets the tile border) into
    the same ``1`` as a full face; reading the collapsed map here fabricated a
    solid 12px wall across the open part of every sloped and curved tile border.
    For tiles 10-13 the collapsed map is byte-identical to tile 1 (fully solid),
    so quarter-circle tiles became sealed boxes and the flood could not enter
    them at all -- squiggly reached 0.6% of its free space. Where the face is
    partial the real geometry is the diagonal/circular segment, which this
    function already emits, so dropping the ``1`` cases removes the fabrication
    without losing a single genuine wall.
    """
    cached = _TILE_LOCAL_SEG_CACHE.get(tile_type)
    if cached is not None:
        return cached
    segs = []
    if tile_type != 0 and tile_type < 34:
        tern = TILE_EDGE_GRID_TERNARY.get(tile_type)
        grid = TILE_GRID_EDGE_MAP.get(tile_type, ())
        for idx, seg in enumerate(_GRID_EDGE_SEGS):
            if tern is not None:
                # _GRID_EDGE_SEGS is indexed 2*outer + inner within each axis
                # block; the ternary table is indexed outer + 3*inner.
                axis, rem = divmod(idx, 6)
                outer, inner = divmod(rem, 2)
                if tern[axis][outer + 3 * inner] != 2:
                    continue
            elif not (idx < len(grid) and grid[idx]):
                continue
            segs.append(seg)
        diag = TILE_SEGMENT_DIAG_MAP.get(tile_type)
        if diag is not None:
            (x1, y1), (x2, y2) = diag
            segs.append((x1, y1, x2, y2))
        circ = TILE_SEGMENT_CIRCULAR_MAP.get(tile_type)
        if circ is not None:
            (cx, cy), (sx, sy), _convex = circ
            r = TILE_PIXEL_SIZE
            prev = None
            for k in range(_ARC_CHORDS + 1):
                ang = (math.pi / 2.0) * (k / _ARC_CHORDS)
                px = cx + sx * r * math.cos(ang)
                py = cy + sy * r * math.sin(ang)
                if prev is not None:
                    segs.append((prev[0], prev[1], px, py))
                prev = (px, py)
    segs = tuple(segs)
    _TILE_LOCAL_SEG_CACHE[tile_type] = segs
    return segs


def _seg_crosses(ax, ay, bx, by, cx, cy, dx, dy) -> bool:
    """True if open segment A->B strictly crosses segment C->D.

    Strict sign test: a step that merely TOUCHES a wall endpoint (or runs
    collinear along it) is NOT treated as blocked — cell centers sit at odd
    multiples of 3px so they never lie on the 12px-aligned ortho walls, keeping
    this robust for the common case while staying optimistic on grazing contact.
    """
    def _o(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    d1 = _o(cx, cy, dx, dy, ax, ay)
    d2 = _o(cx, cy, dx, dy, bx, by)
    d3 = _o(ax, ay, bx, by, cx, cy)
    d4 = _o(ax, ay, bx, by, dx, dy)
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)

# Sub-tile grid (TILE_PIXEL_SIZE/4 = 6px): fine enough to follow a 1-tile corridor.
_CELL = TILE_PIXEL_SIZE / 4.0
# Cover the full *bordered* playable area: the sim places raw tile (c,r) at grid
# (c+1,r+1) (see map_loader), so playable pixels span [0, (MAP_TILE_WIDTH+1)*24].
_W = int((MAP_TILE_WIDTH + 1) * TILE_PIXEL_SIZE / _CELL) + 2
_H = int((MAP_TILE_HEIGHT + 1) * TILE_PIXEL_SIZE / _CELL) + 2

# HORIZONTAL_AUGMENTATION_PLAN §2. Process-wide toggle (a run has one setting,
# and threading it through a disk/mem-cached constructor would buy nothing).
#
# When True this module becomes exactly equivariant under a left-right
# reflection of the level, via two changes:
#
#   1. ``_field_at`` interpolates against cell CENTRES instead of cell left/top
#      edges. A registration FIX in its own right -- ``_dijkstra`` seeds cells by
#      their centres, so the legacy read-out is offset by half a cell (3 px).
#   2. The whole field is BUILT on the level's canonical orientation and
#      reflected back (see ``_build_cached``). Dijkstra's heap breaks ties on
#      ``(cost, x, y)``, so the two orientations relax cells in a different order
#      and their float sums differ in the last bits; the nav hint's ``argmin``
#      over 8 near-tied hops turns that into a different direction. Building once
#      and reflecting the answer removes the whole class.
#
# Default False: the field is a PBRS potential, so changing it changes shaped
# rewards, and a run resumed from a checkpoint trained without it would see a
# different potential. Opt in for fresh trains via
# ``ReachabilityConfig.mirror_equivariant_obs``.
_REFLECTION_EQUIVARIANT = False

#: Reflection axis in pixels -- the full bordered level width.
_AXIS_PX = float((MAP_TILE_WIDTH + 2) * TILE_PIXEL_SIZE)


def set_reflection_equivariant(enabled: bool) -> None:
    """Enable/disable reflection-equivariant field construction process-wide."""
    global _REFLECTION_EQUIVARIANT
    _REFLECTION_EQUIVARIANT = bool(enabled)


def reflection_equivariant() -> bool:
    """Current reflection-equivariance setting."""
    return _REFLECTION_EQUIVARIANT


def _reflect_point(p):
    """Reflect a ``(x, y)`` pixel pair about the level's vertical axis."""
    if p is None:
        return None
    return (_AXIS_PX - float(p[0]), float(p[1]))


def _mirror_grid_x(arr: np.ndarray, fill) -> np.ndarray:
    """Reflect a ``(_W, ...)`` sub-cell grid about the level's vertical axis.

    Not a plain ``np.flip``: the grid is ``_W = 174`` cells wide (px ``[0,
    1044)``) while the level is 1056 px, so cell ``i`` (centre ``(i+0.5)*_CELL``)
    reflects to index ``175 - i`` -- two columns past the end. Pad to the full
    176-cell span, flip, and truncate. The padded columns cover px ``[1044,
    1056)``, which is inside the solid right border, so ``fill`` is the genuine
    value there (not free / zero density / infinite cost) rather than a
    convenience.
    """
    full = int(round(_AXIS_PX / _CELL))  # 176
    padded = np.full((full,) + arr.shape[1:], fill, dtype=arr.dtype)
    padded[: arr.shape[0]] = arr
    return np.ascontiguousarray(padded[::-1][: arr.shape[0]])


_MINE_KILL = 14.0          # NINJA_RADIUS + TOGGLE_MINE_RADIUS_TOGGLED
_MINE_STATE_DEADLY = 0
_MINE_SOFT_K = 8.0         # mine-density penalty on grid edges


# ---------------------------------------------------------------------------
# PHYSICS COST MODEL (opt-in; see ``physics_cost`` on the constructor).
#
# The legacy flood charges ``1.0`` per orthogonal step and ``1.41`` per diagonal
# — a pure PIXEL-GEOMETRY metric. Up costs the same as down, gravity is not free,
# a 5-tile gap costs the same as a walkway, and an unclimbable wall beside a
# ledge is one 1.41 step. That makes the potential rank states by PROXIMITY when
# what the agent needs is REACHABILITY, and it is why the field happily points
# straight up through open air at an exit the ninja cannot possibly reach.
#
# The physics model changes only the EDGE COST and adds a climb-admissibility
# rule. The state stays 2D (no velocity/height dimension), so the flood is the
# same ~17k-cell Dijkstra at the same cost — the anisotropy is carried by
# direction-dependent weights plus two static O(W*H) fields (``_floor_rise``,
# ``_wall_adj``). Costs are in FRAMES, which is also the unit the surface
# lattice's d* reports, so the two are directly comparable.
#
# Everything below is derived from the sim's own constants rather than tuned:
# integrating the jump/fall equations that ``ninja.py`` runs.
# ---------------------------------------------------------------------------


def _integrate_rise(v0: float) -> Tuple[float, int]:
    """Apex height (px) and frames-to-apex for a jump launched at ``v0`` (up = -).

    Mirrors the sim's integration order: jump gravity applies while the jump
    input is held and the ninja is still rising (capped at MAX_JUMP_DURATION),
    fall gravity after. Returns ``(rise_px, frames)``.
    """
    from ..constants.physics_constants import (
        GRAVITY_JUMP, GRAVITY_FALL, MAX_JUMP_DURATION,
    )
    v, y, f = float(v0), 0.0, 0
    for t in range(512):
        v += GRAVITY_JUMP if (t < MAX_JUMP_DURATION and v < 0) else GRAVITY_FALL
        y += v
        f += 1
        if v >= 0:
            break
    return -y, f


def _physics_cost_constants():
    """Per-cell frame costs + the climb/fall limits, derived from sim constants."""
    from ..constants.physics_constants import (
        MAX_HOR_SPEED, GRAVITY_FALL, MAX_SURVIVABLE_IMPACT,
        JUMP_FLOOR_Y, JUMP_WALL_REGULAR_Y,
    )
    # Floor jump: 94.6 px in 68 frames -> 15.77 cells, 4.31 frames/cell.
    jump_rise_px, jump_frames = _integrate_rise(JUMP_FLOOR_Y)
    jump_rise_cells = jump_rise_px / _CELL
    # Wall jump: 57.1 px in 59 frames. A CHAIN also has to kick away from the
    # wall and drift back into contact before the next launch, so the sustained
    # climb rate is worse than the single-jump rate; ``_WALL_RECONTACT_MULT``
    # prices that (calibrated against the surface lattice's wall-chain edges,
    # see [[wall-jump-chaining-recontact]]).
    wall_rise_px, wall_frames = _integrate_rise(JUMP_WALL_REGULAR_Y)
    # Lethal drop onto a FLAT floor — the least forgiving surface. ninja.py kills
    # when ``|n_y| * v > MAX_SURVIVABLE_IMPACT - (4/3) * |n_y|``; at n_y = 1 that
    # is v > 4.667, i.e. a free fall of v^2 / (2 * GRAVITY_FALL) = 163.3 px.
    # (Same formula as gym_environment/fall_lethality.py, recomputed here rather
    # than imported: value_function must not depend on gym_environment.)
    lethal_v = MAX_SURVIVABLE_IMPACT - (4.0 / 3.0)
    lethal_drop_px = lethal_v * lethal_v / (2.0 * GRAVITY_FALL)
    return {
        "run": _CELL / MAX_HOR_SPEED,                       # 1.80 f/cell
        "air": (_CELL / MAX_HOR_SPEED) * _AIR_DRIFT_MULT,   # 2.16 f/cell
        "fall": _CELL / _MEAN_FALL_SPEED,                   # 1.50 f/cell
        "jump": jump_frames / jump_rise_cells,              # 4.31 f/cell
        "wall": (wall_frames / (wall_rise_px / _CELL)) * _WALL_RECONTACT_MULT,
        "jump_reach_cells": jump_rise_cells,                # 15.77
        "lethal_drop_cells": lethal_drop_px / _CELL,        # 27.2
    }


#: Horizontal travel through open air costs a little more than along a surface.
#: Not a speed difference (hspeed carries) — it biases the field toward routes
#: that stay on geometry, which is where execution is reliable.
_AIR_DRIFT_MULT = 1.2
#: Extra cost on horizontal travel through UNSUPPORTED air, scaled by how many
#: jump-heights above the floor the cell is (capped by ``_AIR_UNSUPPORTED_CAP``).
#:
#: Without this the field has a real leak: nothing stops the flood wall-climbing
#: to the ceiling of an open arena and then drifting horizontally across it
#: forever at ~air cost. The ninja cannot — sustained horizontal travel needs a
#: launch, a launch needs ground or wall contact, and mid-arena there is neither;
#: any real traverse at height is a DESCENDING arc. Pricing the move by altitude
#: is the stateless stand-in for that (the exact version wants a fall-time
#: dimension in the flood state, which is what makes the surface lattice
#: expensive). Deliberately a tunable, not a derived constant — it is the one
#: number in this model that physics does not hand us.
#:
#: 2026-08-03: RETUNED 1.0 -> 0.25 on evidence, and the story is a warning about
#: the oracle. Sweeping it against the lattice's d* moved rho by 0.001, so
#: ``tools/calibrate_physics_cost.py`` reported it as a free parameter and it was
#: left at 1.0. Sweeping it against 40 VERIFIED DEMOS
#: (``tools/demo_potential_eval.py``) shows it monotonically wrecking every
#: metric, because a demo spends its jumps exactly in the unsupported air this
#: penalty taxes — so the potential fights the expert mid-arc:
#:
#:     K      rho    progress   |F| std        (geometric: 0.995 / 0.955 / 0.00397)
#:     0.0    0.992    0.962     0.00412
#:     0.25   0.991    0.963     0.00421   <- best per-step agreement of ANY field
#:     0.5    0.986    0.938     0.00483
#:     1.0    0.972    0.906     0.00647   <- previous default
#:     2.0    0.950    0.878     0.00954
#:
#: 0.25 rather than 0.0 because the leak described above is real; the demo corpus
#: is platforming levels and does not contain the open arena that K exists to
#: guard, so zeroing it would be optimising against a blind spot in the test set.
#: This constant only affects the opt-in physics field (NCLONE_FS_PHYSICS_COST).
_AIR_UNSUPPORTED_K = 0.25
_AIR_UNSUPPORTED_CAP = 4.0
#: Representative mid-fall speed (px/frame). A fall accelerates from 0 toward
#: TERMINAL_VELOCITY = 12; over the drop heights that actually occur (a few
#: tiles) the ninja is doing ~3.5-4.7 px/frame, so 4.0 is the honest average.
_MEAN_FALL_SPEED = 4.0
#: Sustained wall-chain climb is slower than a single wall jump's rise rate.
_WALL_RECONTACT_MULT = 1.6

#: Long-range wall-launch ARC edges, on top of the 8-connected step model.
#:
#: The step model can only price moves between ADJACENT cells, so every cell an
#: arc passes through has to be individually legal. Mid-flight cells are neither
#: wall-adjacent nor within a floor jump's reach, so the up-gate refuses them and
#: a genuine wall jump becomes unrepresentable. On ``long-jump-reachable`` that
#: severed the level exactly: the only way out of the lower chamber is a wall
#: jump up through a gap, every cell of the gap has ``floor_rise = 48`` (the
#: chamber below) and ``wall_adj = 0``, so half the level was unreachable.
#:
#: The fix is an edge class that skips the intervening air: from each
#: wall-adjacent cell, launch the sim's own wall-jump impulse, integrate with
#: ``analytical_physics.compute_jump_arc_relative`` (the same routine the ghost
#: arc obs uses, so the trajectory cannot drift from the simulator), and add one
#: edge to wherever the arc terminates, costed at its flight time in frames.
_ARC_EDGES = True
#: (hold_frames, h_after) per launch. ``h_during`` is always the wall normal --
#: the ninja is pushing away from the wall it just left. Full hold is the
#: reaching jump; the short hold is the controlled hop that clears a lip without
#: overshooting. ``h_after = 0`` lets the arc coast; ``= normal`` keeps driving.
_ARC_VARIANTS = ((45, 0), (45, 1), (12, 0))
#: Samples per frame when walking an arc across the 6px grid. A fall reaches
#: 12 px/frame at terminal velocity, i.e. two cells, so 1 sample/frame tunnels
#: straight through thin floors.
_ARC_SUBSAMPLES = 3
#: Hard cap on arc length. A regular wall jump apexes around frame 59; anything
#: still airborne past this is a fall the step model already handles.
_ARC_MAX_FRAMES = 150
#: (normal, hold, h_after) -> (offsets_px, frame_index). Level-independent.
_ARC_SHAPE_CACHE: dict = {}
#: Cost added ONCE when stepping off a ledge into a drop that would kill on a
#: flat floor. Finite, not a block: the flat-floor threshold is the CONSERVATIVE
#: one (a 45 deg slope survives 383 px, not 163), so an infinite cost would
#: sever fields on levels whose landing happens to be sloped.
#:
#: DEFAULT 0 (OFF), and a cost-to-go field is probably the wrong home for this
#: signal at all. Two findings, in order:
#:
#: 1. As written it is inert. The trigger needs a cell that is ON the ground
#:    (``rise <= 1``) adjacent to a column whose floor is 27+ cells below; across
#:    the test-map corpus that is ~10 edges per level, all on the outer boundary,
#:    none on a routed path. Sweeping 0 -> 300 moved Spearman rho against the
#:    lattice by 0.000 on all 8 levels tried.
#:
#: 2. Fixing the trigger does not fix the idea. A potential is "frames to the
#:    goal"; death is terminal, not expensive, and a STATIC per-cell field cannot
#:    tell "falls into this pit" from "arcs safely over it". Any penalty on the
#:    air above a pit therefore taxes every gap-crossing jump too — the exact
#:    manoeuvre PLATFORM_STAIRS needs the agent to attempt. That is a false
#:    positive the flood has no state to disambiguate.
#:
#: The signal is already delivered where it belongs: gym_environment/
#: fall_lethality.py computes a per-cell survivability margin at this same 6px
#: resolution (correctly accounting for the LANDING SURFACE NORMAL — a 45 deg
#: slope survives 383 px where a flat floor kills at 163), and the env exposes it
#: to the POLICY as ``fall_lethality_map`` plus a ninja-centred crop. The policy
#: can condition on it; the potential should not double-count it.
#:
#: Left wired (cost: one predictable branch) so the claim stays A/B-able.
_LETHAL_FALL_PENALTY = 0.0

# ---------------------------------------------------------------------------
# Build cache. Building the source is ~2 Dijkstras over the 6px grid (~0.1s).
# Under multi-level curriculum training every reset draws a NEW level, so this
# rebuilds on (almost) every reset across hundreds of workers — it was the
# dominant SubprocVecEnv per-reset cost (the synchronous vec-step blocks on any
# worker mid-rebuild). We cache the built state two tiers deep: a per-process
# in-memory LRU (cheap re-visits within a worker) backed by a shared on-disk
# cache (each level built ONCE per node, then read in ~ms by every worker and
# across restarts). The built grids are read-only after construction, so sharing
# array references across instances on a mem-cache hit is safe.
# Bump _FS_CACHE_VERSION if _build's output semantics change.
_FS_CACHE_VERSION = "fs10"  # bumped: locked_doors, _threat_mul and the directional edge masks joined the cached state
_FS_STATE_ATTRS = (
    "available", "switch", "exit", "_d_switch", "_d_exit",
    "_d_max_switch", "_d_max_exit", "_deadly", "exit_cost_at_switch",
    "_free", "_mden", "_block_r", "_block_u", "_block_dr", "_block_ur",
    # Physics-cost mode only (None in geometric mode); see _build_climb_fields.
    "_floor_rise", "_wall_adj",
    # Wall-launch arc edges, CSR keyed by landing cell; see _build_arc_edges.
    "_arc_rev_start", "_arc_rev_src", "_arc_rev_cost",
    # Locked-door inventory (see the ``doors`` build mode).  Cached because a
    # caller enumerating door configurations needs it on a cache HIT too, where
    # ``_build`` never runs.
    "locked_doors",
    # Time-marginalized moving-hazard multiplier (see ``threat_occupancy``).
    # Cached with the fields it produced: rebuilding it on a cache hit would
    # cost a 600-frame rollout for a value nothing reads after the flood.
    "_threat_mul",
    # False when the exit flood cannot reach the switch, i.e. the phase-1 tail
    # is the shrunken-component fallback rather than a real cost.
    "exit_reachable_from_switch",
    # Directional twins of the four blocked-edge masks; None unless the level
    # has one-way platforms and they are enabled.  See ``_step_blocked``.
    "_block_r_neg", "_block_u_neg", "_block_dr_neg", "_block_ur_neg",
    "one_ways",
)
_FS_MEM_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_FS_MEM_CAP = int(os.environ.get("NCLONE_FREESPACE_MEM_CAP", "256"))  # v14 worker-RSS audit


_SUBNODE_LUT: Optional[np.ndarray] = None


def _subnode_validity_lut(table) -> np.ndarray:
    """``SUBNODE_VALIDITY_TABLE`` as a dense ``(n_tile_types, 4, 4)`` bool array.

    The dict-of-dicts costs a two-level lookup per cell, which is the whole
    reason the free-mask build used to be a 17k-iteration Python loop. Built
    once per process from the same dict, so it cannot drift from it. Tile types
    absent from the table read False everywhere, matching ``.get(t, {})``.
    """
    global _SUBNODE_LUT
    if _SUBNODE_LUT is None:
        n = max(int(t) for t in table) + 1 if table else 1
        lut = np.zeros((n, 4, 4), bool)
        for t, sub in table.items():
            for (sx, sy), ok in sub.items():
                lut[int(t), int(sx), int(sy)] = bool(ok)
        _SUBNODE_LUT = lut
    return _SUBNODE_LUT


def _fs_cache_dir() -> Optional[str]:
    """Shared on-disk cache dir, or None to disable the disk tier.

    Reuses the training graph-cache dir (a ``freespace`` subdir) when set so the
    one env var enables both; otherwise falls back to a node-local /tmp dir.
    """
    d = os.environ.get("NCLONE_FREESPACE_CACHE_DIR")
    if not d:
        graph = os.environ.get("NCLONE_GRAPH_CACHE_DIR")
        d = os.path.join(graph, "freespace") if graph else "/tmp/nclone_freespace_cache"
    try:
        os.makedirs(d, exist_ok=True)
        return d
    except OSError:
        return None


_NATIVE_FINGERPRINT: Optional[str] = None


def _native_fingerprint() -> str:
    """Identity of the compiled extension, folded into the disk-cache key.

    The blocked-edge masks come from EITHER the Python body here or the Rust port
    in ``nclone-native/src/free_space.rs``, and a geometry change has to land in
    both. Between the two landings the halves disagree while the cache version
    does not, so a build can write a poisoned entry that a later, correct build
    reads straight back — which is exactly how a fixed quarter-circle field kept
    coming back sealed. Keying on the extension's identity turns a native rebuild
    into a cache miss, which is the safe direction, and costs one ``stat``.

    Also protects the deploy path: rsyncing Python without rebuilding the native
    extension would otherwise poison a cache dir shared across the fleet.
    """
    global _NATIVE_FINGERPRINT
    if _NATIVE_FINGERPRINT is None:
        _NATIVE_FINGERPRINT = "none"
        try:
            import nclone_native

            # The package __init__ is a 135-byte re-export stub that never
            # changes on rebuild; the compiled submodule is what must be
            # fingerprinted.
            ext = getattr(nclone_native, "nclone_native", nclone_native)
            path = getattr(ext, "__file__", None)
            if path:
                st = os.stat(path)
                _NATIVE_FINGERPRINT = f"{st.st_size}:{st.st_mtime_ns}"
        except Exception:
            pass
    return _NATIVE_FINGERPRINT


#: Half-length of a door's collision segment, in px (``EntityDoorBase.DOOR_RADIUS``).
_DOOR_RADIUS = 12.0

#: Door orientations that make the 24px barrier VERTICAL; every other value is
#: horizontal.  Mirrors ``EntityDoorBase.is_vertical``.
_DOOR_VERTICAL_ORIENTS = (0, 4)

#: Half-side of a one-way platform's face, in px
#: (``ONE_WAY_PLATFORM_SEMI_SIDE``); the face is 24px like a door.
_ONE_WAY_SEMI = 12.0

#: Outward normal per orientation, for the four axis-aligned values that are the
#: only ones the official corpus uses.  Stated here rather than taken from
#: ``physics.map_orientation_to_vector`` so this module does not acquire a
#: dependency on the entity layer for four constants.
_ORIENT_NORMAL = {0: (1, 0), 2: (0, 1), 4: (-1, 0), 6: (0, -1)}


def _fs_cache_key(map_data: List[int], mine_hard: bool = False,
                  physics_cost: bool = False,
                  doors: str = "open",
                  threat_occupancy: bool = False,
                  one_way: bool = False) -> str:
    h = hashlib.md5()
    h.update(_FS_CACHE_VERSION.encode())
    h.update(_native_fingerprint().encode())
    # Grid/build params are part of the key so a param change can't return stale.
    # ``mine_hard`` selects the mine-FAITHFUL variant (deadly-mine cells cut from
    # the free mask, so the field routes AROUND mines instead of soft-costing
    # through them); it is a distinct field, so it must key distinctly.
    h.update(repr((_CELL, _W, _H, _MINE_KILL, _MINE_SOFT_K, bool(mine_hard))).encode())
    # ``physics_cost`` swaps the edge-cost model (frames, anisotropic) for the
    # geometric one (grid units, isotropic). Different units AND different
    # ordering, so it must key distinctly; the tuning constants key too, so a
    # calibration sweep can never read a stale field.
    if physics_cost:
        h.update(repr((
            "phys1", _AIR_DRIFT_MULT, _MEAN_FALL_SPEED,
            _WALL_RECONTACT_MULT, _LETHAL_FALL_PENALTY,
            _AIR_UNSUPPORTED_K, _AIR_UNSUPPORTED_CAP,
            # Wall-launch arcs change the edge SET, not just weights.
            bool(_ARC_EDGES), _ARC_VARIANTS, _ARC_SUBSAMPLES, _ARC_MAX_FRAMES,
        )).encode())
    # ``doors`` changes the EDGE SET (a closed locked door is a wall), so a
    # doors-closed field must never be served from a doors-open cache entry.
    # Sorted, never ``repr``'d as a set: set iteration order is not part of
    # Python's contract, and a key that varies run to run silently disables the
    # cache instead of failing.
    if doors != "open":
        tag = ("set", sorted(doors)) if not isinstance(doors, str) else doors
        h.update(repr(("doors", tag)).encode())
    # The threat multiplier reweights every edge, and its strength is a tuning
    # constant, so both the flag and the constant have to key.
    if threat_occupancy:
        from .threat_occupancy import THREAT_SOFT_K, DEFAULT_WINDOW_FRAMES
        h.update(repr(("threat", THREAT_SOFT_K, DEFAULT_WINDOW_FRAMES)).encode())
    # One-way platforms make edges DIRECTIONAL, which changes the edge set and
    # adds four arrays to the cached state.
    if one_way:
        h.update(repr(("oneway", _ONE_WAY_SEMI)).encode())
    h.update(np.asarray([int(v) for v in map_data], dtype=np.int64).tobytes())
    return h.hexdigest()


class FreeSpaceDistanceSource:
    """Per-level free-space grid cost-to-go to the switch and exit.

    Best-effort: any failure leaves ``available == False`` so the caller falls back.
    """

    #: Accepted values for the ``doors`` build mode.  See ``__init__``.
    DOOR_MODES = ("open", "closed")

    # Class-level defaults for the DIRECTIONAL edge masks.  Tests (and the
    # blocked-edge helpers) build instances with ``__new__`` to exercise the
    # dispatch against hand-made masks without paying for a real build; those
    # never run ``__init__``, so ``_step_blocked`` would raise AttributeError on
    # its first lookup.  ``None`` also happens to be the correct value — it means
    # "no one-ways, every edge is symmetric", which is the safe reading.
    _block_r_neg: Optional[np.ndarray] = None
    _block_u_neg: Optional[np.ndarray] = None
    _block_dr_neg: Optional[np.ndarray] = None
    _block_ur_neg: Optional[np.ndarray] = None

    # Class-level defaults for the BUILD-MODE flags, for the same reason: an
    # instance made with ``__new__`` never runs ``__init__``, and ``_build``
    # reads every one of these.  Which one raises first depends on the MAP --
    # ``_one_way`` is only reached by a level that carries a type-11 platform,
    # so a hand-made test fixture would fail at ``_mine_hard`` instead, and a
    # different fixture at a different attribute.  Declaring the whole set makes
    # ``__new__`` construction safe regardless of map content.
    #
    # Each value is the documented ``__init__`` default.  For the three flags
    # ``__init__`` resolves from an env var (``_physics_cost``,
    # ``_threat_occupancy``, ``_one_way``), ``False`` is the correct reading:
    # a ``__new__`` instance was never configured, and every one of them is
    # opt-in.
    _mine_hard: bool = False
    _physics_cost: bool = False
    _threat_occupancy: bool = False
    _one_way: bool = False
    _doors = "open"

    # The remaining build state, for the same reason.  ``_threat_mul`` is the one
    # that actually bites a ``__new__`` instance: it is assigned only under
    # ``_threat_occupancy`` but read unconditionally when the flood picks between
    # the native and Python kernels.  The rest are the "must always exist" set
    # that ``__init__`` documents -- ``_build`` leaves them untouched when it
    # returns early (no switch / no exit) or when it is not in physics mode.
    #
    # All immutable, so sharing them on the class is safe.  The mutable
    # containers (``one_ways``, ``locked_doors``, ``_deadly``) are deliberately
    # NOT here -- see ``_build``, which now owns them.
    _threat_mul: Optional[np.ndarray] = None
    _floor_rise: Optional[np.ndarray] = None
    _wall_adj: Optional[np.ndarray] = None
    _d_switch: Optional[np.ndarray] = None
    _d_exit: Optional[np.ndarray] = None
    _d_max_switch = 1.0
    _d_max_exit = 1.0
    exit_cost_at_switch = 0.0
    switch: Optional[Tuple[float, float]] = None
    exit: Optional[Tuple[float, float]] = None
    available = False

    def __init__(self, map_data: List[int], *, name: Optional[str] = None,
                 mine_hard: bool = False, physics_cost: Optional[bool] = None,
                 doors: str = "open",
                 threat_occupancy: Optional[bool] = None,
                 one_way: Optional[bool] = None):
        self.available = False
        # Mine-FAITHFUL mode (v24, for the POLICY observation): cut deadly-mine
        # cells from the free mask so the cost-to-go / nav gradient the policy
        # sees routes AROUND mines. Default False keeps the legacy mine-SOFT
        # field (mines traversable at ~9x cost) that every existing caller uses.
        self._mine_hard = bool(mine_hard)
        # PHYSICS cost model (frames, anisotropic) instead of the legacy
        # geometric one (grid units, isotropic). Opt-in per instance; the env var
        # flips the process-wide default so a training run can A/B it without
        # threading a kwarg through every construction site.
        if physics_cost is None:
            physics_cost = os.environ.get("NCLONE_FS_PHYSICS_COST", "0") == "1"
        self._physics_cost = bool(physics_cost)
        # LOCKED-door awareness.  ``"open"`` is the historical behaviour and
        # stays the default: every door is ignored, so the flood can route
        # straight through a barrier that is shut.
        #
        # Only type 6 belongs here, and that is a deliberate reading of this
        # project's own asymmetry (lethality -> conservative, traversability ->
        # optimistic), not an oversight:
        #
        #   * type 5 REGULAR opens on 10px proximity to its own switch and
        #     re-closes 5 frames later.  It yields to any ninja that walks up to
        #     it, so for ROUTING it is permanently open.
        #   * type 8 TRAP starts OPEN and shuts only if the ninja touches its
        #     switch — an avoidable, self-inflicted barrier.  Optimistically
        #     open.
        #   * type 6 LOCKED starts SHUT and opens only once its switch is
        #     touched.  This is the one genuine barrier, and it is
        #     PHASE-DEPENDENT, which is why it is a mode rather than an
        #     unconditional cut.
        #
        # ``"closed"`` treats every locked door as shut — the phase-0 topology.
        # Measured over all 808 official levels that carry a locked door
        # (``tools/door_audit.py``):
        #
        #   * 648 have at least one PHANTOM cell — free space the doors-open
        #     flood prices finitely but which is reachable only by walking
        #     through a shut door.  Mean 23.3% of free cells, p90 98.3%.
        #   * 355 have MISPRICED cells; where mispriced, the median cost is
        #     5.6% too cheap and the p90 is 58% too cheap.
        #   * 205 (25.4%) are GATED at phase 0: with every door shut the exit
        #     switch is unreachable from the spawn at all.
        #
        # That last number is why ``"closed"`` is not simply the right default.
        # On those 205 levels it would replace an over-promising gradient with
        # NO gradient, and a potential that misses the route is far worse than
        # one that slightly over-promises.  On the other 603 it is free and
        # strictly better.
        #
        # ``doors`` may therefore also be a frozenset of indices into
        # ``locked_doors`` naming the doors that are OPEN — the per-leg topology
        # ``door_dependency.py`` already computes for the surface lattice.  An
        # empty set is exactly ``"closed"``.
        if isinstance(doors, (set, frozenset)):
            doors = frozenset(int(i) for i in doors)
        elif doors not in self.DOOR_MODES:
            raise ValueError(
                f"doors must be one of {self.DOOR_MODES} or a frozenset of open-door "
                f"indices, got {doors!r}")
        self._doors = doors
        #: ``[(door_x, door_y, orientation, switch_x, switch_y), ...]`` for every
        #: type-6 door on the level, in simulator order.  Populated by ``_build``
        #: regardless of mode so callers can enumerate door configurations.
        self.locked_doors: List[Tuple[float, float, int, float, float]] = []
        # Time-marginalized moving-hazard cost; see ``threat_occupancy``.  Only
        # meaningful in physics mode (the geometric model has no frame scale to
        # inflate), and opt-in because it reweights every edge in the flood.
        from .threat_occupancy import threat_occupancy_enabled
        self._threat_occupancy = bool(
            threat_occupancy_enabled() if threat_occupancy is None
            else threat_occupancy)
        self._threat_mul: Optional[np.ndarray] = None
        # ONE-WAY platforms.  Opt-in for now because it is the first thing in
        # this module that makes an edge directional, and a directional edge
        # changes what a BACKWARD flood means for every consumer at once.
        if one_way is None:
            one_way = os.environ.get("NCLONE_FS_ONE_WAY", "0") == "1"
        self._one_way = bool(one_way)
        #: ``[(x, y, normal_x, normal_y), ...]`` for every type-11 platform.
        self.one_ways: List[Tuple[float, float, int, int]] = []
        # Directional twins of the blocked-edge masks.  They must exist as
        # attributes on EVERY instance: ``_step_blocked`` reads them on the hot
        # path and a level with no one-ways never builds them.
        self._block_r_neg: Optional[np.ndarray] = None
        self._block_u_neg: Optional[np.ndarray] = None
        self._block_dr_neg: Optional[np.ndarray] = None
        self._block_ur_neg: Optional[np.ndarray] = None
        # Static climb/fall geometry; populated by _build_climb_fields in physics
        # mode only, and left None otherwise (they are part of the cached state,
        # so they must always exist as attributes).
        self._floor_rise: Optional[np.ndarray] = None
        self._wall_adj: Optional[np.ndarray] = None
        self.switch: Optional[Tuple[float, float]] = None
        self.exit: Optional[Tuple[float, float]] = None
        self._d_switch: Optional[np.ndarray] = None
        self._d_exit: Optional[np.ndarray] = None
        self._d_max_switch = 1.0
        self._d_max_exit = 1.0
        self._deadly: List[Tuple[float, float]] = []  # deadly mine centers (viz)
        # Exit cost-to-go evaluated AT the switch: the constant phase-1 tail added
        # to the phase-0 (pre-switch) field so ``cost_to_completion`` is a single
        # continuous potential across switch activation (no spike at the switch).
        self.exit_cost_at_switch = 0.0
        try:
            self._build_cached(list(map_data))
        except Exception:
            self.available = False

    # ------------------------------------------------------------------
    def _restore_state(self, state: dict) -> None:
        # ``.get`` rather than ``[]``: cache files written before the physics
        # fields existed lack those keys. They can only be read back by a
        # GEOMETRIC instance (physics_cost keys the cache separately), and a
        # geometric instance never reads them, so None is the correct value.
        for k in _FS_STATE_ATTRS:
            setattr(self, k, state.get(k))

    def _capture_state(self) -> dict:
        return {k: getattr(self, k) for k in _FS_STATE_ATTRS}

    def _build_cached(self, map_data: List[int]) -> None:
        """Build the cost-to-go grids, hitting the mem/disk caches first.

        Lookup order: per-process LRU -> shared on-disk cache -> compute. A hit
        avoids the ~0.1s Dijkstra rebuild entirely (the per-reset bottleneck under
        multi-level training). Cache failures degrade silently to a live build.

        With :func:`set_reflection_equivariant` on, the build runs on the level's
        CANONICAL orientation and the resulting fields are reflected back. That
        is what makes the field exactly equivariant: Dijkstra's heap breaks ties
        on ``(cost, x, y)``, so a level and its mirror relax cells in a different
        ORDER and their float sums differ in the last bits -- enough for the nav
        hint's ``argmin`` over 8 near-tied hops to pick a different direction. No
        tie-break rule fixes that; running the search once and reflecting the
        answer does. It also makes a level and its mirror SHARE a cache entry.
        """
        mirrored = False
        # Door indices name positions in the list ``_build`` reads off the map
        # bytes it is HANDED.  Canonicalising the map first would renumber them,
        # so a caller's ``{0}`` would silently open a different door.  Levels
        # with a door topology therefore opt out of mirror-sharing; nothing else
        # changes, and the equivariance guarantee still holds for every level
        # that reaches this on the default path.
        if _REFLECTION_EQUIVARIANT and self._doors == "open":
            try:
                from ..map_augmentation.transforms import canonical_map_data

                map_data, mirrored = canonical_map_data(map_data)
            except Exception:  # noqa: BLE001 - never fail a build over this
                mirrored = False
        self._built_mirrored = mirrored

        try:
            key = _fs_cache_key(map_data, self._mine_hard, self._physics_cost,
                                self._doors, self._threat_occupancy,
                                self._one_way)
        except Exception:
            self._build(map_data)
            self._reflect_state_if_mirrored()
            return

        cached = _FS_MEM_CACHE.get(key)
        if cached is not None:
            _FS_MEM_CACHE.move_to_end(key)
            self._restore_state(cached)
            self._reflect_state_if_mirrored()
            return

        cache_dir = _fs_cache_dir()
        path = os.path.join(cache_dir, key + ".pkl") if cache_dir else None
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    state = pickle.load(f)
                self._restore_state(state)
                self._mem_put(key, state)
                self._reflect_state_if_mirrored()
                return
            except Exception:
                pass  # corrupt/partial cache file -> rebuild

        self._build(map_data)

        # Capture BEFORE reflecting: the cache holds the canonical state, which
        # is what both orientations of the level look it up by.
        state = self._capture_state()
        self._mem_put(key, state)
        if path:
            try:  # atomic write so concurrent workers never read a partial file
                tmp = f"{path}.tmp.{os.getpid()}"
                with open(tmp, "wb") as f:
                    pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, path)
            except Exception:
                pass
        self._reflect_state_if_mirrored()

    def _reflect_state_if_mirrored(self) -> None:
        """Reflect the canonical-orientation build back into the level's frame.

        Only the OUTPUT state is reflected. The directional blocked-edge masks
        (``_block_r`` / ``_block_u`` / ``_block_dr`` / ``_block_ur``) are
        deliberately left in the canonical frame: they are consumed only by
        ``_dijkstra`` during the build, which has already run, and reflecting a
        directional mask means re-deriving four index-shifted neighbours -- new
        code with a real off-by-one surface, for state nothing reads afterwards.

        Never mutates in place: ``_capture_state`` hands the cache references to
        these very arrays, so an in-place flip would poison the shared cache
        entry with one orientation's fields.
        """
        if not getattr(self, "_built_mirrored", False):
            return
        self.switch = _reflect_point(self.switch)
        self.exit = _reflect_point(self.exit)
        self._deadly = [_reflect_point(p) for p in self._deadly]
        # Positions reflect; the raw orientation code deliberately does not.  The
        # only thing read off it is ``orient in _DOOR_VERTICAL_ORIENTS``, and a
        # reflection about the VERTICAL axis maps vertical doors to vertical and
        # horizontal to horizontal, so that bit is invariant.  Running the code
        # through ``_ORIENT_FLIP`` would be both unnecessary and wrong — that
        # table reflects axis-aligned orientations backwards.
        self.locked_doors = [
            (_AXIS_PX - dx, dy, orient, _AXIS_PX - sx, sy)
            for (dx, dy, orient, sx, sy) in (self.locked_doors or [])
        ]
        self._d_switch = _mirror_grid_x(self._d_switch, np.inf)
        self._d_exit = _mirror_grid_x(self._d_exit, np.inf)
        self._free = _mirror_grid_x(self._free, False)
        self._mden = _mirror_grid_x(self._mden, 0.0)

    @staticmethod
    def _mem_put(key: str, state: dict) -> None:
        _FS_MEM_CACHE[key] = state
        _FS_MEM_CACHE.move_to_end(key)
        while len(_FS_MEM_CACHE) > _FS_MEM_CAP:
            _FS_MEM_CACHE.popitem(last=False)

    # ------------------------------------------------------------------
    def _build(self, map_data: List[int]) -> None:
        import nclone_native as Nnative
        # Some datasets store entity coords as FLOATS (e.g. 144.0); the native sim
        # loader needs ints and floors floats anyway. Coerce so the free-space
        # potential builds on float-coord levels instead of silently failing
        # (otherwise --free-space-pbrs has NO dense signal on those levels).
        map_data = [int(v) for v in map_data]
        # The per-level entity containers belong to the BUILD, not to __init__.
        # Owning them here does two things: it makes _build self-sufficient on an
        # instance made with __new__ (the loop below appends to both), and it
        # makes a REBUILD idempotent -- the appends are unguarded, so a second
        # _build on the same instance used to double every door and one-way,
        # silently doubling the barriers the flood routes around.
        self.one_ways = []
        self.locked_doors = []
        sim = Nnative.Simulator.load(map_data)
        st = sim.get_state()
        switch = exit_door = None
        deadly = []
        for e in st.entities:
            if e.entity_type == 4:
                switch = (float(e.x), float(e.y))
            elif e.entity_type == 3:
                exit_door = (float(e.x), float(e.y))
            elif e.entity_type in (1, 21) and int(getattr(e, "state", 1)) == _MINE_STATE_DEADLY:
                deadly.append((float(e.x), float(e.y)))
            elif e.entity_type == 11 and self._one_way:
                n = _ORIENT_NORMAL.get(int(getattr(e, "orientation", -1)))
                if n is not None:
                    self.one_ways.append((float(e.x), float(e.y), n[0], n[1]))
            elif e.entity_type == 6:
                # Locked door.  ``x, y`` is its SWITCH (the base class overwrites
                # the entity position with the switch position after building the
                # segment); the barrier is at ``door_x, door_y``.
                self.locked_doors.append((
                    float(getattr(e, "door_x", e.x)),
                    float(getattr(e, "door_y", e.y)),
                    int(getattr(e, "door_orientation", 0)),
                    float(e.x), float(e.y),
                ))
        self._deadly = deadly  # kept for visualization / introspection
        if switch is None or exit_door is None:
            return
        self.switch, self.exit = switch, exit_door

        # Tile-free mask at 6px sub-cell resolution, using the SAME canonical
        # tile-geometry table the reachability graph and the surface lattice use
        # (SUBNODE_VALIDITY_TABLE: for each tile type, which of the 4x4 6px sub-cells
        # lie in the non-solid area). This faithfully follows slopes / half-tiles
        # instead of treating any non-empty tile as a fully-solid 24px block.
        from nclone.graph.reachability.graph_builder import SUBNODE_VALIDITY_TABLE

        tiles = map_data[184:1150]
        # Vectorised over the whole grid. Cell centre (ix, iy) sits at
        # ((ix + 0.5) * _CELL, (iy + 0.5) * _CELL); the raw tile index is the
        # sim-grid column minus the +1 border the sim applies (map_loader: tile
        # (c, r) -> grid (c+1, r+1)), and the sub-cell is the 6px quarter within
        # it. Cells outside the raw tile array read False, matching the scalar
        # loop's bounds guard.
        cxs = (np.arange(_W, dtype=np.float64) + 0.5) * _CELL
        cys = (np.arange(_H, dtype=np.float64) + 0.5) * _CELL
        rtx = (cxs // TILE_PIXEL_SIZE).astype(np.int64) - 1
        rty = (cys // TILE_PIXEL_SIZE).astype(np.int64) - 1
        sub_x = np.minimum(((cxs % TILE_PIXEL_SIZE) // _CELL).astype(np.int64), 3)
        sub_y = np.minimum(((cys % TILE_PIXEL_SIZE) // _CELL).astype(np.int64), 3)
        in_x, in_y = (rtx >= 0) & (rtx < 42), (rty >= 0) & (rty < 23)
        tarr = np.asarray(tiles, dtype=np.int64).reshape(23, 42)
        ttype = tarr[np.clip(rty, 0, 22)[None, :], np.clip(rtx, 0, 41)[:, None]]
        lut = _subnode_validity_lut(SUBNODE_VALIDITY_TABLE)
        free = lut[np.clip(ttype, 0, lut.shape[0] - 1),
                   sub_x[:, None], sub_y[None, :]]
        free &= in_x[:, None] & in_y[None, :]

        # Per-cell deadly-mine density: how many deadly kill discs cover the cell
        # centre. Vectorised as a stack of disc tests over the cell-centre grid.
        mden = np.zeros((_W, _H), np.float32)
        if deadly:
            r2 = (_MINE_KILL + _CELL) ** 2
            mxy = np.asarray(deadly, dtype=np.float64)
            dx = cxs[:, None, None] - mxy[None, None, :, 0]
            dy = cys[None, :, None] - mxy[None, None, :, 1]
            mden = ((dx * dx + dy * dy) <= r2).sum(axis=2).astype(np.float32)
        # Mine-FAITHFUL variant (v24 policy obs): CUT every cell whose centre lies
        # within a deadly-mine kill disc from the free mask, so backward Dijkstra
        # routes strictly AROUND mines and the gradient the policy follows never
        # points through a kill zone. (Legacy mine-SOFT mode leaves these cells
        # free at ~9x cost, which is exactly what steered the agent into mines.)
        # On a mine-severed level the goals become unreachable -> available=False,
        # and the obs falls back to the Euclidean field (same as a failed build);
        # the REWARD path still uses the 2px MineThreadingDistanceSource there.
        if self._mine_hard:
            free = free & (mden == 0.0)
        self._free, self._mden = free, mden

        # Segment-aware blocked-edge masks: for every step between two free cells,
        # precompute whether the cell-center-to-cell-center move crosses a SOLID
        # collision segment (canonical tile geometry). This stops the flood from
        # leaking through the thin orthogonal grid-edge walls on slope / steep-slope
        # / quarter-pipe boundaries that the (center-sampled) node-free mask cannot
        # see. block_r/_u = orthogonal +x/+y steps; block_dr/_ur = the two diagonal
        # steps, folded to ALSO block a corner squeeze past any adjacent ortho wall
        # (the ninja can only cut a corner when all four surrounding edges are open).
        self._build_blocked_edges(tiles, free)
        if self._doors != "open" and self.locked_doors:
            self._apply_door_edges(free)
        if self._one_way and self.one_ways:
            # After the doors: the negative masks are copies of the positive
            # ones, so every barrier already applied has to be in place first.
            self._apply_one_way_edges(free)


        # Goal cost-to-go reaches 0 at the ACTIVATION boundary, not the entity
        # center: the ninja triggers the switch / exit by edge contact when its
        # center is within (NINJA_RADIUS + entity_radius) of the entity center
        # (overlap_circle_vs_circle). Seeding the activation disc keeps this
        # potential consistent with the surface lattice and the reachability A*.
        sw_r = NINJA_RADIUS + EXIT_SWITCH_RADIUS
        ex_r = NINJA_RADIUS + EXIT_DOOR_RADIUS
        # Climb fields and arc edges are INPUTS to either flood, so build them up
        # front rather than inside one of the two paths — that is what keeps the
        # native and Python fields identical.
        self._arc_rev_start = self._arc_rev_src = self._arc_rev_cost = None
        if self._physics_cost:
            self._build_climb_fields(free)
            self._build_arc_edges()
            if self._threat_occupancy:
                from .threat_occupancy import marginal_occupancy, step_multiplier
                self._threat_mul = step_multiplier(marginal_occupancy(map_data))
        # The native kernel takes the four UNDIRECTED masks and no threat term,
        # so a level carrying either must relax in Python — otherwise the cost
        # model would silently revert to the hazard-blind, direction-blind one.
        # Levels with neither keep the fast path, which is most of them.
        native = (None if (self._threat_mul is not None
                           or self._block_r_neg is not None)
                  else self._flood_native(switch, exit_door, sw_r, ex_r))
        if native is not None:
            self._d_switch, self._d_exit = native
        else:
            flood = self._dijkstra_physics if self._physics_cost else self._dijkstra
            self._d_switch = flood(switch, sw_r)
            self._d_exit = flood(exit_door, ex_r)
        self._d_max_switch = self._finite_max(self._d_switch)
        self._d_max_exit = self._finite_max(self._d_exit)
        # Available only if both goals reached some free cells.
        self.available = (np.isfinite(self._d_switch).any()
                          and np.isfinite(self._d_exit).any())
        # Phase-1 tail constant: exit cost at the switch position. Robust to the
        # switch being surface-mounted (its own cell solid) via the nearest-free
        # ring search; falls back to the field's finite max on genuine severance.
        #
        # WARNING — that fallback is NOT a safe upper bound once the mask can
        # sever the level, which is exactly what ``doors="closed"`` and
        # ``one_way=True`` make possible. ``_d_max_exit`` is the max over the
        # component the exit can still reach, so severing the switch off from the
        # exit SHRINKS it, and the phase-1 tail gets *cheaper*: the agent is told
        # the level is nearly finished at the moment the exit became unreachable.
        #
        # Measured on `065 striving upwards`: cost_to_completion at the spawn
        # falls 325.5 -> 77.7 when one-ways are enforced, purely through this
        # fallback. It is a pre-existing defect, not one the constraints
        # introduced — `258 mad dash` already takes the fallback with no
        # constraints at all (145.9) — but a hard constraint is what makes it
        # common. ``exit_reachable_from_switch`` exposes it so a caller
        # composing per-leg topologies can refuse a leg whose goal is severed
        # rather than silently descend a bogus tail.
        self.exit_reachable_from_switch = True
        if self.available:
            c = self.cost_at_nearest_free(switch[0], switch[1], True)
            self.exit_reachable_from_switch = bool(np.isfinite(c))
            self.exit_cost_at_switch = (
                float(c) if np.isfinite(c) else float(self._d_max_exit)
            )

    def _build_blocked_edges(self, tiles, free: np.ndarray) -> None:
        """Precompute per-edge blocking from the canonical tile collision segments.

        Builds four (_W, _H) bool masks consulted by ``_dijkstra`` so a flood step
        never crosses a solid wall (orthogonal grid-edge, slope diagonal, or curve
        chord). Only free->free boundaries are tested; solid cells are already
        excluded by the node-free mask.

        Prefers the native implementation (nclone-native/src/free_space.rs),
        which reads the SAME three tile tables the simulator collides against;
        the Python body below is the fallback and the reference.
        """
        if self._blocked_edges_native(tiles, free):
            return
        block_r = np.zeros((_W, _H), bool)   # step (ix,iy) -> (ix+1,iy)
        block_u = np.zeros((_W, _H), bool)   # step (ix,iy) -> (ix,iy+1)
        block_dr = np.zeros((_W, _H), bool)  # step (ix,iy) -> (ix+1,iy+1)
        block_ur = np.zeros((_W, _H), bool)  # step (ix,iy) -> (ix+1,iy-1)

        # Global-pixel wall segments per raw tile (only non-empty tiles carry any).
        walls: dict = {}
        for rty in range(23):
            base = rty * 42
            oy = (rty + 1) * TILE_PIXEL_SIZE
            for rtx in range(42):
                tt = int(tiles[rtx + base])
                local = _tile_local_solid_segments(tt)
                if not local:
                    continue
                ox = (rtx + 1) * TILE_PIXEL_SIZE
                walls[(rtx, rty)] = [
                    (x1 + ox, y1 + oy, x2 + ox, y2 + oy) for (x1, y1, x2, y2) in local
                ]

        if not walls:  # fully open (or empty) level: nothing to block
            self._block_r, self._block_u = block_r, block_u
            self._block_dr, self._block_ur = block_dr, block_ur
            return

        def _crosses(ax, ay, bx, by) -> bool:
            # A short step near a tile corner can pass through up to 4 tiles; the
            # solid segment it crosses may belong to any of them (e.g. a diagonal
            # step clipping a slope/arc corner whose bounding segment is in the
            # diagonal-neighbour tile). Test every tile in the step's bounding box.
            tx0 = int(min(ax, bx) // TILE_PIXEL_SIZE) - 1
            tx1 = int(max(ax, bx) // TILE_PIXEL_SIZE) - 1
            ty0 = int(min(ay, by) // TILE_PIXEL_SIZE) - 1
            ty1 = int(max(ay, by) // TILE_PIXEL_SIZE) - 1
            for tx in range(tx0, tx1 + 1):
                for ty in range(ty0, ty1 + 1):
                    for (cx, cy, dxx, dyy) in walls.get((tx, ty), ()):
                        if _seg_crosses(ax, ay, bx, by, cx, cy, dxx, dyy):
                            return True
            return False

        half = _CELL / 2.0
        # Pass 1: orthogonal boundaries.
        for ix in range(_W):
            ax = ix * _CELL + half
            for iy in range(_H):
                if not free[ix, iy]:
                    continue
                ay = iy * _CELL + half
                if ix + 1 < _W and free[ix + 1, iy] and _crosses(ax, ay, ax + _CELL, ay):
                    block_r[ix, iy] = True
                if iy + 1 < _H and free[ix, iy + 1] and _crosses(ax, ay, ax, ay + _CELL):
                    block_u[ix, iy] = True
        # Pass 2: diagonals = direct crossing OR a squeeze past any surrounding
        # ortho wall (all four edges of the 2x2 corner block must be open).
        for ix in range(_W):
            ax = ix * _CELL + half
            for iy in range(_H):
                if not free[ix, iy]:
                    continue
                ay = iy * _CELL + half
                if ix + 1 < _W and iy + 1 < _H and free[ix + 1, iy + 1]:
                    block_dr[ix, iy] = (
                        _crosses(ax, ay, ax + _CELL, ay + _CELL)
                        or block_r[ix, iy] or block_r[ix, iy + 1]
                        or block_u[ix, iy] or block_u[ix + 1, iy]
                    )
                if ix + 1 < _W and iy - 1 >= 0 and free[ix + 1, iy - 1]:
                    block_ur[ix, iy] = (
                        _crosses(ax, ay, ax + _CELL, ay - _CELL)
                        or block_r[ix, iy - 1] or block_r[ix, iy]
                        or block_u[ix, iy - 1] or block_u[ix + 1, iy - 1]
                    )
        self._block_r, self._block_u = block_r, block_u
        self._block_dr, self._block_ur = block_dr, block_ur

    def _apply_one_way_edges(self, free: np.ndarray) -> None:
        """Make one-way platforms block the flood in one direction only.

        A one-way is a 24px face with an outward normal ``n``.  The game
        depenetrates a ninja that is on the ``+n`` side and moving toward the
        face (``v.n <= 0``) and does nothing at all from behind, so the edge is
        genuinely directional — and the four undirected masks cannot say that.
        This builds the ``*_neg`` twins (identical for every tile wall) and sets
        each one-way crossing in whichever of the two the step direction earns.

        Applied to 887 of the 3,568 official levels — the same order as locked
        doors, and a different 887, so the two together reach most of the corpus
        the mask was previously blind on.

        Only the four axis-aligned orientations occur in the official corpus
        (measured: 0/2/4/6, 9,010 platforms, zero diagonals), so ``n`` is always
        a unit axis vector and the face is always axis-aligned.
        """
        if not self.one_ways:
            return
        # Copies, not zeros: a tile wall blocks both directions, and the negative
        # masks are the authority for every reverse traversal once they exist.
        self._block_r_neg = self._block_r.copy()
        self._block_u_neg = self._block_u.copy()
        self._block_dr_neg = self._block_dr.copy()
        self._block_ur_neg = self._block_ur.copy()

        half = _CELL / 2.0
        pad = 2
        # (mask attr, neg attr, step direction, index offset from (ix, iy))
        dirs = (
            ("_block_r", "_block_r_neg", (1, 0)),
            ("_block_u", "_block_u_neg", (0, 1)),
            ("_block_dr", "_block_dr_neg", (1, 1)),
            ("_block_ur", "_block_ur_neg", (1, -1)),
        )
        for (px_, py_, nx, ny) in self.one_ways:
            # Face is perpendicular to the normal, centred on the platform.
            if nx:  # normal along x -> vertical face
                x1, y1, x2, y2 = px_, py_ - _ONE_WAY_SEMI, px_, py_ + _ONE_WAY_SEMI
            else:
                x1, y1, x2, y2 = px_ - _ONE_WAY_SEMI, py_, px_ + _ONE_WAY_SEMI, py_
            cx0 = max(0, int(min(x1, x2) / _CELL) - pad)
            cx1 = min(_W - 1, int(max(x1, x2) / _CELL) + pad)
            cy0 = max(0, int(min(y1, y2) / _CELL) - pad)
            cy1 = min(_H - 1, int(max(y1, y2) / _CELL) + pad)
            for ix in range(cx0, cx1 + 1):
                ax = ix * _CELL + half
                for iy in range(cy0, cy1 + 1):
                    if not free[ix, iy]:
                        continue
                    ay = iy * _CELL + half
                    for (pos_attr, neg_attr, (sdx, sdy)) in dirs:
                        jx, jy = ix + sdx, iy + sdy
                        if not (0 <= jx < _W and 0 <= jy < _H and free[jx, jy]):
                            continue
                        if not _seg_crosses(ax, ay, ax + sdx * _CELL,
                                            ay + sdy * _CELL, x1, y1, x2, y2):
                            continue
                        # Blocked only when travelling AGAINST the normal, i.e.
                        # arriving from the face's outward side.  A step exactly
                        # parallel to the face (dot == 0) cannot be crossing it in
                        # a blocking sense and is left open, matching the module's
                        # strict-crossing convention elsewhere.
                        dot = sdx * nx + sdy * ny
                        if dot < 0:
                            getattr(self, pos_attr)[ix, iy] = True
                        elif dot > 0:
                            getattr(self, neg_attr)[ix, iy] = True

    def _apply_door_edges(self, free: np.ndarray) -> None:
        """OR closed locked-door barriers into the four blocked-edge masks.

        A door is a 24px collision segment exactly like a tile wall, so it
        belongs in the EDGE masks and not in the node-free mask.  Cutting cells
        instead would erase occupiable ground — the §7.2 failure that cost a
        spawn — and a door is a place the ninja can legitimately stand right up
        against.

        Applied as a local overlay after ``_build_blocked_edges``, which lets the
        native tile kernel keep ownership of the tile geometry and keeps this
        change out of Rust entirely.  Only the cells within a segment's dilated
        bounding box are re-tested, so the cost is proportional to the door count
        rather than to the grid.
        """
        open_set = self._doors if isinstance(self._doors, frozenset) else frozenset()
        segs = []
        for idx, (dx_, dy_, orient, _sx, _sy) in enumerate(self.locked_doors):
            if idx in open_set:
                continue
            if orient in _DOOR_VERTICAL_ORIENTS:
                segs.append((dx_, dy_ - _DOOR_RADIUS, dx_, dy_ + _DOOR_RADIUS))
            else:
                segs.append((dx_ - _DOOR_RADIUS, dy_, dx_ + _DOOR_RADIUS, dy_))
        if not segs:
            return
        half = _CELL / 2.0
        # A step spans one cell, so a crossing can only start within one cell of
        # the segment's bounding box; +2 is that plus the diagonal reach.
        pad = 2
        touched = set()
        for (x1, y1, x2, y2) in segs:
            cx0 = max(0, int(min(x1, x2) / _CELL) - pad)
            cx1 = min(_W - 1, int(max(x1, x2) / _CELL) + pad)
            cy0 = max(0, int(min(y1, y2) / _CELL) - pad)
            cy1 = min(_H - 1, int(max(y1, y2) / _CELL) + pad)
            for ix in range(cx0, cx1 + 1):
                ax = ix * _CELL + half
                for iy in range(cy0, cy1 + 1):
                    if not free[ix, iy]:
                        continue
                    ay = iy * _CELL + half
                    touched.add((ix, iy))
                    if (ix + 1 < _W and free[ix + 1, iy]
                            and _seg_crosses(ax, ay, ax + _CELL, ay, x1, y1, x2, y2)):
                        self._block_r[ix, iy] = True
                    if (iy + 1 < _H and free[ix, iy + 1]
                            and _seg_crosses(ax, ay, ax, ay + _CELL, x1, y1, x2, y2)):
                        self._block_u[ix, iy] = True
                    if (ix + 1 < _W and iy + 1 < _H and free[ix + 1, iy + 1]
                            and _seg_crosses(ax, ay, ax + _CELL, ay + _CELL,
                                             x1, y1, x2, y2)):
                        self._block_dr[ix, iy] = True
                    if (ix + 1 < _W and iy - 1 >= 0 and free[ix + 1, iy - 1]
                            and _seg_crosses(ax, ay, ax + _CELL, ay - _CELL,
                                             x1, y1, x2, y2)):
                        self._block_ur[ix, iy] = True
        # Re-apply the corner-squeeze rule over the touched neighbourhood: a
        # diagonal is blocked when the ninja would have to slip past an
        # orthogonal wall, and the orthogonal walls just changed.  Without this,
        # a door that blocks both of a corner's ortho edges still admits the
        # diagonal step around it.
        for (ix, iy) in touched:
            if ix + 1 < _W and iy + 1 < _H and free[ix + 1, iy + 1]:
                if (self._block_r[ix, iy] or self._block_r[ix, iy + 1]
                        or self._block_u[ix, iy] or self._block_u[ix + 1, iy]):
                    self._block_dr[ix, iy] = True
            if ix + 1 < _W and iy - 1 >= 0 and free[ix + 1, iy - 1]:
                if (self._block_r[ix, iy - 1] or self._block_r[ix, iy]
                        or self._block_u[ix, iy - 1] or self._block_u[ix + 1, iy - 1]):
                    self._block_ur[ix, iy] = True

    def _step_blocked(self, x: int, y: int, dx: int, dy: int) -> bool:
        """Is the flood step (x,y)->(x+dx,y+dy) crossing a solid wall segment?

        Each of the four masks stores ONE undirected edge and is read from either
        end, which is exactly right for a tile wall: rock blocks both ways.  A
        one-way platform does not, so when the level has any, a parallel set of
        ``*_neg`` masks holds the same edges as traversed in the NEGATIVE
        direction.  They are built as copies of the positive masks, so every tile
        wall is still symmetric and only the one-way edges differ.
        """
        neg = self._block_r_neg is not None
        if dy == 0:
            if dx == 1:
                return bool(self._block_r[x, y])
            return bool((self._block_r_neg if neg else self._block_r)[x - 1, y])
        if dx == 0:
            if dy == 1:
                return bool(self._block_u[x, y])
            return bool((self._block_u_neg if neg else self._block_u)[x, y - 1])
        if dx == dy:  # (1,1) forward or (-1,-1) reverse -> same DR diagonal
            if dx == 1:
                return bool(self._block_dr[x, y])
            return bool((self._block_dr_neg if neg else self._block_dr)[x - 1, y - 1])
        # (1,-1) forward or (-1,1) reverse -> same UR diagonal
        if dx == 1:
            return bool(self._block_ur[x, y])
        return bool((self._block_ur_neg if neg else self._block_ur)[x - 1, y + 1])

    # ------------------------------------------------------------------
    # Physics cost model
    # ------------------------------------------------------------------
    def _blocked_edges_native(self, tiles, free: np.ndarray) -> bool:
        """Fill the four blocked-edge masks natively. False -> caller does it."""
        if os.environ.get("NCLONE_FS_NATIVE", "1") != "1":
            return False
        try:
            import nclone_native as Nnative

            br, bu, bdr, bur = Nnative.free_space_blocked_edges(
                _W, _H, _CELL,
                np.ascontiguousarray(tiles, dtype=np.int32).ravel(),
                np.ascontiguousarray(free, dtype=np.uint8).tobytes(),
            )
        except Exception:  # noqa: BLE001 - any native failure -> Python path
            return False
        def _m(v):
            return np.asarray(v, np.uint8).reshape(_W, _H).astype(bool)
        self._block_r, self._block_u = _m(br), _m(bu)
        self._block_dr, self._block_ur = _m(bdr), _m(bur)
        return True

    def _flood_native(self, switch, exit_door, sw_r: float, ex_r: float):
        """Both cost-to-go fields via ``nclone_native``, or None to fall back.

        The flood is ~95% of this class's build time and is pure integer/float
        work over masks the caller already has, so it ports to Rust with no
        geometry duplication (see nclone-native/src/free_space.rs). Returns None
        — never raises — on any import/ABI/shape problem, so a stale or missing
        native module degrades to the Python flood instead of losing the dense
        reward signal entirely.

        The native path also fills ``_floor_rise`` / ``_wall_adj`` in physics
        mode, so the cached state is identical whichever path built it.
        """
        if os.environ.get("NCLONE_FS_NATIVE", "1") != "1":
            return None
        try:
            import nclone_native as Nnative

            flood = Nnative.free_space_flood
        except (ImportError, AttributeError):
            return None
        try:
            def _b(a):
                return np.ascontiguousarray(a, dtype=np.uint8).tobytes()

            params = None
            if self._physics_cost:
                K = _physics_cost_constants()
                params = [
                    K["run"], K["air"], K["fall"], K["jump"], K["wall"],
                    K["jump_reach_cells"], K["lethal_drop_cells"],
                    _LETHAL_FALL_PENALTY, _AIR_UNSUPPORTED_K,
                    _AIR_UNSUPPORTED_CAP,
                ]
            # Arc edges are extra PREDECESSOR edges the flood must relax; passed
            # in as CSR rather than re-derived, same as the blocked-edge masks.
            arcs = None
            if self._arc_rev_start is not None:
                arcs = (
                    np.ascontiguousarray(self._arc_rev_start, np.int64),
                    np.ascontiguousarray(self._arc_rev_src, np.int64),
                    np.ascontiguousarray(self._arc_rev_cost, np.float64),
                )
            ds, de = flood(
                _W, _H, _CELL, _b(self._free),
                np.ascontiguousarray(self._mden, dtype=np.float32).ravel(),
                _b(self._block_r), _b(self._block_u),
                _b(self._block_dr), _b(self._block_ur),
                _MINE_SOFT_K,
                (float(switch[0]), float(switch[1])),
                (float(exit_door[0]), float(exit_door[1])),
                float(sw_r), float(ex_r), params, arcs,
            )
            return (np.asarray(ds, np.float64).reshape(_W, _H),
                    np.asarray(de, np.float64).reshape(_W, _H))
        except Exception:  # noqa: BLE001 - any native failure -> Python flood
            return None

    def _build_climb_fields(self, free: np.ndarray) -> None:
        """``_floor_rise`` / ``_wall_adj``, natively when available."""
        if os.environ.get("NCLONE_FS_NATIVE", "1") == "1":
            try:
                import nclone_native as Nnative

                rise, wall = Nnative.free_space_climb_fields(
                    _W, _H, np.ascontiguousarray(free, dtype=np.uint8).tobytes()
                )
                self._floor_rise = np.asarray(rise, np.int32).reshape(_W, _H)
                self._wall_adj = (
                    np.asarray(wall, np.uint8).reshape(_W, _H).astype(bool)
                )
                return
            except Exception:
                pass
        self._build_climb_fields_py(free)

    def _build_climb_fields_py(self, free: np.ndarray) -> None:
        """Static per-cell climb geometry consumed by ``_dijkstra_physics``.

        ``_floor_rise[x, y]`` — how many free cells lie directly BELOW this one
        before the first solid. It is simultaneously "how high above the floor am
        I" (which bounds whether a jump could have put me here) and "how far
        would I fall from here" (which prices committing to a drop). Note ``y``
        grows DOWNWARD on this grid, so "below" is ``y + 1``; the scan therefore
        runs bottom-up. Out-of-range below counts as solid — the sim's 1-tile
        border is solid, so a column is never genuinely bottomless.

        ``_wall_adj[x, y]`` — a solid cell sits immediately left or right, so the
        ninja can wall-jump/wall-slide here. This is the escape hatch that keeps
        shafts and chimneys climbable when the open-air jump-reach rule would
        otherwise seal them.
        """
        floor_rise = np.zeros((_W, _H), np.int32)
        for iy in range(_H - 1, -1, -1):
            below_free = free[:, iy + 1] if iy + 1 < _H else np.zeros(_W, bool)
            prev = floor_rise[:, iy + 1] if iy + 1 < _H else np.zeros(_W, np.int32)
            floor_rise[:, iy] = np.where(below_free, prev + 1, 0)
        floor_rise[~free] = 0

        solid = ~free
        left_solid = np.ones((_W, _H), bool)
        left_solid[1:] = solid[:-1]
        right_solid = np.ones((_W, _H), bool)
        right_solid[:-1] = solid[1:]
        self._floor_rise = floor_rise
        self._wall_adj = free & (left_solid | right_solid)

    @staticmethod
    def _arc_shape(normal: int, hold: int, h_after: int) -> np.ndarray:
        """Sampled wall-jump arc offsets in px, relative to the launch cell centre.

        Level-independent, so it is computed once per (normal, hold, h_after) and
        reused for every launch cell. Returns ``(n_samples, 2)`` of ``(dx, dy)``
        together with the frame index each sample belongs to, via
        :meth:`_arc_shape_frames`.
        """
        key = (normal, hold, h_after)
        cached = _ARC_SHAPE_CACHE.get(key)
        if cached is not None:
            return cached[0]
        from ..constants.physics_constants import (
            JUMP_WALL_REGULAR_X_MULTIPLIER, JUMP_WALL_REGULAR_Y,
        )
        from .analytical_physics import compute_jump_arc_relative

        impulse_x = JUMP_WALL_REGULAR_X_MULTIPLIER * normal
        # wall_jump() from rest: vx <- impulse_x, vy <- JUMP_WALL_REGULAR_Y, and
        # the position is nudged by the same impulse on the firing tick.
        frames = compute_jump_arc_relative(
            float(impulse_x), float(JUMP_WALL_REGULAR_Y), hold,
            h_during=normal, h_after=h_after, max_frames=_ARC_MAX_FRAMES,
        )
        pts = [(float(impulse_x), float(JUMP_WALL_REGULAR_Y))]
        idx = [0]
        for fi, fr in enumerate(frames, start=1):
            pts.append((impulse_x + fr.rel_x, JUMP_WALL_REGULAR_Y + fr.rel_y))
            idx.append(fi)
        pts = np.asarray(pts, np.float64)
        # Interpolate between frames so a fast fall cannot tunnel a thin floor.
        segs, fidx = [], []
        for i in range(len(pts) - 1):
            for k in range(_ARC_SUBSAMPLES):
                t = k / _ARC_SUBSAMPLES
                segs.append(pts[i] + (pts[i + 1] - pts[i]) * t)
                fidx.append(idx[i] + t)
        segs.append(pts[-1])
        fidx.append(float(idx[-1]))
        out = (np.asarray(segs, np.float64), np.asarray(fidx, np.float64))
        _ARC_SHAPE_CACHE[key] = out
        return out[0]

    @staticmethod
    def _arc_shape_frames(normal: int, hold: int, h_after: int) -> np.ndarray:
        FreeSpaceDistanceSource._arc_shape(normal, hold, h_after)
        return _ARC_SHAPE_CACHE[(normal, hold, h_after)][1]

    def _build_arc_edges(self) -> None:
        """Wall-launch arc edges, as a CSR reverse adjacency keyed by LANDING cell.

        Backward Dijkstra pops a node and relaxes its PREDECESSORS, so the flood
        needs "which launches land here", not "where does this launch go" — hence
        the inversion at the end.

        Vectorised over launch cells: each arc shape is a fixed offset array, so
        one arc variant costs a couple of gathers over all its launch sites
        rather than a Python loop per site.
        """
        self._arc_rev_start = None
        self._arc_rev_src = None
        self._arc_rev_cost = None
        free = self._free
        wall = self._wall_adj
        if not _ARC_EDGES or free is None:
            return

        solid = ~free
        left_solid = np.zeros((_W, _H), bool)
        left_solid[1:] = solid[:-1]
        right_solid = np.zeros((_W, _H), bool)
        right_solid[:-1] = solid[1:]

        # The cells the step model cannot be entered from below at all — see the
        # up-gate in _dijkstra_physics. Arc edges only ever target these.
        reach = _physics_cost_constants()["jump_reach_cells"]
        unlockable = free & ~self._wall_adj & (self._floor_rise > reach)

        src_list, dst_list, cost_list = [], [], []
        for normal in (1, -1):
            # normal points AWAY from the wall: solid on the left -> jump right.
            launch = free & (left_solid if normal == 1 else right_solid)
            lx, ly = np.nonzero(launch)
            if lx.size == 0:
                continue
            cx0 = (lx + 0.5) * _CELL
            cy0 = (ly + 0.5) * _CELL
            for hold, h_after in _ARC_VARIANTS:
                off = self._arc_shape(normal, hold, h_after)
                fr = self._arc_shape_frames(normal, hold, h_after)
                # (n_launch, n_sample) absolute cell indices along every arc.
                gx = np.floor((cx0[:, None] + off[None, :, 0]) / _CELL).astype(np.int32)
                gy = np.floor((cy0[:, None] + off[None, :, 1]) / _CELL).astype(np.int32)
                inside = (gx >= 0) & (gx < _W) & (gy >= 0) & (gy < _H)
                ok = inside & free[np.clip(gx, 0, _W - 1), np.clip(gy, 0, _H - 1)]
                # First blocked sample ends the arc; everything after is behind
                # geometry. argmin on a bool gives the first False (0 if none).
                blocked = ~ok
                first_bad = np.where(blocked.any(axis=1), blocked.argmax(axis=1),
                                     ok.shape[1])
                # Every cell the arc REACHES, not just where it ends. A wall jump
                # in a deep chamber arcs up through the gap and falls back down,
                # so its terminus is back where it started; the cell that matters
                # is the one it passes through at apex.
                flying = np.arange(ok.shape[1])[None, :] < first_bad[:, None]
                # Keep only destinations the step model REFUSES to enter from
                # below — not wall-adjacent and out of a floor jump's reach. Those
                # are exactly the cells an arc exists to unlock; anywhere else the
                # 8-connected steps already price a route, and emitting those too
                # would multiply the edge set for nothing.
                gate_fails = unlockable[np.clip(gx, 0, _W - 1), np.clip(gy, 0, _H - 1)]
                take = flying & gate_fails
                if not take.any():
                    continue
                rows, cols = np.nonzero(take)
                dx, dy = gx[rows, cols], gy[rows, cols]
                moved = (dx != lx[rows]) | (dy != ly[rows])
                if not moved.any():
                    continue
                rows, cols = rows[moved], cols[moved]
                dx, dy = dx[moved], dy[moved]
                src_list.append(lx[rows].astype(np.int64) * _H + ly[rows])
                dst_list.append(dx.astype(np.int64) * _H + dy)
                cost_list.append(np.maximum(fr[cols], 1.0))

        if not src_list:
            return
        src = np.concatenate(src_list)
        dst = np.concatenate(dst_list)
        cost = np.concatenate(cost_list).astype(np.float64)
        # Keep only the cheapest edge per (launch, landing) pair.
        order = np.lexsort((cost, dst, src))
        src, dst, cost = src[order], dst[order], cost[order]
        uniq = np.ones(src.size, bool)
        uniq[1:] = (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])
        src, dst, cost = src[uniq], dst[uniq], cost[uniq]

        # Invert to CSR keyed by landing cell.
        order = np.argsort(dst, kind="stable")
        src, dst, cost = src[order], dst[order], cost[order]
        self._arc_rev_start = np.searchsorted(dst, np.arange(_W * _H + 1)).astype(np.int64)
        self._arc_rev_src = src.astype(np.int64)
        self._arc_rev_cost = cost

    def _dijkstra_physics(self, goal: Tuple[float, float],
                          contact_radius: float) -> np.ndarray:
        """Anisotropic frame-cost cost-to-go (8-connected), backward from ``goal``.

        Same grid, same seeding, same wall-segment blocking as ``_dijkstra`` —
        only the edge weight changes, from "1.0 / 1.41 grid units" to "estimated
        FRAMES, direction-dependent":

        * **down** is cheap and always legal (gravity is free propulsion);
        * **up** costs ~2.4x a run and is only legal into a cell that is either
          wall-adjacent (wall-jump chain, priced higher still) or within a floor
          jump's 15.8-cell reach of the floor beneath it — otherwise the step is
          BLOCKED, which is what stops the field flooding straight up through
          open air at an exit the ninja cannot reach;
        * **horizontal** costs a run along a surface, slightly more mid-air;
        * stepping off a ledge into a flat-floor-fatal drop costs a one-time
          penalty.

        Because the weights are ASYMMETRIC, a backward flood must relax REVERSE
        edges: popping ``n`` we consider each predecessor ``p = n - (dx, dy)``
        and charge the cost of the forward move ``p -> n``. Relaxing ``n -> p``
        instead (what the geometric flood does, harmlessly, because its weights
        are symmetric) would compute the cost of walking AWAY from the goal.
        """
        d = np.full((_W, _H), np.inf, np.float64)
        pq = self._seed_activation_disc(d, goal, contact_radius)
        if not pq:
            return d
        return self._relax_physics(d, pq)

    def flood_physics_from_seeds(
        self, seeds: Dict[Tuple[int, int], float],
    ) -> np.ndarray:
        """Anisotropic frame-cost flood from arbitrary pre-costed seed cells.

        Same relaxation as :meth:`_dijkstra_physics`, but the sources are supplied
        rather than derived from a goal entity's activation disc.  This is what
        lets the SURFACE LATTICE's exact, physics-derived ``d*`` be pushed out into
        free space: seed each cell holding a surface point with that point's ``d*``
        and the flood extends it to the ~80% of level cells that sit further than
        the lattice's 18px surface-lookup radius from any surface point, where
        ``cost_to_go`` otherwise returns ``inf``.

        Args:
            seeds: ``{(cell_x, cell_y): initial_cost_in_frames}``.  Non-free or
                out-of-range cells are ignored; duplicates keep the cheapest.

        Returns:
            ``(_W, _H)`` float64 cost field; ``inf`` where unreachable.
        """
        d = np.full((_W, _H), np.inf, np.float64)
        free = self._free
        if free is None:
            return d
        pq: list = []
        for (cx, cy), cost in seeds.items():
            if not (0 <= cx < _W and 0 <= cy < _H) or not free[cx, cy]:
                continue
            c = float(cost)
            if not math.isfinite(c) or c >= d[cx, cy]:
                continue
            d[cx, cy] = c
            pq.append((c, cx, cy))
        if not pq:
            return d
        return self._relax_physics(d, pq)

    def _relax_physics(self, d: np.ndarray, pq: list) -> np.ndarray:
        """Shared anisotropic relaxation loop for the physics cost model.

        ``d`` holds the seed costs already; ``pq`` is the unheapified source list.
        See :meth:`_dijkstra_physics` for the edge-weight semantics and for why a
        backward flood must relax REVERSE edges.
        """
        free, mden = self._free, self._mden
        tmul = self._threat_mul
        rise, wall = self._floor_rise, self._wall_adj
        rev_start = self._arc_rev_start
        rev_src, rev_cost = self._arc_rev_src, self._arc_rev_cost
        K = _physics_cost_constants()
        c_run, c_air, c_fall = K["run"], K["air"], K["fall"]
        c_jump, c_wall = K["jump"], K["wall"]
        reach, lethal = K["jump_reach_cells"], K["lethal_drop_cells"]

        heapq.heapify(pq)
        while pq:
            c, x, y = heapq.heappop(pq)
            if c > d[x, y]:
                continue
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    # Predecessor of the forward move (dx, dy) that lands on (x, y).
                    px, py = x - dx, y - dy
                    if not (0 <= px < _W and 0 <= py < _H and free[px, py]):
                        continue
                    # Corner squeeze + solid-segment crossing, both evaluated on
                    # the forward move p -> n exactly as _dijkstra does.
                    if dx and dy and not (free[px + dx, py] and free[px, py + dy]):
                        continue
                    if self._step_blocked(px, py, dx, dy):
                        continue

                    if dy < 0:  # rising into (x, y)
                        if wall[x, y]:
                            vcost = c_wall
                        elif rise[x, y] <= reach:
                            vcost = c_jump
                        else:
                            continue  # unreachable from below: no free climb
                    elif dy > 0:
                        vcost = c_fall
                    else:
                        vcost = 0.0
                    if not dx:
                        hcost = 0.0
                    elif rise[x, y] == 0 or wall[x, y]:
                        hcost = c_run  # running on a floor / working a wall
                    else:
                        # Unsupported air: the higher above the floor, the less
                        # of this traverse the ninja could actually hold.
                        h = min(rise[x, y] / reach, _AIR_UNSUPPORTED_CAP)
                        hcost = c_air * (1.0 + _AIR_UNSUPPORTED_K * h)
                    step = (math.hypot(hcost, vcost) if (dx and dy)
                            else (hcost + vcost))
                    step *= 1.0 + _MINE_SOFT_K * mden[x, y]
                    if tmul is not None:
                        # Time-MARGINALIZED moving-hazard cost (threat_occupancy):
                        # the fraction of a hazard's cycle this cell is lethal,
                        # applied exactly like the static mine density above.  It
                        # is a preference for corridors that are USUALLY safe, not
                        # a safety system, and it lives inside Phi so the shaping
                        # term still telescopes.
                        step *= tmul[x, y]
                    # Committing to a fatal drop: charged once, on the move that
                    # LEAVES the ground (rise[p] <= 1) into a column whose floor
                    # is a lethal distance below. Wall-adjacent targets are exempt
                    # — the ninja can wall-slide a shaft of any depth.
                    if (rise[px, py] <= 1 and rise[x, y] > lethal
                            and not wall[x, y]):
                        step += _LETHAL_FALL_PENALTY

                    nd = c + step
                    if nd < d[px, py]:
                        d[px, py] = nd
                        heapq.heappush(pq, (nd, px, py))

            # Arc predecessors: wall launches that LAND on (x, y). These skip the
            # intervening air, which is the whole point — every mid-flight cell
            # would fail the up-gate above.
            if rev_start is not None:
                node = x * _H + y
                for e in range(rev_start[node], rev_start[node + 1]):
                    lidx = rev_src[e]
                    lx, ly = divmod(int(lidx), _H)
                    nd = c + rev_cost[e]
                    if nd < d[lx, ly]:
                        d[lx, ly] = nd
                        heapq.heappush(pq, (nd, lx, ly))
        return d

    @staticmethod
    def _finite_max(d: np.ndarray) -> float:
        fin = d[np.isfinite(d)]
        return float(fin.max()) if fin.size else 1.0

    def _cell(self, x: float, y: float) -> Tuple[int, int]:
        return int(x / _CELL), int(y / _CELL)

    def _seed_activation_disc(self, d: np.ndarray, goal: Tuple[float, float],
                              contact_radius: float) -> list:
        """Zero-cost sources for a flood toward ``goal``; mutates ``d`` in place.

        Multi-source seed: every free cell whose centre lies inside the
        activation disc (``NINJA_RADIUS + entity_radius``), so the cost-to-go
        bottoms out at the contact boundary rather than the exact entity centre.

        Fallback when the disc covers no free cell — the entity is RECESSED behind
        a wall, so its 16px contact disc reaches the corridor only at sub-cell
        resolution (e.g. the niche-mounted switch in loop corridors, ~18-24px from
        the nearest free cell CENTRE). Seeding the single nearest free cell leaves
        it corner-isolated -> a DEGENERATE 1-cell field -> a FLAT, gradientless
        cost-to-go (the agent gets NO dense signal toward the switch). Instead
        expand the seed radius until we capture the adjacent free corridor and seed
        that whole cluster as cost-0, so the field propagates and the gradient is
        restored. Only triggers in the recessed case; the normal disc-seeded path
        (exit, most switches) is unchanged.

        Returns the unheapified source list (empty iff the goal is unreachable at
        every radius tried).
        """
        free = self._free
        gx, gy = self._cell(*goal)
        pq: list = []
        for mult in (1.0, 2.0, 3.0, 4.0, 6.0):
            rr = contact_radius * mult
            rr2 = rr * rr
            rc = int(rr / _CELL) + 1
            for ix in range(max(0, gx - rc), min(_W, gx + rc + 1)):
                cx = ix * _CELL + _CELL / 2.0
                for iy in range(max(0, gy - rc), min(_H, gy + rc + 1)):
                    cy = iy * _CELL + _CELL / 2.0
                    if free[ix, iy] and (cx - goal[0]) ** 2 + (cy - goal[1]) ** 2 <= rr2:
                        d[ix, iy] = 0.0
                        pq.append((0.0, ix, iy))
            if pq:
                break
        return pq

    def _dijkstra(self, goal: Tuple[float, float],
                  contact_radius: float) -> np.ndarray:
        """Mine-soft cost-to-go over the tile-free grid (8-connected).

        ``contact_radius`` (= NINJA_RADIUS + entity_radius) defines the activation
        disc: every free cell whose center is within it is a cost-0 source, so the
        cost-to-go bottoms out at the activation boundary rather than the exact
        entity center.
        """
        free, mden = self._free, self._mden
        d = np.full((_W, _H), np.inf, np.float64)
        pq = self._seed_activation_disc(d, goal, contact_radius)
        if not pq:
            return d
        heapq.heapify(pq)
        while pq:
            c, x, y = heapq.heappop(pq)
            if c > d[x, y]:
                continue
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < _W and 0 <= ny < _H and free[nx, ny]):
                        continue
                    # Don't cut through a wall corner: a diagonal step is only
                    # valid if both orthogonally-adjacent cells are also free
                    # (else the ninja-sized body can't pass the corner). Keeps the
                    # field — and its gradient — from leaking through tile geometry.
                    if dx and dy and not (free[x + dx, y] and free[x, y + dy]):
                        continue
                    # Don't cross a solid collision segment sitting BETWEEN two free
                    # cells (thin grid-edge wall on a slope/steep-slope/quarter-pipe
                    # boundary, or a slope diagonal). The node-free mask can't see
                    # these; this is the fix for paths leaking through solid tiles.
                    # This flood is seeded AT the goal and grows outward, so
                    # writing ``d[nx, ny] = d[x, y] + step`` prices the agent's
                    # move (nx, ny) -> (x, y) — the OPPOSITE of ``(dx, dy)``.
                    # For a tile wall that distinction is invisible (rock blocks
                    # both ways), but a one-way platform would be applied
                    # backwards, letting the flood through the face it blocks and
                    # sealing the face it does not.  Same edge, stated in the
                    # travelled direction.
                    if self._step_blocked(nx, ny, -dx, -dy):
                        continue
                    step = (1.41 if dx and dy else 1.0) * (1.0 + _MINE_SOFT_K * mden[nx, ny])
                    if c + step < d[nx, ny]:
                        d[nx, ny] = c + step
                        heapq.heappush(pq, (d[nx, ny], nx, ny))
        return d

    def _nearest_free(self, gx, gy):
        best, bd = (None, None), 1e18
        fy, fx = np.where(self._free.T)  # not used; simple scan below
        for ix in range(_W):
            for iy in range(_H):
                if self._free[ix, iy]:
                    q = (ix - gx) ** 2 + (iy - gy) ** 2
                    if q < bd:
                        bd, best = q, (ix, iy)
        return best

    # ------------------------------------------------------------------
    def _field_at(self, d: np.ndarray, x: float, y: float) -> float:
        # Bilinear interpolation over the 4 surrounding cells so the potential is
        # CONTINUOUS in (x, y) -- a sub-cell (<6px) movement still changes Phi, so
        # PBRS rewards even small steps instead of only when crossing a 6px cell
        # boundary (which would make the dense signal "steppy"). Falls back to the
        # nearest-cell / nearest-finite-in-window logic when any of the 4 corners
        # is non-finite (ninja inside a mine band / tile sliver / level edge).
        # Cell (i, j) carries the value at its CENTRE, ``(i + 0.5) * _CELL``
        # (see _dijkstra's activation disc). Interpolating with ``x / _CELL``
        # therefore reads the field half a cell (3 px) to the left/up of where it
        # actually lives. ``reflection_equivariant`` corrects the offset,
        # which also makes the field exactly equivariant under a left-right
        # reflection: ``gx -> (_W_SYM - 1) - gx`` maps cell centres onto cell
        # centres, whereas the legacy anchor set is not closed under reflection.
        # Off by default because it shifts a PBRS potential (see the setter).
        if _REFLECTION_EQUIVARIANT:
            gx, gy = x / _CELL - 0.5, y / _CELL - 0.5
        else:
            gx, gy = x / _CELL, y / _CELL
        ix, iy = int(math.floor(gx)), int(math.floor(gy))
        if not (0 <= ix < _W and 0 <= iy < _H):
            return float("inf")
        if 0 <= ix < _W - 1 and 0 <= iy < _H - 1:
            v00 = float(d[ix, iy]); v10 = float(d[ix + 1, iy])
            v01 = float(d[ix, iy + 1]); v11 = float(d[ix + 1, iy + 1])
            if (math.isfinite(v00) and math.isfinite(v10)
                    and math.isfinite(v01) and math.isfinite(v11)):
                fx, fy = gx - ix, gy - iy
                return (v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy)
                        + v01 * (1 - fx) * fy + v11 * fx * fy)
        v = float(d[ix, iy])
        if math.isfinite(v):
            return v
        # Nearest finite cell within a small window (the ninja may be momentarily
        # inside a mine band / tile sliver); keeps the signal continuous.
        best = float("inf")
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < _W and 0 <= jy < _H:
                    w = float(d[jx, jy])
                    if math.isfinite(w) and w < best:
                        best = w
        return best

    def _arrays(self, switch_activated: bool):
        if switch_activated:
            return self._d_exit, self._d_max_exit
        return self._d_switch, self._d_max_switch

    def cost_to_go(self, x: float, y: float, switch_activated: bool) -> float:
        """Mine-soft grid cost-to-go (grid units; the HWM/PBRS normalise it)."""
        if not self.available:
            return float("inf")
        d, dmax = self._arrays(switch_activated)
        c = self._field_at(d, x, y)
        if not math.isfinite(c):
            return float(dmax)
        return c

    def cost_to_completion(self, x: float, y: float, switch_activated: bool) -> float:
        """Single continuous cost-to-completion potential across both phases.

        Unlike the lattice's ``d_comb``, the free-space source floods the switch
        and exit independently, so its phase-0 field is cost-to-*switch* only.
        Composing ``phase-0 = cost_to_switch + exit_cost_at_switch`` (a per-level
        constant tail) and ``phase-1 = cost_to_exit`` yields one global potential
        that is continuous at switch activation — essential for PBRS, which would
        otherwise read the basis change as a large spike and punish the switch.
        """
        if switch_activated:
            return self.cost_to_go(x, y, True)
        return self.cost_to_go(x, y, False) + self.exit_cost_at_switch

    def spawn_finite(self, spawn_x: float, spawn_y: float) -> bool:
        if not self.available:
            return False
        return np.isfinite(self._field_at(self._d_switch, spawn_x, spawn_y))

    def cost_at_nearest_free(
        self, x: float, y: float, switch_activated: bool, max_cells: int = 6
    ) -> float:
        """Cost-to-go at the nearest free cell to ``(x, y)`` (expanding-ring
        search up to ``max_cells``), with NO ``dmax`` fallback.

        ``cost_to_go``'s fixed 2-cell slack is tuned for the ninja's live,
        in-corridor position; it returns ``inf`` for points *attached to a
        surface* — a wall-mounted exit switch, the ninja standing on a floor —
        whose own cell is solid. This bridges such points to the free corridor
        they border, so it is the right reader for static difficulty labelling
        and the geometric solvability prefilter. Returns ``inf`` only on a
        genuine severance (no free cell within range, or the goal unreachable
        from it)."""
        if not self.available:
            return float("inf")
        d, _ = self._arrays(switch_activated)
        ix, iy = self._cell(x, y)
        for r in range(max_cells + 1):
            best = float("inf")
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:  # only the ring at radius r
                        continue
                    jx, jy = ix + dx, iy + dy
                    if 0 <= jx < _W and 0 <= jy < _H:
                        v = float(d[jx, jy])
                        if np.isfinite(v) and v < best:
                            best = v
            if np.isfinite(best):
                return best
        return float("inf")

    _NAV_DIRS = (
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (0.70710678, 0.70710678), (0.70710678, -0.70710678),
        (-0.70710678, 0.70710678), (-0.70710678, -0.70710678),
    )

    def nav_feature(self, x: float, y: float, switch_activated: bool,
                    step_px: float = 12.0) -> np.ndarray:
        """4-dim ``[dx_hop, dy_hop, cost_norm, phase]`` steepest-descent hint over
        the free-space cost-to-go (mirrors LatticeDistanceSource.nav_feature)."""
        if not self.available:
            return np.zeros(4, dtype=np.float32)
        d, dmax = self._arrays(switch_activated)
        c0 = self._field_at(d, x, y)
        if not math.isfinite(c0):
            return np.zeros(4, dtype=np.float32)
        best_c, best_dir = c0, (0.0, 0.0)
        for ux, uy in self._NAV_DIRS:
            c = self._field_at(d, x + ux * step_px, y + uy * step_px)
            if math.isfinite(c) and c < best_c:
                best_c, best_dir = c, (ux, uy)
        return np.array([best_dir[0], best_dir[1],
                         min(c0 / max(dmax, 1.0), 1.0),
                         1.0 if switch_activated else 0.0], dtype=np.float32)

    #: Move classes reported by :meth:`descent_path`, in the order
    #: ``_dijkstra_physics`` decides them. A step is named for its VERTICAL
    #: component when it has one, because that is what the cost model gates on;
    #: only a purely horizontal step is named for its horizontal component.
    #: A GEOMETRIC field distinguishes none of this -- its rising moves are
    #: reported as plain ``up`` rather than borrowing physics labels it did not
    #: earn, and it never reports ``air``/``jump``/``wall``.
    MOVE_CLASSES = ("run", "air", "fall", "jump", "wall", "up", "arc")

    def _arc_forward(self):
        """Launch-keyed CSR view of the arc edges, built on demand.

        The flood only ever needs the landing-keyed direction, so that is what is
        persisted; descending the field needs the opposite. Derived here rather
        than stored so the cached state stays one copy.
        """
        if getattr(self, "_arc_fwd", None) is None:
            if self._arc_rev_start is None:
                self._arc_fwd = (None, None, None)
            else:
                start, src, cost = (self._arc_rev_start, self._arc_rev_src,
                                    self._arc_rev_cost)
                # Expand CSR back to (dst per edge), then re-sort by launch.
                counts = np.diff(start)
                dst = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
                order = np.argsort(src, kind="stable")
                s2, d2, c2 = src[order], dst[order], cost[order]
                fwd_start = np.searchsorted(s2, np.arange(_W * _H + 1)).astype(np.int64)
                self._arc_fwd = (fwd_start, d2, c2)
        return self._arc_fwd

    def descent_path(self, x: float, y: float, switch_activated: bool,
                     max_steps: int = 4000) -> Optional[dict]:
        """Steepest-descent route from ``(x, y)`` to the active goal.

        This is the route the PBRS gradient actually pushes along, so it is the
        honest thing to draw: it descends the SAME field the potential reads, and
        it is restricted to the SAME moves the flood was willing to charge for.
        Legality is re-derived here rather than trusted from the field, because
        picking the lowest-``d`` neighbour alone would happily step straight up
        through open air — the field value at a cell reflects that cell's best
        move, not the legality of arriving there from where we stand.

        Per-step cost is reported as the field DROP ``d[p] - d[n]`` rather than a
        recomputed edge weight: that is exactly the quantity PBRS pays out, so a
        class breakdown answers "where does this route spend its frames".

        Returns ``None`` when unavailable or the start cell has no finite cost;
        otherwise a dict with ``points`` (sim pixels, cell centres), ``moves``
        and ``costs`` (both ``len(points) - 1``), ``total`` (cost-to-go in frames
        for a physics field, grid units for a geometric one), ``by_class`` and
        ``truncated``.
        """
        if not self.available:
            return None
        d, _dmax = self._arrays(switch_activated)
        if d is None:
            return None
        free = self._free
        rise, wall = self._floor_rise, self._wall_adj
        physics = bool(self._physics_cost and rise is not None and wall is not None)
        reach = _physics_cost_constants()["jump_reach_cells"] if physics else 0.0

        ix, iy = self._cell(x, y)
        if not (0 <= ix < _W and 0 <= iy < _H):
            return None
        if not np.isfinite(d[ix, iy]):
            # The ninja can sit a cell or two inside geometry (tile slivers, mine
            # bands), so nudge to the nearest cell with a FINITE cost -- not the
            # nearest FREE one, which on a level whose flood never reached here
            # would still be inf. Bounded: beyond a few cells "no route" is the
            # honest answer, and this runs every frame.
            best, bq = None, None
            for jx in range(max(0, ix - 4), min(_W, ix + 5)):
                for jy in range(max(0, iy - 4), min(_H, iy + 5)):
                    if not np.isfinite(d[jx, jy]):
                        continue
                    q = (jx - ix) ** 2 + (jy - iy) ** 2
                    if bq is None or q < bq:
                        bq, best = q, (jx, jy)
            if best is None:
                return None
            ix, iy = best

        start_cost = float(d[ix, iy])
        points = [((ix + 0.5) * _CELL, (iy + 0.5) * _CELL)]
        moves: List[str] = []
        costs: List[float] = []
        by_class = {k: 0.0 for k in self.MOVE_CLASSES}
        seen = {(ix, iy)}
        truncated = False

        for _ in range(max_steps):
            cur = d[ix, iy]
            if cur <= 0.0:
                break
            best = None
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ix + dx, iy + dy
                    if not (0 <= nx < _W and 0 <= ny < _H) or not free[nx, ny]:
                        continue
                    # Same corner-squeeze and solid-segment tests the flood
                    # applied to this move, in the same (forward) orientation.
                    if dx and dy and not (free[ix + dx, iy] and free[ix, iy + dy]):
                        continue
                    if self._step_blocked(ix, iy, dx, dy):
                        continue
                    if physics and dy < 0 and not (wall[nx, ny] or rise[nx, ny] <= reach):
                        continue  # no free climb: the flood refused to price it
                    nd = d[nx, ny]
                    if not np.isfinite(nd) or nd >= cur:
                        continue
                    if best is None or nd < best[0]:
                        best = (nd, nx, ny, dx, dy, False)
            # Arc successors. Without these the route stalls wherever a wall
            # launch is the ONLY way on — which is exactly the situation the arc
            # edges were added for, so the drawn route would contradict the field
            # it is descending.
            fwd_start, fwd_dst, _fwd_cost = self._arc_forward()
            if fwd_start is not None:
                node = ix * _H + iy
                for e in range(fwd_start[node], fwd_start[node + 1]):
                    tx, ty = divmod(int(fwd_dst[e]), _H)
                    nd = d[tx, ty]
                    if not np.isfinite(nd) or nd >= cur:
                        continue
                    if best is None or nd < best[0]:
                        best = (nd, tx, ty, 0, 0, True)

            if best is None:
                break  # local minimum: the field cannot be descended further
            nd, nx, ny, dx, dy, is_arc = best
            if (nx, ny) in seen:
                break  # cycle guard; steepest descent on a strict drop shouldn't
            seen.add((nx, ny))

            if is_arc:
                cls = "arc"
            elif dy < 0:
                if not physics:
                    cls = "up"
                else:
                    cls = "wall" if wall[nx, ny] else "jump"
            elif dy > 0:
                cls = "fall"
            elif physics and rise[ix, iy] != 0 and not wall[ix, iy]:
                cls = "air"
            else:
                cls = "run"
            drop = float(cur - nd)
            moves.append(cls)
            costs.append(drop)
            by_class[cls] += drop
            points.append(((nx + 0.5) * _CELL, (ny + 0.5) * _CELL))
            ix, iy = nx, ny
        else:
            truncated = True

        return {
            "points": points,
            "moves": moves,
            "costs": costs,
            "by_class": by_class,
            "total": start_cost,
            "reached_goal": bool(d[ix, iy] <= 0.0),
            "physics": physics,
            "truncated": truncated,
        }
