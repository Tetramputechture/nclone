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
from typing import List, Optional, Tuple

import numpy as np

from ..constants.physics_constants import (
    MAP_TILE_WIDTH, MAP_TILE_HEIGHT, TILE_PIXEL_SIZE,
    NINJA_RADIUS, EXIT_SWITCH_RADIUS, EXIT_DOOR_RADIUS,
)
from ..tile_definitions import (
    TILE_GRID_EDGE_MAP, TILE_SEGMENT_DIAG_MAP, TILE_SEGMENT_CIRCULAR_MAP,
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

    Union of the orthogonal grid-edge walls (TILE_GRID_EDGE_MAP), the diagonal
    slope segment (TILE_SEGMENT_DIAG_MAP) and a chord approximation of the curved
    segment (TILE_SEGMENT_CIRCULAR_MAP). Empty tile (0) and the glitched tiles
    34-37 have NO collision (matching the simulator, which treats 34-37 as empty).
    """
    cached = _TILE_LOCAL_SEG_CACHE.get(tile_type)
    if cached is not None:
        return cached
    segs = []
    if tile_type != 0 and tile_type < 34:
        for present, seg in zip(TILE_GRID_EDGE_MAP.get(tile_type, ()), _GRID_EDGE_SEGS):
            if present:
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
_MINE_KILL = 14.0          # NINJA_RADIUS + TOGGLE_MINE_RADIUS_TOGGLED
_MINE_STATE_DEADLY = 0
_MINE_SOFT_K = 8.0         # mine-density penalty on grid edges

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
_FS_CACHE_VERSION = "fs5"  # bumped: segment-aware edge blocking + tile-8 node-leak fix (no through-solid)
_FS_STATE_ATTRS = (
    "available", "switch", "exit", "_d_switch", "_d_exit",
    "_d_max_switch", "_d_max_exit", "_deadly", "exit_cost_at_switch",
    "_free", "_mden", "_block_r", "_block_u", "_block_dr", "_block_ur",
)
_FS_MEM_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_FS_MEM_CAP = int(os.environ.get("NCLONE_FREESPACE_MEM_CAP", "256"))  # v14 worker-RSS audit


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


def _fs_cache_key(map_data: List[int]) -> str:
    h = hashlib.md5()
    h.update(_FS_CACHE_VERSION.encode())
    # Grid/build params are part of the key so a param change can't return stale.
    h.update(repr((_CELL, _W, _H, _MINE_KILL, _MINE_SOFT_K)).encode())
    h.update(np.asarray([int(v) for v in map_data], dtype=np.int64).tobytes())
    return h.hexdigest()


class FreeSpaceDistanceSource:
    """Per-level free-space grid cost-to-go to the switch and exit.

    Best-effort: any failure leaves ``available == False`` so the caller falls back.
    """

    def __init__(self, map_data: List[int], *, name: Optional[str] = None):
        self.available = False
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
        for k in _FS_STATE_ATTRS:
            setattr(self, k, state[k])

    def _capture_state(self) -> dict:
        return {k: getattr(self, k) for k in _FS_STATE_ATTRS}

    def _build_cached(self, map_data: List[int]) -> None:
        """Build the cost-to-go grids, hitting the mem/disk caches first.

        Lookup order: per-process LRU -> shared on-disk cache -> compute. A hit
        avoids the ~0.1s Dijkstra rebuild entirely (the per-reset bottleneck under
        multi-level training). Cache failures degrade silently to a live build.
        """
        try:
            key = _fs_cache_key(map_data)
        except Exception:
            self._build(map_data)
            return

        cached = _FS_MEM_CACHE.get(key)
        if cached is not None:
            _FS_MEM_CACHE.move_to_end(key)
            self._restore_state(cached)
            return

        cache_dir = _fs_cache_dir()
        path = os.path.join(cache_dir, key + ".pkl") if cache_dir else None
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    state = pickle.load(f)
                self._restore_state(state)
                self._mem_put(key, state)
                return
            except Exception:
                pass  # corrupt/partial cache file -> rebuild

        self._build(map_data)

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
        if switch is None or exit_door is None:
            return
        self.switch, self.exit = switch, exit_door
        self._deadly = deadly  # kept for visualization / introspection

        # Tile-free mask at 6px sub-cell resolution, using the SAME canonical
        # tile-geometry table the reachability graph and the surface lattice use
        # (SUBNODE_VALIDITY_TABLE: for each tile type, which of the 4x4 6px sub-cells
        # lie in the non-solid area). This faithfully follows slopes / half-tiles
        # instead of treating any non-empty tile as a fully-solid 24px block.
        from nclone.graph.reachability.graph_builder import SUBNODE_VALIDITY_TABLE

        tiles = map_data[184:1150]
        free = np.zeros((_W, _H), bool)
        for ix in range(_W):
            cx = ix * _CELL + _CELL / 2.0
            # Pixel cx lands in sim-grid col cx//24; raw tile index is that minus the
            # +1 border the sim applies (map_loader: tile (c,r) -> grid (c+1,r+1)).
            rtx = int(cx // TILE_PIXEL_SIZE) - 1
            sub_x = min(int((cx % TILE_PIXEL_SIZE) // _CELL), 3)
            for iy in range(_H):
                cy = iy * _CELL + _CELL / 2.0
                rty = int(cy // TILE_PIXEL_SIZE) - 1
                if 0 <= rtx < 42 and 0 <= rty < 23:
                    ttype = int(tiles[rtx + rty * 42])
                    sub_y = min(int((cy % TILE_PIXEL_SIZE) // _CELL), 3)
                    free[ix, iy] = SUBNODE_VALIDITY_TABLE.get(ttype, {}).get(
                        (sub_x, sub_y), False
                    )
        # Per-cell deadly-mine density.
        mden = np.zeros((_W, _H), np.float32)
        kc = int(_MINE_KILL / _CELL) + 2
        for (mx, my) in deadly:
            cx, cy = int(mx / _CELL), int(my / _CELL)
            for ix in range(max(0, cx - kc), min(_W, cx + kc + 1)):
                for iy in range(max(0, cy - kc), min(_H, cy + kc + 1)):
                    if (ix * _CELL + _CELL / 2 - mx) ** 2 + (iy * _CELL + _CELL / 2 - my) ** 2 <= (_MINE_KILL + _CELL) ** 2:
                        mden[ix, iy] += 1.0
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

        # Goal cost-to-go reaches 0 at the ACTIVATION boundary, not the entity
        # center: the ninja triggers the switch / exit by edge contact when its
        # center is within (NINJA_RADIUS + entity_radius) of the entity center
        # (overlap_circle_vs_circle). Seeding the activation disc keeps this
        # potential consistent with the surface lattice and the reachability A*.
        self._d_switch = self._dijkstra(switch, NINJA_RADIUS + EXIT_SWITCH_RADIUS)
        self._d_exit = self._dijkstra(exit_door, NINJA_RADIUS + EXIT_DOOR_RADIUS)
        self._d_max_switch = self._finite_max(self._d_switch)
        self._d_max_exit = self._finite_max(self._d_exit)
        # Available only if both goals reached some free cells.
        self.available = (np.isfinite(self._d_switch).any()
                          and np.isfinite(self._d_exit).any())
        # Phase-1 tail constant: exit cost at the switch position. Robust to the
        # switch being surface-mounted (its own cell solid) via the nearest-free
        # ring search; falls back to the field's finite max on genuine severance.
        if self.available:
            c = self.cost_at_nearest_free(switch[0], switch[1], True)
            self.exit_cost_at_switch = (
                float(c) if np.isfinite(c) else float(self._d_max_exit)
            )

    def _build_blocked_edges(self, tiles, free: np.ndarray) -> None:
        """Precompute per-edge blocking from the canonical tile collision segments.

        Builds four (_W, _H) bool masks consulted by ``_dijkstra`` so a flood step
        never crosses a solid wall (orthogonal grid-edge, slope diagonal, or curve
        chord). Only free->free boundaries are tested; solid cells are already
        excluded by the node-free mask.
        """
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

    def _step_blocked(self, x: int, y: int, dx: int, dy: int) -> bool:
        """Is the flood step (x,y)->(x+dx,y+dy) crossing a solid wall segment?"""
        if dy == 0:
            return bool(self._block_r[x, y] if dx == 1 else self._block_r[x - 1, y])
        if dx == 0:
            return bool(self._block_u[x, y] if dy == 1 else self._block_u[x, y - 1])
        if dx == dy:  # (1,1) forward or (-1,-1) reverse -> same DR diagonal
            ox, oy = (x, y) if dx == 1 else (x - 1, y - 1)
            return bool(self._block_dr[ox, oy])
        # (1,-1) forward or (-1,1) reverse -> same UR diagonal
        ox, oy = (x, y) if dx == 1 else (x - 1, y + 1)
        return bool(self._block_ur[ox, oy])

    @staticmethod
    def _finite_max(d: np.ndarray) -> float:
        fin = d[np.isfinite(d)]
        return float(fin.max()) if fin.size else 1.0

    def _cell(self, x: float, y: float) -> Tuple[int, int]:
        return int(x / _CELL), int(y / _CELL)

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
        gx, gy = self._cell(*goal)
        # Multi-source seed: all free cells inside the activation disc.
        pq = []
        r_cells = int(contact_radius / _CELL) + 1
        r2 = contact_radius * contact_radius
        for ix in range(max(0, gx - r_cells), min(_W, gx + r_cells + 1)):
            cx = ix * _CELL + _CELL / 2.0
            for iy in range(max(0, gy - r_cells), min(_H, gy + r_cells + 1)):
                cy = iy * _CELL + _CELL / 2.0
                if free[ix, iy] and (cx - goal[0]) ** 2 + (cy - goal[1]) ** 2 <= r2:
                    d[ix, iy] = 0.0
                    pq.append((0.0, ix, iy))
        # Fallback: the activation disc covered no free cell — the entity is
        # RECESSED behind a wall so its 16px contact disc reaches the corridor only
        # at sub-cell resolution (e.g. the niche-mounted switch in loop corridors,
        # ~18-24px from the nearest free cell CENTER). Seeding the single nearest
        # free cell leaves it corner-isolated -> a DEGENERATE 1-cell field -> a FLAT,
        # gradientless cost-to-go (the agent gets NO dense signal toward the switch).
        # Instead expand the seed radius until we capture the adjacent free corridor
        # and seed that whole cluster as cost-0, so the field propagates and the
        # gradient is restored. Only triggers in the recessed case; the normal
        # disc-seeded path (exit, most switches) is unchanged.
        if not pq:
            for mult in (2.0, 3.0, 4.0, 6.0):
                rr2 = (contact_radius * mult) ** 2
                rc = int(contact_radius * mult / _CELL) + 1
                for ix in range(max(0, gx - rc), min(_W, gx + rc + 1)):
                    cx = ix * _CELL + _CELL / 2.0
                    for iy in range(max(0, gy - rc), min(_H, gy + rc + 1)):
                        cy = iy * _CELL + _CELL / 2.0
                        if free[ix, iy] and (cx - goal[0]) ** 2 + (cy - goal[1]) ** 2 <= rr2:
                            d[ix, iy] = 0.0
                            pq.append((0.0, ix, iy))
                if pq:
                    break
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
                    if self._step_blocked(x, y, dx, dy):
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
        gx, gy = x / _CELL, y / _CELL
        ix, iy = int(gx), int(gy)
        if not (0 <= ix < _W and 0 <= iy < _H):
            return float("inf")
        if 0 <= ix < _W - 1 and 0 <= iy < _H - 1:
            v00 = float(d[ix, iy]); v10 = float(d[ix + 1, iy])
            v01 = float(d[ix, iy + 1]); v11 = float(d[ix + 1, iy + 1])
            if (np.isfinite(v00) and np.isfinite(v10)
                    and np.isfinite(v01) and np.isfinite(v11)):
                fx, fy = gx - ix, gy - iy
                return (v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy)
                        + v01 * (1 - fx) * fy + v11 * fx * fy)
        v = float(d[ix, iy])
        if np.isfinite(v):
            return v
        # Nearest finite cell within a small window (the ninja may be momentarily
        # inside a mine band / tile sliver); keeps the signal continuous.
        best = float("inf")
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < _W and 0 <= jy < _H:
                    w = float(d[jx, jy])
                    if np.isfinite(w) and w < best:
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
        if not np.isfinite(c):
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
        if not np.isfinite(c0):
            return np.zeros(4, dtype=np.float32)
        best_c, best_dir = c0, (0.0, 0.0)
        for ux, uy in self._NAV_DIRS:
            c = self._field_at(d, x + ux * step_px, y + uy * step_px)
            if np.isfinite(c) and c < best_c:
                best_c, best_dir = c, (ux, uy)
        return np.array([best_dir[0], best_dir[1],
                         min(c0 / max(dmax, 1.0), 1.0),
                         1.0 if switch_activated else 0.0], dtype=np.float32)
