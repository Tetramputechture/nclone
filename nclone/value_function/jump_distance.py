"""Jump-aware cost-to-go DistanceSource (Approach A).

A drop-in replacement for :class:`FreeSpaceDistanceSource` that produces a
PHYSICS-AWARE cost-to-go: instead of an 8-connected geometric flood over free
cells (which floods through the air between platforms and points the gradient
into lethal gaps), it builds a graph of standable SUPPORT cells connected by
WALK edges and ballistic JUMP/FALL edges (validated per-level against the level's
6px solid mask using the level-INDEPENDENT trajectory library), and runs an exact
backward Dijkstra from each goal. The dense (W, H) ``_d_switch`` / ``_d_exit``
fields it sets are consumed unchanged by the PBRS reward AND the ``cost_to_go_map``
observation (it subclasses the free-space source, so every scalar method —
``cost_to_go``, ``cost_to_completion``, ``nav_feature``, ``spawn_finite`` — is
inherited and just reads the new fields).

Air (non-support) cells inherit the cost-to-go of their nearest support cell, so
the field's gradient points toward platforms/launch points rather than through
gaps. The result is robust-by-construction (no learning), generalizes to any
procgen geometry, and is correct on flat/slope/curve/maze AND floating-platform
levels.
"""

from __future__ import annotations

import hashlib
import heapq
import os
import pickle
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

from nclone.value_function.free_space_distance import (
    FreeSpaceDistanceSource,
    _W,
    _H,
    _CELL,
)
from nclone.value_function.jump_trajectory_lib import (
    build_floor_trajectories,
    build_wall_trajectories,
    _mirror,
    Trajectory,
)

# Process-wide cache: the trajectory library is level-independent + deterministic.
_LIBRARY: Optional[Dict[int, List[Trajectory]]] = None
# Wall-jump library, per orientation (LEFT-wall base + its RIGHT-wall mirror),
# each keyed by launch vy-bin (arrival vertical velocity -> trick-jump reach).
_WALL_LIB_LEFT: Optional[Dict[int, List[Trajectory]]] = None
_WALL_LIB_RIGHT: Optional[Dict[int, List[Trajectory]]] = None


def _library() -> Dict[int, List[Trajectory]]:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = build_floor_trajectories()
    return _LIBRARY


def _wall_library() -> Tuple[Dict[int, List[Trajectory]], Dict[int, List[Trajectory]]]:
    """(left-wall arcs, right-wall arcs), each a vy-bin -> [Trajectory] dict.
    Right-wall arcs are the horizontal mirror of the left-wall arcs."""
    global _WALL_LIB_LEFT, _WALL_LIB_RIGHT
    if _WALL_LIB_LEFT is None:
        left = build_wall_trajectories()
        _WALL_LIB_LEFT = left
        _WALL_LIB_RIGHT = {bi: [_mirror(t) for t in ts] for bi, ts in left.items()}
    return _WALL_LIB_LEFT, _WALL_LIB_RIGHT


# Cost-to-go is in FRAMES (edge weights are frame costs). A walk step (one 6px
# cell) at ~run speed is ~2 frames; keep walk cheap relative to jumps.
_WALK_COST = 2.0

# ---------------------------------------------------------------------------
# Build cache (mirrors free_space_distance's two-tier cache, but under a DISTINCT
# namespace/subdir so the jump field never collides with the geometric one). The
# jump build is SLOW (6-11 s/level: validating every trajectory per support cell
# against the level mask), so under multi-level training a per-reset rebuild would
# stall the synchronous vec-step badly. We cache the built state two tiers deep:
# a per-process in-memory LRU backed by a shared on-disk cache (each level built
# ONCE per node, then read in ~ms by every worker and across restarts). Built
# grids are read-only after construction, so sharing array references on a mem-
# cache hit is safe. Bump _JUMP_CACHE_VERSION if _build's output semantics change
# (e.g. the trajectory library or edge model changes).
_JUMP_CACHE_VERSION = "jump5"  # jump5: mine-HARD field (deadly kill-radius cells block support/wall/arcs/fall-fill) — Audit II P0-2/F4
_JUMP_MEM_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_JUMP_MEM_CAP = int(os.environ.get("NCLONE_JUMPFIELD_MEM_CAP", "256"))  # v14 worker-RSS audit


def _jump_cache_dir() -> Optional[str]:
    """Shared on-disk cache dir for the jump field, or None to disable disk tier.

    Distinct from the free-space cache (a ``jumpfield`` subdir under the shared
    graph cache) so one env var enables both without collision; otherwise falls
    back to a node-local /tmp dir.
    """
    d = os.environ.get("NCLONE_JUMPFIELD_CACHE_DIR")
    if not d:
        graph = os.environ.get("NCLONE_GRAPH_CACHE_DIR")
        d = os.path.join(graph, "jumpfield") if graph else "/tmp/nclone_jumpfield_cache"
    try:
        os.makedirs(d, exist_ok=True)
        return d
    except OSError:
        return None


def _jump_cache_key(map_data: List[int]) -> str:
    h = hashlib.md5()
    h.update(_JUMP_CACHE_VERSION.encode())
    # Grid/build params are part of the key so a param change can't return stale.
    h.update(repr((_CELL, _W, _H, _WALK_COST)).encode())
    h.update(np.asarray([int(v) for v in map_data], dtype=np.int64).tobytes())
    return h.hexdigest()


class JumpDistanceSource(FreeSpaceDistanceSource):
    def _build_cached(self, map_data: List[int]) -> None:
        """Build the jump cost-to-go fields, hitting the mem/disk caches first.

        Lookup order: per-process LRU -> shared on-disk cache -> compute. A hit
        avoids the 6-11 s jump-field rebuild entirely. Uses a DEDICATED namespace
        (``jumpfield``) so it never shares the geometric free-space disk cache.
        Cache failures degrade silently to a live build."""
        try:
            key = _jump_cache_key(map_data)
        except Exception:
            self._build(map_data)
            return

        cached = _JUMP_MEM_CACHE.get(key)
        if cached is not None:
            _JUMP_MEM_CACHE.move_to_end(key)
            self._restore_state(cached)
            return

        cache_dir = _jump_cache_dir()
        path = os.path.join(cache_dir, key + ".pkl") if cache_dir else None
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    state = pickle.load(f)
                self._restore_state(state)
                self._jump_mem_put(key, state)
                return
            except Exception:
                pass  # corrupt/partial cache file -> rebuild

        self._build(map_data)

        # Reuse the free-space state-attr set: every attr it captures
        # (_d_switch/_d_exit/_d_max_*/available/exit_cost_at_switch/_free/_mden/...)
        # is set by _build (which calls the parent _build then overwrites the
        # fields), so a cache hit restores a fully-working source.
        state = self._capture_state()
        self._jump_mem_put(key, state)
        if path:
            try:  # atomic write so concurrent workers never read a partial file
                tmp = f"{path}.tmp.{os.getpid()}"
                with open(tmp, "wb") as f:
                    pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, path)
            except Exception:
                pass

    @staticmethod
    def _jump_mem_put(key: str, state: dict) -> None:
        _JUMP_MEM_CACHE[key] = state
        _JUMP_MEM_CACHE.move_to_end(key)
        while len(_JUMP_MEM_CACHE) > _JUMP_MEM_CAP:
            _JUMP_MEM_CACHE.popitem(last=False)

    def _build(self, map_data: List[int]) -> None:
        # Reuse the parent for geometry: sets self._free, self._mden, self.switch,
        # self.exit, self._deadly (and a geometric _d_switch/_d_exit we overwrite).
        FreeSpaceDistanceSource._build(self, map_data)
        if self.switch is None or self.exit is None or self._free is None:
            self.available = False
            return

        self._build_jump_fields()

        self._d_max_switch = self._finite_max(self._d_switch)
        self._d_max_exit = self._finite_max(self._d_exit)
        self.available = (
            np.isfinite(self._d_switch).any() and np.isfinite(self._d_exit).any()
        )
        if self.available:
            c = self.cost_at_nearest_free(self.switch[0], self.switch[1], True)
            self.exit_cost_at_switch = (
                float(c) if np.isfinite(c) else float(self._d_max_exit)
            )

    # ------------------------------------------------------------------
    def _build_jump_fields(self) -> None:
        free = self._free
        solid = ~free
        W, H = free.shape

        # Mine-HARD block mask (Audit II P0-2 / F4): cells within the deadly-
        # mine kill radius are untraversable for the jump field — no support,
        # no wall-cling, arcs blocked, fall-fill blocked. This is what lets the
        # jump-aware field take PRECEDENCE over MineThreadingDistanceSource on
        # mined levels (previously the threading source pre-empted it there, so
        # the jump field was silently unused on exactly the mined precision
        # levels). Kill radius = ninja(10) + mine(4) px, + one cell of margin.
        mine_block = np.zeros((W, H), bool)
        for (mx, my) in getattr(self, "_deadly", []) or []:
            r = 14.0 + _CELL
            x0 = max(0, int((mx - r) / _CELL))
            x1 = min(W, int((mx + r) / _CELL) + 1)
            y0 = max(0, int((my - r) / _CELL))
            y1 = min(H, int((my + r) / _CELL) + 1)
            for ix in range(x0, x1):
                cxp = ix * _CELL + _CELL / 2.0
                for iy in range(y0, y1):
                    cyp = iy * _CELL + _CELL / 2.0
                    if (cxp - mx) ** 2 + (cyp - my) ** 2 <= 14.0 ** 2:
                        mine_block[ix, iy] = True
        self._mine_block = mine_block
        # Arcs and landings treat mine cells like solid; free/standable exclude them.
        blocked = solid | mine_block

        # SUPPORT cells: a free cell with a solid cell directly below (a floor to
        # stand on). These are the only cells the ninja can be grounded on.
        # Mine-hard: a support inside a kill radius is not standable.
        support = np.zeros((W, H), bool)
        support[:, : H - 1] = free[:, : H - 1] & solid[:, 1:H]
        support &= ~mine_block
        self._support = support

        # WALL-CLING cells: a free cell with a solid cell immediately to a SIDE.
        # The ninja can cling to / wall-jump off / slide down these. A corner cell
        # can be both support and wall.
        wall_left = np.zeros((W, H), bool)   # solid immediately to the LEFT
        wall_left[1:W, :] = free[1:W, :] & solid[0 : W - 1, :]
        wall_right = np.zeros((W, H), bool)  # solid immediately to the RIGHT
        wall_right[0 : W - 1, :] = free[0 : W - 1, :] & solid[1:W, :]
        wall_left &= ~mine_block
        wall_right &= ~mine_block
        wall = wall_left | wall_right
        self._wall = wall

        lib = _library()
        wall_lib_left, wall_lib_right = _wall_library()
        sup_cells = list(map(tuple, np.argwhere(support)))
        wall_cells = list(map(tuple, np.argwhere(wall)))

        # REVERSE adjacency (predecessors): backward Dijkstra from a goal needs
        # "who can reach node v". rev[v] = {u: cost(u->v)}.
        rev: Dict[Tuple[int, int], Dict[Tuple[int, int], float]] = {}

        def _add(u, v, w):
            d = rev.setdefault(v, {})
            if u not in d or w < d[u]:
                d[u] = w

        for (cx, cy) in sup_cells:
            # WALK edges: horizontally adjacent support cells.
            for dx in (-1, 1):
                nx = cx + dx
                if 0 <= nx < W and support[nx, cy]:
                    _add((cx, cy), (nx, cy), _WALK_COST)
            # FLOOR JUMP/FALL edges: first descending SUPPORT landing along each
            # clear trajectory, plus the first WALL cell the arc grabs (the ninja
            # can catch a wall it arcs into before it lands).
            for trajs in lib.values():
                for t in trajs:
                    offs = t.offsets
                    desc = t.descending
                    frs = t.frames
                    grabbed = False
                    for k in range(offs.shape[0]):
                        px = cx + int(offs[k, 0])
                        py = cy + int(offs[k, 1])
                        if not (0 <= px < W and 0 <= py < H):
                            break
                        if blocked[px, py]:
                            break  # wall/ceiling OR deadly-mine kill radius
                        if not grabbed and wall[px, py] and (px, py) != (cx, cy):
                            _add((cx, cy), (px, py), float(frs[k]))  # wall grab
                            grabbed = True
                        if desc[k] and py + 1 < H and solid[px, py + 1]:
                            _add((cx, cy), (px, py), float(frs[k]))
                            break  # first valid support landing on this trajectory

        # WALL-CLING edges: wall-jumps (up/across to a support or another wall) and
        # a free slide/fall straight down the wall.
        for (cx, cy) in wall_cells:
            # Slide/fall down one cell along the wall (available unless a mine
            # kill radius sits in the way).
            if cy + 1 < H and free[cx, cy + 1] and not mine_block[cx, cy + 1]:
                _add((cx, cy), (cx, cy + 1), _WALK_COST)
            if wall_left[cx, cy]:
                self._sweep_wall_edges(
                    cx, cy, wall_lib_left, blocked, solid, wall, W, H, _add
                )
            if wall_right[cx, cy]:
                self._sweep_wall_edges(
                    cx, cy, wall_lib_right, blocked, solid, wall, W, H, _add
                )

        self._d_switch = self._jump_field(rev, support, self.switch)
        self._d_exit = self._jump_field(rev, support, self.exit)

    @staticmethod
    def _sweep_wall_edges(cx, cy, traj_lib, blocked, solid, wall, W, H, _add):
        """Add wall-jump edges from cling cell (cx, cy). ``traj_lib`` is the vy-bin
        -> [Trajectory] dict; we sweep ALL bins (optimistic over the ninja's
        arrival vertical velocity, exactly like the floor loop iterates every vx
        bin), so chained TRICK jumps that arrive with retained upward velocity
        reach far-up cells. For each arc, the FIRST landing — either a support cell
        (descending onto solid) or ANY wall cell it contacts (a re-grab, incl.
        while still ascending, which is how a shaft is climbed) — ends it.
        ``blocked`` = solid | deadly-mine kill radius (mine-hard arcs); support
        detection still keys on real ``solid`` ground (a mine below is lethal,
        not standable)."""
        for trajs in traj_lib.values():
            for t in trajs:
                offs = t.offsets
                desc = t.descending
                frs = t.frames
                for k in range(offs.shape[0]):
                    px = cx + int(offs[k, 0])
                    py = cy + int(offs[k, 1])
                    if not (0 <= px < W and 0 <= py < H):
                        break
                    if blocked[px, py]:
                        break  # wall/ceiling OR deadly-mine kill radius
                    if (px, py) == (cx, cy):
                        continue
                    is_support = desc[k] and py + 1 < H and solid[px, py + 1]
                    if wall[px, py] or is_support:
                        _add((cx, cy), (px, py), float(frs[k]))
                        break

    def _jump_field(self, rev, support, goal_xy) -> np.ndarray:
        """Backward jump/walk Dijkstra from the goal's nearest support cell, then
        a GEODESIC air-fill over the reachable free region.

        Air (non-support) cells inherit the cost-to-go of the nearest support cell
        measured THROUGH free space (8-connected, corner-safe, mine-soft) — NOT by
        straight-line Euclidean distance, which leaked a low cost across walls into
        far-open regions (a wide-open side of the map would wrongly read a small
        cost-to-go even with the goals on the far side). Solid tiles and free cells
        NOT connected to a valued support stay ``inf``: cost-to-go is defined only
        within the flood-fill reachability (matching the parent free-space field),
        never inside geometry or in disconnected open space.
        """
        W, H = support.shape
        field = np.full((W, H), np.inf, dtype=np.float32)
        sup_cells = list(map(tuple, np.argwhere(support)))
        if not sup_cells:
            return field
        gx, gy = goal_xy[0] / _CELL, goal_xy[1] / _CELL
        goal = min(sup_cells, key=lambda c: (c[0] - gx) ** 2 + (c[1] - gy) ** 2)

        dist = {goal: 0.0}
        pq = [(0.0, goal)]
        while pq:
            c, u = heapq.heappop(pq)
            if c > dist.get(u, np.inf):
                continue
            for p, w in rev.get(u, {}).items():
                nc = c + w
                if nc < dist.get(p, np.inf):
                    dist[p] = nc
                    heapq.heappush(pq, (nc, p))

        for (cx, cy), v in dist.items():
            field[cx, cy] = v

        self._fall_project_fill(field, support)
        return field

    def _fall_project_fill(self, field: np.ndarray, support: np.ndarray) -> None:
        """Extend support-cell costs to AIR cells by GRAVITY fall-projection
        (in place).

        Physical model: a ninja in the air falls straight down to the first floor
        below, then routes from there. So every free non-support (air) cell
        inherits the cost-to-go of the support cell it would land on — found by
        scanning DOWN its column to the first support. Implemented as a single
        bottom-up pass per column (``support`` at (x, y) means free with solid at
        (x, y+1), so larger y is DOWN):

            land[x, y] = dist(x, y)              if support[x, y]
                       = land[x, y+1]            elif free[x, y] (fall one step)
                       = inf                     if solid (a wall blocks the fall)

        This is exact for what happens to an airborne ninja, and by construction:
        * never leaks a cost sideways through a wall (projection is vertical only),
          so wide-open regions far from the goal read a HIGH cost, not a spuriously
          low one inherited from a goal-adjacent platform across a wall;
        * stays inside the flood-fill reachability — an air cell whose fall column
          has no reachable support (or lands in a dead pocket) keeps ``inf``, and
          solid tiles keep ``inf`` (we never define cost-to-go inside geometry).
        """
        free = self._free
        if free is None:
            return
        # Mine-hard (Audit II P0-2/F4): a fall through a deadly-mine kill
        # radius is death, so mine cells block the projection exactly like
        # solid — the column above a mine keeps inf instead of inheriting the
        # landing cost from below it.
        mine_block = getattr(self, "_mine_block", None)
        if mine_block is not None:
            free = free & ~mine_block
        W, H = field.shape
        # land[x] tracks the cost of the support a faller from the current row
        # would land on; reset to inf at any solid cell (the fall can't pass it).
        land = np.full(W, np.inf, dtype=np.float32)
        for y in range(H - 1, -1, -1):  # bottom row up to top
            col_free = free[:, y]
            col_support = support[:, y]
            # solid cells block the fall -> reset the running landing cost.
            land = np.where(col_free, land, np.inf).astype(np.float32)
            # support cells ARE a landing surface -> adopt their own jump cost.
            land = np.where(col_support, field[:, y], land).astype(np.float32)
            # free air cells (free, not support) inherit the landing cost below;
            # solid cells stay inf. Support cells already hold their exact value.
            # WALL-CLING cells are graph nodes too (they hold a finite wall-climb
            # cost from the Dijkstra), so take the MIN rather than overwriting —
            # this preserves the up-the-wall gradient while still filling pure-air
            # cells (inf -> land) exactly as before.
            air = col_free & ~col_support
            field[air, y] = np.minimum(field[air, y], land[air])


def get_jump_distance_source(map_data, name: Optional[str] = None) -> JumpDistanceSource:
    return JumpDistanceSource(list(map_data), name=name)
