"""Mine-threading cost-to-go — a connected, MINE-FAITHFUL PBRS potential for levels
whose route threads a narrow channel between deadly mines (e.g. mines-extremely-difficult).

The surface lattice severs on these levels (every goal-adjacent surface node sits in a
mine kill-disc, so backward Dijkstra has no seed -> ``available=False``). The free-space
grid stays finite only by treating mines as a SOFT cost — it lets the route pass through
lethal cells, so it is not mine-faithful.

This source builds a cost-to-go on the ninja-free region with a HARD mine constraint:

* **Mine-safe traversable region** (3px grid): a cell is traversable iff the ninja CENTRE
  there is >= ``KILL_RADIUS`` (= NINJA_RADIUS + toggle-mine radius = 14px) from every deadly
  mine. Mines/entities are 6px-grid-aligned, so a 3px lattice resolves the channel. This
  region is, empirically, a SINGLE connected component spanning spawn / switch / exit — the
  threadable route exists; the lattice/free-space just failed to represent it.
* **Tile-solidity is a strong COST penalty, not a hard cut.** The real route is a chain of
  curved ballistic arcs that jump off small exposed surfaces and fly *over/past* tile
  features through mine-free air. Cutting tile-solid cells fragments the region (the arcs
  cross them); penalising them keeps the region connected while biasing the potential toward
  open air. A mild mine-proximity penalty also biases toward channel centrelines.
* No control tape / executable arc is computed — only the cost-to-go scalar field. The RL
  agent learns the frame-precise threading; PBRS only needs a connected, mine-faithful Φ.

The public API mirrors ``FreeSpaceDistanceSource`` (cost_to_go / cost_to_completion /
spawn_finite / nav_feature / available) so it is a drop-in PBRS potential source.
"""

from __future__ import annotations

import heapq
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..constants.physics_constants import (
    MAP_TILE_WIDTH, MAP_TILE_HEIGHT, TILE_PIXEL_SIZE,
    NINJA_RADIUS, EXIT_SWITCH_RADIUS, EXIT_DOOR_RADIUS,
)

logger = logging.getLogger(__name__)

# Fine sub-tile grid. Mines/entities are on a 6px grid (multiples of 6), so they
# rasterise exactly. 2px (TILE/12) is needed to resolve razor approaches such as
# the exit door's upper-right corner — the ONLY way to complete some mine levels:
# its single activatable mine-safe cell sits ~21.3px from the door centre (a 0.7px
# margin inside the 22px reach), which a 3px lattice misses entirely. 2px captures
# it and connects it to the right-hand channel (cost-to-exit there: 2494 -> 396).
_CELL = TILE_PIXEL_SIZE / 12.0         # 2px
# +1 tile for the sim's border (raw tile (c,r) lives at grid (c+1,r+1)).
_W = int((MAP_TILE_WIDTH + 1) * TILE_PIXEL_SIZE / _CELL) + 1   # 517
_H = int((MAP_TILE_HEIGHT + 1) * TILE_PIXEL_SIZE / _CELL) + 1  # 289

_TOGGLE_MINE_RADIUS_TOGGLED = 4.0
_KILL_RADIUS = NINJA_RADIUS + _TOGGLE_MINE_RADIUS_TOGGLED   # 14px deadly disc
_MINE_STATE_DEADLY = 0

# Entity (mine/switch/exit/ninja) positions from the sim are in WORLD space, which
# includes the 1-tile (24px) border; the tile array map_data[184:1150] is in INNER
# space (tile (0,0) at inner pixel 0). The whole grid here is built in INNER space
# (so it aligns with tile geometry); world coords are converted at the API boundary.
try:
    from ..graph.reachability.pathfinding_utils import NODE_WORLD_COORD_OFFSET as _OFFSET
except Exception:  # pragma: no cover
    _OFFSET = 24.0

# Tile-solid traversal penalty: the route prefers open air but may fly over/past a
# tile feature (a jump off a surface). Strong enough to bias toward air, finite so
# the region stays connected through unavoidable crossings.
_TILE_SOLID_PENALTY = 5.0
# Mild mine-proximity penalty (on top of the hard >=14px constraint) so the path
# hugs channel centrelines (safer to execute). Cost factor = 1 + k*(1 - clr/ref).
_MINE_PROX_K = 1.5
_MINE_PROX_REF = 24.0   # px clearance at/above which there is no proximity penalty


class MineThreadingDistanceSource:
    """Hard-mine, tile-soft cost-to-go to the switch and exit.

    Best-effort: any failure leaves ``available == False`` so callers fall back.
    """

    def __init__(self, map_data: List[int], *, name: Optional[str] = None):
        self.available = False
        self.name = name
        self.switch: Optional[Tuple[float, float]] = None
        self.exit: Optional[Tuple[float, float]] = None
        self._deadly: List[Tuple[float, float]] = []
        self._safe = np.zeros((_W, _H), bool)        # mine-safe traversable region
        self._tile_solid = np.zeros((_W, _H), bool)  # tile geometry (cost penalty)
        self._clear = np.zeros((_W, _H), np.float32)  # px clearance to nearest mine
        self._d_switch = np.full((_W, _H), np.inf)
        self._d_exit = np.full((_W, _H), np.inf)
        self._d_max_switch = 1.0
        self._d_max_exit = 1.0
        self.exit_cost_at_switch = 0.0
        # Whether ``exit_cost_at_switch`` is a real switch->exit tail or the
        # ``_d_max_exit`` fallback. The fallback SHRINKS with the severed
        # component (see the WARNING on the free-space source), so a consumer
        # that needs a route LENGTH must refuse it rather than believe it.
        self.exit_reachable_from_switch = True
        try:
            self._build(list(map_data))
        except Exception as e:  # pragma: no cover
            logger.debug("MineThreadingDistanceSource build failed: %r", e)
            self.available = False

    # ------------------------------------------------------------------
    @staticmethod
    def _cell(x: float, y: float) -> Tuple[int, int]:
        return int(round(x / _CELL)), int(round(y / _CELL))

    def _build(self, map_data: List[int]) -> None:
        import nclone_native as Nnative
        from ..graph.reachability.graph_builder import SUBNODE_VALIDITY_TABLE

        # Float entity coords (some datasets) -> native loader needs ints (floors
        # anyway); coerce so this source builds instead of silently failing.
        map_data = [int(v) for v in map_data]
        st = Nnative.Simulator.load(map_data).get_state()
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
        # Public attributes stay in WORLD space (consistent with the sim and the
        # cost_to_go API). The internal grid / tile array live in INNER space, so
        # convert to local inner coords for the build; cost_to_go converts queries.
        self.switch, self.exit, self._deadly = switch, exit_door, deadly
        switch_in = (switch[0] - _OFFSET, switch[1] - _OFFSET)
        exit_in = (exit_door[0] - _OFFSET, exit_door[1] - _OFFSET)
        deadly_in = [(mx - _OFFSET, my - _OFFSET) for (mx, my) in deadly]

        tiles = map_data[184:1150]

        # --- Mine clearance (px to nearest deadly mine) + hard mine-safe mask ---
        clear = self._mine_clearance(deadly_in)
        safe = clear >= _KILL_RADIUS
        # Restrict to the playfield interior.
        safe[:, :] &= self._playfield_mask()
        self._clear = clear
        self._safe = safe
        if not safe.any():
            return

        # --- Tile-solidity (canonical 6px geometry), used as a COST penalty ---
        tile_solid = np.zeros((_W, _H), bool)
        for ix in range(_W):
            px = ix * _CELL
            rtx = int(px // TILE_PIXEL_SIZE) - 1  # -1: sim border (grid = raw + 1)
            if not (0 <= rtx < 42):
                tile_solid[ix, :] = True
                continue
            sx = min(int((px % TILE_PIXEL_SIZE) // (TILE_PIXEL_SIZE / 4)), 3)
            for iy in range(_H):
                py = iy * _CELL
                rty = int(py // TILE_PIXEL_SIZE) - 1
                if not (0 <= rty < 23):
                    tile_solid[ix, iy] = True
                    continue
                tt = int(tiles[rtx + rty * 42])
                sy = min(int((py % TILE_PIXEL_SIZE) // (TILE_PIXEL_SIZE / 4)), 3)
                tile_solid[ix, iy] = not SUBNODE_VALIDITY_TABLE.get(tt, {}).get((sx, sy), False)
        self._tile_solid = tile_solid

        # --- Per-cell traversal multiplier: tile-solid penalty + mine-proximity ---
        mult = np.ones((_W, _H), np.float32)
        mult[tile_solid] *= _TILE_SOLID_PENALTY
        prox = 1.0 + _MINE_PROX_K * np.clip(1.0 - clear / _MINE_PROX_REF, 0.0, 1.0)
        mult *= prox

        # --- Walk graph as reverse adjacency (edge u->v stored at v) ---
        radj: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]] = {}
        diag = _CELL * 1.41421356
        orth = _CELL
        for ix in range(_W):
            for iy in range(_H):
                if not safe[ix, iy]:
                    continue
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = ix + dx, iy + dy
                        if not (0 <= nx < _W and 0 <= ny < _H and safe[nx, ny]):
                            continue
                        base = diag if (dx and dy) else orth
                        # Travel cost into the destination cell (avg of the two
                        # endpoints' multipliers keeps the field smooth).
                        cost = base * 0.5 * (mult[ix, iy] + mult[nx, ny])
                        radj.setdefault((nx, ny), []).append(((ix, iy), cost))

        # --- Goal seeds: mine-safe cells inside the activation discs (inner) ---
        sw_seed = self._disc_cells(safe, switch_in, NINJA_RADIUS + EXIT_SWITCH_RADIUS) \
            or [self._nearest_safe(safe, switch_in)]
        ex_seed = self._disc_cells(safe, exit_in, NINJA_RADIUS + EXIT_DOOR_RADIUS) \
            or [self._nearest_safe(safe, exit_in)]
        sw_seed = [c for c in sw_seed if c is not None]
        ex_seed = [c for c in ex_seed if c is not None]
        if not sw_seed or not ex_seed:
            return

        self._d_switch = self._dijkstra(radj, sw_seed)
        self._d_exit = self._dijkstra(radj, ex_seed)
        self._d_max_switch = self._finite_max(self._d_switch)
        self._d_max_exit = self._finite_max(self._d_exit)
        # Phase-1 tail constant: exit cost at the switch (continuity across
        # activation). Evaluated in inner space (the field's frame).
        c = self._field_at(self._d_exit, *switch_in)
        self.exit_reachable_from_switch = bool(np.isfinite(c))
        self.exit_cost_at_switch = float(c) if np.isfinite(c) else float(self._d_max_exit)

        self.available = (np.isfinite(self._d_switch).any()
                          and np.isfinite(self._d_exit).any())

    # ------------------------------------------------------------------
    @staticmethod
    def _playfield_mask() -> np.ndarray:
        m = np.zeros((_W, _H), bool)
        pw = int(MAP_TILE_WIDTH * TILE_PIXEL_SIZE / _CELL)
        ph = int(MAP_TILE_HEIGHT * TILE_PIXEL_SIZE / _CELL)
        m[:pw, :ph] = True
        return m

    def _mine_clearance(self, deadly) -> np.ndarray:
        """Px distance from each grid cell centre to the nearest deadly mine.

        Mines are 6px-grid-aligned, so they rasterise EXACTLY onto the 3px grid
        (a multiple of 6 / 3 is an even integer). A Euclidean distance transform
        then gives the exact clearance field in O(cells) — far cheaper than the
        per-mine disc stamp on a 13k-mine level.
        """
        BIG = 1e6
        if not deadly:
            return np.full((_W, _H), BIG, np.float32)
        from scipy.ndimage import distance_transform_edt
        grid = np.ones((_W, _H), dtype=np.uint8)  # 0 marks a mine cell
        for (mx, my) in deadly:
            ix, iy = int(round(mx / _CELL)), int(round(my / _CELL))
            if 0 <= ix < _W and 0 <= iy < _H:
                grid[ix, iy] = 0
        # EDT: distance from each nonzero cell to the nearest zero (mine) cell.
        return (distance_transform_edt(grid) * _CELL).astype(np.float32)

    @staticmethod
    def _disc_cells(safe, center, radius) -> List[Tuple[int, int]]:
        cx, cy = int(round(center[0] / _CELL)), int(round(center[1] / _CELL))
        rc = int(radius / _CELL) + 1
        r2 = radius * radius
        out = []
        for ix in range(max(0, cx - rc), min(_W, cx + rc + 1)):
            for iy in range(max(0, cy - rc), min(_H, cy + rc + 1)):
                if safe[ix, iy] and (ix * _CELL - center[0]) ** 2 + (iy * _CELL - center[1]) ** 2 <= r2:
                    out.append((ix, iy))
        return out

    @staticmethod
    def _nearest_safe(safe, center):
        cx, cy = int(round(center[0] / _CELL)), int(round(center[1] / _CELL))
        best, bd = None, 1e18
        for r in range(0, 60):
            for ix in range(max(0, cx - r), min(_W, cx + r + 1)):
                for iy in range(max(0, cy - r), min(_H, cy + r + 1)):
                    if max(abs(ix - cx), abs(iy - cy)) != r or not safe[ix, iy]:
                        continue
                    d = (ix - cx) ** 2 + (iy - cy) ** 2
                    if d < bd:
                        bd, best = d, (ix, iy)
            if best is not None:
                return best
        return None

    @staticmethod
    def _dijkstra(radj, seeds) -> np.ndarray:
        d = np.full((_W, _H), np.inf)
        pq = []
        for s in seeds:
            d[s[0], s[1]] = 0.0
            pq.append((0.0, s))
        heapq.heapify(pq)
        while pq:
            c, (x, y) = heapq.heappop(pq)
            if c > d[x, y]:
                continue
            for (u, w) in radj.get((x, y), ()):
                nc = c + w
                if nc < d[u[0], u[1]]:
                    d[u[0], u[1]] = nc
                    heapq.heappush(pq, (nc, u))
        return d

    @staticmethod
    def _finite_max(d) -> float:
        fin = d[np.isfinite(d)]
        return float(fin.max()) if fin.size else 1.0

    def _field_at(self, d, x, y) -> float:
        ix, iy = self._cell(x, y)
        if not (0 <= ix < _W and 0 <= iy < _H):
            return float("inf")
        v = float(d[ix, iy])
        if np.isfinite(v):
            return v
        # The ninja may be transiently inside an unsafe cell, and the switch/exit
        # CENTRES sit ~15px deep in mines (the agent triggers them from the safe
        # activation-disc boundary). Search out to the kill radius for the nearest
        # finite (safe) cell so goal-proximate cost is robust.
        rng = int(_KILL_RADIUS / _CELL) + 2
        best = float("inf")
        for dx in range(-rng, rng + 1):
            for dy in range(-rng, rng + 1):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < _W and 0 <= jy < _H:
                    w = float(d[jx, jy])
                    if np.isfinite(w) and w < best:
                        best = w
        return best

    def _field_at_interp(self, d, x, y) -> float:
        """Bilinearly-interpolated field value at the CONTINUOUS inner-pixel point
        ``(x, y)``. Unlike the nearest-cell ``_field_at`` (which is a 2px step
        function, so sub-cell motion produces NO cost change and a dead PBRS
        gradient), this varies smoothly with position: any movement toward the goal
        lowers the cost, so PBRS rewards even sub-pixel progress.

        Interpolates over the FINITE corners only, renormalizing the bilinear
        weights so a mine-masked (inf) corner never leaks inf into the threadable
        region — the value at a channel edge cleanly approaches the finite side.
        Falls back to the nearest-finite search when all four corners are inf (e.g.
        transiently inside a mine / at a goal disc centre).
        """
        fx, fy = x / _CELL, y / _CELL
        i0, j0 = int(np.floor(fx)), int(np.floor(fy))
        tx, ty = fx - i0, fy - j0
        num = 0.0
        wsum = 0.0
        for di, wi in ((0, 1.0 - tx), (1, tx)):
            ix = i0 + di
            if not (0 <= ix < _W):
                continue
            for dj, wj in ((0, 1.0 - ty), (1, ty)):
                jy = j0 + dj
                if not (0 <= jy < _H):
                    continue
                w = wi * wj
                if w <= 0.0:
                    continue
                v = float(d[ix, jy])
                if np.isfinite(v):
                    num += w * v
                    wsum += w
        if wsum > 0.0:
            return num / wsum
        # All surrounding corners are inf -> nearest-finite fallback.
        return self._field_at(d, x, y)

    # ---- Public API (mirrors FreeSpaceDistanceSource) ----------------
    def _arrays(self, switch_activated: bool):
        if switch_activated:
            return self._d_exit, self._d_max_exit
        return self._d_switch, self._d_max_switch

    #: The mine-threading flood charges geometric grid steps (with a mine-
    #: clearance multiplier), not frames -- see ``_build``'s ``orth``/``diag``.
    route_cost_unit = "grid"
    route_cost_px_per_unit = float(_CELL)

    def cost_to_go(self, x: float, y: float, switch_activated: bool) -> float:
        if not self.available:
            return float("inf")
        d, dmax = self._arrays(switch_activated)
        # Bilinear (continuous) lookup so PBRS gets a live gradient for arbitrarily
        # small movements — not a 2px staircase.
        c = self._field_at_interp(d, x - _OFFSET, y - _OFFSET)  # world -> inner
        return c if np.isfinite(c) else float(dmax)

    def cost_to_completion(self, x: float, y: float, switch_activated: bool) -> float:
        """Continuous cost-to-completion across switch activation (see free-space)."""
        if switch_activated:
            return self.cost_to_go(x, y, True)
        return self.cost_to_go(x, y, False) + self.exit_cost_at_switch

    def spawn_finite(self, spawn_x: float, spawn_y: float) -> bool:
        if not self.available:
            return False
        return bool(np.isfinite(
            self._field_at(self._d_switch, spawn_x - _OFFSET, spawn_y - _OFFSET)))

    # ------------------------------------------------------------------
    @staticmethod
    def _nearest_finite_cell(d: np.ndarray, cx: int, cy: int,
                             max_r: int = 120) -> Tuple[Optional[int], Optional[int]]:
        """Nearest grid cell to ``(cx, cy)`` whose field value is finite (a ring
        search outward). Used to snap a descent start that lands transiently
        inside a mine / outside the safe region onto the connected field."""
        for r in range(0, max_r):
            x0, x1 = max(0, cx - r), min(_W, cx + r + 1)
            y0, y1 = max(0, cy - r), min(_H, cy + r + 1)
            for ix in range(x0, x1):
                for iy in range(y0, y1):
                    if max(abs(ix - cx), abs(iy - cy)) != r:
                        continue
                    if np.isfinite(d[ix, iy]):
                        return ix, iy
        return None, None

    def _descend(self, d: np.ndarray, start_inner: Tuple[float, float],
                 max_steps: int) -> List[Tuple[float, float]]:
        """Steepest-descent path (8-connected) on a Dijkstra cost field from
        ``start_inner`` (inner pixels) to a goal seed (cost 0). A shortest-path
        distance field has no interior local minima, so greedy descent always
        terminates at the seed; ``visited``/``max_steps`` guard against ties.
        Returns a list of inner-pixel points (cell centres)."""
        ix, iy = self._cell(*start_inner)
        if not (0 <= ix < _W and 0 <= iy < _H and np.isfinite(d[ix, iy])):
            ix, iy = self._nearest_finite_cell(d, ix, iy)
            if ix is None:
                return []
        path = [((ix + 0.5) * _CELL, (iy + 0.5) * _CELL)]
        visited = {(ix, iy)}
        for _ in range(max_steps):
            if d[ix, iy] <= 0.0:
                break
            best, best_d = None, d[ix, iy]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ix + dx, iy + dy
                    if (0 <= nx < _W and 0 <= ny < _H
                            and np.isfinite(d[nx, ny]) and d[nx, ny] < best_d):
                        best_d, best = d[nx, ny], (nx, ny)
            if best is None or best in visited:
                break
            ix, iy = best
            visited.add(best)
            path.append(((ix + 0.5) * _CELL, (iy + 0.5) * _CELL))
        return path

    def centerline(self, spawn_world: Tuple[float, float], *,
                   max_steps: int = 6000) -> List[Tuple[float, float]]:
        """Route centerline as a WORLD-coord polyline spawn -> switch -> exit.

        Traced by steepest descent on the connected cost-to-go fields: leg A
        descends ``_d_switch`` from the spawn, leg B descends ``_d_exit`` from the
        switch. The path lives entirely in the mine-safe region (every cell is
        >= 14px from any deadly mine), so it is the centre of the threadable
        channel — exactly the axis to displace mines away from. Empty if the
        source is unavailable.
        """
        if not self.available:
            return []
        sx, sy = spawn_world[0] - _OFFSET, spawn_world[1] - _OFFSET
        leg_a = self._descend(self._d_switch, (sx, sy), max_steps)
        swx, swy = self.switch[0] - _OFFSET, self.switch[1] - _OFFSET
        leg_b = self._descend(self._d_exit, (swx, swy), max_steps)
        inner = leg_a + leg_b
        return [(x + _OFFSET, y + _OFFSET) for (x, y) in inner]

    _NAV_DIRS = (
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (0.70710678, 0.70710678), (0.70710678, -0.70710678),
        (-0.70710678, 0.70710678), (-0.70710678, -0.70710678),
    )

    def nav_feature(self, x: float, y: float, switch_activated: bool,
                    step_px: float = 12.0) -> np.ndarray:
        if not self.available:
            return np.zeros(4, dtype=np.float32)
        d, dmax = self._arrays(switch_activated)
        x, y = x - _OFFSET, y - _OFFSET  # world -> inner
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
