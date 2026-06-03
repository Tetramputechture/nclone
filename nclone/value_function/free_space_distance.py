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

import heapq
from typing import List, Optional, Tuple

import numpy as np

from ..constants.physics_constants import (
    MAP_TILE_WIDTH, MAP_TILE_HEIGHT, TILE_PIXEL_SIZE,
)

# Sub-tile grid (TILE_PIXEL_SIZE/4 = 6px): fine enough to follow a 1-tile corridor.
_CELL = TILE_PIXEL_SIZE / 4.0
_W = int(MAP_TILE_WIDTH * TILE_PIXEL_SIZE / _CELL) + 2
_H = int(MAP_TILE_HEIGHT * TILE_PIXEL_SIZE / _CELL) + 2
_MINE_KILL = 14.0          # NINJA_RADIUS + TOGGLE_MINE_RADIUS_TOGGLED
_MINE_STATE_DEADLY = 0
_MINE_SOFT_K = 8.0         # mine-density penalty on grid edges


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
        try:
            self._build(list(map_data))
        except Exception:
            self.available = False

    # ------------------------------------------------------------------
    def _build(self, map_data: List[int]) -> None:
        import nclone_native as Nnative
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

        # Tile-free mask: a cell is free if its centre is not inside a solid tile.
        tiles = map_data[184:1150]
        free = np.zeros((_W, _H), bool)
        for ix in range(_W):
            cx = ix * _CELL + _CELL / 2.0
            tx = int(cx // TILE_PIXEL_SIZE)
            for iy in range(_H):
                cy = iy * _CELL + _CELL / 2.0
                ty = int(cy // TILE_PIXEL_SIZE)
                if 0 <= tx < 42 and 0 <= ty < 23:
                    free[ix, iy] = tiles[tx + ty * 42] == 0
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

        self._d_switch = self._dijkstra(switch)
        self._d_exit = self._dijkstra(exit_door)
        self._d_max_switch = self._finite_max(self._d_switch)
        self._d_max_exit = self._finite_max(self._d_exit)
        # Available only if both goals reached some free cells.
        self.available = (np.isfinite(self._d_switch).any()
                          and np.isfinite(self._d_exit).any())

    @staticmethod
    def _finite_max(d: np.ndarray) -> float:
        fin = d[np.isfinite(d)]
        return float(fin.max()) if fin.size else 1.0

    def _cell(self, x: float, y: float) -> Tuple[int, int]:
        return int(x / _CELL), int(y / _CELL)

    def _dijkstra(self, goal: Tuple[float, float]) -> np.ndarray:
        """Mine-soft cost-to-go over the tile-free grid (8-connected)."""
        free, mden = self._free, self._mden
        d = np.full((_W, _H), np.inf, np.float64)
        gx, gy = self._cell(*goal)
        if not (0 <= gx < _W and 0 <= gy < _H and free[gx, gy]):
            gx, gy = self._nearest_free(gx, gy)
            if gx is None:
                return d
        d[gx, gy] = 0.0
        pq = [(0.0, gx, gy)]
        while pq:
            c, x, y = heapq.heappop(pq)
            if c > d[x, y]:
                continue
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < _W and 0 <= ny < _H and free[nx, ny]:
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
        ix, iy = self._cell(x, y)
        if not (0 <= ix < _W and 0 <= iy < _H):
            return float("inf")
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

    def spawn_finite(self, spawn_x: float, spawn_y: float) -> bool:
        if not self.available:
            return False
        return np.isfinite(self._field_at(self._d_switch, spawn_x, spawn_y))

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
