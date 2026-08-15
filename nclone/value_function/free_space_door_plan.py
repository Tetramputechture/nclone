"""Locked-door dependency legs computed on the FLOOD instead of the lattice.

Why this exists
---------------
``door_dependency.solve_door_dependencies`` resolves which locked doors must be
opened, in what order, and produces a cost-to-go field per leg.  Its backend was
the surface lattice, enumerating physics edges once per door configuration.

Two things make that backend the wrong one now:

* The lattice is being demoted.  Its seeds were measured to add ~2% to the
  potential and rescue nothing on >=90% of levels, for 44x the precompute
  (``tools/seed_ablation.py``), so nothing downstream should acquire a new
  dependency on it.
* Per-configuration ARC ENUMERATION is the expensive half of the lattice, and a
  door plan needs one configuration per leg.  On the flood a configuration costs
  ~0.07s; the brief's own §6 notes the lattice can cost 380-480s on a bad level,
  and that is *per configuration*.

The dependency algorithm itself was never graph-specific — only four primitives
were.  Those now live behind ``door_dependency.SurfaceGraphOracle``, and this
module supplies the flood equivalent, so both backends run the identical solver.

What the legs are for
---------------------
The agent's phase = how many necessary doors it has opened.  While phase < k it
descends ``door_leg_dstar[phase]`` toward the next required switch; then the
level's normal switch->exit potential takes over.  Without this, the flood is
door-blind: measured over all 808 official levels carrying a locked door,
648 have free space that is priced finitely only by routing through a shut door,
and 205 cannot reach the exit switch at all with the doors shut
(``tools/door_audit.py``).

Reachability, and why it is not the cost field
----------------------------------------------
The solver asks "can the ninja get from the spawn to this switch", which is
FORWARD reachability.  The cost fields are BACKWARD floods (cost from a cell to
a goal), and on a grid whose edges are directional — one-way platforms — those
are different questions.  So reachability is its own forward pass rather than a
finiteness test on a backward field, and it applies exactly the edge rules the
physics flood applies: the corner squeeze, the blocked-segment test in the
travelled direction, and the up-gate that refuses a rise no jump can clear.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from .free_space_distance import (
    FreeSpaceDistanceSource,
    _CELL,
    _W,
    _H,
    _physics_cost_constants,
)
from ..constants.physics_constants import (
    NINJA_RADIUS, EXIT_SWITCH_RADIUS, EXIT_DOOR_RADIUS, LOCKED_DOOR_SWITCH_RADIUS,
)

#: Contact radius per goal kind, matching what the flood seeds with.
_CONTACT = {
    "switch": NINJA_RADIUS + EXIT_SWITCH_RADIUS,
    "exit": NINJA_RADIUS + EXIT_DOOR_RADIUS,
    "door_switch": NINJA_RADIUS + LOCKED_DOOR_SWITCH_RADIUS,
}


def _forward_arcs(src: FreeSpaceDistanceSource):
    """``(start, landings)`` CSR of wall-launch arcs keyed by LAUNCH cell.

    ``_build_arc_edges`` stores the inverse — keyed by landing — because a
    BACKWARD Dijkstra relaxes predecessors.  A forward BFS needs the other
    direction, so this re-inverts it once per configuration.

    These edges are the whole reason the reachability answer changes: the 8
    connected steps refuse any rise a floor jump cannot clear, and an arc is
    exactly a wall launch that clears one.  Without them the BFS declares a
    chamber unreachable that the ninja wall-jumps out of, and the door solver
    then reports the level unsolvable — which it did on 9 levels that have a
    verified winning demo.
    """
    start, s = src._arc_rev_start, src._arc_rev_src
    if start is None or s is None or len(s) == 0:
        return None
    n_cells = _W * _H
    # Landing cell for each stored edge, recovered from the CSR row lengths.
    landings = np.repeat(np.arange(n_cells, dtype=np.int64), np.diff(start))
    order = np.argsort(s, kind="stable")
    launches, landings = np.asarray(s)[order], landings[order]
    fstart = np.searchsorted(launches, np.arange(n_cells + 1))
    return fstart, landings


def forward_reachable(src: FreeSpaceDistanceSource,
                      x: float, y: float) -> Optional[np.ndarray]:
    """``(_W, _H)`` bool mask of cells reachable FORWARD from ``(x, y)``.

    A plain BFS, not a Dijkstra: the solver only asks yes/no, and the cost model
    cannot change the reachable SET (every edge it prices is finite).  Uses the
    same three gates ``_relax_physics`` uses on a move ``p -> n``, in the
    travelled direction:

    * the corner squeeze — a diagonal needs both orthogonal neighbours free;
    * ``_step_blocked(p, direction)`` — solid segments, shut doors, one-ways;
    * the up-gate — a rising move is legal only into a wall-adjacent cell or one
      within a floor jump's reach of the floor below it;
    * plus the wall-launch ARC edges, which are precisely the moves that bypass
      that up-gate legitimately (see :func:`_forward_arcs`).

    Returns ``None`` if the source is off-grid or the level has no free mask.
    """
    free = src._free
    if free is None:
        return None
    sx, sy = int(x / _CELL), int(y / _CELL)
    if not (0 <= sx < _W and 0 <= sy < _H):
        return None
    if not free[sx, sy]:
        start = _nearest_free(free, sx, sy)
        if start is None:
            return None
        sx, sy = start

    physics = bool(src._physics_cost)
    rise, wall = src._floor_rise, src._wall_adj
    reach_cells = _physics_cost_constants()["jump_reach_cells"] if physics else 0.0

    arcs = _forward_arcs(src) if physics else None

    seen = np.zeros((_W, _H), bool)
    seen[sx, sy] = True
    q: deque = deque([(sx, sy)])
    while q:
        px, py = q.popleft()
        if arcs is not None:
            # Wall-launch arcs: the one move that legitimately bypasses the
            # up-gate below. Skipping these is what made the solver call
            # demonstrably winnable levels unsolvable.
            fstart, landings = arcs
            flat = px * _H + py
            for k in range(fstart[flat], fstart[flat + 1]):
                d = int(landings[k])
                ax, ay = divmod(d, _H)
                if not seen[ax, ay] and free[ax, ay]:
                    seen[ax, ay] = True
                    q.append((ax, ay))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = px + dx, py + dy
                if not (0 <= nx < _W and 0 <= ny < _H) or seen[nx, ny]:
                    continue
                if not free[nx, ny]:
                    continue
                if dx and dy and not (free[px + dx, py] and free[px, py + dy]):
                    continue
                if src._step_blocked(px, py, dx, dy):
                    continue
                if physics and dy < 0 and rise is not None:
                    # Rising: the flood refuses to price a climb no jump can
                    # make, and reachability must agree or the plan will claim
                    # switches the potential can never lead the agent to.
                    if not (wall[nx, ny] or rise[nx, ny] <= reach_cells):
                        continue
                seen[nx, ny] = True
                q.append((nx, ny))
    return seen


def _nearest_free(free: np.ndarray, cx: int, cy: int, r: int = 3):
    """Nearest free cell within ``r``; goals and spawns can sit on geometry."""
    best, best_d2 = None, None
    for ix in range(max(0, cx - r), min(_W, cx + r + 1)):
        for iy in range(max(0, cy - r), min(_H, cy + r + 1)):
            if not free[ix, iy]:
                continue
            d2 = (ix - cx) ** 2 + (iy - cy) ** 2
            if best_d2 is None or d2 < best_d2:
                best, best_d2 = (ix, iy), d2
    return best


def _touches(reach: np.ndarray, free: np.ndarray, x: float, y: float,
             radius: float) -> bool:
    """Is any reachable free cell inside the goal's activation disc?

    Mirrors ``_seed_activation_disc``: contact is by disc overlap, not by
    landing on the entity's own cell, and an entity recessed behind geometry may
    own no free cell of its own at all — hence the nearest-free fallback.
    """
    r2 = radius * radius
    rc = int(radius / _CELL) + 1
    cx0, cy0 = int(x / _CELL), int(y / _CELL)
    hit_any_free = False
    for ix in range(max(0, cx0 - rc), min(_W, cx0 + rc + 1)):
        px = ix * _CELL + _CELL / 2.0
        for iy in range(max(0, cy0 - rc), min(_H, cy0 + rc + 1)):
            if not free[ix, iy]:
                continue
            py = iy * _CELL + _CELL / 2.0
            if (px - x) ** 2 + (py - y) ** 2 <= r2:
                hit_any_free = True
                if reach[ix, iy]:
                    return True
    if hit_any_free:
        return False
    nf = _nearest_free(free, cx0, cy0)
    return bool(nf is not None and reach[nf[0], nf[1]])


class FreeSpaceDoorOracle:
    """Flood backend for :func:`door_dependency.solve_door_dependencies`.

    One ``FreeSpaceDistanceSource`` per door configuration, cached — the same
    shape as the lattice backend's per-config graph cache, three orders of
    magnitude cheaper.  ``kind`` is threaded through to pick the right contact
    radius; a locked-door switch is 5px where the exit switch is 6px, and using
    the wrong one moves the contact boundary by a cell.
    """

    def __init__(self, map_data, *, mine_hard: bool = True,
                 physics_cost: bool = True, one_way: Optional[bool] = None,
                 mine_split_topology: bool = False,
                 name: Optional[str] = None, reach_on_soft_mask: bool = False):
        self._map_data = list(map_data)
        self._kw = dict(mine_hard=mine_hard, physics_cost=physics_cost,
                        one_way=one_way,
                        mine_split_topology=mine_split_topology,
                        name=name)
        # Reachability asks an OCCUPANCY question, so it SHOULD run on the
        # undilated mask -- but doing that alone is a net regression and the
        # option is off by default. See ``_reach_config`` for the measurement.
        self._reach_soft = bool(reach_on_soft_mask and mine_hard)
        self._cache: Dict[frozenset, FreeSpaceDistanceSource] = {}
        self._soft_cache: Dict[frozenset, FreeSpaceDistanceSource] = {}
        self._reach: Dict[Tuple[frozenset, int, int], np.ndarray] = {}

    def config(self, open_set) -> FreeSpaceDistanceSource:
        key = frozenset(int(i) for i in open_set)
        src = self._cache.get(key)
        if src is None:
            src = FreeSpaceDistanceSource(
                self._map_data, doors=key, **self._kw)
            self._cache[key] = src
        return src

    def _reach_config(self, cfg: FreeSpaceDistanceSource) -> FreeSpaceDistanceSource:
        """The source whose mask answers "can the ninja BE here", not "may a
        route relax through here".

        The mine-hard mask cuts every cell whose CENTRE is within
        ``_MINE_KILL + _CELL`` = 20px of a deadly mine — the true 14px kill
        radius dilated by a whole cell so no part of a 6px cell can be lethal.
        That dilation is right for ROUTING and wrong for OCCUPANCY, and
        conflating the two is a defect this project has already paid for once
        (it erased a survivable spawn standing 21.63px from two mines).

        At corpus scale it is not an edge case: **183 of 327 verified winning
        routes** spend time in cells the dilation erased — 7,418 frames — and
        all four levels the door solver still wrongly called unsolvable failed
        for exactly this reason, 100% mine-cut and 0% tile-cut. Asking a
        reachability question on the routing mask makes the solver deny routes
        the game demonstrably allows.

        Running reachability on the mine-SOFT mask fixes the solver and makes
        the whole thing WORSE, which is why this is off by default. Measured
        over all 808 locked-door levels:

            unsolvable "exit unreachable"      23 -> 3
            "closed field unusable"             3 -> 24
            "exit unreachable in final config"  7 -> 14
            door-aware                        734 -> 726
            build p50                        0.41s -> 1.03s

        The solver stops rejecting levels and the DOWNSTREAM checks reject them
        instead, because those run on the hard mask and the two masks now
        disagree about the same level. Splitting the question across two masks
        moves the contradiction rather than resolving it.

        The real fix is to stop the hard mask over-cutting at all: cut at the
        true 14px kill radius for OCCUPANCY and keep the 20px dilation in the
        EDGE test for ROUTING -- the same split that already works for doors and
        one-ways (geometry in edges, occupancy in nodes). That is a change to the
        mask every consumer shares, so it needs its own measurement and A/B;
        this flag exists to keep the evidence reproducible until then.
        """
        if not self._reach_soft:
            return cfg
        key = frozenset(cfg._doors) if not isinstance(cfg._doors, str) else cfg._doors
        soft = self._soft_cache.get(key)
        if soft is None:
            kw = dict(self._kw)
            kw["mine_hard"] = False
            try:
                soft = FreeSpaceDistanceSource(self._map_data, doors=key, **kw)
            except Exception:
                soft = None
            if soft is None or not soft.available or soft._free is None:
                soft = cfg  # degrade to the hard mask rather than lose the plan
            self._soft_cache[key] = soft
        return soft

    def reachable_from(self, cfg: FreeSpaceDistanceSource, x: float, y: float):
        # Keyed by cell, not by float position: the solver asks from the same
        # spawn every time, and a BFS over ~15k cells is not free.
        key = (frozenset(cfg._doors) if not isinstance(cfg._doors, str) else cfg._doors,
               int(x / _CELL), int(y / _CELL))
        r = self._reach.get(key)
        if r is None:
            r = forward_reachable(self._reach_config(cfg), x, y)
            if r is None:
                r = np.zeros((_W, _H), bool)
            self._reach[key] = r
        return r

    def reaches(self, cfg: FreeSpaceDistanceSource, reach, x: float, y: float,
                kind: str) -> bool:
        # The free mask here must be the SAME one the reach mask was computed
        # on, or a goal sitting in a mine-dilated cell would be tested against
        # cells that mask calls solid and always come back unreachable.
        rc = self._reach_config(cfg)
        if rc._free is None:
            return False
        return _touches(
            reach, rc._free, x, y, _CONTACT.get(kind, _CONTACT["switch"])
        )

    def leg_field(self, cfg: FreeSpaceDistanceSource, x: float, y: float,
                  kind: str) -> np.ndarray:
        radius = _CONTACT.get(kind, _CONTACT["switch"])
        flood = cfg._dijkstra_physics if cfg._physics_cost else cfg._dijkstra
        return flood((float(x), float(y)), radius)


def solve(map_data, *, mine_hard: bool = True, physics_cost: bool = True,
          one_way: Optional[bool] = None,
          mine_split_topology: bool = False,
          name: Optional[str] = None):
    """``(DoorPlan, FreeSpaceDoorOracle)`` for a level, or ``(None, None)``.

    ``None`` when the level has no locked doors, no goals, or will not build —
    all of which mean "use the ordinary single-phase field", not "error".
    """
    from .door_dependency import DoorDescriptor, solve_door_dependencies

    oracle = FreeSpaceDoorOracle(map_data, mine_hard=mine_hard,
                                 physics_cost=physics_cost, one_way=one_way,
                                 mine_split_topology=mine_split_topology,
                                 name=name)
    base = oracle.config(frozenset())
    if not base.available or not base.locked_doors:
        return None, None
    try:
        import nclone_native
        sim = nclone_native.Simulator.load([int(v) for v in map_data])
        spawn = sim.get_ninja_pos()
    except Exception:
        return None, None

    doors = [DoorDescriptor(switch_x=sx, switch_y=sy)
             for (_dx, _dy, _o, sx, sy) in base.locked_doors]
    plan = solve_door_dependencies(
        None, doors,
        float(spawn[0]), float(spawn[1]),
        float(base.switch[0]), float(base.switch[1]),
        float(base.exit[0]), float(base.exit[1]),
        oracle=oracle,
    )
    return plan, oracle
