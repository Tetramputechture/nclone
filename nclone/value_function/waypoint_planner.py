"""Lattice-backed subgoal/waypoint extraction for long-horizon RL.

Turns a level's physics-aware lattice ``d*`` route into an ordered list of local
**subgoals** (waypoints), and provides a runtime :class:`WaypointTracker` that
advances through them as the ninja arrives, exposing the *current local
waypoint* and a normalized global cost-to-go.

**Why this exists.** A feedforward policy conditioned on the distant exit
suffers a steep success-rate dropoff as spawn→switch→exit distance grows
(long-horizon credit assignment + exploration). Decomposing the route into
nearby subgoals keeps the per-step learning signal strong and distance-invariant
(each leg is short), turning one long-horizon problem into a sequence of
short-horizon ones (SoRB / HIGL / TTGS). The lattice already computes an exact,
physics-aware route, so we read the subgoals straight off it rather than learning
a distance metric.

**Two consumers, one extractor:**
  1. *Runtime hierarchy* — :class:`WaypointTracker` conditions the policy on the
     current local waypoint and backs a per-leg PBRS term.
  2. *Offline data generation* — dumping ``(level -> waypoint sequence)`` pairs
     over a corpus is the training set for a future **amortized** subgoal
     generator (a learned model predicting these waypoints from the level in a
     cheap forward pass, removing the per-level lattice build at procgen scale).

Building the lattice is expensive (seconds–minutes/level), so this is an
*offline / cached* path; at training scale the lattice is precomputed over the
level pool (or eventually replaced by the amortized generator). See
``docs/LONG_HORIZON_DISTANCE_HANDOFF.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Waypoint:
    """One subgoal along the lattice-optimal route.

    Attributes:
        x, y: Target position (ninja center, pixels).
        tol: Arrival radius (pixels). The tracker advances once the ninja is
            within ``tol``. Objective waypoints use the entity contact range;
            intermediate arc landings use a small fixed tolerance.
        phase: 0 = pre-exit-switch leg, 1 = post-switch leg.
        cost_to_go: Lattice ``d*`` (frames) from this waypoint to the FINAL goal
            (the exit). Monotonically decreases along the route; ~0 at the exit.
            This is a single global metric across both phases (phase-0 uses the
            combined ``d_comb`` tree, so it already counts the switch→exit tail).
        is_objective: True for the exit switch and the exit door (the
            phase-final, must-touch targets); False for intermediate landings.
    """

    x: float
    y: float
    tol: float
    phase: int
    cost_to_go: float
    is_objective: bool
    # Audit II P2-11 executable annotations (defaults keep old callers valid):
    # surface normal at the waypoint's landing (the NEXT arc's launch normal —
    # landing of arc k == launch of arc k+1), and whether the leg INTO this
    # waypoint is an airborne arc (jump/fall) vs a grounded run.
    normal_x: float = 0.0
    normal_y: float = 0.0
    airborne_leg: bool = False


def plan_waypoints(
    map_data: Sequence[int],
    *,
    workers: int = 4,
    use_native: bool = True,
    use_cache: bool = True,
    name: Optional[str] = None,
    momentum_model: str = "legacy",
    slope_jump_modes: Optional[str] = None,
    leg_tol: float = 14.0,
    max_leg_px: Optional[float] = None,
    max_nodes: Optional[int] = None,
) -> Tuple[List[Waypoint], bool]:
    """Convenience wrapper: ``(waypoints, available)`` for one level.

    ``available`` is False (and ``waypoints`` empty) when the lattice could not
    be built or has no route — the caller falls back to whole-level conditioning.
    """
    p = LatticeWaypointPlanner(
        map_data,
        workers=workers,
        use_native=use_native,
        use_cache=use_cache,
        name=name,
        momentum_model=momentum_model,
        slope_jump_modes=slope_jump_modes,
        leg_tol=leg_tol,
        max_leg_px=max_leg_px,
        max_nodes=max_nodes,
    )
    return p.waypoints, p.available


class LatticeWaypointPlanner:
    """Build a lattice-backed ordered subgoal list from a level's ``map_data``.

    Construction is best-effort, mirroring :class:`LatticeDistanceSource`: any
    failure (unsupported entities, enumeration error, unreachable goals) leaves
    ``available == False`` and ``waypoints == []`` so the caller cleanly falls
    back to whole-level (single-goal) conditioning.
    """

    def __init__(
        self,
        map_data: Sequence[int],
        *,
        workers: int = 4,
        use_native: bool = True,
        use_cache: bool = True,
        name: Optional[str] = None,
        momentum_model: str = "legacy",
        slope_jump_modes: Optional[str] = None,
        leg_tol: float = 14.0,
        max_leg_px: Optional[float] = None,
        max_nodes: Optional[int] = None,
    ):
        self.available = False
        self.waypoints: List[Waypoint] = []
        self.switch: Optional[Tuple[float, float]] = None
        self.exit: Optional[Tuple[float, float]] = None
        self.total_cost: float = float("inf")

        try:
            from .tools.lattice_audit import LatticeAudit
            from .edge_risk import (
                DEFAULT_W_RISK,
                DEFAULT_W_LOOP,
                DEFAULT_RISK_CAP,
                DEFAULT_JUMP_TAX,
            )
            from ..constants.physics_constants import (
                NINJA_RADIUS,
                EXIT_SWITCH_RADIUS,
                EXIT_DOOR_RADIUS,
            )

            audit = LatticeAudit(
                map_data=list(map_data),
                name=name,
                workers=workers,
                use_native=use_native,
                w_risk=DEFAULT_W_RISK,
                w_loop=DEFAULT_W_LOOP,
                risk_cap=DEFAULT_RISK_CAP,
                jump_tax=DEFAULT_JUMP_TAX,
                momentum_model=momentum_model,
                slope_jump_modes=slope_jump_modes,
                max_nodes=max_nodes,
            )
            pc = audit.precompute(use_cache=use_cache)
            phase0, phase1, info = audit.build_route(pc)
            self.switch = audit.switch
            self.exit = audit.exit

            wps = self._assemble(
                phase0,
                phase1,
                info,
                leg_tol=leg_tol,
                switch_tol=NINJA_RADIUS + EXIT_SWITCH_RADIUS,
                exit_tol=NINJA_RADIUS + EXIT_DOOR_RADIUS,
                max_leg_px=max_leg_px,
            )
            if wps:
                self.waypoints = wps
                self.total_cost = wps[0].cost_to_go
                self.available = True
        except Exception:
            self.available = False

    def _assemble(
        self,
        phase0,
        phase1,
        info: dict,
        *,
        leg_tol: float,
        switch_tol: float,
        exit_tol: float,
        max_leg_px: Optional[float],
    ) -> List[Waypoint]:
        """Compose the ordered waypoint list from the two-phase arc route.

        Mirrors :meth:`SegmentedDStarExecutor._waypoints`: intermediate arc
        landings become subgoals, and the switch / exit entities are the
        phase-final must-touch targets (with entity contact ranges as tolerance).
        """
        wps: List[Waypoint] = []

        # Phase 0: intermediate landings (drop the last arc; the switch entity is
        # the real phase-0 target). P2-11: annotate each landing with the NEXT
        # arc's launch surface normal (== the normal at this landing) and
        # whether the leg INTO it was airborne (a jump/fall arc).
        for i, arc in enumerate(phase0[:-1] if phase0 else []):
            nxt = phase0[i + 1]
            wps.append(
                Waypoint(
                    float(arc.landing_x),
                    float(arc.landing_y),
                    leg_tol,
                    0,
                    float(arc.cumulative_cost),
                    False,
                    normal_x=float(getattr(nxt, "surface_normal_x", 0.0)),
                    normal_y=float(getattr(nxt, "surface_normal_y", 0.0)),
                    airborne_leg=bool(
                        getattr(arc, "jump_hold_frames", 0)
                        or abs(
                            float(arc.landing_y)
                            - float(getattr(arc, "launch_y", arc.landing_y))
                        ) > 1.0
                    ),
                )
            )

        # Exit switch: cost-to-go from the switch contact = exact d* switch->exit.
        if self.switch is not None:
            sc = float(info.get("spawn_d_exit_from_switch", float("inf")))
            if not math.isfinite(sc):
                # fall back to the first phase-1 arc's cumulative cost if present
                sc = float(phase1[0].cumulative_cost) if phase1 else 0.0
            wps.append(
                Waypoint(
                    float(self.switch[0]),
                    float(self.switch[1]),
                    switch_tol,
                    0,
                    sc,
                    True,
                )
            )

        # Phase 1: intermediate landings toward the exit (same P2-11 annotations).
        for i, arc in enumerate(phase1[:-1] if phase1 else []):
            nxt = phase1[i + 1]
            wps.append(
                Waypoint(
                    float(arc.landing_x),
                    float(arc.landing_y),
                    leg_tol,
                    1,
                    float(arc.cumulative_cost),
                    False,
                    normal_x=float(getattr(nxt, "surface_normal_x", 0.0)),
                    normal_y=float(getattr(nxt, "surface_normal_y", 0.0)),
                    airborne_leg=bool(
                        getattr(arc, "jump_hold_frames", 0)
                        or abs(
                            float(arc.landing_y)
                            - float(getattr(arc, "launch_y", arc.landing_y))
                        ) > 1.0
                    ),
                )
            )

        # Exit: the final goal, cost-to-go 0.
        if self.exit is not None:
            wps.append(
                Waypoint(
                    float(self.exit[0]),
                    float(self.exit[1]),
                    exit_tol,
                    1,
                    0.0,
                    True,
                )
            )

        if max_leg_px is not None and max_leg_px > 0:
            wps = _densify(wps, max_leg_px)
        return wps


def _densify(waypoints: List[Waypoint], max_leg_px: float) -> List[Waypoint]:
    """Split any leg longer than ``max_leg_px`` into evenly spaced intermediate
    subgoals (linear in position and cost-to-go), keeping every leg short so the
    per-step shaping gradient stays strong. Objective waypoints are preserved.
    """
    if len(waypoints) < 2:
        return list(waypoints)
    out: List[Waypoint] = [waypoints[0]]
    for prev, wp in zip(waypoints[:-1], waypoints[1:]):
        d = math.hypot(wp.x - prev.x, wp.y - prev.y)
        n_seg = max(1, math.ceil(d / max_leg_px))  # segments so each leg <= max
        for k in range(1, n_seg):
            t = k / n_seg
            out.append(
                Waypoint(
                    prev.x + t * (wp.x - prev.x),
                    prev.y + t * (wp.y - prev.y),
                    wp.tol,
                    wp.phase,
                    prev.cost_to_go + t * (wp.cost_to_go - prev.cost_to_go),
                    False,
                )
            )
        out.append(wp)
    return out


class FreeSpaceWaypointPlanner:
    """Waypoint chain from the CHEAP free-space cost-to-go field (no lattice).

    Steepest-descends the mine-aware free-space cost-to-go from spawn -> switch
    (phase 0) then switch -> exit (phase 1), sampling a subgoal every ~max_leg_px
    of path length. The free-space source is disk-cached (~ms), so this builds on
    reset at procgen scale without the lattice's seconds-per-level cost. The
    waypoints FOLLOW the winding corridor (a Dijkstra cost field has no spurious
    interior local minima, so steepest descent reaches the goal), so on a loop
    they lead the agent AWAY from the adjacent exit toward the far switch first --
    exactly what the single far gradient fails to do. Best-effort: any failure
    leaves ``available == False`` so the caller falls back to whole-level shaping.
    """

    def __init__(
        self,
        map_data: Sequence[int],
        spawn: Tuple[float, float],
        *,
        name: Optional[str] = None,
        leg_tol: float = 14.0,
        max_leg_px: Optional[float] = 48.0,
    ):
        self.available = False
        self.waypoints: List[Waypoint] = []
        self.switch: Optional[Tuple[float, float]] = None
        self.exit: Optional[Tuple[float, float]] = None
        self.total_cost: float = float("inf")
        # Descent cell paths (free cells, corridor-following) — kept for validation
        # / visualization. Every cell is free and no step cuts a wall corner.
        self.path_cells0: List[Tuple[int, int]] = []
        self.path_cells1: List[Tuple[int, int]] = []
        try:
            from .free_space_distance import (
                FreeSpaceDistanceSource, _CELL, _W, _H,
            )
            from ..constants.physics_constants import (
                NINJA_RADIUS, EXIT_SWITCH_RADIUS, EXIT_DOOR_RADIUS,
            )

            src = FreeSpaceDistanceSource(list(map_data), name=name)
            if not src.available or src.switch is None or src.exit is None:
                return
            self._src, self._CELL, self._W, self._H = src, _CELL, _W, _H
            self.switch, self.exit = src.switch, src.exit

            cells0 = self._descend(src._d_switch, spawn, src.switch)
            cells1 = self._descend(src._d_exit, src.switch, src.exit)
            self.path_cells0, self.path_cells1 = cells0, cells1
            wps = self._assemble(
                cells0, cells1, src,
                leg_tol=leg_tol,
                switch_tol=NINJA_RADIUS + EXIT_SWITCH_RADIUS,
                exit_tol=NINJA_RADIUS + EXIT_DOOR_RADIUS,
                max_leg_px=(max_leg_px or 48.0),
            )
            if wps:
                self.waypoints = wps
                self.total_cost = wps[0].cost_to_go
                self.available = True
        except Exception:
            self.available = False

    def _cell(self, x: float, y: float) -> Tuple[int, int]:
        return (min(max(int(x / self._CELL), 0), self._W - 1),
                min(max(int(y / self._CELL), 0), self._H - 1))

    def _start_cell(self, field, px: Tuple[float, float]) -> Tuple[int, int]:
        """Nearest free cell with a FINITE field value. A surface-mounted spawn /
        switch sits in a solid (or boundary) cell whose field is inf even though
        the connected finite region is one ring out — descending from there would
        die immediately, so snap to the nearest reachable cell."""
        import numpy as np
        c = self._cell(px[0], px[1])
        free = self._src._free
        if free[c[0], c[1]] and np.isfinite(field[c[0], c[1]]):
            return c
        for r in range(1, 14):
            best, bd = None, 1e18
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = c[0] + dx, c[1] + dy
                    if (0 <= nx < self._W and 0 <= ny < self._H
                            and free[nx, ny] and np.isfinite(field[nx, ny])):
                        q = dx * dx + dy * dy
                        if q < bd:
                            bd, best = q, (nx, ny)
            if best is not None:
                return best
        return c

    def _descend(self, field, start_px, goal_px) -> List[Tuple[int, int]]:
        """8-connected steepest descent on ``field`` from spawn cell to goal."""
        free = self._src._free
        W, H = self._W, self._H
        cur = self._start_cell(field, start_px)
        goal = self._cell(goal_px[0], goal_px[1])
        path = [cur]
        visited = {cur}
        for _ in range(W * H):
            cx, cy = cur
            cv = field[cx, cy]
            if not math.isfinite(cv) or cv <= 0.0 or cur == goal:
                break
            best, bestv = None, cv
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if not (0 <= nx < W and 0 <= ny < H and free[nx, ny]):
                        continue
                    # No diagonal corner-cutting through a wall: a diagonal step is
                    # valid only if BOTH orthogonally-adjacent cells are also free
                    # (a ninja-sized body can't pass a solid corner). Matches the
                    # free-space Dijkstra so the path can never clip solid geometry.
                    if dx and dy and not (free[cx + dx, cy] and free[cx, cy + dy]):
                        continue
                    v = field[nx, ny]
                    if math.isfinite(v) and v < bestv and (nx, ny) not in visited:
                        bestv, best = v, (nx, ny)
            if best is None:
                break
            cur = best
            visited.add(cur)
            path.append(cur)
        return path

    def _seg_free(self, a, b, step: float = 3.0) -> bool:
        """True iff the straight segment a->b stays in free cells (sampled). Used
        so each waypoint leg is a LINE-OF-SIGHT free-space segment (the per-leg PBRS
        gradient toward the next waypoint then never points across a wall corner)."""
        free = self._src._free
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(d / step))
        for k in range(n + 1):
            t = k / n
            ix = int((a[0] + t * (b[0] - a[0])) / self._CELL)
            iy = int((a[1] + t * (b[1] - a[1])) / self._CELL)
            if not (0 <= ix < self._W and 0 <= iy < self._H and free[ix, iy]):
                return False
        return True

    def _sample(self, cells: List[Tuple[int, int]], max_leg_px: float):
        """Line-of-sight (string-pulling) subgoals along the corridor path: commit a
        waypoint at the FURTHEST cell still in straight free-space view of the last
        anchor (and within max_leg_px). Every leg is then a straight free segment —
        no corner-clipping — so the per-leg PBRS gradient always points through open
        corridor. Endpoints are dropped (the objectives are added separately)."""
        if len(cells) < 3:
            return []
        pts = [((c[0] + 0.5) * self._CELL, (c[1] + 0.5) * self._CELL) for c in cells]
        out: List[Tuple[float, float]] = []
        anchor = pts[0]
        i = 1
        n = len(pts)
        while i < n - 1:
            # Extend as far as line-of-sight AND the leg cap allow.
            j = i
            while (j + 1 < n - 1
                   and math.hypot(pts[j + 1][0] - anchor[0], pts[j + 1][1] - anchor[1]) <= max_leg_px
                   and self._seg_free(anchor, pts[j + 1])):
                j += 1
            out.append(pts[j])
            anchor = pts[j]
            i = j + 1
        return out

    def _assemble(self, cells0, cells1, src, *, leg_tol, switch_tol, exit_tol, max_leg_px):
        wps: List[Waypoint] = []
        # Phase 0: spawn -> switch intermediate landings (cost-to-completion is the
        # global metric, monotone-decreasing along the descent).
        for px, py in self._sample(cells0, max_leg_px):
            c = src.cost_to_completion(px, py, False)
            if math.isfinite(c):
                wps.append(Waypoint(px, py, leg_tol, 0, float(c), False))
        # Exit switch objective: cost-to-go from the switch = the phase-1 tail.
        wps.append(Waypoint(float(src.switch[0]), float(src.switch[1]), switch_tol,
                            0, float(src.exit_cost_at_switch), True))
        # Phase 1: switch -> exit intermediate landings.
        for px, py in self._sample(cells1, max_leg_px):
            c = src.cost_to_completion(px, py, True)
            if math.isfinite(c):
                wps.append(Waypoint(px, py, leg_tol, 1, float(c), False))
        # Exit objective: the final goal, cost-to-go 0.
        wps.append(Waypoint(float(src.exit[0]), float(src.exit[1]), exit_tol, 1, 0.0, True))
        return wps


class WaypointTracker:
    """Runtime cursor over a precomputed waypoint list.

    Advances to the next subgoal as the ninja arrives, fast-forwards past the
    pre-switch leg once the switch is activated, and exposes a 4-dim navigation
    hint matching the existing lattice ``nav_feature`` contract
    (``[dx_hat, dy_hat, cost_to_go_norm, phase]``) so it is a drop-in policy
    observation. Pure bookkeeping — no physics, no lattice — so it is cheap and
    deterministic on the hot path.
    """

    def __init__(
        self,
        waypoints: Sequence[Waypoint],
        total_cost: Optional[float] = None,
    ):
        self.waypoints: List[Waypoint] = list(waypoints)
        # Normalizer for cost_to_go_norm: total route cost (first waypoint's
        # cost-to-go). Kept fixed per level so the norm is route-length aware.
        if total_cost is not None and total_cost > 0:
            self._total = float(total_cost)
        elif self.waypoints and self.waypoints[0].cost_to_go > 0:
            self._total = float(self.waypoints[0].cost_to_go)
        else:
            self._total = 1.0
        self._i = 0

    def reset(self) -> None:
        self._i = 0

    @property
    def index(self) -> int:
        return self._i

    @property
    def done(self) -> bool:
        return self._i >= len(self.waypoints)

    def current(self) -> Optional[Waypoint]:
        if not self.waypoints or self.done:
            return None
        return self.waypoints[self._i]

    def update(self, x: float, y: float, switch_activated: bool) -> Optional[Waypoint]:
        """Advance the cursor given the ninja position + switch state.

        Rules:
          * Once the switch is active, skip any remaining phase-0 waypoints
            (the switch leg is over regardless of position).
          * Advance through any intermediate waypoint within its ``tol``, but
            never past the final waypoint.
          * Do NOT auto-skip the phase-0 switch *objective* on proximity alone —
            wait for ``switch_activated`` (being near the switch is not touching
            it). The exit objective is never auto-skipped (it is the last entry).
        """
        if not self.waypoints:
            return None

        if switch_activated:
            while self._i < len(self.waypoints) and self.waypoints[self._i].phase == 0:
                self._i += 1
            if self._i >= len(self.waypoints):
                self._i = len(self.waypoints) - 1

        while self._i < len(self.waypoints) - 1:
            wp = self.waypoints[self._i]
            if wp.is_objective and wp.phase == 0 and not switch_activated:
                break  # must actually hit the switch; gated on switch_activated
            # Robust monotonic advancement: advance when the ninja is within tol of
            # this waypoint OR has clearly PASSED it (closer to the next waypoint
            # along the route). Strict within-tol-only stalls the cursor whenever the
            # agent's valid trajectory deviates >tol from a densified waypoint, which
            # silently kills the per-leg dense reward on the long return leg. The
            # "closer to next" carrot keeps the cursor (and the dense gradient)
            # tracking forward route progress without allowing a loop shortcut
            # (advancement is still strictly in route order, one waypoint at a time).
            nxt = self.waypoints[self._i + 1]
            d_cur = math.hypot(x - wp.x, y - wp.y)
            d_nxt = math.hypot(x - nxt.x, y - nxt.y)
            if d_cur <= wp.tol or d_nxt < d_cur:
                self._i += 1
            else:
                break
        return self.current()

    def leg_distance(self, x: float, y: float) -> float:
        """Euclidean pixels from ``(x, y)`` to the current waypoint (``inf`` if
        done). Use this to back a per-leg PBRS term (short, bounded leg → strong,
        distance-invariant per-step gradient)."""
        wp = self.current()
        if wp is None:
            return float("inf")
        return math.hypot(x - wp.x, y - wp.y)

    def nav_feature(
        self, x: float, y: float, switch_activated: bool
    ) -> np.ndarray:
        """9-dim policy hint (Audit II P2-11)
        ``[dx_hat, dy_hat, cost_to_go_norm, phase, leg_len, tol, airborne_leg,
        normal_x, normal_y]``.

        Dims 0-3 are the legacy hint: ``(dx_hat, dy_hat)`` is the unit
        direction to the current LOCAL waypoint (not the distant goal),
        ``cost_to_go_norm = wp.cost_to_go / total`` in ``[0, 1]`` is the global
        remaining-cost fraction, ``phase`` is the current leg phase. Dims 4-8
        are the EXECUTABLE annotations the planner previously discarded:
        ``leg_len`` = distance to the waypoint / 192 px (clipped), ``tol`` =
        arrival radius / 24 px (clipped — small for precision landings, larger
        for entity contacts), ``airborne_leg`` = 1 when the leg is a jump/fall
        arc, and the landing surface normal. All zeros when no plan/route
        remains, so the feature is a safe no-op on lattice-less levels.
        Advances the cursor as a side effect (idempotent within a frame for a
        fixed ``(x, y)``)."""
        self.update(x, y, switch_activated)
        wp = self.current()
        if wp is None:
            return np.zeros(9, dtype=np.float32)
        dx, dy = wp.x - x, wp.y - y
        dist = math.hypot(dx, dy)
        if dist > 1e-6:
            ux, uy = dx / dist, dy / dist
        else:
            ux, uy = 0.0, 0.0
        ctg = min(max(wp.cost_to_go / self._total, 0.0), 1.0)
        phase = 1.0 if (switch_activated or wp.phase == 1) else 0.0
        leg_len = min(dist / 192.0, 1.0)
        tol = min(max(float(wp.tol), 0.0) / 24.0, 1.0)
        return np.array(
            [
                ux, uy, ctg, phase,
                leg_len, tol,
                1.0 if getattr(wp, "airborne_leg", False) else 0.0,
                float(getattr(wp, "normal_x", 0.0)),
                float(getattr(wp, "normal_y", 0.0)),
            ],
            dtype=np.float32,
        )
