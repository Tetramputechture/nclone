"""Lattice cost-to-go as the reward's distance signal (physics-aware potential).

The HWM reward shaping consumes a per-step *distance to switch* (phase 0) and
*distance to exit* (phase 1).  Historically those come from a runtime **geometric
A*** over the tile adjacency graph — the "misleading 6px potential" (memory
``reward-potential-not-mechanism``): it ignores momentum, walls-as-launch-points,
and which jumps are physically makeable.

This module provides the drop-in replacement: the **lattice ``d*``** (backward
Dijkstra over the velocity-augmented surface graph, risk-reweighted for
completion>loops>speed).  Where ``d*`` is finite (~90% of supported levels) the
agent gets a physics-aware cost-to-go in *frames*; the caller falls back to the
geometric A* only when the lattice has no plan (robustness on the ~10% no-route
levels — see ``LatticeDistanceSource.spawn_finite``).

Design notes that make it safe for the hot training path:
  * **Velocity-agnostic lookup.** The env caches reachability features by ninja
    *cell* (24px) — not velocity — so the distance must depend only on position
    (+phase).  We take the ``min`` ``d*`` over all velocity bins at the nearest
    surface point: the best-case cost-to-go reachable from here, a valid (lower
    bound) potential that is stable within a cell.
  * **Phase mapping.** Phase 0 uses ``d_comb`` (jointly optimal cost
    *node→switch→exit*); phase 1 uses ``d_ex`` (cost *node→exit*).  Both decrease
    monotonically along the optimal route and are normalised per-phase by HWM, so
    the (large) absolute scale is irrelevant — only the gradient matters.
  * **No mid-episode scale jumps.** A transient ``inf`` (ninja drifted to a
    disconnected node) is clamped to the per-level max finite ``d*`` rather than
    handing off to the geometric scale; the lattice-vs-geometric choice is made
    once per level by the caller.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .surface_graph import (
    SurfaceGraph,
    find_nearest_point,
    node_id,
    num_velocity_bins,
)
from .surface_extraction import SURFACE_FLOOR, SURFACE_WALL


def _finite_max(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)]
    return float(finite.max()) if finite.size else 1.0


class LatticeDistanceSource:
    """Per-level holder of the lattice ``d*`` exposing a position→cost-to-go map.

    Construction is best-effort: any failure (unsupported entities, enumeration
    error, unreachable goals) leaves ``available == False`` so the caller cleanly
    falls back to the geometric A* — the level still trains, just without the
    physics-aware potential.
    """

    def __init__(
        self,
        map_data: List[int],
        *,
        w_risk: Optional[float] = None,
        w_loop: Optional[float] = None,
        risk_cap: Optional[float] = None,
        jump_tax: Optional[float] = None,
        momentum_model: str = "legacy",
        use_native: bool = True,
        use_cache: bool = True,
        name: Optional[str] = None,
        workers: int = 4,
    ):
        self.available = False
        self._graph: Optional[SurfaceGraph] = None
        self._d_comb: Optional[np.ndarray] = None
        self._d_ex: Optional[np.ndarray] = None
        self._d_max_comb = 1.0
        self._d_max_ex = 1.0
        self.switch: Optional[Tuple[float, float]] = None
        self.exit: Optional[Tuple[float, float]] = None
        try:
            from .tools.lattice_audit import LatticeAudit
            from .edge_risk import (
                DEFAULT_W_RISK, DEFAULT_W_LOOP, DEFAULT_RISK_CAP,
                DEFAULT_JUMP_TAX,
            )
            audit = LatticeAudit(
                map_data=list(map_data), name=name, workers=workers,
                use_native=use_native,
                w_risk=DEFAULT_W_RISK if w_risk is None else float(w_risk),
                w_loop=DEFAULT_W_LOOP if w_loop is None else float(w_loop),
                risk_cap=DEFAULT_RISK_CAP if risk_cap is None else float(risk_cap),
                jump_tax=DEFAULT_JUMP_TAX if jump_tax is None else float(jump_tax),
                momentum_model=momentum_model,
            )
            pc = audit.precompute(use_cache=use_cache)
            self._graph = pc["graph"]
            self._d_comb = pc["d_comb"]
            self._d_ex = pc["d_ex"]
            self._d_max_comb = _finite_max(self._d_comb)
            self._d_max_ex = _finite_max(self._d_ex)
            self.switch = audit.switch
            self.exit = audit.exit
            self.available = True
        except Exception:
            self.available = False

    # ------------------------------------------------------------------
    def _min_dstar_at(self, x: float, y: float, d_star: np.ndarray) -> Optional[float]:
        """Min finite ``d*`` over all velocity bins at the nearest floor/wall
        point to ``(x, y)``.  ``None`` if no surface point is found at all;
        ``inf`` if a point is found but every bin is unreachable."""
        g = self._graph
        best = float("inf")
        found_point = False
        for stype in (SURFACE_FLOOR, SURFACE_WALL):
            pt = find_nearest_point(g, x, y, stype)
            if pt is None:
                continue
            found_point = True
            for vb in range(num_velocity_bins(g.surface_points[pt].surface_type)):
                nid = node_id(pt, vb, g.max_vel_bins)
                c = float(d_star[nid])
                if np.isfinite(c) and c < best:
                    best = c
        if not found_point:
            return None
        return best  # may be inf if all bins unreachable

    def _phase_arrays(self, switch_activated: bool) -> Tuple[np.ndarray, float]:
        if switch_activated:
            return self._d_ex, self._d_max_ex
        return self._d_comb, self._d_max_comb

    def cost_to_go(self, x: float, y: float, switch_activated: bool) -> float:
        """Phase-aware lattice cost-to-go (frames) at ``(x, y)``.

        ``inf`` only when no surface point exists near ``(x, y)``; a transient
        unreachable lookup is clamped to the per-level max finite ``d*`` to avoid
        a scale discontinuity within an episode.
        """
        if not self.available:
            return float("inf")
        d_star, d_max = self._phase_arrays(switch_activated)
        c = self._min_dstar_at(x, y, d_star)
        if c is None:
            return float("inf")
        if not np.isfinite(c):
            return float(d_max)
        return float(c)

    def spawn_finite(self, spawn_x: float, spawn_y: float) -> bool:
        """True iff the spawn has a finite phase-0 (``d_comb``) cost-to-go — the
        per-level gate for using the lattice potential this episode."""
        if not self.available:
            return False
        c = self._min_dstar_at(spawn_x, spawn_y, self._d_comb)
        return c is not None and np.isfinite(c)

    # 8-neighbour unit directions (axis + diagonal) for next-hop sampling.
    _NAV_DIRS = (
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (0.70710678, 0.70710678), (0.70710678, -0.70710678),
        (-0.70710678, 0.70710678), (-0.70710678, -0.70710678),
    )

    def nav_feature(
        self, x: float, y: float, switch_activated: bool, step_px: float = 24.0,
    ) -> np.ndarray:
        """4-dim physics-aware navigation hint for the POLICY observation:
        ``[dx_hop, dy_hop, cost_to_go_norm, phase]``.

        ``(dx_hop, dy_hop)`` is the unit direction of steepest ``d*`` descent —
        sampling the 8 neighbours one tile away and picking the one that most
        reduces the lattice cost-to-go (the direction the physics-aware plan
        wants to go, unlike the geometric Euclidean-to-goal that can point into a
        wall).  ``cost_to_go_norm = d*/d_max`` in ``[0, 1]``.  ``phase`` = 1 once
        the switch is collected.  All-zeros when no finite plan exists here, so
        the feature is a safe no-op on no-route levels.
        """
        if not self.available:
            return np.zeros(4, dtype=np.float32)
        d_star, d_max = self._phase_arrays(switch_activated)
        c0 = self._min_dstar_at(x, y, d_star)
        if c0 is None or not np.isfinite(c0):
            return np.zeros(4, dtype=np.float32)
        best_c = c0
        best_dir = (0.0, 0.0)
        for ux, uy in self._NAV_DIRS:
            c = self._min_dstar_at(x + ux * step_px, y + uy * step_px, d_star)
            if c is not None and np.isfinite(c) and c < best_c:
                best_c = c
                best_dir = (ux, uy)
        cost_norm = float(min(1.0, c0 / max(d_max, 1.0)))
        phase = 1.0 if switch_activated else 0.0
        return np.array([best_dir[0], best_dir[1], cost_norm, phase],
                        dtype=np.float32)
