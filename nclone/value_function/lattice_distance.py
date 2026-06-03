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
        slope_jump_modes: Optional[str] = None,
    ):
        self.available = False
        self._graph: Optional[SurfaceGraph] = None
        self._d_comb: Optional[np.ndarray] = None
        self._d_ex: Optional[np.ndarray] = None
        self._d_max_comb = 1.0
        self._d_max_ex = 1.0
        self.switch: Optional[Tuple[float, float]] = None
        self.exit: Optional[Tuple[float, float]] = None
        # Locked-door N-phase plan (None when the level has no locked doors).
        self._door_plan = None
        self._door_leg_max = []  # per-leg finite max for normalization
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
                slope_jump_modes=slope_jump_modes,
            )
            pc = audit.precompute(use_cache=use_cache)
            self._graph = pc["graph"]
            self._d_comb = pc["d_comb"]
            self._d_ex = pc["d_ex"]
            self._d_max_comb = _finite_max(self._d_comb)
            self._d_max_ex = _finite_max(self._d_ex)
            self.switch = audit.switch
            self.exit = audit.exit
            plan = pc.get("door_plan")
            if plan is not None and plan.solvable and plan.necessary_order:
                self._door_plan = plan
                self._door_leg_max = [_finite_max(d) for d in plan.door_leg_dstar]
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

    def door_phase(
        self, switch_activated: bool, open_door_indices=None
    ) -> Optional[int]:
        """Index of the current locked-door leg (0..k-1), or ``None`` once all
        necessary doors are open (then the level's normal switch->exit potential
        takes over) / when there is no door plan.

        ``open_door_indices`` is the set of locked-door indices (audit order) whose
        switch has been collected. The phase = how many necessary doors are open;
        it only advances PAST the door legs once ALL k necessary doors are open —
        so a pre-collected exit switch does NOT skip the door-opening legs.
        """
        plan = getattr(self, "_door_plan", None)
        if plan is None:
            return None
        idx = set(open_door_indices or ())
        k = len(plan.necessary_order)
        n_open = sum(1 for d in plan.necessary_order if d in idx)
        return n_open if n_open < k else None

    def _phase_arrays(
        self, switch_activated: bool, open_door_indices=None
    ) -> Tuple[np.ndarray, float]:
        ph = self.door_phase(switch_activated, open_door_indices)
        if ph is not None:  # still opening necessary doors -> guide to next switch
            leg_max = getattr(self, "_door_leg_max", [])
            dstar = self._door_plan.door_leg_dstar[ph]
            return dstar, (leg_max[ph] if ph < len(leg_max) else 0.0) or 1.0
        # All necessary doors open (or no door plan): normal switch -> exit.
        if switch_activated:
            return self._d_ex, self._d_max_ex
        return self._d_comb, self._d_max_comb

    def cost_to_go(
        self, x: float, y: float, switch_activated: bool, open_door_indices=None
    ) -> float:
        """Phase-aware lattice cost-to-go (frames) at ``(x, y)``.

        With locked doors, ``open_door_indices`` selects the correct N-phase leg
        (the agent's current sub-goal); without doors it falls back to the
        two-phase switch/exit d*.

        ``inf`` only when no surface point exists near ``(x, y)``; a transient
        unreachable lookup is clamped to the per-level max finite ``d*`` to avoid
        a scale discontinuity within an episode.
        """
        if not self.available:
            return float("inf")
        d_star, d_max = self._phase_arrays(switch_activated, open_door_indices)
        c = self._min_dstar_at(x, y, d_star)
        if c is None:
            return float("inf")
        if not np.isfinite(c):
            return float(d_max)
        return float(c)

    @staticmethod
    def gold_orienteering_tail(
        x: float, y: float, remaining_gold, gold_frame_value: float = 120.0
    ) -> float:
        """Frames to detour through only the WORTHWHILE remaining gold from (x, y).

        Greedy nearest-neighbour tour / max-horizontal-speed, but a gold is added
        to the tail only while the nearest remaining gold's marginal detour is
        below ``gold_frame_value`` frames — i.e. only gold whose 120-frame credit
        outweighs the time cost of grabbing it.  Because the tour is built
        nearest-first, once the closest remaining gold is too far to be worth it,
        every farther gold is too, so we stop: the tail covers exactly the
        prize-collecting (orienteering) prefix.

        This is the upgrade over the old all-gold tour (which guided the agent
        toward gold it should skip — the 'misleading potential' failure mode):
        unworthwhile far-flung gold never enters the tail.

        Depends only on the remaining-gold SET (not position alone), so collecting
        a worthwhile gold strictly lowers it — the monotonicity a valid PBRS gold
        potential needs.  Geometric (gold floats off surfaces so it isn't in the
        lattice graph); exactness doesn't matter for a potential, only the SET
        monotonicity and the worthwhile filter.

        NB: the +gold_frame_value CREDIT per collected gold is realised through the
        R_GOLD pickup reward, NOT subtracted here — this quantity is consumed as an
        HWM *distance* (non-negative, lower=better), so folding the credit in would
        double-count and could drive the distance negative.
        """
        from ..constants.physics_constants import MAX_HOR_SPEED
        speed = max(MAX_HOR_SPEED, 1e-6)
        pts = [(float(gx), float(gy)) for (gx, gy) in remaining_gold]
        cx, cy = float(x), float(y)
        total = 0.0
        while pts:
            j = min(range(len(pts)),
                    key=lambda k: (pts[k][0] - cx) ** 2 + (pts[k][1] - cy) ** 2)
            gx, gy = pts[j]
            marginal = ((gx - cx) ** 2 + (gy - cy) ** 2) ** 0.5 / speed
            if marginal >= gold_frame_value:
                break  # nearest gold isn't worth the detour => none beyond it are
            pts.pop(j)
            total += marginal
            cx, cy = gx, gy
        return total

    def gold_cost_to_go(self, x: float, y: float, switch_activated: bool,
                        remaining_gold, gold_weight: float = 1.0,
                        gold_frame_value: float = 120.0) -> float:
        """Gold-aware phase potential: base lattice cost-to-go (frames) PLUS the
        worthwhile-remaining-gold orienteering tail-cost.  Reduces to
        ``cost_to_go`` when no (worthwhile) gold remains.  ``inf`` only when the
        base is ``inf`` (no plan near (x, y)).

        ``gold_frame_value`` is the per-gold time-saving exchange rate (mirror of
        reward_constants.GOLD_FRAME_VALUE): only gold whose detour costs fewer than
        this many frames is folded into the potential."""
        base = self.cost_to_go(x, y, switch_activated)
        if not np.isfinite(base):
            return base
        tail = self.gold_orienteering_tail(x, y, remaining_gold, gold_frame_value)
        return base + float(gold_weight) * tail

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
        remaining_gold=None, gold_frame_value: float = 120.0,
        open_door_indices=None,
    ) -> np.ndarray:
        """4-dim physics-aware navigation hint for the POLICY observation:
        ``[dx_hop, dy_hop, cost_to_go_norm, phase]``.

        ``(dx_hop, dy_hop)`` is the unit direction of steepest descent —
        sampling the 8 neighbours one tile away and picking the one that most
        reduces the lattice cost-to-go (the direction the physics-aware plan
        wants to go, unlike the geometric Euclidean-to-goal that can point into a
        wall).  ``cost_to_go_norm = c/d_max`` in ``[0, 1]``.  ``phase`` = 1 once
        the switch is collected.  All-zeros when no finite plan exists here, so
        the feature is a safe no-op on no-route levels.

        When ``remaining_gold`` is supplied the descent follows the ORIENTEERING
        potential (base cost-to-go + worthwhile-gold tail) instead of the plain
        cost-to-go, so the hint bends the policy toward gold worth its detour —
        encoding the 120-frame objective directly in the observation. ``d_max`` is
        still the gold-free per-level scale, so the norm can exceed the no-gold
        value while a worthwhile detour is pending (clamped to 1.0).
        """
        if not self.available:
            return np.zeros(4, dtype=np.float32)
        _, d_max = self._phase_arrays(switch_activated, open_door_indices)
        gold = list(remaining_gold) if remaining_gold else None

        def cost_at(px: float, py: float) -> Optional[float]:
            if gold:
                c = self.gold_cost_to_go(px, py, switch_activated, gold,
                                         gold_frame_value=gold_frame_value)
                return c if np.isfinite(c) else None
            d_star, _ = self._phase_arrays(switch_activated, open_door_indices)
            c = self._min_dstar_at(px, py, d_star)
            return c if (c is not None and np.isfinite(c)) else None

        c0 = cost_at(x, y)
        if c0 is None:
            return np.zeros(4, dtype=np.float32)
        best_c = c0
        best_dir = (0.0, 0.0)
        for ux, uy in self._NAV_DIRS:
            c = cost_at(x + ux * step_px, y + uy * step_px)
            if c is not None and c < best_c:
                best_c = c
                best_dir = (ux, uy)
        cost_norm = float(min(1.0, c0 / max(d_max, 1.0)))
        phase = 1.0 if switch_activated else 0.0
        return np.array([best_dir[0], best_dir[1], cost_norm, phase],
                        dtype=np.float32)
