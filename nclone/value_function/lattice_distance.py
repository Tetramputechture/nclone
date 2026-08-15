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
from ..constants.physics_constants import (
    NINJA_RADIUS, EXIT_SWITCH_RADIUS, EXIT_DOOR_RADIUS,
)

#: Sentinel for "not yet computed" in the goal-spec cache, so a legitimately
#: ``None`` spec (the locked-door legs) is cached rather than recomputed forever.
_MISSING = object()

#: Opt-in: prefer the (monotone) flood over the raw surface d* even where the
#: surface lookup can answer.  See ``_cost_at``.  Off by default — it changes
#: previously-finite values, so it needs its own training A/B.
_PREFER_FIELD_ENV = "NCLONE_LATTICE_PREFER_FIELD"


def _prefer_field_enabled() -> bool:
    import os
    return os.environ.get(_PREFER_FIELD_ENV, "0") not in ("0", "false", "False", "")


def _finite_max(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)]
    return float(finite.max()) if finite.size else 1.0


class LatticeDistanceSource:
    """Per-level holder of the lattice ``d*`` exposing a position→cost-to-go map.

    Construction is best-effort: any failure (unsupported entities, enumeration
    error, unreachable goals) is swallowed, and the level still trains — just
    without the physics-aware potential.

    ``available`` IS NOT A "THE PLANNER BUILT" FLAG. It is the union of the
    lattice and the airborne fallback::

        self.available = bool(self.lattice_available or self._air is not None)

    so a level whose ``precompute`` RAISED still reports ``available is True``
    and silently runs on the degraded air field. Measured 2026-08-15 on four
    levels that raise ``Switch unreachable``: all four had ``available=True``
    with ``lattice_available=False``. An offline audit that gated on
    ``available`` therefore recorded 22 unbuildable levels as healthy and sent
    the investigation after a non-existent cache-write bug.

    **Check ``lattice_available`` whenever the question is "did the lattice
    build".** Use ``available`` only for "is there any potential at all".
    """

    # Class-level defaults for the airborne extension.  Tests (and any other
    # caller) build a source with ``__new__`` to exercise the lookup logic against
    # hand-made d* arrays without paying for a real precompute; those instances
    # never run ``__init__``, so the extension state has to exist on the class or
    # every such construction raises AttributeError.  Defaulting to "no field"
    # also makes the surface-only path the safe fallback everywhere.
    _air = None
    _cont_cache: Optional[float] = None
    _goal_spec_cache: dict = {}
    _prefer_field = False

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
        max_nodes: Optional[int] = None,
        airborne_field: Optional[bool] = None,
        prefer_field: Optional[bool] = None,
        seed_mode: Optional[str] = None,
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
                max_nodes=max_nodes,
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
            self.lattice_available = True
        except Exception:
            self.lattice_available = False

        # The surface lattice fails outright on ~13% of official levels — measured
        # over 60: 8.3% ``LatticeTooLargeError`` (graph over the node budget) and
        # 5% ``ValueError`` (a goal with no surface contact node and no arc through
        # its contact zone).  Every one of those levels is still a connected free
        # space with a goal in it, so the field is built REGARDLESS, and the goal
        # is always seeded.  With no graph it degenerates to a pure goal-seeded
        # physics flood: finite everywhere free space reaches the goal, and a
        # discrete navigation function (its only minimum on that connected
        # component IS the goal, which Dijkstra gives for free).  With a graph the
        # exact surface d* seeds dominate and nothing changes.
        #
        # One potential, one scale (frames), no per-level source switching and so
        # no mid-episode scale discontinuity.
        self._air = None
        self._cont_cache: Optional[float] = None
        self._goal_spec_cache = {}
        self._prefer_field = (
            _prefer_field_enabled() if prefer_field is None else bool(prefer_field)
        )
        try:
            from .airborne_field import AirborneCostField, airborne_field_enabled
            want = (airborne_field_enabled() if airborne_field is None
                    else bool(airborne_field))
            if want:
                if self.switch is None or self.exit is None:
                    self._read_goals_from_map(map_data)
                if self.switch is not None and self.exit is not None:
                    air = AirborneCostField(
                        list(map_data), self._graph, name=name,
                        seed_mode=seed_mode,
                        # Only the "spawn" ablation arm needs this, and loading a
                        # simulator is not free, so it is not paid on the
                        # default path.
                        spawn=(self._spawn_pos(map_data)
                               if seed_mode == "spawn" else None))
                    self._air = air if air.available else None
        except Exception:
            self._air = None

        if not self.lattice_available and self._air is not None:
            # No lattice, but a usable field: expose distinct empty arrays as the
            # per-phase tokens.  They carry no surface seeds (so ``_build_field``
            # falls through to goal seeds only) while still giving ``_goal_spec``
            # and the field memo their ``is`` / ``id()`` identity per phase.
            self._d_comb = np.empty(0, dtype=np.float32)
            self._d_ex = np.empty(0, dtype=np.float32)
            self._d_max_comb = self._field_scale(self._d_comb)
            self._d_max_ex = self._field_scale(self._d_ex)

        self.available = bool(self.lattice_available or self._air is not None)

    # ------------------------------------------------------------------
    @staticmethod
    def _spawn_pos(map_data) -> Optional[Tuple[float, float]]:
        """Ninja spawn, for the ``spawn`` seed-ablation arm.  ``None`` on error."""
        try:
            import nclone_native
            sim = nclone_native.Simulator.load(list(map_data))
            x, y = sim.get_ninja_pos()
            return (float(x), float(y))
        except Exception:
            return None

    def _read_goals_from_map(self, map_data) -> None:
        """Recover switch/exit positions without a successful lattice build."""
        try:
            import nclone_native
            from .tools.physics_route_spike import _goal_positions
            sim = nclone_native.Simulator.load(list(map_data))
            switch, exit_door, _mines = _goal_positions(sim)
            self.switch = switch
            self.exit = exit_door
        except Exception:
            pass

    def _field_scale(self, d_star: np.ndarray) -> float:
        """Per-level normalisation constant taken from the field.

        ``_finite_max`` of the empty phase token would return 1.0, which would
        make ``nav_feature``'s ``cost/d_max`` saturate at 1.0 everywhere and
        destroy the very gradient this is for.
        """
        if self._air is None:
            return 1.0
        f = self._air.field_for(d_star, self._goal_spec(d_star))
        if f is None:
            return 1.0
        finite = f[np.isfinite(f)]
        return float(finite.max()) if finite.size else 1.0

    # ------------------------------------------------------------------
    def _min_dstar_at(self, x: float, y: float, d_star: np.ndarray) -> Optional[float]:
        """Min finite ``d*`` over all velocity bins at the nearest floor/wall
        point to ``(x, y)``.  ``None`` if no surface point is found at all;
        ``inf`` if a point is found but every bin is unreachable."""
        g = self._graph
        if g is None:
            return None  # no lattice on this level; the field answers instead
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

    def _goal_spec(self, d_star: np.ndarray):
        """Memoised :meth:`_goal_spec_uncached`, keyed by array identity.

        ``_cost_at`` consults this on every lookup that misses the surface graph,
        which on an airborne frame is every lookup, and ``nav_feature`` makes nine
        of those per step.  Recomputing the spec each time (an inline import plus
        the switch-continuation lookup) measured 160us -> 310us per nav_feature on
        ``000 hilltops``; the spec is constant per level and phase, so it is
        computed once and cached.
        """
        key = id(d_star)
        spec = self._goal_spec_cache.get(key, _MISSING)
        if spec is _MISSING:
            spec = self._goal_spec_uncached(d_star)
            self._goal_spec_cache[key] = spec
        return spec

    def _goal_spec_uncached(self, d_star: np.ndarray):
        """``(x, y, contact_radius, cost_at_contact)`` for the leg ``d_star`` solves.

        Feeds spec D3's airborne goal nodes.  Two legs are handled; anything else
        (the locked-door legs) returns ``None`` and the field falls back to
        surface-only seeding, which is the pre-existing behaviour.

        * phase 1 (``d_ex``): the goal is the exit and reaching it ends the level,
          so the contact cost is 0.
        * phase 0 (``d_comb``): the goal is the switch, and touching it leaves the
          remaining switch->exit journey.  Seeding the contact cost with that
          continuation is what keeps the two phases on one continuous scale —
          seeding 0 would make the potential drop to zero at the switch and then
          jump back up the instant the phase flips.
        """
        if d_star is self._d_ex and self.exit is not None:
            return (self.exit[0], self.exit[1],
                    NINJA_RADIUS + EXIT_DOOR_RADIUS, 0.0)
        if d_star is self._d_comb and self.switch is not None:
            cont = self._switch_continuation()
            if cont is None:
                return None
            return (self.switch[0], self.switch[1],
                    NINJA_RADIUS + EXIT_SWITCH_RADIUS, cont)
        return None

    def _switch_continuation(self) -> Optional[float]:
        """Remaining frames from the switch to the exit.

        Prefers the surface lattice's own answer at the switch; falls back to the
        exit FIELD at the switch position, which is the only source that exists
        when the switch floats in open space — exactly the case D3 is for.
        """
        if self._cont_cache is not None:
            return self._cont_cache if self._cont_cache >= 0 else None
        val = None
        sx, sy = self.switch
        c = self._min_dstar_at(sx, sy, self._d_ex)
        if c is not None and np.isfinite(c):
            val = float(c)
        elif self._air is not None and self.exit is not None:
            ca = self._air.cost_at(
                sx, sy, self._d_ex,
                (self.exit[0], self.exit[1],
                 NINJA_RADIUS + EXIT_DOOR_RADIUS, 0.0),
            )
            if np.isfinite(ca):
                val = float(ca)
        self._cont_cache = val if val is not None else -1.0
        return val

    def _cost_at(
        self, x: float, y: float, d_star: np.ndarray,
    ) -> Optional[float]:
        """Surface ``d*`` at ``(x, y)``, falling back to the airborne field.

        The surface answer is returned UNCHANGED whenever it is finite, so every
        position the lattice could already price keeps its exact previous value
        (spec A7).  The airborne field is consulted only in the two cases that
        previously produced no usable signal:

        * ``None`` — no floor/wall point within the 18px lookup radius.  This is
          the dominant case by a wide margin: measured 54-83% of level cells, and
          it is what silences the potential for most airborne frames.
        * ``inf`` — a point was found but every velocity bin is unreachable
          (genuine graph severance at that node).

        Returns ``None`` only when neither source has anything, so callers keep
        their existing "no plan here" branch.
        """
        air = self._air
        if air is not None and self._prefer_field:
            # MONOTONE mode.  Take the flood even where the surface lookup could
            # answer.  Measured over 8 official levels, 20.21% of surface-sourced
            # cells are spurious LOCAL MINIMA of the potential, against 0.25% of
            # field-sourced cells: ``_min_dstar_at`` is a piecewise-constant snap
            # to the nearest surface point, so neighbouring query cells can land
            # on different points whose (risk-reweighted, non-metric) d* jumps
            # around.  The flood is a Dijkstra cost-to-go and therefore monotone
            # by construction.
            #
            # Measured effect on 47.5k cells: coverage IDENTICAL (0 lost, 0
            # gained), median difference 0, values lowered on 19.7% of cells and
            # raised on only 1.7% (max +33 frames).  It is not monotonically
            # lower, and the reason is worth knowing: ``_min_dstar_at`` snaps to
            # any surface point within +/-18px and charges NOTHING for that
            # teleport, whereas the flood makes you pay the travel.  So on the
            # small fraction of cells where the free teleport was doing the work,
            # the flood is (correctly) a little more expensive.
            #
            # The trade is monotonicity against calibration: flood travel costs
            # are approximate, so values get more optimistic.  For a PBRS
            # potential that is the right side of the trade — HWM normalises per
            # phase, so "only the gradient matters" (see this module's header) —
            # but it is off by default until it has had a training A/B.
            ca = air.cost_at(x, y, d_star, self._goal_spec(d_star))
            if np.isfinite(ca):
                return float(ca)
        if air is not None and not air.surface_lookup_can_hit(x, y):
            # Provably no surface point in range, so ``_min_dstar_at`` would
            # return None after two empty 7x7 scans.  Skip straight to the field;
            # the result is identical, just without the wasted scans.
            ca = air.cost_at(x, y, d_star, self._goal_spec(d_star))
            return float(ca) if np.isfinite(ca) else None
        c = self._min_dstar_at(x, y, d_star)
        if c is not None and np.isfinite(c):
            return c
        if air is not None:
            ca = air.cost_at(x, y, d_star, self._goal_spec(d_star))
            if np.isfinite(ca):
                return float(ca)
        return c

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
        c = self._cost_at(x, y, d_star)
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
        c = self._cost_at(spawn_x, spawn_y, self._d_comb)
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
            c = self._cost_at(px, py, d_star)
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
