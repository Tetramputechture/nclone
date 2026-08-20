"""Phase-aware, door-aware free-space potential.

What this composes
------------------
``FreeSpaceDistanceSource`` prices one topology.  A level with locked doors has
several: the exit switch may be unreachable until door A is open, and door A's
switch unreachable until door B is.  ``free_space_door_plan`` resolves that into
a sequence of legs; this class turns the legs into a single continuous
cost-to-go the agent can descend, and decides what to do when a leg is not
well-defined.

The governing rule: NEVER emit no-gradient
------------------------------------------
Every failure here falls back to the doors-OPEN field, which is the historical
behaviour.  That is not laziness, it is this project's stated asymmetry —
lethality conservative, traversability optimistic — and it is forced by
measurement:

* Of 105 locked-door levels that have a **verified winning demo**, the solver
  calls **9 unsolvable**.  A demo wins those, so the verdict is provably
  over-conservative: the 6px flood's move set cannot express every wall-jump
  chain and ballistic route the ninja actually has.  Trusting "unsolvable" would
  delete the potential on levels that are demonstrably winnable.
* The 105 checkable levels are the *easy* tail — the planner solved them — so the
  true false-unsolvable rate over all 155 unsolvable verdicts is at least 8.6%
  and probably higher.

An over-promising gradient is recoverable; an absent one is not.

The severed-tail trap this also closes
--------------------------------------
``FreeSpaceDistanceSource.exit_cost_at_switch`` falls back to ``_d_max_exit``
when the exit flood cannot reach the switch.  That is not an upper bound once a
mask can sever the level: severance SHRINKS the component the max is taken over,
so the phase-1 tail gets *cheaper* and the agent is told the level is nearly
finished at the exact moment the exit became unreachable (measured: `065
striving upwards`, 325.5 -> 77.7).  A leg is therefore accepted only when its
goal is genuinely reachable; otherwise that leg falls back too.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .free_space_distance import FreeSpaceDistanceSource, _CELL, _W, _H


class DoorAwareFreeSpaceField:
    """Per-level, phase-aware cost-to-go over locked-door configurations.

    ``available`` is True whenever *some* usable field exists — including the
    pure fallback — because the caller's alternative is no potential at all.
    ``door_aware`` distinguishes "the legs are real" from "we fell back", which
    is what an audit wants and what a training A/B has to be able to split on.
    """

    def __init__(self, map_data, *, mine_hard: bool = True,
                 physics_cost: bool = True, one_way: Optional[bool] = None,
                 mine_split_topology: bool = False,
                 name: Optional[str] = None):
        self.available = False
        self.door_aware = False
        self.fallback_reason: Optional[str] = None
        self.necessary_order: List[int] = []
        self._legs: List[np.ndarray] = []
        self._leg_max: List[float] = []
        self._exit_leg: Optional[np.ndarray] = None
        self._base: Optional[FreeSpaceDistanceSource] = None

        # The doors-OPEN source is built first and unconditionally: it is both
        # the fallback and the thing every failure path returns to, so a level
        # that cannot have legs still gets exactly today's behaviour.
        try:
            self._base = FreeSpaceDistanceSource(
                list(map_data), mine_hard=mine_hard, physics_cost=physics_cost,
                one_way=one_way,
                mine_split_topology=mine_split_topology,
                name=name, doors="open")
        except Exception as exc:
            self.fallback_reason = f"base build failed: {type(exc).__name__}: {exc}"
            return
        if not self._base.available:
            self.fallback_reason = "base field unavailable"
            return
        self.available = True

        if not self._base.locked_doors:
            self.fallback_reason = "no locked doors"
            return

        try:
            from .free_space_door_plan import solve
            plan, oracle = solve(map_data, mine_hard=mine_hard,
                                 physics_cost=physics_cost, one_way=one_way,
                                 mine_split_topology=mine_split_topology,
                                 name=name)
        except Exception as exc:
            self.fallback_reason = f"solver failed: {type(exc).__name__}: {exc}"
            return
        if plan is None:
            self.fallback_reason = "no plan"
            return
        if not plan.solvable:
            # See the module docstring: measured to be wrong at least 8.6% of the
            # time, so it is a reason to fall back, never a reason to emit inf.
            self.fallback_reason = f"plan unsolvable ({plan.reason or 'no reason'})"
            return
        if not plan.necessary_order:
            # Doors exist but none must be opened; the doors-CLOSED topology is
            # then both correct and strictly more informative than doors-open.
            closed = oracle.config(frozenset())
            if closed.available and closed.exit_reachable_from_switch:
                self._base = closed
                self.door_aware = True
                self.fallback_reason = None
            else:
                self.fallback_reason = "no necessary doors, closed field unusable"
            return

        legs = list(plan.door_leg_dstar)
        if len(legs) != len(plan.necessary_order) or plan.exit_leg_dstar is None:
            self.fallback_reason = "plan legs incomplete"
            return
        # A leg is only usable if it actually prices something; an all-inf leg
        # would hand the agent a flat potential for a whole phase.
        for i, d in enumerate(legs):
            if not np.isfinite(d).any():
                self.fallback_reason = f"leg {i} is empty"
                return
        if not np.isfinite(plan.exit_leg_dstar).any():
            self.fallback_reason = "exit leg is empty"
            return

        # The final configuration must be able to reach the exit FROM the switch,
        # or the phase-1 tail is the shrunken-component fallback described above.
        final = oracle.config(frozenset(plan.necessary_order))
        if not (final.available and final.exit_reachable_from_switch):
            self.fallback_reason = "exit unreachable from switch in final config"
            return

        self.necessary_order = list(plan.necessary_order)
        self._legs = legs
        self._leg_max = [self._finite_max(d) for d in legs]
        self._exit_leg = plan.exit_leg_dstar
        self._final = final
        self.door_aware = True
        self.fallback_reason = None

    @staticmethod
    def _finite_max(a: np.ndarray) -> float:
        f = a[np.isfinite(a)]
        return float(f.max()) if f.size else 1.0

    # ------------------------------------------------------------------
    def door_phase(self, open_door_indices: Optional[Sequence[int]] = None
                   ) -> Optional[int]:
        """Index of the current leg, or ``None`` once every necessary door is open.

        The phase counts only NECESSARY doors, and advances past the legs only
        when all of them are open — so collecting the exit switch early does not
        skip a door the level still requires. Mirrors
        ``LatticeDistanceSource.door_phase`` so a consumer can swap backends.
        """
        if not self.door_aware or not self.necessary_order:
            return None
        idx = set(int(i) for i in (open_door_indices or ()))
        n_open = sum(1 for d in self.necessary_order if d in idx)
        return n_open if n_open < len(self.necessary_order) else None

    @property
    def route_cost_unit(self) -> str:
        """Delegated from whichever field actually answers a cost query."""
        src = getattr(self, "_final", None) or self._base
        return str(getattr(src, "route_cost_unit", "grid"))

    @property
    def route_cost_px_per_unit(self) -> float:
        src = getattr(self, "_final", None) or self._base
        return float(getattr(src, "route_cost_px_per_unit", 1.0))

    @property
    def exit_reachable_from_switch(self) -> bool:
        """False when ANY field in the composition took the shrinking
        ``_d_max_exit`` tail fallback. Conservative on purpose: a route length
        composed from a degraded leg is not a route length."""
        srcs = [self._base, getattr(self, "_final", None)]
        return all(bool(getattr(s, "exit_reachable_from_switch", True))
                   for s in srcs if s is not None)

    def spawn_finite(self, spawn_x: float, spawn_y: float) -> bool:
        """Whether the phase-0 cost-to-completion is usable at the spawn."""
        return bool(np.isfinite(
            self.cost_to_completion(float(spawn_x), float(spawn_y), False, ())
        ))

    def cost_to_go(self, x: float, y: float, switch_activated: bool,
                   open_door_indices: Optional[Sequence[int]] = None
                   ) -> float:
        """DistanceSource-compatible phase cost.

        While a necessary locked-door leg is active, both reward slots must use
        the same leg basis; mixing an exit-only field with a door-leg field would
        make ``_potential_substituted`` claim continuity it does not have.
        """
        ph = self.door_phase(open_door_indices)
        if ph is not None:
            return self.cost_to_completion(
                x, y, switch_activated, open_door_indices
            )
        src = getattr(self, "_final", None) or self._base
        if src is None:
            return float("inf")
        return float(src.cost_to_go(x, y, switch_activated))

    def cost_to_completion(self, x: float, y: float, switch_activated: bool,
                           open_door_indices: Optional[Sequence[int]] = None
                           ) -> float:
        """Frames-to-finish at ``(x, y)`` for the agent's current phase."""
        if not self.available:
            return float("inf")
        ph = self.door_phase(open_door_indices)
        if ph is None:
            src = getattr(self, "_final", None) or self._base
            return float(src.cost_to_completion(x, y, switch_activated))
        cx, cy = int(x / _CELL), int(y / _CELL)
        if not (0 <= cx < _W and 0 <= cy < _H):
            return float("inf")
        v = float(self._legs[ph][cx, cy])
        if not np.isfinite(v):
            # Off this leg's reachable set — usually transient (mid-jump, or a
            # cell the leg's tighter topology cut). Snapping to the leg's global
            # MAX is the obvious clamp and it is wrong: it made the potential
            # RISE when a door opened on 52 of 219 levels, i.e. told the agent
            # that a correct, irreversible action was negative progress. Prefer
            # the nearest priced cell, which is local and keeps the phases
            # ordered; the global max is the last resort.
            v = self._nearest_finite(self._legs[ph], cx, cy)
        # Offset by the legs still to come so the potential does not JUMP at a
        # door: each leg is measured to its own switch, and without this the
        # value would reset to ~0 every time one opened.
        return v + self._remaining_after(ph)

    _NAV_DIRS = (
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (0.70710678, 0.70710678), (0.70710678, -0.70710678),
        (-0.70710678, 0.70710678), (-0.70710678, -0.70710678),
    )

    def nav_feature(self, x: float, y: float, switch_activated: bool,
                    step_px: float = 12.0,
                    open_door_indices: Optional[Sequence[int]] = None
                    ) -> np.ndarray:
        """Four-value steepest-descent hint over the live door phase."""
        c0 = self.cost_to_completion(
            x, y, switch_activated, open_door_indices
        )
        if not np.isfinite(c0):
            return np.zeros(4, dtype=np.float32)
        best, direction = c0, (0.0, 0.0)
        for ux, uy in self._NAV_DIRS:
            c = self.cost_to_completion(
                x + ux * step_px,
                y + uy * step_px,
                switch_activated,
                open_door_indices,
            )
            if np.isfinite(c) and c < best:
                best, direction = c, (ux, uy)
        ph = self.door_phase(open_door_indices)
        if ph is None:
            dmax = self._final_potential_max()
        else:
            dmax = self._leg_max[ph] + self._remaining_after(ph)
        return np.asarray(
            [direction[0], direction[1], min(c0 / max(dmax, 1.0), 1.0),
             1.0 if switch_activated else 0.0],
            dtype=np.float32,
        )

    #: Snap radius in cells, matching the surface lookup's tolerance.
    _SNAP_CELLS = 3

    def _nearest_finite(self, field: np.ndarray, cx: int, cy: int) -> float:
        r = self._SNAP_CELLS
        sub = field[max(0, cx - r):cx + r + 1, max(0, cy - r):cy + r + 1]
        fin = sub[np.isfinite(sub)]
        if fin.size:
            return float(fin.min())
        return self._finite_max(field)

    def _remaining_after(self, ph: int) -> float:
        """Upper bound on everything after leg ``ph``, so phases compose monotonically.

        The exact remaining cost would be "distance from door ph's switch to door
        ph+1's switch, then ...", which is another flood per leg. An UPPER BOUND
        is enough and is what makes the composition provably monotone: with

            value(ph, x) = leg[ph](x) + sum_{i>ph} leg_max[i] + final_max

        each transition is non-increasing, because ``leg[ph+1](x) <=
        leg_max[ph+1]`` and ``final(x) <= final_max``. PBRS is policy-invariant
        for ANY potential, so only the gradient WITHIN a phase drives learning
        and that is exact; the offset exists purely to stop a step at a boundary.

        ``final_max`` is the max of the phase-``None`` potential — which is
        ``cost_to_switch + exit_cost_at_switch``, not the exit leg alone. Using
        the exit leg's max here was a genuine bug: it left the last transition
        short by a whole switch leg, so the potential ROSE when the final door
        opened on 52 of 219 levels, telling the agent an irreversible correct
        action was negative progress.
        """
        tail = sum(self._leg_max[i] for i in range(ph + 1, len(self._legs)))
        return tail + self._final_potential_max()

    def _final_potential_max(self) -> float:
        """Max of the all-doors-open potential, i.e. of what phase ``None`` returns."""
        cached = getattr(self, "_final_max", None)
        if cached is not None:
            return cached
        src = getattr(self, "_final", None) or self._base
        v = 1.0
        if src is not None:
            v = float(src._d_max_switch) + float(src.exit_cost_at_switch)
        self._final_max = v
        return v
