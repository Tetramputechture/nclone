"""Airborne extension of the surface lattice's cost-to-go.

Why this exists
---------------
``LatticeDistanceSource.cost_to_go`` answers by snapping ``(x, y)`` to the
nearest floor/wall *surface point* (``_min_dstar_at`` ->
``surface_graph.find_nearest_point``).  That search covers +/-3 spatial cells of
6px, i.e. a +/-18px box.  Any query further than that from a surface finds no
point at all and the lattice returns ``inf`` — no potential, no gradient.

Measured on official levels, that blind spot is most of the level::

    000 hilltops    17.3% of cells have a surface point within the lookup radius
    023 chicken     46.1%

so the lattice is silent at 54-83% of positions.  It is silent in exactly the
places that matter: 85.8% of frames in validated winning routes are airborne
(SPEC_airborne_lattice §1), and an airborne ninja mid-arc is usually further
than 18px from anything.  The surface graph itself is healthy — on both levels
above, *every* position that resolved to a surface point had a finite ``d*``.
The failure is the projection to the observation (spec §2.4), not the graph.

What this does
--------------
Push the lattice's exact, physics-derived surface ``d*`` out into free space:

1. Seed a cell for every surface point with that point's ``min`` over velocity
   bins of ``d*`` — the same optimistic "best achievable if you arrive with the
   right speed" quantity ``_min_dstar_at`` already returns.
2. Flood outward over the free-space grid with
   :meth:`FreeSpaceDistanceSource.flood_physics_from_seeds`, whose anisotropic
   per-cell frame costs (cheap down, expensive up, up-gated by jump reach,
   wall-launch arc edges) are already calibrated against 40 verified demos.

The result, ``field[cell] = min over surface points p ( frames(cell -> p) +
d*[p] )``, is a dense finite cost-to-go wherever free space is connected to any
reachable surface, and it *inherits* the lattice's physics-awareness (run-ups,
wall chains, mine pruning) instead of being an independent geometric flood.

The guarantee, and how it is reached
------------------------------------
The goal is that **every level has a finite cost-to-go at its spawn** unless the
goal is genuinely unreachable (e.g. gated by a locked door).  That is a
*coverage* property, not an accuracy one, so it is reached by making each tier
strictly more complete and strictly less exact than the one above it, and by
letting the more-complete tier SEED the more-exact one rather than switching
between them:

1. **Surface lattice d-star** — exact, physics-derived, velocity-augmented.  Covers
   what the arc enumeration can prove.  Fails outright on ~13% of official
   levels (8.3% graph over the node budget, 5% goal with no contact node).
2. **This field, seeded from surface d-star + the goal disc** — covers all free
   space connected to a reachable surface *or* to the goal.  With no graph at
   all it degenerates to a pure goal-seeded flood, which is still a valid
   potential: a shortest-path cost-to-go has its only minimum AT the goal on the
   connected component (the discrete analogue of a Rimon-Koditschek navigation
   function), which Dijkstra gives for free.
3. **The mine-soft rescue** (``_soft_field_for``) — covers regions the hard mine
   mask disconnected at 6px resolution even though the ninja can thread them.

Every tier produces frames on one scale, so there is no per-level source switch
and no mid-episode scale discontinuity — the failure mode the module docstring
of ``lattice_distance`` warns about.  Measured: 8/8 of the levels tier 1 loses
are priced by tier 2.

Known approximations
--------------------
* **Mixed scale.** The seeds are the lattice's RISK-REWEIGHTED d* (``edge_risk``
  applies ``w_risk`` / ``jump_tax`` multipliers on top of raw frames), while the
  flood adds unweighted physics frames.  So a route's surface leg is priced with
  risk and its airborne leg without, and the field slightly under-weights risk
  relative to the surface values it extends.  Left as-is deliberately: PBRS is
  policy-invariant for ANY potential (Ng, Harada & Russell 1999), so this cannot
  make the optimal policy wrong, and the gradient WITHIN the flooded region is a
  consistent frame cost.  Worth revisiting only if the field is ever used as a
  value estimate rather than a shaping potential.
* **Velocity-agnostic.** Like ``_min_dstar_at``, the field answers "best
  achievable if you arrive with the right speed" (spec §2.4).  It is a lower
  bound, not the cost from the ninja's current momentum.

Entity awareness, and where it plugs in
---------------------------------------
The field inherits its free/blocked mask from ``FreeSpaceDistanceSource``, which
reads only entity types 4 (switch), 3 (exit) and 1/21 (mines).  The arcs that
produce the seed ``d*`` are likewise enumerated with ``tick_tiles_only`` /
``tick_physics_mines_only``, i.e. tiles + mine death and no other entity
physics.  So one-way platforms, bounce blocks, thwumps and launch/boost pads are
invisible to BOTH layers (traversal entities are out of scope here — see
SPEC_airborne_lattice §7 and ENTITY_INTEGRATION_PLAN.md).

Two consequences worth knowing before extending this:

* The seeds improve for free.  The field is seeded from whatever ``d*`` the
  surface graph produces, so when the arc tick becomes entity-aware (the natural
  move is a ``tick_physics_static`` alongside the existing mines-only tick) this
  module needs no change at all.
* The MASK does not improve for free.  A one-way platform is currently neither
  solid nor one-directional in the free mask, so the flood can pass through it
  in the blocked direction and under-price the route.  That mask is the single
  integration point for static traversal entities; ``_step_blocked`` in
  free_space_distance is where a directional test belongs.

Deliberately NOT a change to the surface values
-----------------------------------------------
The field is consulted only where the surface lookup returns ``None`` or
``inf``.  Every position that already had a finite lattice value keeps it
byte-for-byte, which is what makes the extension inert where it should be
(spec A7's bit-identical regression) and keeps this a strict widening of the
domain rather than a re-pricing of the existing one.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

from .free_space_distance import (
    FreeSpaceDistanceSource,
    _CELL,
    _W,
    _H,
)
from .surface_graph import SurfaceGraph, node_id, num_velocity_bins
from ..constants.physics_constants import (
    NINJA_RADIUS, TOGGLE_MINE_RADIUS_TOGGLED,
)

#: The game's exact lethal contact distance, centre to centre.  Note this is the
#: UNDILATED radius: ``free_space_distance`` cuts its routing mask at
#: ``_MINE_KILL + _CELL`` (20px) so that no part of a 6px cell can be lethal,
#: which is right for routing and wrong for asking "can the ninja be here".
_MINE_KILL_RADIUS = NINJA_RADIUS + TOGGLE_MINE_RADIUS_TOGGLED  # 14.0 px

#: Process-wide kill switch.  Set ``NCLONE_AIRBORNE_FIELD=0`` to restore the
#: surface-only lookup (the pre-extension behaviour) without a code change —
#: the A/B control for "is the airborne extension responsible for X".
_ENV_FLAG = "NCLONE_AIRBORNE_FIELD"

#: Which surface seeds the flood is allowed to use.  This is the ablation lever
#: for "does the surface lattice earn its cost" (RESEARCH_pbrs_under_entities
#: §6): the lattice is expensive (p50 3.5s, up to 206s, fails outright on ~13%
#: of levels) and its VALUES are already known to be ignorable, so the only
#: remaining case for it is that its seeds make the flood physics-aware.
#:
#: Seeds are merged into the flood by ``min``, so the modes are strictly
#: ordered: ``full`` <= ``spawn`` <= ``none`` pointwise.  ``none`` is a pure
#: goal-seeded flood — no surface graph needed at all.
_SEED_MODE_ENV = "NCLONE_AIRBORNE_SEED_MODE"
SEED_MODES = ("full", "spawn", "none")

#: Radius, in px, of the ``spawn`` mode's surface-seed window.  Sized as "the
#: part of the level the agent can plausibly reach before the flood takes over",
#: not as a tuned constant — the arm exists to bracket ``full`` and ``none``.
_SPAWN_SEED_RADIUS_PX = 120.0


def airborne_field_enabled(default: bool = True) -> bool:
    """Whether the airborne extension is active for newly built sources."""
    v = os.environ.get(_ENV_FLAG)
    if v is None:
        return default
    return v not in ("0", "false", "False", "")


def airborne_seed_mode(default: str = "full") -> str:
    """Seed mode for newly built fields; see :data:`_SEED_MODE_ENV`."""
    v = os.environ.get(_SEED_MODE_ENV)
    if v is None or v not in SEED_MODES:
        return default
    return v


class AirborneCostField:
    """Per-level free-space extension of one or more lattice ``d*`` arrays.

    Construction builds the free-space grid once (cached on disk / in-process by
    ``FreeSpaceDistanceSource``); each distinct ``d*`` array then costs one flood
    over ~17.6k cells, which is milliseconds.  Fields are memoised per array so
    the per-phase arrays (``d_comb``, ``d_ex``, each locked-door leg) are each
    flooded at most once.

    Lookup is a bounds-checked array index, so the per-step cost on the training
    hot path is unchanged (spec A5).
    """

    def __init__(
        self,
        map_data,
        graph: Optional[SurfaceGraph] = None,
        *,
        name: Optional[str] = None,
        mine_hard: bool = True,
        seed_mode: Optional[str] = None,
        spawn: Optional[Tuple[float, float]] = None,
    ) -> None:
        # ``graph=None`` is a first-class case, not a degenerate one.  The surface
        # lattice fails outright on ~13% of official levels (measured: 8.3%
        # LatticeTooLargeError, 5% "goal has no contact node"), and on every one
        # of those the level is still a connected free space with a goal in it —
        # a goal-seeded flood alone priced the spawn on 8/8 of them.  So the field
        # is built from the map whether or not a graph exists; the graph only ever
        # ADDS exact surface seeds on top.
        self.available = False
        self._graph = graph
        self._fields: Dict[tuple, np.ndarray] = {}
        self._fs: Optional[FreeSpaceDistanceSource] = None
        # Mine-soft rescue tier (see _soft_field_for): built lazily, only if a
        # query is unreachable under the hard mask.
        self._fs_soft: Optional[FreeSpaceDistanceSource] = None
        self._soft_fields: Dict[tuple, np.ndarray] = {}
        self._soft_failed = False
        self._map_data = list(map_data)
        self._name = name
        self._seed_mode = (
            airborne_seed_mode() if seed_mode is None else str(seed_mode)
        )
        self._spawn = spawn
        try:
            # ``physics_cost=True`` unconditionally: the whole point is to extend
            # a *physics* cost-to-go, so the geometric grid-unit model would put
            # the field on a different scale from the lattice values seeding it
            # and the two would not compose.
            # ``mine_hard=True``: deadly-mine cells are cut from the free mask
            # rather than made merely expensive, so the extension can never hand
            # back a finite cost for standing inside a kill radius.  Spec trap #3
            # — severance is fixed by adding airborne representation, never by
            # permitting the agent into a kill zone.
            fs = FreeSpaceDistanceSource(
                list(map_data), name=name, mine_hard=mine_hard,
                physics_cost=True,
            )
            if not fs.available or fs._free is None:
                return
            self._fs = fs
            use_graph = graph if self._seed_mode != "none" else None
            self._seed_cells = (
                self._build_seed_cells(use_graph, self._seed_mode, spawn)
                if use_graph else []
            )
            self._surface_near = (
                self._build_surface_near(graph) if graph
                else np.zeros((_W, _H), dtype=bool)
            )
            # Availability keys on the FREE-SPACE GRID, not on having surface
            # seeds: with no graph the goal seeds alone still produce a valid,
            # finite, single-minimum potential.
            self.available = True
        except Exception:
            self.available = False

    @staticmethod
    def _build_surface_near(graph: SurfaceGraph) -> np.ndarray:
        """``[gx, gy]`` -> could ``find_nearest_point`` find anything here?

        ``_min_dstar_at`` runs ``find_nearest_point`` twice (floor, then wall) and
        each call scans the 7x7 spatial-index neighbourhood of the query cell.  On
        an airborne query both scans are guaranteed to come back empty, and that
        wasted 98-cell scan is the dominant cost of the lookup — which now runs
        nine times per step, because ``nav_feature`` no longer bails out early.

        The surface graph's spatial index uses the same 6px cell size as the
        free-space grid, so "is there ANY surface point within the search radius"
        is exactly a 7x7 dilation of the occupied index cells.  Where this says
        False, ``_min_dstar_at`` provably returns ``None`` and the scans can be
        skipped outright — an exact short-circuit, not an approximation.
        """
        near = np.zeros((_W, _H), dtype=bool)
        occupied = np.zeros((_W, _H), dtype=bool)
        for sp in graph.surface_points:
            gx = int(sp.x // _CELL)
            gy = int(sp.y // _CELL)
            if 0 <= gx < _W and 0 <= gy < _H:
                occupied[gx, gy] = True
        # 7x7 dilation == find_nearest_point's search_radius of 3 cells.
        r = 3
        xs, ys = np.nonzero(occupied)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx = np.clip(xs + dx, 0, _W - 1)
                ny = np.clip(ys + dy, 0, _H - 1)
                near[nx, ny] = True
        return near

    def surface_lookup_can_hit(self, x: float, y: float) -> bool:
        """False only when ``_min_dstar_at`` is guaranteed to return ``None``."""
        gx = int(x // _CELL)
        gy = int(y // _CELL)
        if not (0 <= gx < _W and 0 <= gy < _H):
            return True  # off-grid: fall through to the real scan
        return bool(self._surface_near[gx, gy])

    # ------------------------------------------------------------------
    @staticmethod
    def _build_seed_cells(graph: SurfaceGraph, seed_mode: str = "full",
                          spawn: Optional[Tuple[float, float]] = None):
        """Map each surface point to its grid cell and its node-id range.

        Returns ``[(cell_x, cell_y, first_node_id, n_bins), ...]``, computed once
        so that flooding a new ``d*`` array is a pure gather with no geometry.

        ``seed_mode="spawn"`` keeps only the points within
        :data:`_SPAWN_SEED_RADIUS_PX` of ``spawn`` — the ablation arm that asks
        whether the lattice needs to cover the whole level or only the region the
        agent starts in.  With no spawn supplied it degrades to ``full`` rather
        than to ``none``, because silently dropping every seed would read as a
        catastrophic result for the lattice when it is really a missing argument.
        """
        keep_r2 = None
        if seed_mode == "spawn" and spawn is not None:
            keep_r2 = _SPAWN_SEED_RADIUS_PX * _SPAWN_SEED_RADIUS_PX
        out = []
        for pt_idx, sp in enumerate(graph.surface_points):
            if keep_r2 is not None:
                dx, dy = sp.x - spawn[0], sp.y - spawn[1]
                if dx * dx + dy * dy > keep_r2:
                    continue
            cx = int(sp.x / _CELL)
            cy = int(sp.y / _CELL)
            if not (0 <= cx < _W and 0 <= cy < _H):
                continue
            n_bins = num_velocity_bins(sp.surface_type)
            out.append((cx, cy, node_id(pt_idx, 0, graph.max_vel_bins), n_bins))
        return out

    def _goal_seeds(
        self, goal: Tuple[float, float, float, float],
    ) -> Dict[Tuple[int, int], float]:
        """Cost-``c`` seeds on every free cell inside a goal's contact disc.

        This is spec D3's "airborne goal nodes".  Without it a switch or exit
        that floats in open space is priced only *through a surface*: the field
        answers "frames to the nearest ledge, plus that ledge's d*", so standing
        with your hand on an air switch reads as expensive as standing on the
        ledge below it.  The gradient near the goal points at the ledge instead
        of the goal, which is the exact mispricing §2.4 and D3 call out.

        Seeding the contact disc directly makes the goal its own source, so the
        field descends toward the goal through open air.

        ``goal`` is ``(x, y, contact_radius, cost_at_contact)``.  The cost is 0
        for the terminal (exit) leg and the remaining switch->exit cost for the
        pre-switch leg, so the two phases compose without a step at the switch.
        """
        gx, gy, radius, cost = goal
        free = self._fs._free
        seeds: Dict[Tuple[int, int], float] = {}
        r2 = radius * radius
        rc = int(radius / _CELL) + 1
        cx0, cy0 = int(gx / _CELL), int(gy / _CELL)
        for cx in range(max(0, cx0 - rc), min(_W, cx0 + rc + 1)):
            px = cx * _CELL + _CELL / 2.0
            for cy in range(max(0, cy0 - rc), min(_H, cy0 + rc + 1)):
                py = cy * _CELL + _CELL / 2.0
                if not free[cx, cy]:
                    continue
                if (px - gx) ** 2 + (py - gy) ** 2 <= r2:
                    seeds[(cx, cy)] = float(cost)
        return seeds

    def _build_field(
        self,
        d_star: np.ndarray,
        goal: Optional[Tuple[float, float, float, float]] = None,
    ) -> np.ndarray:
        """Flood one ``d*`` array out into free space."""
        seeds: Dict[Tuple[int, int], float] = {}
        n = int(d_star.shape[0])
        for cx, cy, first, n_bins in self._seed_cells:
            hi = min(first + n_bins, n)
            if hi <= first:
                continue
            # min over the point's velocity bins — the same optimistic quantity
            # ``_min_dstar_at`` returns, so the seed is on the lattice's scale.
            vals = d_star[first:hi]
            best = float(vals.min())
            if not np.isfinite(best):
                continue
            key = (cx, cy)
            prev = seeds.get(key)
            if prev is None or best < prev:
                seeds[key] = best
        if goal is not None:
            # Merged by min, like the surface seeds: whichever route out of a
            # contact cell is cheaper wins.  In practice the goal seed dominates
            # (0 for the exit leg), which is the point — it makes the goal its own
            # source instead of something reachable only via a surface.
            for key, cost in self._goal_seeds(goal).items():
                prev = seeds.get(key)
                if prev is None or cost < prev:
                    seeds[key] = cost
        if not seeds:
            return np.full((_W, _H), np.inf, np.float64)
        return self._fs.flood_physics_from_seeds(seeds)

    def field_for(
        self,
        d_star: np.ndarray,
        goal: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[np.ndarray]:
        """Memoised flood for ``d_star`` (keyed by array identity + goal)."""
        if not self.available or d_star is None:
            return None
        key = (id(d_star), goal)
        f = self._fields.get(key)
        if f is None:
            f = self._build_field(d_star, goal)
            self._fields[key] = f
        return f

    def _soft_field_for(
        self,
        d_star: np.ndarray,
        goal: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[np.ndarray]:
        """Mine-SOFT rescue flood, built lazily and only where the hard one is inf.

        ``mine_hard=True`` cuts every cell within a deadly mine's kill radius out
        of the free mask.  That is the right default — it is what stops the field
        ever pricing a position inside a kill disc.  But on the dense mine
        corridors this whole workstream exists for, a 6px grid with those cells
        removed can be *disconnected* even though the real ninja threads the gap:
        the discretisation is coarser than the manoeuvre.

        So: hard mask first, and only if it leaves a query unreachable do we fall
        back to the soft mask (mines traversable at ~9x cost).  The soft field is
        strongly mine-averse rather than mine-proof, which is the correct trade
        for a LAST-RESORT potential — an unreachable goal means no gradient at
        all, whereas a mine-averse gradient still points the right way, and PBRS
        policy-invariance means an approximate potential cannot make the optimal
        policy wrong.  Order matters: a position priced by the hard field never
        consults this.
        """
        if self._fs_soft is None:
            if self._soft_failed:
                return None
            try:
                fs = FreeSpaceDistanceSource(
                    list(self._map_data), name=self._name, mine_hard=False,
                    physics_cost=True,
                )
                if not fs.available or fs._free is None:
                    self._soft_failed = True
                    return None
                self._fs_soft = fs
            except Exception:
                self._soft_failed = True
                return None
        key = (id(d_star), goal)
        f = self._soft_fields.get(key)
        if f is None:
            hard, self._fs = self._fs, self._fs_soft
            try:
                f = self._build_field(d_star, goal)
            finally:
                self._fs = hard
            self._soft_fields[key] = f
        return f

    def cost_at(
        self,
        x: float,
        y: float,
        d_star: np.ndarray,
        goal: Optional[Tuple[float, float, float, float]] = None,
    ) -> float:
        """Flooded cost-to-go (frames) at ``(x, y)``; ``inf`` off-grid/unreachable."""
        cx = int(x / _CELL)
        cy = int(y / _CELL)
        if not (0 <= cx < _W and 0 <= cy < _H):
            return float("inf")
        f = self.field_for(d_star, goal)
        if f is not None:
            v = float(f[cx, cy])
            if np.isfinite(v):
                return v
        # The soft tier rescues UNREACHABILITY, never LETHALITY.  A cell the hard
        # mask cut out is either solid or inside a deadly mine's kill disc, and
        # both must stay ``inf`` — otherwise the rescue would hand back a finite
        # cost for standing in a kill radius on every mine cell in the level,
        # which is precisely what spec trap #3 forbids.  Only a cell that IS free
        # under the hard mask, yet unreachable because the mask disconnected the
        # region around it, falls through to the soft flood.
        fs = self._fs
        if fs is None or fs._free is None:
            return float("inf")
        if not fs._free[cx, cy]:
            # The query cell is cut from the hard mask.  That does NOT mean the
            # position is lethal, and conflating the two is a real bug.
            #
            # The mask is deliberately conservative for ROUTING: it cuts any cell
            # whose centre is within ``_MINE_KILL + _CELL`` = 20px of a deadly
            # mine, dilating the true 14px kill radius by a whole cell so that no
            # part of a 6px cell can be lethal.  It also samples the cell CENTRE,
            # which can sit several px nearer the mine than the query does.
            #
            # Measured on "550 bamboo training forest": the game spawns the ninja
            # at (258, 468), between two deadly mines each **21.63px** away — a
            # 7.6px margin over the 14px kill radius, i.e. comfortably SURVIVABLE
            # (the ninja is standing on a surface between them).  Its cell centre
            # (261, 471) is 17.49px from the nearest mine, under the 20px dilated
            # cut, so the mask erases a perfectly good standing position.
            #
            # So test LETHALITY at the actual position rather than trusting the
            # routing mask.  Genuinely inside a kill disc -> ``inf``.  Solid rock
            # -> ``inf`` (the ninja centre cannot be there; pricing it would
            # invent geometry).  Otherwise the position is occupiable and the mask
            # was merely being conservative, so snap to the nearest routable
            # value, bounded by the same +/-3 cell radius the surface lookup uses.
            #
            # Spec trap #3 is untouched: it protects ROUTING, and the flood still
            # never relaxes through a cut cell, so no path gains a shortcut
            # through a mine.  Only the LOOKUP recovers a survivable position.
            if self._position_is_lethal(x, y) or self._is_solid_cell(cx, cy):
                return float("inf")
            return self._nearest_free_value(cx, cy, f)
        soft = self._soft_field_for(d_star, goal)
        if soft is None:
            return float("inf")
        return float(soft[cx, cy])

    #: Snap radius in cells for a query that lands on a non-free cell.  Matches
    #: ``find_nearest_point``'s search_radius so the field and the surface lookup
    #: tolerate the same amount of "not quite in valid space".
    _SNAP_CELLS = 3

    def _position_is_lethal(self, x: float, y: float) -> bool:
        """Is the ninja centre at ``(x, y)`` actually inside a deadly kill disc?

        The exact game condition — ``distance(ninja_centre, mine) < NINJA_RADIUS +
        TOGGLE_MINE_RADIUS_TOGGLED`` — evaluated at the real position, not at a
        6px cell centre and without the mask's ``+_CELL`` routing dilation.
        """
        deadly = getattr(self._fs, "_deadly", None)
        if not deadly:
            return False
        r2 = _MINE_KILL_RADIUS * _MINE_KILL_RADIUS
        for mx, my in deadly:
            dx = x - mx
            dy = y - my
            if dx * dx + dy * dy < r2:
                return True
        return False

    def _is_solid_cell(self, cx: int, cy: int) -> bool:
        """True when a hard-mask-cut cell is SOLID rather than a mine kill cell.

        The soft mask keeps mine cells (traversable at ~9x cost) and drops only
        real geometry, so "cut by hard, kept by soft" is exactly "kill disc" and
        "cut by both" is exactly "solid".  Falls back to treating the cell as
        solid when the soft mask cannot be built, which is the conservative
        answer (no value invented).
        """
        if self._fs_soft is None and not self._soft_failed:
            try:
                fs = FreeSpaceDistanceSource(
                    list(self._map_data), name=self._name, mine_hard=False,
                    physics_cost=True,
                )
                self._fs_soft = fs if (fs.available and fs._free is not None) else None
                self._soft_failed = self._fs_soft is None
            except Exception:
                self._soft_failed = True
        if self._fs_soft is None:
            return True
        return not bool(self._fs_soft._free[cx, cy])

    def _nearest_free_value(self, cx: int, cy: int, f) -> float:
        """Min finite field value among free cells within ``_SNAP_CELLS``."""
        if f is None:
            return float("inf")
        free = self._fs._free
        best = float("inf")
        r = self._SNAP_CELLS
        for ix in range(max(0, cx - r), min(_W, cx + r + 1)):
            for iy in range(max(0, cy - r), min(_H, cy + r + 1)):
                if not free[ix, iy]:
                    continue
                v = float(f[ix, iy])
                if v < best:
                    best = v
        return best

    def coverage(self, d_star: np.ndarray, goal=None) -> float:
        """Fraction of FREE cells that the field prices finitely (diagnostics)."""
        f = self.field_for(d_star, goal)
        if f is None or self._fs is None or self._fs._free is None:
            return 0.0
        free = self._fs._free
        n_free = int(free.sum())
        if n_free == 0:
            return 0.0
        return float(np.isfinite(f[free]).sum()) / n_free
