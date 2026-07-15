"""Free-space route cost — a CHEAP difficulty proxy for synthetic level gen.

Difficulty is approximated by the geometric route the ninja must traverse:

    spawn ──▶ exit switch ──▶ exit door

measured as a **mine-soft geodesic distance over the fast collision/free-space
grid** (:class:`~nclone.value_function.free_space_distance.FreeSpaceDistanceSource`),
*not* the physics lattice. This is ~100–1000× cheaper than the lattice ``d*`` (one
Dijkstra per goal over a fixed grid) and a good monotone estimate of how far /
contorted the required route is and whether deadly mines obstruct it:

* trivial   — switch adjacent to door, spawn right beside the switch  → small cost
* hard      — switch and door far apart and/or a mine field between them → large cost

Caveats (see ``docs/LEVEL_POOL_UED_PLAN.md``):

* It is a *geometric* estimate. Geometric reachability is **necessary but not
  sufficient** for physical solvability — a free-space corridor can still be
  unjumpable. So ``route_cost`` is the right *difficulty* signal and a good
  *necessary-condition* reject (``inf`` ⇒ no geometric path ⇒ almost certainly
  unsolvable), but final solvability still needs the physics gate.
* ``inf`` means the switch is unreachable from the spawn, or the door from the
  switch, over free space — i.e. a wall or mine wall severs the route.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from nclone.value_function.free_space_distance import FreeSpaceDistanceSource


def route_cost(map_data: List[int]) -> float:
    """``spawn → switch → door`` free-space cost; ``inf`` if no geometric path.

    Uses the level's own ninja spawn (read from a native sim, matching how
    ``generate_levels`` reads it). Both legs must be geometrically reachable.
    """
    md = list(map_data)
    src = FreeSpaceDistanceSource(md)
    if not src.available or src.switch is None or src.exit is None:
        return float("inf")
    try:
        import nclone_native as Nnative

        sim = Nnative.Simulator.load(md)
        sx, sy = sim.get_ninja_pos()
    except Exception:
        return float("inf")
    return cost_from_parts(src, float(sx), float(sy))


def cost_from_parts(
    src: FreeSpaceDistanceSource, spawn_x: float, spawn_y: float
) -> float:
    """``route_cost`` given a prebuilt source + an explicit candidate spawn.

    Lets a caller reuse one :class:`FreeSpaceDistanceSource` (one pair of Dijkstra
    fields) to score many candidate spawns — the difficulty-targeted spawn
    placement in ``generate_levels`` does exactly this. Query points are snapped
    to the nearest free cell (they sit on surfaces), so a bare exact-pixel read
    is not enough.
    """
    if not src.available or src.switch is None or src.exit is None:
        return float("inf")
    leg1 = src.cost_at_nearest_free(float(spawn_x), float(spawn_y), switch_activated=False)
    leg2 = src.cost_at_nearest_free(src.switch[0], src.switch[1], switch_activated=True)
    if not (math.isfinite(leg1) and math.isfinite(leg2)):
        return float("inf")
    return float(leg1 + leg2)


def fit_edges(costs, n_bands: int = 5) -> List[float]:
    """``n_bands-1`` internal quantile edges over the FINITE route costs.

    Mirrors :func:`nclone.levelgen.labeling.difficulty_edges` but over float
    costs, dropping the ``inf`` (geometrically-unsolvable) tail so it does not
    skew the quantiles.
    """
    vals = sorted(float(c) for c in costs if c is not None and math.isfinite(c) and c > 0)
    if len(vals) < n_bands:
        return []
    edges: List[float] = []
    for k in range(1, n_bands):
        q = k / n_bands
        idx = min(len(vals) - 1, int(round(q * (len(vals) - 1))))
        edges.append(vals[idx])
    return edges


def assign_band(cost: float, edges) -> str:
    """Map a route cost to a ``DIFFICULTY_BANDS`` name; ``inf`` → ``"unknown"``.

    Higher cost ⇒ harder, so this matches the demo-length banding semantics in
    :func:`nclone.levelgen.labeling.assign_band` (reused for finite costs).
    """
    if cost is None or not math.isfinite(cost):
        return "unknown"
    from nclone.levelgen.labeling import assign_band as _assign

    return _assign(cost, edges)


__all__ = ["route_cost", "cost_from_parts", "fit_edges", "assign_band"]
