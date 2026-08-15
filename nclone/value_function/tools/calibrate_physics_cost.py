"""Score the free-space PHYSICS cost model against the surface lattice's d*.

The physics flood (``FreeSpaceDistanceSource(physics_cost=True)``) exists to buy
most of the surface lattice's physics awareness at ~0.1 s/level instead of the
lattice's 5 s - 18 min. This tool measures how much of it we actually bought.

Ground truth is ``LatticeAudit``'s ``d_comb`` — cost-to-completion in frames at
every surface point. A PBRS potential only has to RANK states, so the metric is
**Spearman rho** between the two fields over the lattice's surface points, and
the baseline to beat is the legacy geometric flood's rho on the same points.

Usage::

    python -m nclone.value_function.tools.calibrate_physics_cost \\
        nclone/test_maps/jump-then-fall nclone/test_maps/di-required

    # sweep the one constant physics does not hand us
    python -m nclone.value_function.tools.calibrate_physics_cost \\
        --sweep-air-k 0,0.5,1,2,4 <maps...>

Levels whose lattice exceeds ``--max-nodes`` are skipped rather than waited on:
those are exactly the levels the physics flood exists to serve, and no oracle is
available for them by construction.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, ties averaged. Avoids a scipy dependency."""
    if a.size < 3:
        return float("nan")

    def _rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="stable")
        ranks = np.empty(v.size, np.float64)
        ranks[order] = np.arange(v.size, dtype=np.float64)
        # Average ranks within tie groups so ties don't fabricate ordering.
        sv = v[order]
        i = 0
        while i < sv.size:
            j = i + 1
            while j < sv.size and sv[j] == sv[i]:
                j += 1
            if j - i > 1:
                ranks[order[i:j]] = ranks[order[i:j]].mean()
            i = j
        return ranks

    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _lattice_truth(map_path: str, max_nodes: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """``(points_xy, d_comb)`` from the lattice oracle, or None if unavailable.

    Hits ``LatticeAudit``'s on-disk d* cache when warm; otherwise builds, which
    is the expensive path this whole exercise is trying to avoid paying at
    training time.
    """
    from .lattice_audit import LatticeAudit

    try:
        audit = LatticeAudit(map_path=map_path, max_nodes=max_nodes)
        pre = audit.precompute()
    except Exception as exc:  # LatticeTooLargeError and friends
        print(f"    lattice unavailable: {type(exc).__name__}: {exc}")
        return None

    d_comb = np.asarray(pre["d_comb"], dtype=np.float64)
    pts = pre["graph"].surface_points
    # d_comb is indexed by NODE (surface point x velocity bin); collapse to the
    # per-point optimum, which is the cost-to-go the point actually offers.
    n_pts = len(pts)
    if d_comb.size != n_pts:
        if d_comb.size % n_pts:
            print(f"    d_comb size {d_comb.size} not a multiple of {n_pts} points")
            return None
        d_comb = d_comb.reshape(n_pts, -1).min(axis=1)
    xy = np.array([(p.x, p.y) for p in pts], dtype=np.float64)
    return xy, d_comb


def _score(map_path: str, max_nodes: int) -> Optional[dict]:
    from ..free_space_distance import FreeSpaceDistanceSource as FS
    from .physics_route_spike import _load_map_data

    name = os.path.basename(map_path)
    truth = _lattice_truth(map_path, max_nodes)
    if truth is None:
        return None
    xy, d_star = truth

    map_data = _load_map_data(map_path)
    out = {"name": name, "n_points": int(xy.shape[0])}
    for tag, physics in (("geo", False), ("phys", True)):
        t0 = time.time()
        src = FS(map_data, physics_cost=physics)
        out[f"{tag}_ms"] = (time.time() - t0) * 1000.0
        if not src.available:
            out[f"{tag}_rho"] = float("nan")
            continue
        # cost_at_nearest_free, not cost_to_go: surface points sit ON geometry,
        # so their own 6px cell is usually solid.
        vals = np.array(
            [src.cost_at_nearest_free(float(x), float(y), False) for x, y in xy],
            dtype=np.float64,
        )
        ok = np.isfinite(vals) & np.isfinite(d_star)
        out[f"{tag}_cov"] = float(ok.mean()) if ok.size else 0.0
        out[f"{tag}_rho"] = _spearman(vals[ok], d_star[ok])
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("maps", nargs="+", help="map files to score")
    ap.add_argument("--max-nodes", type=int, default=18000,
                    help="skip levels whose lattice exceeds this (default 18000)")
    ap.add_argument("--sweep-air-k", default=None,
                    help="comma-separated _AIR_UNSUPPORTED_K values to sweep")
    args = ap.parse_args(argv)

    from .. import free_space_distance as fsd

    k_values = ([float(v) for v in args.sweep_air_k.split(",")]
                if args.sweep_air_k else [fsd._AIR_UNSUPPORTED_K])

    for k in k_values:
        fsd._AIR_UNSUPPORTED_K = k
        if len(k_values) > 1:
            print(f"\n=== _AIR_UNSUPPORTED_K = {k} ===")
        print(f"{'level':30s} {'pts':>6s} {'geo rho':>9s} {'phys rho':>9s} "
              f"{'delta':>8s} {'geo ms':>8s} {'phys ms':>8s}")
        rows = []
        for m in args.maps:
            r = _score(m, args.max_nodes)
            if r is None:
                continue
            rows.append(r)
            print(f"{r['name']:30s} {r['n_points']:6d} {r['geo_rho']:9.3f} "
                  f"{r['phys_rho']:9.3f} {r['phys_rho'] - r['geo_rho']:+8.3f} "
                  f"{r['geo_ms']:8.0f} {r['phys_ms']:8.0f}")
        if rows:
            g = np.nanmean([r["geo_rho"] for r in rows])
            p = np.nanmean([r["phys_rho"] for r in rows])
            print(f"{'MEAN':30s} {len(rows):6d} {g:9.3f} {p:9.3f} {p - g:+8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
