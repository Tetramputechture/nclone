"""Does the surface lattice earn its cost, or is it only a seed generator?

The question (RESEARCH_pbrs_under_entities §6)
----------------------------------------------
Tier 1 of the potential — the exact, physics-derived surface lattice — is the
expensive tier.  It costs p50 3.5s and up to 206s per level, fails outright on
~13% of official levels, contributes 20.21% spurious local minima against the
flood's 0.25%, and its *values* are already known to be ignorable: turning on
``prefer_field`` (answer from the flood even where the surface lookup could
answer) moved Spearman by +0.006 and lost zero cells.

Its one remaining justification is that its ``d*`` **seeds** are what make tier
2 physics-aware rather than geometric.  Making it entity-aware is strictly more
expensive (more primitives, more arcs, per-door-configuration re-enumeration),
so that justification is worth about a whole workstream and should be measured
before it is paid for.

The structural fact that makes this cheap to measure
----------------------------------------------------
Seeds are merged into the flood by ``min`` and the goal disc is seeded in every
arm.  So the arms' seed sets are nested, and therefore::

    field(full) <= field(spawn) <= field(none)     pointwise, everywhere

Surface seeds can only ever *lower* a value.  That collapses "what does tier 1
contribute" into two measurable quantities:

* **bind rate** — the fraction of free cells where ``field(full) <
  field(none)``, i.e. where routing via a surface point genuinely beat flooding
  straight from the goal.  Where this is ~0, tier 1 changed nothing.
* **rescue** — cells that are ``inf`` under ``none`` but finite under ``full``.
  This is the coverage argument for tier 1 and it is the one that cannot be
  recovered by tuning the flood.

Both are computed on the same array, so they cost one extra flood per level.

The oracle
----------
Agreement is scored against **real winning play**, not against the lattice —
grading a potential by rank-correlating it with tier 1 is exactly the circular
oracle ``demo_potential_eval`` warns about, and here it would beg the question.
A verified demo gives the true cost-to-go at every frame for free: the number of
frames the expert actually needed from there.

Four metrics per demo per arm (same definitions as ``demo_potential_eval``):

``rho``       Spearman(cost, true frames remaining) — global ordering.
``progress``  fraction of expert steps where the cost strictly decreases.  PBRS
              pays on the per-step delta, so a field can rank well globally and
              still fight the expert on half its own moves.
``f_std``     std of the shaping term with Phi normalised as the runtime does.
``cover``     fraction of demo frames priced finitely at all.

Traces are re-simulated ONCE per demo and cached, because the arms differ only
in the potential and re-running ``NPlayHeadless`` per arm would quadruple the
cost of the sweep for identical trajectories.

Usage::

    python -m nclone.value_function.tools.seed_ablation \\
        datasets/phase2_planner_demos --limit 120 --workers 6 \\
        --out /tmp/seed_ablation.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: The ablation arms.  ``full`` is today's default; ``none`` needs no surface
#: graph at all, so if it matches ``full`` the entire tier-1 entity workstream
#: can be skipped.
ARMS = ("full", "spawn", "none")

#: Cells inside this radius of the goal are excluded from the local-minimum
#: count — the goal IS a minimum and counting it would put a floor under every
#: arm.  Matches the goal-disc exclusion used in the §3.2 measurement.
_GOAL_EXCLUDE_CELLS = 4


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation with tie averaging (avoids a scipy dependency)."""
    if a.size < 3:
        return float("nan")

    def _rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="stable")
        ranks = np.empty(v.size, np.float64)
        ranks[order] = np.arange(v.size, dtype=np.float64)
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


def _load_replay(path: str) -> Optional[Tuple[List[int], List[Tuple[int, int]]]]:
    """``(map_data, [(horizontal, jump), ...])`` from a compact V1 ``.replay``."""
    try:
        data = open(path, "rb").read()
        _ver, mlen, ilen = struct.unpack("<III", data[0:12])
        map_data = list(data[13:13 + mlen])
        raw = data[13 + mlen:13 + mlen + ilen]
        if not map_data or not raw:
            return None
        inputs = [(1 if b & 2 else (-1 if b & 4 else 0), b & 1) for b in raw]
        return map_data, inputs
    except Exception:
        return None


def _trace(map_data: List[int], inputs: Sequence[Tuple[int, int]]):
    """Re-simulate a tape; ``(positions, switch_flags)`` up to the winning frame.

    Returns None unless the tape actually wins on the CURRENT simulator.  The
    replay's own success flag is not trusted: a stale tape carries success=True
    while the re-simulation dies, which would admit a trajectory whose "true
    frames remaining" is a fiction.
    """
    from ...nplay_headless import NPlayHeadless

    nph = NPlayHeadless(render_mode=None, enable_animation=False,
                        enable_logging=False, enable_debug_overlay=False)
    try:
        nph.load_map_from_map_data(list(map_data))
        nph.reset()
        xs, sw = [], []
        for hor, jump in inputs:
            x, y = nph.ninja_position()
            xs.append((float(x), float(y)))
            sw.append(bool(nph.exit_switch_activated()))
            nph.tick(hor, jump)
            if nph.ninja_has_won():
                x, y = nph.ninja_position()
                xs.append((float(x), float(y)))
                sw.append(bool(nph.exit_switch_activated()))
                return np.asarray(xs, np.float64), np.asarray(sw, bool)
        return None
    finally:
        try:
            nph.exit()
        except Exception:
            pass


def _trace_one(args) -> dict:
    """Worker: load a replay, re-simulate it, persist the trajectory."""
    path, cache_dir = args
    name = os.path.basename(path)
    out = os.path.join(cache_dir, name + ".npz")
    rec = {"name": name, "path": path, "cache": out}
    if os.path.exists(out):
        rec["status"] = "cached"
        return rec
    loaded = _load_replay(path)
    if loaded is None:
        rec["status"] = "unreadable"
        return rec
    map_data, inputs = loaded
    try:
        tr = _trace(map_data, inputs)
    except Exception as exc:
        rec["status"] = f"error: {type(exc).__name__}: {exc}"
        return rec
    if tr is None:
        rec["status"] = "no_win"
        return rec
    pos, sw = tr
    # Written through a file handle, not a path: ``savez_compressed`` appends
    # ".npz" to any path that lacks it, so a ".tmp.<pid>" suffix would land the
    # archive somewhere the rename cannot find it.
    tmp = out + f".tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, pos=pos, sw=sw,
                            map_data=np.asarray(map_data, np.int32))
    os.replace(tmp, out)
    rec["status"] = "ok"
    rec["frames"] = int(len(pos))
    return rec


def _score_trajectory(costs: np.ndarray, gamma: float, weight: float
                      ) -> Optional[Dict[str, float]]:
    """The four demo metrics for one already-evaluated cost trajectory."""
    finite = np.isfinite(costs)
    cover = float(finite.mean())
    if int(finite.sum()) < 3:
        return None
    true_left = np.arange(len(costs) - 1, -1, -1, dtype=np.float64)
    rho = _spearman(costs[finite], true_left[finite])
    step_ok = finite[:-1] & finite[1:]
    progress = (float((costs[1:][step_ok] < costs[:-1][step_ok]).mean())
                if step_ok.any() else float("nan"))
    d0 = costs[0] if (np.isfinite(costs[0]) and costs[0] > 0) else np.nanmax(
        costs[finite])
    if not np.isfinite(d0) or d0 <= 0:
        return None
    phi = -np.clip(costs / d0, 0.0, 1.0)
    f = weight * (gamma * phi[1:] - phi[:-1])
    f_std = float(np.std(f[step_ok])) if step_ok.any() else float("nan")
    return {"rho": rho, "progress": progress, "f_std": f_std, "cover": cover}


def _local_minima(field: np.ndarray, free: np.ndarray,
                  goal_cell: Optional[Tuple[int, int]]) -> Optional[float]:
    """Fraction of finite free cells with no strictly-lower 8-neighbour.

    A spurious minimum is a trap: the shaping gradient points into it from every
    direction and there is no descending step out.  Vectorised with a shifted
    stack rather than a Python loop over ~58k cells.
    """
    fin = np.isfinite(field) & free
    if not fin.any():
        return None
    if goal_cell is not None:
        gx, gy = goal_cell
        r = _GOAL_EXCLUDE_CELLS
        xs = np.arange(field.shape[0])[:, None]
        ys = np.arange(field.shape[1])[None, :]
        fin &= ((xs - gx) ** 2 + (ys - gy) ** 2) > r * r
    if not fin.any():
        return None
    big = np.where(np.isfinite(field), field, np.inf)
    padded = np.full((field.shape[0] + 2, field.shape[1] + 2), np.inf)
    padded[1:-1, 1:-1] = big
    lower = np.zeros_like(fin)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nb = padded[1 + dx:1 + dx + field.shape[0],
                        1 + dy:1 + dy + field.shape[1]]
            lower |= nb < big
    return float((fin & ~lower).sum()) / float(fin.sum())


def _build_source(map_data, name, arm, workers):
    from ..lattice_distance import LatticeDistanceSource
    return LatticeDistanceSource(
        list(map_data), name=name, use_native=True, use_cache=True,
        workers=workers, airborne_field=True,
        # prefer_field is held ON in EVERY arm on purpose.  The question is
        # about the seeds, and leaving the surface READOUT in would let tier 1
        # answer directly on ~20% of cells and confound the two effects that
        # §3.3 already showed are separable.
        prefer_field=True, seed_mode=arm)


def _measure_one(args) -> dict:
    """Score every arm on one level, plus the bind/rescue diagnostics."""
    cache_path, gamma, weight, workers, arms = args
    rec: Dict[str, object] = {"name": os.path.basename(cache_path)[:-4]}
    try:
        z = np.load(cache_path)
        pos, sw = z["pos"], z["sw"]
        map_data = [int(v) for v in z["map_data"]]
    except Exception as exc:
        rec["error"] = f"load: {type(exc).__name__}: {exc}"
        return rec

    fields: Dict[str, np.ndarray] = {}
    free_mask = None
    for arm in arms:
        t0 = time.time()
        try:
            src = _build_source(map_data, rec["name"], arm, workers)
        except Exception as exc:
            rec[arm] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        build_s = time.time() - t0
        if not src.available:
            rec[arm] = {"error": "unavailable", "build_s": build_s}
            continue
        costs = np.array(
            [src.cost_to_go(float(x), float(y), bool(s))
             for (x, y), s in zip(pos, sw)], dtype=np.float64)
        m = _score_trajectory(costs, gamma, weight) or {}
        m["build_s"] = build_s
        m["lattice_available"] = bool(src.lattice_available)
        m["n_seed_cells"] = int(len(getattr(src._air, "_seed_cells", []) or []))
        m["d_spawn"] = (float(costs[0]) if np.isfinite(costs[0]) else None)
        # The pre-switch phase array is what the agent actually descends first.
        air = src._air
        if air is not None:
            d_star = src._d_comb
            f = air.field_for(d_star, src._goal_spec(d_star))
            if f is not None:
                fields[arm] = f
                free_mask = air._fs._free
                gc = None
                if src.switch is not None:
                    from ..free_space_distance import _CELL
                    gc = (int(src.switch[0] / _CELL), int(src.switch[1] / _CELL))
                m["local_minima"] = _local_minima(f, free_mask, gc)
                m["field_coverage"] = float(
                    np.isfinite(f[free_mask]).mean()) if free_mask is not None else None
        rec[arm] = m

    # Bind / rescue: exactly what tier 1's seeds bought over a pure goal flood.
    if "full" in fields and "none" in fields and free_mask is not None:
        a, c = fields["full"], fields["none"]
        both = np.isfinite(a) & np.isfinite(c) & free_mask
        n_free = int(free_mask.sum())
        rescue = np.isfinite(a) & ~np.isfinite(c) & free_mask
        # A tolerance, not equality: both arms run the same flood, so a cell the
        # seeds did not change comes back bit-identical; anything above this is
        # a genuine re-route, not float noise.  Differenced only where BOTH are
        # finite — elsewhere it is inf-inf = nan, which is not a comparison.
        bind = np.zeros_like(both)
        bind[both] = (c[both] - a[both]) > 1e-6
        rec["diag"] = {
            "n_free_cells": n_free,
            "frac_rescued_by_seeds": float(rescue.sum()) / n_free if n_free else None,
            "frac_bind": float(bind.sum()) / int(both.sum()) if both.any() else None,
            "bind_rel_p50": (float(np.median((c[bind] - a[bind]) / np.maximum(c[bind], 1e-9)))
                             if bind.any() else None),
            "bind_abs_p50": float(np.median(c[bind] - a[bind])) if bind.any() else None,
            # Sanity check on the nesting claim.  A violation means the seed sets
            # are not nested and every comparison below is invalid.
            "monotone_violations": int((a[both] - c[both] > 1e-6).sum()),
        }
    return rec


def _agg(vals: List[float]) -> Dict[str, float]:
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return {"n": 0}
    s = sorted(v)
    return {"n": len(v), "mean": sum(v) / len(v),
            "p10": s[int(0.10 * (len(s) - 1))],
            "p50": s[int(0.50 * (len(s) - 1))],
            "p90": s[int(0.90 * (len(s) - 1))]}


def summarize(records: List[dict], arms: Sequence[str]) -> dict:
    out: Dict[str, object] = {"n_levels": len(records)}
    for arm in arms:
        ms = [r[arm] for r in records if isinstance(r.get(arm), dict)
              and "error" not in r[arm]]
        out[arm] = {
            "n_scored": len(ms),
            "n_failed": sum(1 for r in records if isinstance(r.get(arm), dict)
                            and "error" in r[arm]),
            "rho": _agg([m.get("rho") for m in ms]),
            "progress": _agg([m.get("progress") for m in ms]),
            "f_std": _agg([m.get("f_std") for m in ms]),
            "cover": _agg([m.get("cover") for m in ms]),
            "local_minima": _agg([m.get("local_minima") for m in ms]),
            "field_coverage": _agg([m.get("field_coverage") for m in ms]),
            "build_s": _agg([m.get("build_s") for m in ms]),
            "n_seed_cells": _agg([m.get("n_seed_cells") for m in ms]),
        }
    # Paired deltas: the arms run on the SAME levels, so a paired comparison is
    # far tighter than differencing two independent percentile summaries.
    for arm in arms:
        if arm == "full":
            continue
        pairs = [(r["full"], r[arm]) for r in records
                 if isinstance(r.get("full"), dict) and isinstance(r.get(arm), dict)
                 and "error" not in r["full"] and "error" not in r[arm]]
        d: Dict[str, object] = {"n_paired": len(pairs)}
        # Direction of goodness per metric.  rho/progress/cover are better HIGH;
        # f_std (shaping variance the critic must absorb) and local_minima
        # (gradient traps) are better LOW.  Labelling every delta "better when
        # positive" would invert the conclusion on exactly the two metrics that
        # discriminate here, so the sign is carried explicitly.
        higher_is_better = {"rho": True, "progress": True, "cover": True,
                            "f_std": False, "local_minima": False}
        for k, hib in higher_is_better.items():
            deltas = [b.get(k) - a.get(k) for a, b in pairs
                      if a.get(k) is not None and b.get(k) is not None
                      and not math.isnan(a.get(k)) and not math.isnan(b.get(k))]
            d[k] = _agg(deltas)
            if deltas:
                pos = sum(1 for x in deltas if x > 1e-9)
                neg = sum(1 for x in deltas if x < -1e-9)
                d[k]["higher_is_better"] = hib
                d[k]["n_better_than_full"] = pos if hib else neg
                d[k]["n_worse_than_full"] = neg if hib else pos
        out[f"delta_{arm}_minus_full"] = d
    diags = [r["diag"] for r in records if isinstance(r.get("diag"), dict)]
    if diags:
        out["seed_contribution"] = {
            "n": len(diags),
            "frac_bind": _agg([d.get("frac_bind") for d in diags]),
            "frac_rescued_by_seeds": _agg([d.get("frac_rescued_by_seeds") for d in diags]),
            "bind_rel_p50": _agg([d.get("bind_rel_p50") for d in diags]),
            "bind_abs_p50": _agg([d.get("bind_abs_p50") for d in diags]),
            "levels_with_monotone_violation": sum(
                1 for d in diags if d.get("monotone_violations")),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("demo_dirs", nargs="+", help="directories of .replay files")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--native-threads", type=int, default=2,
                    help="rayon threads per level; keep workers*this <= nproc")
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--weight", type=float, default=1.0)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--cache-dir", default="/tmp/seed_ablation_traces")
    ap.add_argument("--trace-only", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    arms = tuple(a for a in args.arms.split(",") if a)
    paths: List[str] = []
    for d in args.demo_dirs:
        for f in sorted(os.listdir(d)):
            if f.endswith(".replay"):
                paths.append(os.path.join(d, f))
    if args.limit:
        paths = paths[:args.limit]
    os.makedirs(args.cache_dir, exist_ok=True)
    print(f"[ablation] {len(paths)} demos, arms={arms}, workers={args.workers}")

    # Stage 1 — trace once per demo.  NPlayHeadless stands up a pygame display
    # per instance; the dummy driver is what keeps that from serialising the
    # pool under software GL.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    t0 = time.time()
    traced: List[str] = []
    stats: Dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_trace_one, (p, args.cache_dir)) for p in paths]
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            stats[str(r.get("status"))] = stats.get(str(r.get("status")), 0) + 1
            if r.get("status") in ("ok", "cached"):
                traced.append(r["cache"])
            if (i + 1) % 50 == 0:
                print(f"  trace {i+1}/{len(futs)} ({time.time()-t0:.0f}s)")
    print(f"[ablation] traced {len(traced)}/{len(paths)} in {time.time()-t0:.0f}s "
          f"{stats}")
    if args.trace_only:
        return 0

    # Stage 2 — score every arm against the cached trajectories.
    t1 = time.time()
    work = [(c, args.gamma, args.weight, max(2, args.native_threads), arms)
            for c in sorted(traced)]
    records: List[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_measure_one, w) for w in work]
        for i, fut in enumerate(as_completed(futs)):
            try:
                records.append(fut.result())
            except Exception as exc:
                records.append({"error": f"{type(exc).__name__}: {exc}"})
            if (i + 1) % 25 == 0:
                print(f"  score {i+1}/{len(futs)} ({time.time()-t1:.0f}s)")

    summary = summarize(records, arms)
    summary["elapsed_s"] = time.time() - t0
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "records": records}, f, indent=2)
        print(f"[ablation] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
