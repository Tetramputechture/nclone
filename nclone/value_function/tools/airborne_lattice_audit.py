"""Measurement harness for the airborne-aware surface lattice (SPEC_airborne_lattice).

Produces the numbers the spec's acceptance criteria are stated against:

* **A2 severance** — the fraction of official levels where
  :meth:`LatticeDistanceSource.spawn_finite` is ``False``, i.e. the lattice has
  no plan at the spawn and the agent silently trains on the free-space fallback.
  This is the number that must drop by >=50% once the airborne extension lands,
  and it has to be recorded *before* changing anything.
* **A5 cost** — per-level precompute wall time (p50 / p90 / p99).  The spec
  warns of a recorded 380-480s pathology on ``horizontal_corridor`` levels where
  the search itself is 0.0s, so we report percentiles, never the mean.
* **A3 oracle agreement** — when a coverage JSON of planner win-frame counts is
  supplied, the Spearman correlation of lattice ``d*`` at spawn against the
  planner's actual frame cost, plus the optimism ratio.

Each level runs in its own subprocess with a hard timeout so a single pathology
cannot stall the sweep; timeouts are reported as a distinct outcome rather than
being silently folded into "unavailable".

Usage::

    python -m nclone.value_function.tools.airborne_lattice_audit \\
        --manifest ../npp-rl/datasets/official_compatible_manifest_native.json \\
        --limit 200 --workers 8 --timeout 120 --out /tmp/a2_baseline.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import signal
import time
from concurrent.futures import (
    ProcessPoolExecutor, as_completed, TimeoutError as FutTimeout,
)
from typing import Dict, List, Optional, Tuple

# Outcome codes recorded per level.
OUTCOME_FINITE = "finite"          # lattice available AND spawn_finite
OUTCOME_SEVERED = "severed"        # lattice available but d* at spawn is inf
OUTCOME_UNAVAILABLE = "unavailable"  # construction failed (unsupported/too large)
OUTCOME_TIMEOUT = "timeout"
OUTCOME_ERROR = "error"

#: Free cells sampled per level for the coverage statistic (see
#: ``_free_cell_coverage``).  400 gives a +/-2.5pp standard error on a fraction
#: near 0.5, which is far finer than the effect being measured (20% -> 100%).
_COVERAGE_SAMPLES = 400


class _LevelTimeout(Exception):
    """Raised by the serial path's SIGALRM handler when a level runs too long."""


@contextlib.contextmanager
def _level_timeout(seconds: float):
    """Per-level wall-clock guard for the serial path.

    SIGALRM only fires in the main thread of the main interpreter, which is
    exactly where the serial path runs.  It interrupts Python bytecode, so a
    level parked inside a long native call is cut at the next check rather than
    instantly — good enough to keep a sweep moving past a pathology.
    """
    if not seconds or seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _fire(_signum, _frame):
        raise _LevelTimeout()

    prev = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev)


def _free_cell_coverage(src, map_data) -> Optional[float]:
    """Fraction of the level's FREE cells at which the lattice has a finite cost.

    This is the headline severance number, and the denominator matters: counting
    *all* grid cells mixes in solid rock, where having no cost-to-go is correct
    and uninteresting.  Over free cells the quantity is exactly "how much of the
    space the ninja can actually occupy does the potential cover".
    """
    import numpy as np
    from nclone.value_function.free_space_distance import (
        FreeSpaceDistanceSource, _CELL, _W, _H,
    )
    try:
        fs = FreeSpaceDistanceSource(
            list(map_data), mine_hard=True, physics_cost=True)
        if not fs.available or fs._free is None:
            return None
        free = fs._free
        d_star, _ = src._phase_arrays(False, None)
        cells = np.argwhere(free)
        if cells.size == 0:
            return None
        # Sample rather than scan.  ``_cost_at`` runs find_nearest_point over a
        # 7x7 cell neighbourhood for two surface types, so an exhaustive pass is
        # ~17.6k such lookups per level and dominates the whole sweep.  A few
        # hundred samples pin a per-level FRACTION to well inside the precision
        # anyone reads it at, and the seed is fixed so the A/B arms sample the
        # SAME cells and their difference is exact.
        rng = np.random.default_rng(12345)
        if len(cells) > _COVERAGE_SAMPLES:
            cells = cells[rng.choice(len(cells), _COVERAGE_SAMPLES, replace=False)]
        n_finite = 0
        for cx, cy in cells:
            v = src._cost_at(int(cx) * _CELL + _CELL / 2.0,
                             int(cy) * _CELL + _CELL / 2.0, d_star)
            if v is not None and np.isfinite(v):
                n_finite += 1
        return n_finite / len(cells)
    except Exception:
        return None


def _diagnose(src, spawn) -> dict:
    """Explain *why* a lattice has (or lacks) a plan at the spawn.

    ``spawn_finite`` collapses two very different failures into one ``False``:

    * ``no_surface_point`` — ``_min_dstar_at`` found no floor/wall point within
      its lookup radius at all, so the query returns ``None``.  This is the
      *projection* failure of spec §2.4: the position is in open space and the
      lattice simply has nothing to say there.  It fires for ANY airborne query,
      not only at spawn, which matters because 85.8% of route frames are airborne.
    * ``all_bins_unreachable`` — a surface point was found but every velocity bin
      at it has ``d* = inf``.  This is genuine graph severance: the node exists
      and is disconnected from the goal.

    The two need opposite fixes (D3 pricing vs D2 connectivity), so the baseline
    has to separate them or the follow-up work is aimed blind.
    """
    import numpy as np
    from nclone.value_function.surface_graph import find_nearest_point
    from nclone.value_function.surface_extraction import SURFACE_FLOOR, SURFACE_WALL

    out: Dict[str, object] = {}
    g = getattr(src, "_graph", None)
    if g is None:
        return out
    sx, sy = float(spawn[0]), float(spawn[1])

    has_point = any(
        find_nearest_point(g, sx, sy, st) is not None
        for st in (SURFACE_FLOOR, SURFACE_WALL)
    )
    out["spawn_has_surface_point"] = has_point

    d_comb = getattr(src, "_d_comb", None)
    if d_comb is not None:
        finite_mask = np.isfinite(d_comb)
        out["n_nodes"] = int(d_comb.size)
        out["frac_nodes_finite_dcomb"] = float(finite_mask.mean())
    d_ex = getattr(src, "_d_ex", None)
    if d_ex is not None:
        out["frac_nodes_finite_dex"] = float(np.isfinite(d_ex).mean())

    if not out.get("spawn_has_surface_point", True):
        out["severance_reason"] = "no_surface_point"
    elif out.get("frac_nodes_finite_dcomb", 1.0) == 0.0:
        out["severance_reason"] = "no_finite_dcomb_anywhere"
    else:
        out["severance_reason"] = "all_bins_unreachable"
    return out


def _measure_one(args) -> dict:
    """Build the lattice for one level and report severance + cost.

    Runs in a worker process: imports are done inside so the parent stays cheap
    and a crashed worker cannot poison the parent's module state.
    """
    map_path, cache_dir, use_native, native_threads, airborne = args
    rec: Dict[str, object] = {"path": map_path, "name": os.path.basename(map_path),
                              "airborne_field": airborne}
    t0 = time.time()
    try:
        from nclone.value_function.lattice_distance import LatticeDistanceSource
        import nclone_native

        # Read the .map bytes directly rather than through NPlayHeadless.  That
        # class stands up a pygame display per instance, and under software GL
        # (llvmpipe/zink here) the display setup serialises across processes:
        # measured 4-9% CPU per worker with the sweep otherwise idle.  load_map
        # itself is only ``[int(b) for b in open(path,'rb').read()]``.
        with open(map_path, "rb") as f:
            map_data = [int(b) for b in f.read()]

        sim = nclone_native.Simulator.load(list(map_data))
        spawn = sim.get_ninja_pos()
        rec["spawn"] = [float(spawn[0]), float(spawn[1])]

        src = LatticeDistanceSource(
            map_data,
            name=os.path.basename(map_path),
            use_native=use_native,
            # Always on: the airborne extension is built AFTER the lattice
            # precompute and does not change it, so an A/B over the same corpus
            # shares one cache and the second pass measures only the extension.
            use_cache=True,
            # ``workers`` doubles as the native kernel's thread count, and
            # ``workers=1`` is the special value that makes it spawn cpu_count()
            # rayon threads.  With P audit processes running concurrently that is
            # P x cpu_count() threads (measured: load 85 on 16 cores), which
            # distorts every wall-clock number A5 is supposed to report.  Keep
            # ``processes x native_threads <= cpu_count()``.
            workers=max(2, int(native_threads)),
            airborne_field=airborne,
        )
        rec["build_s"] = time.time() - t0
        if not src.available:
            rec["outcome"] = OUTCOME_UNAVAILABLE
            return rec

        finite = bool(src.spawn_finite(spawn[0], spawn[1]))
        rec["outcome"] = OUTCOME_FINITE if finite else OUTCOME_SEVERED
        rec.update(_diagnose(src, spawn))
        # d* at spawn (phase 0) — the A3 oracle quantity.  Recorded even when
        # severed so the distribution of inf-vs-finite is inspectable.
        c = src.cost_to_go(spawn[0], spawn[1], switch_activated=False)
        rec["d_spawn"] = float(c) if math.isfinite(c) else None
        rec["free_cell_coverage"] = _free_cell_coverage(src, map_data)
        return rec
    except Exception as exc:  # pragma: no cover - diagnostics path
        rec["outcome"] = OUTCOME_ERROR
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["build_s"] = time.time() - t0
        return rec


def _percentiles(vals: List[float], qs=(50, 90, 99)) -> Dict[str, float]:
    if not vals:
        return {f"p{q}": float("nan") for q in qs}
    s = sorted(vals)
    out = {}
    for q in qs:
        k = (len(s) - 1) * q / 100.0
        lo, hi = int(math.floor(k)), int(math.ceil(k))
        out[f"p{q}"] = float(s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo))
    return out


def _spearman(xs: List[float], ys: List[float]) -> float:
    """Spearman rank correlation with average ranks for ties."""
    n = len(xs)
    if n < 3:
        return float("nan")

    def ranks(v: List[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return float(num / (dx * dy)) if dx > 0 and dy > 0 else float("nan")


def load_manifest(path: str, root_override: Optional[str] = None) -> List[str]:
    """Return absolute map paths from a compatible-manifest JSON."""
    with open(path) as f:
        man = json.load(f)
    entries = man.get("compatible") or []
    # The manifest stores paths relative to the *repo root* of whatever tool wrote
    # it, which is rarely the manifest's own directory.  Try the plausible bases
    # (manifest dir, its parent, cwd) and keep the first that actually resolves;
    # --official-root short-circuits the whole guess.
    mdir = os.path.dirname(os.path.abspath(path))
    bases = [mdir, os.path.dirname(mdir), os.getcwd()]
    paths = []
    for e in entries:
        if root_override and isinstance(e, dict):
            paths.append(os.path.join(root_override, e["folder"], e["name"]))
            continue
        p = e["path"] if isinstance(e, dict) else str(e)
        if os.path.isabs(p):
            paths.append(p)
            continue
        resolved = None
        for b in bases:
            cand = os.path.normpath(os.path.join(b, p))
            if os.path.exists(cand):
                resolved = cand
                break
        paths.append(resolved or os.path.normpath(os.path.join(mdir, p)))
    return paths


def summarize(records: List[dict], oracle: Optional[Dict[str, float]] = None) -> dict:
    """Roll per-level records up into the spec's A2 / A5 / A3 numbers."""
    n = len(records)
    counts: Dict[str, int] = {}
    for r in records:
        counts[str(r.get("outcome"))] = counts.get(str(r.get("outcome")), 0) + 1

    n_avail = counts.get(OUTCOME_FINITE, 0) + counts.get(OUTCOME_SEVERED, 0)
    severed_frac = (counts.get(OUTCOME_SEVERED, 0) / n_avail) if n_avail else float("nan")
    # Severance over ALL attempted levels too: an "unavailable" lattice is just as
    # absent from training as a severed one, so both denominators are reported.
    no_signal = n - counts.get(OUTCOME_FINITE, 0)

    build_times = [float(r["build_s"]) for r in records if r.get("build_s") is not None]

    out = {
        "n_levels": n,
        "counts": counts,
        "a2_severed_fraction_of_available": severed_frac,
        "a2_no_signal_fraction_of_all": (no_signal / n) if n else float("nan"),
        "a5_build_seconds": _percentiles(build_times),
        "a5_build_max": max(build_times) if build_times else float("nan"),
    }

    # Why the severed levels are severed — D3 (pricing) vs D2 (connectivity).
    reasons: Dict[str, int] = {}
    for r in records:
        if r.get("outcome") == OUTCOME_SEVERED:
            k = str(r.get("severance_reason", "unknown"))
            reasons[k] = reasons.get(k, 0) + 1
    out["a2_severance_reasons"] = reasons

    cov = [float(r["free_cell_coverage"]) for r in records
           if r.get("free_cell_coverage") is not None]
    if cov:
        out["free_cell_coverage"] = {
            "n": len(cov),
            "mean": sum(cov) / len(cov),
            **_percentiles(cov, (10, 50, 90)),
        }

    if oracle:
        pairs: List[Tuple[float, float]] = []
        for r in records:
            if r.get("outcome") != OUTCOME_FINITE or r.get("d_spawn") is None:
                continue
            key = str(r.get("name"))
            if key in oracle:
                pairs.append((float(r["d_spawn"]), float(oracle[key])))
        if pairs:
            ds = [p[0] for p in pairs]
            fr = [p[1] for p in pairs]
            ratios = [d / f for d, f in pairs if f > 0]
            out["a3"] = {
                "n_joined": len(pairs),
                "spearman": _spearman(ds, fr),
                "ratio_p50": _percentiles(ratios, (50,))["p50"],
                # A3: d* must not be systematically optimistic by >2x, and must
                # clear 0.5x planner frames on >=90% of levels.
                "frac_ratio_ge_0p5": sum(1 for r_ in ratios if r_ >= 0.5) / len(ratios),
                "frac_ratio_ge_2": sum(1 for r_ in ratios if r_ >= 2.0) / len(ratios),
            }
        else:
            out["a3"] = {"n_joined": 0}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True,
                    help="compatible-manifest JSON (uses its 'compatible' list)")
    ap.add_argument("--official-root", default=None,
                    help="override the manifest's map root")
    ap.add_argument("--limit", type=int, default=200,
                    help="sample at most N levels (0 = all)")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="per-level hard timeout in seconds")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--no-native", action="store_true")
    ap.add_argument("--native-threads", type=int, default=2,
                    help="rayon threads per level; keep workers*this <= nproc")
    ap.add_argument("--airborne-field", dest="airborne_field",
                    action="store_true", default=None,
                    help="force the airborne extension ON (A2 'after' arm)")
    ap.add_argument("--no-airborne-field", dest="airborne_field",
                    action="store_false",
                    help="force the airborne extension OFF (A2 baseline arm)")
    ap.add_argument("--oracle", default=None,
                    help="JSON {level_name: planner_win_frames} for A3")
    ap.add_argument("--out", default=None, help="write per-level records + summary")
    args = ap.parse_args(argv)

    paths = load_manifest(args.manifest, args.official_root)
    paths = [p for p in paths if os.path.exists(p)]
    if args.limit and args.limit < len(paths):
        random.Random(args.seed).shuffle(paths)
        paths = paths[: args.limit]
    print(f"[audit] {len(paths)} levels, {args.workers} workers, "
          f"timeout {args.timeout}s")

    work = [(p, args.cache_dir, not args.no_native, args.native_threads,
             args.airborne_field)
            for p in paths]
    records: List[dict] = []
    t0 = time.time()
    if args.workers <= 1:
        # Genuine serial path, not a 1-worker pool.  The native kernel is already
        # rayon-parallel inside a level, so process-level fan-out mostly buys
        # contention: measured 0.06-0.64s/level serially against workers sitting
        # at 4-9% CPU under the pool.  Serial is also the only mode where a
        # per-level wall time means anything for A5.
        for i, w in enumerate(work):
            try:
                with _level_timeout(args.timeout):
                    records.append(_measure_one(w))
            except _LevelTimeout:
                # A5 exists because of a recorded 380-480s precompute pathology.
                # Without this the sweep silently parks on one such level and
                # reports nothing at all, which is how the p99 stays unmeasured.
                print(f"  TIMEOUT after {args.timeout:.0f}s: "
                      f"{os.path.basename(w[0])}")
                records.append({"path": w[0], "name": os.path.basename(w[0]),
                                "outcome": OUTCOME_TIMEOUT,
                                "build_s": float(args.timeout)})
            except Exception as exc:
                records.append({"path": w[0], "name": os.path.basename(w[0]),
                                "outcome": OUTCOME_ERROR,
                                "error": f"{type(exc).__name__}: {exc}"})
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(work)} done ({time.time()-t0:.0f}s)")
        return _finish(records, args, t0)
    # Wall-clock budget for the whole sweep.  A genuine *per-level* timeout would
    # need to kill the worker mid-native-call, which ProcessPoolExecutor cannot
    # do; instead we drain completions against one overall deadline and record
    # every future still outstanding when it expires as a timeout.  Sized as
    # "every level gets its share, plus slack for the known 380-480s pathology".
    deadline = t0 + args.timeout * max(1, len(work) / max(1, args.workers))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_measure_one, w): w[0] for w in work}
        done = 0
        try:
            for fut in as_completed(list(futs), timeout=max(1.0, deadline - time.time())):
                path = futs[fut]
                try:
                    records.append(fut.result())
                except Exception as exc:
                    records.append({"path": path, "name": os.path.basename(path),
                                    "outcome": OUTCOME_ERROR,
                                    "error": f"{type(exc).__name__}: {exc}"})
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(work)} done ({time.time()-t0:.0f}s)")
        except FutTimeout:
            print(f"[audit] overall deadline hit at {time.time()-t0:.0f}s")
        for fut, path in futs.items():
            if not fut.done():
                records.append({"path": path, "name": os.path.basename(path),
                                "outcome": OUTCOME_TIMEOUT})
                fut.cancel()

    return _finish(records, args, t0)


def _finish(records: List[dict], args, t0: float) -> int:
    """Summarise, print and optionally persist a completed sweep."""
    oracle = None
    if args.oracle:
        with open(args.oracle) as f:
            oracle = {str(k): float(v) for k, v in json.load(f).items()}

    summary = summarize(records, oracle)
    summary["elapsed_s"] = time.time() - t0
    print(json.dumps(summary, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "records": records}, f, indent=2)
        print(f"[audit] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
