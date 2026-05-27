#!/usr/bin/env python3
"""Batch-precompute the lattice ``d*`` for a corpus of maps (ML/RL pipeline).

Wiring the lattice ``d*`` into the reward (``ReachabilityConfig.use_lattice_potential``)
makes each training env build the per-level ``d*`` on its first reset.  That
precompute is fast (~2–8 s, native) and **cached on disk** by the map_data hash
(see ``LatticeAudit._cache_path``; relocate with ``NCLONE_LATTICE_CACHE_DIR``).
This tool **pre-populates that shared cache** for a whole training corpus so the
workers pay ~0 at reset, and reports the **d\\* coverage** (fraction of levels
with a finite spawn cost-to-go) — the planner's guidance-coverage metric for RL
(memory ``planner-benchmark-execution-bound``: ~90% on supported levels).

Throughput note (memory ``corpus-throughput-conserved-work``): work is conserved
across cores, so don't over-slice.  Run a modest number of map-level processes,
each with a healthy native-thread count; the real scaling lever is across
MACHINES (levels are independent — shard the list and run this per machine).

Engine API (importable, e.g. from an npp-rl corpus script):

    from nclone.value_function.tools.precompute_corpus import precompute_corpus
    summary = precompute_corpus(
        [(level_id, map_data), ...],     # (name, raw map_data list)
        processes=6, workers=4,
    )

CLI (map files / a directory / a list file)::

    NCLONE_LATTICE_CACHE_DIR=/data/lattice_cache \
    python -m nclone.value_function.tools.precompute_corpus \
        --maps-dir nclone/maps/official/SL --processes 6 --workers 4
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .lattice_audit import LatticeAudit, _load_map_data
from ..edge_risk import DEFAULT_W_RISK, DEFAULT_W_LOOP, DEFAULT_RISK_CAP


def precompute_one(
    name: str,
    map_data: List[int],
    *,
    w_risk: float = DEFAULT_W_RISK,
    w_loop: float = DEFAULT_W_LOOP,
    risk_cap: float = DEFAULT_RISK_CAP,
    workers: int = 4,
    use_native: bool = True,
    use_cache: bool = True,
) -> Dict:
    """Precompute (and cache) the lattice ``d*`` for one map.

    Fault-tolerant: returns a result dict with ``ok=False`` + ``error`` on any
    failure instead of raising, so one bad level never aborts the corpus.
    Reports ``spawn_finite`` — whether the spawn has a finite cost-to-go — which
    aggregates into the corpus d\\* coverage metric.
    """
    t0 = time.time()
    try:
        audit = LatticeAudit(
            map_data=list(map_data), name=name, workers=workers,
            use_native=use_native, w_risk=w_risk, w_loop=w_loop,
            risk_cap=risk_cap,
        )
        pc = audit.precompute(use_cache=use_cache)
        d_comb = pc["d_comb"]
        spawn = audit._spawn_node(pc["graph"], d_comb)
        spawn_finite = bool(
            spawn is not None and spawn >= 0 and np.isfinite(d_comb[spawn])
        )
        spawn_d = float(d_comb[spawn]) if spawn_finite else float("inf")
        return {
            "name": name, "ok": True, "cached": bool(pc.get("cached", False)),
            "spawn_finite": spawn_finite, "spawn_d": spawn_d,
            "n_edges": int(len(pc["edges"])),
            "enum_time": float(pc.get("enum_time", 0.0)),
            "time": time.time() - t0, "error": None,
        }
    except Exception as e:  # noqa: BLE001 - corpus jobs must not abort on one map
        return {
            "name": name, "ok": False, "cached": False, "spawn_finite": False,
            "spawn_d": float("inf"), "n_edges": 0, "enum_time": 0.0,
            "time": time.time() - t0, "error": f"{type(e).__name__}: {e}",
        }


# Top-level worker wrapper so ProcessPoolExecutor can pickle it.
def _worker(args: Tuple) -> Dict:
    name, map_data, kw = args
    return precompute_one(name, map_data, **kw)


def precompute_corpus(
    items: Iterable[Tuple[str, List[int]]],
    *,
    processes: Optional[int] = None,
    workers: int = 4,
    w_risk: float = DEFAULT_W_RISK,
    w_loop: float = DEFAULT_W_LOOP,
    risk_cap: float = DEFAULT_RISK_CAP,
    use_native: bool = True,
    use_cache: bool = True,
    verbose: bool = True,
) -> Dict:
    """Precompute + cache ``d*`` for every ``(name, map_data)`` in ``items``.

    ``processes`` map-level workers (default ``max(1, cpu // workers)`` so the
    per-level native threads × processes ≈ cores — see the throughput note).
    Returns a summary dict (counts, coverage, timing, per-map results).
    """
    items = list(items)
    n = len(items)
    if processes is None:
        cpu = os.cpu_count() or 4
        processes = max(1, cpu // max(1, workers))
    kw = dict(w_risk=w_risk, w_loop=w_loop, risk_cap=risk_cap,
              workers=workers, use_native=use_native, use_cache=use_cache)

    results: List[Dict] = []
    t0 = time.time()
    if processes <= 1 or n <= 1:
        for i, (name, md) in enumerate(items):
            r = precompute_one(name, md, **kw)
            results.append(r)
            if verbose:
                _print_progress(i + 1, n, r)
    else:
        with ProcessPoolExecutor(max_workers=processes) as ex:
            futs = {ex.submit(_worker, (name, md, kw)): name
                    for (name, md) in items}
            for i, fut in enumerate(as_completed(futs)):
                r = fut.result()
                results.append(r)
                if verbose:
                    _print_progress(i + 1, n, r)

    ok = [r for r in results if r["ok"]]
    covered = [r for r in ok if r["spawn_finite"]]
    summary = {
        "total": n,
        "ok": len(ok),
        "failed": n - len(ok),
        "covered": len(covered),
        "coverage": (len(covered) / n) if n else 0.0,
        "cached_hits": sum(1 for r in results if r["cached"]),
        "wall_time_s": time.time() - t0,
        "processes": processes,
        "workers": workers,
        "results": results,
    }
    if verbose:
        print(
            f"\n=== corpus d* precompute ===\n"
            f"  maps={n}  ok={summary['ok']}  failed={summary['failed']}  "
            f"cache_hits={summary['cached_hits']}\n"
            f"  d* COVERAGE (finite spawn cost-to-go): {summary['covered']}/{n} "
            f"= {100*summary['coverage']:.1f}%  (RL guidance coverage)\n"
            f"  wall={summary['wall_time_s']:.1f}s  "
            f"processes={processes} workers/level={workers}  "
            f"cache_dir={os.environ.get('NCLONE_LATTICE_CACHE_DIR', '/tmp/lattice_audit_cache')}"
        )
        fails = [r for r in results if not r["ok"]]
        if fails:
            print(f"  failures ({len(fails)}):")
            for r in fails[:20]:
                print(f"    {r['name']}: {r['error']}")
    return summary


def _print_progress(done: int, total: int, r: Dict) -> None:
    status = ("OK " if r["ok"] else "ERR") + ("·cache" if r["cached"] else "")
    cov = "✓" if r["spawn_finite"] else ("·" if r["ok"] else "✗")
    print(f"  [{done}/{total}] {status} {cov} {r['name']}  "
          f"({r['time']:.1f}s)" + (f"  {r['error']}" if not r["ok"] else ""))


# ---------------------------------------------------------------------------
def _gather_map_paths(args) -> List[str]:
    paths: List[str] = []
    if args.maps_dir:
        for ext in ("", ".txt"):  # official maps are extensionless
            paths.extend(p for p in glob.glob(os.path.join(args.maps_dir, "*" + ext))
                         if os.path.isfile(p))
    if args.list:
        with open(args.list) as f:
            paths.extend(line.strip() for line in f if line.strip())
    if args.maps:
        paths.extend(args.maps)
    # De-dup, keep order.
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("maps", nargs="*", help="explicit map file paths")
    p.add_argument("--maps-dir", help="directory of map files to glob")
    p.add_argument("--list", help="text file with one map path per line")
    p.add_argument("--processes", type=int, default=None,
                   help="map-level parallel processes (default cpu//workers)")
    p.add_argument("--workers", type=int, default=4,
                   help="native threads per level (keep healthy; default 4)")
    p.add_argument("--w-risk", type=float, default=DEFAULT_W_RISK)
    p.add_argument("--w-loop", type=float, default=DEFAULT_W_LOOP)
    p.add_argument("--risk-cap", type=float, default=DEFAULT_RISK_CAP)
    p.add_argument("--no-native", action="store_true")
    p.add_argument("--no-cache", action="store_true",
                   help="force recompute (ignore existing cache)")
    args = p.parse_args()

    paths = _gather_map_paths(args)
    if not paths:
        p.error("no maps given (use positional paths, --maps-dir, or --list)")

    items: List[Tuple[str, List[int]]] = []
    for path in paths:
        try:
            md = _load_map_data(path)
            items.append((os.path.basename(path.rstrip("/")), md))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {path}: {type(e).__name__}: {e}")

    summary = precompute_corpus(
        items, processes=args.processes, workers=args.workers,
        w_risk=args.w_risk, w_loop=args.w_loop, risk_cap=args.risk_cap,
        use_native=not args.no_native, use_cache=not args.no_cache,
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
