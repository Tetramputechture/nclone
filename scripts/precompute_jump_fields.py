#!/usr/bin/env python3
"""Batch-precompute the jump-aware cost-to-go (``JumpDistanceSource``) for a
mechanic-benchmark dataset, populating the shared on-disk cache.

Wiring the jump field into the env (``ReachabilityConfig.use_jump_potential``)
makes each training env build the per-level jump cost-to-go on its first reset.
That build is SLOW (6-11 s/level: every standable support cell's ballistic
trajectories are validated against the level mask), so without a warm cache it
stalls the synchronous vec-step on every reset under multi-level training. The
``JumpDistanceSource`` is two-tier cached (per-process mem LRU + a shared on-disk
cache under ``<NCLONE_GRAPH_CACHE_DIR>/jumpfield/`` or
``NCLONE_JUMPFIELD_CACHE_DIR``), keyed by the map_data hash. This tool
**pre-populates that disk cache** for a whole dataset so the workers pay ~0 at
reset (cache hit <0.5 s vs multi-second build).

Set the cache dir the training run will use, e.g.::

    NCLONE_GRAPH_CACHE_DIR=/data/graph_cache \
    python -m scripts.precompute_jump_fields \
        --dataset-root datasets/mechanic_benchmark_v4 --split train \
        --processes 8

Optionally restrict to specific buckets / tiers::

    ... --buckets PLATFORM_STAIRS SLOPES --tiers T2 T3

Levels are independent, so the real scaling lever is across MACHINES — shard the
bucket/tier list and run this per machine, all writing the same shared cache dir.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

# Allow running as a plain script (python scripts/precompute_jump_fields.py).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _gather_levels(
    dataset_root: str,
    split: str,
    buckets: Optional[List[str]],
    tiers: Optional[List[str]],
) -> List[Tuple[str, str]]:
    """Return (level_id, pkl_path) for every dataset level matching the filters."""
    split_dir = os.path.join(dataset_root, split)
    if not os.path.isdir(split_dir):
        raise SystemExit(f"split dir not found: {split_dir}")
    out: List[Tuple[str, str]] = []
    for bucket in sorted(os.listdir(split_dir)):
        if buckets and bucket not in buckets:
            continue
        bdir = os.path.join(split_dir, bucket)
        if not os.path.isdir(bdir):
            continue
        for tier in sorted(os.listdir(bdir)):
            if tiers and tier not in tiers:
                continue
            tdir = os.path.join(bdir, tier)
            if not os.path.isdir(tdir):
                continue
            for fn in sorted(os.listdir(tdir)):
                if fn.endswith(".pkl"):
                    out.append((f"{bucket}/{tier}/{fn}", os.path.join(tdir, fn)))
    return out


def precompute_one(name: str, pkl_path: str) -> Dict:
    """Build (and disk-cache) the JumpDistanceSource for one level.

    Fault-tolerant: returns ``ok=False`` + ``error`` on any failure so one bad
    level never aborts the dataset. Reports ``available`` (the source produced a
    finite jump field) and whether the build was a cache hit (fast)."""
    t0 = time.time()
    try:
        with open(pkl_path, "rb") as f:
            level = pickle.load(f)
        map_data = level["map_data"]
        from nclone.value_function.jump_distance import JumpDistanceSource

        src = JumpDistanceSource(list(map_data), name=name)
        dt = time.time() - t0
        return {
            "name": name, "ok": True,
            "available": bool(getattr(src, "available", False)),
            "cache_hit": dt < 0.5,  # a real build is multi-second; a hit is ~ms
            "time": dt, "error": None,
        }
    except Exception as e:  # noqa: BLE001 - dataset jobs must not abort on one level
        return {
            "name": name, "ok": False, "available": False, "cache_hit": False,
            "time": time.time() - t0, "error": f"{type(e).__name__}: {e}",
        }


def _worker(args: Tuple[str, str]) -> Dict:
    return precompute_one(*args)


def precompute_jump_fields(
    items: List[Tuple[str, str]],
    *,
    processes: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """Precompute + cache the jump field for every ``(name, pkl_path)``."""
    n = len(items)
    if processes is None:
        processes = max(1, (os.cpu_count() or 4))
    results: List[Dict] = []
    t0 = time.time()
    if processes <= 1 or n <= 1:
        for i, (name, path) in enumerate(items):
            r = precompute_one(name, path)
            results.append(r)
            if verbose:
                _print_progress(i + 1, n, r)
    else:
        with ProcessPoolExecutor(max_workers=processes) as ex:
            futs = {ex.submit(_worker, it): it[0] for it in items}
            for i, fut in enumerate(as_completed(futs)):
                r = fut.result()
                results.append(r)
                if verbose:
                    _print_progress(i + 1, n, r)

    wall = time.time() - t0
    ok = [r for r in results if r["ok"]]
    avail = [r for r in ok if r["available"]]
    hits = [r for r in results if r["cache_hit"]]
    summary = {
        "total": n, "ok": len(ok), "failed": n - len(ok),
        "available": len(avail), "cache_hits": len(hits),
        "wall_time_s": wall, "throughput": (n / wall) if wall else 0.0,
        "processes": processes, "results": results,
    }
    if verbose:
        cache_dir = os.environ.get("NCLONE_JUMPFIELD_CACHE_DIR") or (
            os.path.join(os.environ["NCLONE_GRAPH_CACHE_DIR"], "jumpfield")
            if os.environ.get("NCLONE_GRAPH_CACHE_DIR") else "/tmp/nclone_jumpfield_cache"
        )
        print(
            f"\n=== jump-field precompute ===\n"
            f"  levels={n}  ok={summary['ok']}  failed={summary['failed']}  "
            f"cache_hits={summary['cache_hits']}\n"
            f"  available (finite jump field): {summary['available']}/{n}\n"
            f"  wall={wall:.1f}s  throughput={summary['throughput']:.2f} levels/s  "
            f"processes={processes}\n"
            f"  cache_dir={cache_dir}"
        )
        fails = [r for r in results if not r["ok"]]
        if fails:
            print(f"  failures ({len(fails)}):")
            for r in fails[:20]:
                print(f"    {r['name']}: {r['error']}")
    return summary


def _print_progress(done: int, total: int, r: Dict) -> None:
    status = "OK " if r["ok"] else "ERR"
    tag = "·cache" if r["cache_hit"] else ""
    av = "✓" if r["available"] else ("·" if r["ok"] else "✗")
    print(f"  [{done}/{total}] {status}{tag} {av} {r['name']}  ({r['time']:.1f}s)"
          + (f"  {r['error']}" if not r["ok"] else ""))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="datasets/mechanic_benchmark_v4",
                   help="dataset root (default datasets/mechanic_benchmark_v4)")
    p.add_argument("--split", default="train",
                   help="dataset split subdir (default train)")
    p.add_argument("--buckets", nargs="*", default=None,
                   help="restrict to these bucket names (default: all)")
    p.add_argument("--tiers", nargs="*", default=None,
                   help="restrict to these tiers, e.g. T2 T3 (default: all)")
    p.add_argument("--processes", type=int, default=None,
                   help="parallel processes (default: cpu_count)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap the number of levels (debug)")
    args = p.parse_args()

    items = _gather_levels(args.dataset_root, args.split, args.buckets, args.tiers)
    if args.limit:
        items = items[: args.limit]
    if not items:
        p.error("no levels matched the dataset/split/bucket/tier filters")

    print(f"jump-field precompute: {len(items)} levels from "
          f"{args.dataset_root}/{args.split}"
          + (f" buckets={args.buckets}" if args.buckets else "")
          + (f" tiers={args.tiers}" if args.tiers else ""))
    summary = precompute_jump_fields(items, processes=args.processes)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
