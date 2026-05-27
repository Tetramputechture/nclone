"""Corpus throughput harness for large-scale lattice generation.

The native kernel parallelizes *within* a level (rayon, all cores).  For a corpus
of many (often small) procgen levels, throughput is usually higher running *many
levels concurrently with fewer threads each* — less per-level rayon overhead and
better load-balance across heterogeneous level sizes.  This sweeps
(processes × native-threads) allocations and reports end-to-end maps/sec, to pick
the throughput-optimal config for a 10k-level run.

Each task runs the full per-level product (surface extraction -> native
enumeration -> two-phase d*) via LatticeAudit.precompute, i.e. the actual
training-data artifact.

Usage:
    .venv/bin/python -m nclone.value_function.tools.corpus_throughput [REPEAT] [maps...]

REPEAT (default 1) tiles the corpus to simulate a larger run.  With no map args
it uses all single-file maps under nclone/test_maps/.
Note: don't import rayon-using code in the parent before forking (fork-unsafe);
workers initialise their own rayon pool on first enumerate.
"""
import glob
import os
import sys
import time
import multiprocessing as mp


def _process_one(args):
    map_path, k = args
    # Imports inside the worker (post-fork) so the parent never touches rayon/pygame.
    from .lattice_audit import LatticeAudit
    try:
        t = time.time()
        audit = LatticeAudit(map_path=map_path, workers=k, use_native=True)
        pc = audit.precompute(use_cache=False)
        return (map_path, time.time() - t, len(pc["edges"]), True)
    except Exception as e:  # keep the batch alive on a bad map
        return (map_path, 0.0, 0, f"FAIL: {type(e).__name__}: {e}")


def run_corpus(maps, n_proc, k_threads):
    tasks = [(m, k_threads) for m in maps]
    t0 = time.time()
    if n_proc == 1:
        results = [_process_one(t) for t in tasks]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(n_proc) as pool:
            results = pool.map(_process_one, tasks)
    total = time.time() - t0
    return total, results


def main(maps, configs):
    n_cores = os.cpu_count() or 1
    print(f"corpus: {len(maps)} maps, {n_cores} cores")
    # warm one config lightly is not needed; report all.
    print(f"  {'procs x thr':>12} {'total(s)':>9} {'maps/s':>8} {'fails':>6}")
    best = None
    for (m, k) in configs:
        total, results = run_corpus(maps, m, k)
        fails = sum(1 for r in results if r[3] is not True)
        mps = len(maps) / total if total > 0 else 0.0
        flag = ""
        if best is None or mps > best[1]:
            best = ((m, k), mps)
            flag = "  <- best"
        print(f"  {f'{m}x{k}':>12} {total:9.2f} {mps:8.2f} {fails:6d}{flag}")
    print(f"\nbest throughput: {best[0][0]}x{best[0][1]} = {best[1]:.2f} maps/s")


if __name__ == "__main__":
    args = sys.argv[1:]
    repeat = 1
    if args and args[0].isdigit():
        repeat = int(args[0])
        args = args[1:]
    if args:
        maps = args
    else:
        maps = sorted(
            f for f in glob.glob("nclone/test_maps/*")
            if os.path.isfile(f) and not f.endswith((".md", ".json"))
        )
    maps = maps * repeat
    n = os.cpu_count() or 1
    # Keep K (threads/level) high: enumeration is heavy (~100M+ ticks/level) so
    # under-threading a level is catastrophic.  The across-level win is only the
    # overlap of the serial Python phases, realised at low M x high K.
    configs = [(1, n), (2, max(1, n // 2)), (3, max(1, n // 3)), (4, max(1, n // 4))]
    seen = set()
    configs = [c for c in configs if not (c in seen or seen.add(c))]
    main(maps, configs)
