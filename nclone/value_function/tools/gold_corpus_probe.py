"""Corpus gold probe: gold-count and spatial-clustering histogram.

Phase 1 of the gold-collection integration.  Scans a corpus of levels, counts
gold entities (EntityType.GOLD == 2) per level, and characterises their spatial
layout (clusters vs scattered).  The point is to pin the planner's exact-vs-
heuristic ordering thresholds with real numbers instead of guessing:

  - gold-count distribution  -> where exact ordering (small N) is even relevant
  - clusteredness            -> whether cluster-first TSP buys anything
  - scatter (mean NN dist)   -> how strong the directional potential must be

Ordering is heuristic for any non-trivial N (2^N is hopeless past ~18), so the
buckets below are about *where the work concentrates*, not a promise of optimality.

Usage:
    .venv/bin/python -m nclone.value_function.tools.gold_corpus_probe [CORPUS] [OUT_JSON] [WORKERS]

CORPUS (default nclone/maps/official) may be a directory (scanned for map files,
nested official/<CAT>/<level> layout and flat dirs both handled) or a newline-
delimited list file of map paths.  OUT_JSON defaults to /tmp/gold_corpus_probe.json.
"""
import glob
import json
import multiprocessing as mp
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

GOLD_TYPE = 2

# Single-linkage cluster radius (px).  One tile is 24px; 48px = two gold within
# ~2 tiles are "the same cluster".  Tunable via env for sensitivity sweeps.
CLUSTER_EPS = float(os.environ.get("NCLONE_GOLD_CLUSTER_EPS", "48.0"))

# Count buckets.  Boundaries chosen around the planner cutoffs: <=12 is where an
# exact Held-Karp ordering would be feasible at all; 13-20 borderline; >20 is
# heuristic-only territory (the regime this corpus is expected to live in).
COUNT_BUCKETS = [
    (0, 0, "0"),
    (1, 4, "1-4"),
    (5, 12, "5-12"),
    (13, 20, "13-20"),
    (21, 50, "21-50"),
    (51, 100, "51-100"),
    (101, 200, "101-200"),
    (201, 10 ** 9, "200+"),
]


def _bucket_label(n):
    for lo, hi, label in COUNT_BUCKETS:
        if lo <= n <= hi:
            return label
    return "200+"


def _cluster(points, eps):
    """Single-linkage clustering via union-find with an eps grid for neighbours.

    Returns a list of cluster sizes (descending).  O(N) buckets, near-linear for
    the typical low-density layouts; worst case bounded by local density.
    """
    n = len(points)
    if n == 0:
        return []
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    grid = {}
    for i, (x, y) in enumerate(points):
        grid.setdefault((int(x // eps), int(y // eps)), []).append(i)

    eps2 = eps * eps
    for i, (x, y) in enumerate(points):
        gx, gy = int(x // eps), int(y // eps)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((gx + dx, gy + dy), ()):
                    if j <= i:
                        continue
                    xj, yj = points[j]
                    if (x - xj) ** 2 + (y - yj) ** 2 <= eps2:
                        union(i, j)

    sizes = {}
    for i in range(n):
        r = find(i)
        sizes[r] = sizes.get(r, 0) + 1
    return sorted(sizes.values(), reverse=True)


def _mean_nn_dist(points):
    """Mean nearest-neighbour distance (scatter metric).  None if <2 gold or too
    many to do the exact O(N^2) pass cheaply (rare; keeps the probe fast)."""
    n = len(points)
    if n < 2:
        return None
    if n > 3000:
        return None
    total = 0.0
    for i, (x, y) in enumerate(points):
        best = float("inf")
        for j, (xj, yj) in enumerate(points):
            if j == i:
                continue
            d2 = (x - xj) ** 2 + (y - yj) ** 2
            if d2 < best:
                best = d2
        total += best ** 0.5
    return total / n


def _category(map_path, corpus_root):
    """Official corpus is official/<CATEGORY>/<level>; return the category dir."""
    parent = os.path.basename(os.path.dirname(map_path))
    root = os.path.basename(os.path.normpath(corpus_root)) if corpus_root else ""
    return parent if parent and parent != root else "_flat"


def _probe_one(nh, map_path, corpus_root):
    nh.load_map(map_path)
    golds = nh.sim.entity_dic.get(GOLD_TYPE, [])
    points = [(float(g.xpos), float(g.ypos)) for g in golds]
    sizes = _cluster(points, CLUSTER_EPS)
    n = len(points)
    n_singletons = sum(1 for s in sizes if s == 1)
    in_clusters = sum(s for s in sizes if s >= 2)
    return {
        "path": map_path,
        "category": _category(map_path, corpus_root),
        "gold_count": n,
        "n_clusters": len(sizes),
        "largest_cluster": sizes[0] if sizes else 0,
        "n_singletons": n_singletons,
        "frac_clustered": (in_clusters / n) if n else 0.0,
        "mean_nn_dist": _mean_nn_dist(points),
    }


def _worker(args):
    chunk, corpus_root = args
    from nclone.nplay_headless import NPlayHeadless

    nh = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    out = []
    for map_path in chunk:
        try:
            out.append(_probe_one(nh, map_path, corpus_root))
        except Exception as e:  # keep the scan alive on a bad map
            out.append({"path": map_path, "error": f"{type(e).__name__}: {e}"})
    return out


def _discover_maps(corpus):
    if os.path.isfile(corpus):
        # treat as a newline-delimited list file
        with open(corpus) as f:
            return [ln.strip() for ln in f if ln.strip()], None
    # directory: match both nested (official/<cat>/<level>) and flat layouts
    nested = glob.glob(os.path.join(corpus, "*", "*"))
    flat = glob.glob(os.path.join(corpus, "*"))
    maps = sorted(
        f
        for f in set(nested) | set(flat)
        if os.path.isfile(f) and not f.endswith((".md", ".json", ".txt"))
    )
    return maps, corpus


def _chunks(seq, n):
    k = max(1, (len(seq) + n - 1) // n)
    return [seq[i : i + k] for i in range(0, len(seq), k)]


def scan(corpus, out_json, workers):
    maps, corpus_root = _discover_maps(corpus)
    print(f"corpus: {len(maps)} maps from {corpus} (cluster eps={CLUSTER_EPS}px)")
    if not maps:
        print("no maps found")
        return

    tasks = [(c, corpus_root) for c in _chunks(maps, workers)]
    if workers == 1:
        results = [r for t in tasks for r in _worker(t)]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
            results = [r for part in pool.map(_worker, tasks) for r in part]

    _report(results, out_json)


def _report(results, out_json):
    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]

    # count histogram
    hist = {}
    for r in ok:
        hist[_bucket_label(r["gold_count"])] = hist.get(_bucket_label(r["gold_count"]), 0) + 1

    with_gold = [r for r in ok if r["gold_count"] > 0]
    counts = sorted(r["gold_count"] for r in with_gold)

    def pct(p):
        if not counts:
            return 0
        return counts[min(len(counts) - 1, int(p * len(counts)))]

    # clusteredness across gold-bearing levels
    frac_clustered = [r["frac_clustered"] for r in with_gold]
    mean_frac_clustered = sum(frac_clustered) / len(frac_clustered) if frac_clustered else 0.0
    mostly_scattered = sum(1 for r in with_gold if r["frac_clustered"] < 0.34)
    mostly_clustered = sum(1 for r in with_gold if r["frac_clustered"] >= 0.67)

    summary = {
        "n_maps": len(results),
        "n_ok": len(ok),
        "n_errors": len(errs),
        "n_with_gold": len(with_gold),
        "cluster_eps_px": CLUSTER_EPS,
        "count_histogram": hist,
        "gold_count_pctiles": {
            "min": counts[0] if counts else 0,
            "p50": pct(0.50),
            "p90": pct(0.90),
            "p99": pct(0.99),
            "max": counts[-1] if counts else 0,
        },
        "n_gt12": sum(1 for c in counts if c > 12),
        "n_gt20": sum(1 for c in counts if c > 20),
        "n_gt100": sum(1 for c in counts if c > 100),
        "mean_frac_clustered": mean_frac_clustered,
        "n_mostly_scattered": mostly_scattered,
        "n_mostly_clustered": mostly_clustered,
    }

    with open(out_json, "w") as f:
        json.dump({"summary": summary, "results": results}, f)

    # stdout report
    print()
    print(f"  maps ok={len(ok)} errors={len(errs)} with_gold={len(with_gold)}")
    print()
    print("  gold-count histogram:")
    for _, _, label in COUNT_BUCKETS:
        if label in hist:
            n = hist[label]
            bar = "#" * min(60, n * 60 // max(1, len(ok)))
            print(f"    {label:>8}: {n:5d}  {bar}")
    print()
    p = summary["gold_count_pctiles"]
    print(f"  gold count (gold-bearing levels): "
          f"min={p['min']} p50={p['p50']} p90={p['p90']} p99={p['p99']} max={p['max']}")
    print(f"  levels with >12 gold: {summary['n_gt12']}  "
          f">20: {summary['n_gt20']}  >100: {summary['n_gt100']}")
    print(f"  clusteredness: mean frac in clusters={mean_frac_clustered:.2f}  "
          f"mostly-scattered={mostly_scattered}  mostly-clustered={mostly_clustered}")
    if errs:
        print(f"\n  {len(errs)} errors (first 3): "
              + "; ".join(f"{os.path.basename(e['path'])}: {e['error']}" for e in errs[:3]))
    print(f"\n  full results -> {out_json}")


if __name__ == "__main__":
    corpus = sys.argv[1] if len(sys.argv) > 1 else "nclone/maps/official"
    out_json = sys.argv[2] if len(sys.argv) > 2 else "/tmp/gold_corpus_probe.json"
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else min(12, os.cpu_count() or 1)
    scan(corpus, out_json, workers)
