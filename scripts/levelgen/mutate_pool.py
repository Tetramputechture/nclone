"""ACCEL mutation producer — the "millions engine" loop (WS-C2 / Phase 1).

A long-running (or one-shot) producer that turns a small set of *structured* parent
levels into many gated, subset-pure, structurally-rich children and admits them to a
packed level pool (:mod:`nclone.levelgen.pool_store`). It is the concrete realisation of
``docs/SYNTHETIC_LEVELGEN_NEXT_STEPS.md`` C2.

Parent sources (``--seed``):

* ``procgen`` — enumerate the entity-subset procgen presets (the structured seed set;
  reuses ``build_procgen_supplement``'s canonical enumeration) and mutate those.
* ``pool`` — read an existing pool / benchmark dir and mutate its levels. With
  ``--curation-log`` (the learnability snapshot written by npp-rl's
  ``LearnabilityCurriculumCallback``) it selects **agent-targeted** parents: levels on
  the learnability frontier (``p∈[lo,hi]``) and *just-mastered* levels (``p≥mastered``,
  pushed harder via difficulty-perturbing mutation). Without it, parents are uniform.

Every child is gated (``Map.clean_placement`` + free-space gate), labelled on the shared
free-space difficulty scale, and written with provenance (``source=mutant``,
``parent=<id>``). Run many instances with distinct ``--writer-id`` for lock-free
parallelism (B1) into one pool dir.

Examples::

    # one-shot: structured procgen seeds -> 5000 mutants
    python -m scripts.levelgen.mutate_pool --out datasets/levelgen/pool \\
        --seed procgen --n 5000 --edits 2 --writer-id 0

    # agent-driven: mutate the live pool toward the agent's frontier
    python -m scripts.levelgen.mutate_pool --out datasets/levelgen/pool \\
        --seed pool --seed-dir datasets/levelgen/pool \\
        --curation-log experiments/run/learnability_levels.json --n 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from nclone.levelgen.free_space_difficulty import assign_band as assign_fs_band
from nclone.levelgen.mutations import mutate
from nclone.levelgen.pool_store import (
    PoolWriter,
    PackedPool,
    is_packed_pool,
    content_id,
)
from nclone.levelgen.tokenizer import DIFFICULTY_BANDS

logger = logging.getLogger("levelgen.mutate_pool")


# --------------------------------------------------------------------- parent sources
def _procgen_parents(
    seeds_per_preset: int, seed_offset: int = 0, seed_stride: int = 1
) -> Iterator[Tuple[List[int], dict]]:
    """Yield ``(map_data, meta)`` for cleaned, subset-pure procgen levels.

    ``seed_offset``/``seed_stride`` partition the seed space across parallel workers so
    each worker generates a *disjoint* set of parents (worker ``w`` of ``N`` uses
    ``seed_offset=w, seed_stride=N``) — no overlap, no wasted dedup.
    """
    from scripts.levelgen.build_procgen_supplement import (
        _SUBSET_BUCKETS,
        _enumerate_presets,
        _gen_map_data,
    )
    from nclone.levelgen.labeling import subset_ok

    presets = _enumerate_presets(list(_SUBSET_BUCKETS))
    logger.info("procgen parents: %d (gen,preset) pairs x %d seeds", len(presets), seeds_per_preset)
    for gen_type, preset in presets:
        for k in range(seeds_per_preset):
            s = seed_offset + k * seed_stride
            md = _gen_map_data(gen_type, preset, seed=s)
            if md is None or not subset_ok(md):
                continue
            yield md, {"bucket": _bucket_for(gen_type), "preset": preset}


def _bucket_for(gen_type: str) -> str:
    return gen_type.upper()


def _pool_parents(
    seed_dir: str,
    curation_log: Optional[str],
    frontier_lo: float,
    frontier_hi: float,
    mastered: float,
    rng: random.Random,
) -> Iterator[Tuple[List[int], dict]]:
    """Yield ``(map_data, meta)`` from a pool/benchmark dir, agent-targeted if a
    learnability snapshot is provided."""
    if is_packed_pool(seed_dir):
        pool = PackedPool(seed_dir)
        id_to_md = {int(r["id"]): None for r in pool.records()}
        meta_by_id = {int(r["id"]): r for r in pool.records()}

        def loader(lid):
            return pool.map_data(lid)
    else:
        from nclone.evaluation.mechanic_benchmark_loader import MechanicBenchmarkLoader

        ld = MechanicBenchmarkLoader(seed_dir, splits=("train",))
        cache = {int(ld.load_payload(i)["level_id"]): i for i in range(len(ld))}
        id_to_md = {lid: None for lid in cache}
        meta_by_id = {}

        def loader(lid):
            return list(ld.load_payload(cache[lid])["map_data"])

    all_ids = list(id_to_md.keys())
    if not all_ids:
        return

    targeted: List[Tuple[int, Optional[float]]] = []  # (level_id, target_cost_hint)
    if curation_log:
        snap = json.loads(Path(curation_log).read_text())
        levels = snap.get("levels", snap)  # tolerate raw {id: {...}} or {"levels": {...}}
        for lid_s, st in levels.items():
            try:
                lid = int(lid_s)
            except (TypeError, ValueError):
                continue
            if lid not in id_to_md:
                continue
            p = float(st.get("p", st if isinstance(st, (int, float)) else 0.0))
            if frontier_lo <= p <= frontier_hi:
                targeted.append((lid, None))          # learnable: preserve difficulty
            elif p >= mastered:
                targeted.append((lid, "harder"))      # mastered: push difficulty up
        logger.info("curation: %d agent-targeted parents (of %d snapshot levels)",
                    len(targeted), len(levels))
    if not targeted:
        # uniform fallback (cold start / no log): all ids, neutral difficulty
        targeted = [(lid, None) for lid in all_ids]

    rng.shuffle(targeted)
    for lid, hint in targeted:
        md = loader(lid)
        meta = {"parent": lid, "diff_hint": hint}
        if lid in meta_by_id:
            meta["bucket"] = meta_by_id[lid].get("bucket", "MUTANT")
        yield md, meta


# --------------------------------------------------------------------- worker
def _produce(args, writer_id: str, n_target: int, worker_idx: int, n_workers: int) -> dict:
    """Run one producer worker: parents -> mutate -> gate -> admit. Returns a summary.

    Module-level (picklable) so it can run in a spawned subprocess for ``--workers``.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = random.Random((args.seed_rng * 1000003) ^ (worker_idx + 1))

    edges = None
    if args.edges_from:
        edges = json.loads(Path(args.edges_from).read_text()).get("difficulty_edges")
        logger.info("difficulty edges: %s", edges)

    def band_targets() -> List[Optional[float]]:
        """Target costs (band midpoints) to spread difficulty across, or [None]."""
        if not (args.spread_difficulty and edges):
            return [None]
        mids: List[Optional[float]] = []
        for di in range(len(edges) + 1):
            lo = edges[di - 1] if di > 0 else 0.0
            hi = edges[di] if di < len(edges) else edges[-1] * 1.5
            mids.append(0.5 * (lo + hi))
        return mids

    if args.seed == "procgen":
        # partition the seed space so workers generate disjoint parents
        parents = _procgen_parents(
            args.seeds_per_preset, seed_offset=worker_idx, seed_stride=max(1, n_workers)
        )
    else:
        parents = _pool_parents(
            args.seed_dir, args.curation_log,
            args.frontier_lo, args.frontier_hi, args.mastered, rng,
        )

    t0 = time.time()
    admitted = attempts = 0
    max_attempts = args.max_attempts or (n_target * 30)
    band_hist: Dict[str, int] = {}

    with PoolWriter(args.out, writer_id=str(writer_id), shard_size=args.shard_size) as w:
        for parent_md, meta in parents:
            if admitted >= n_target or attempts >= max_attempts:
                break
            if args.time_budget and (time.time() - t0) >= args.time_budget:
                break
            parent_id = meta.get("parent")
            if parent_id is None:
                try:
                    parent_id = content_id(parent_md)
                except Exception:
                    parent_id = None
            targets = band_targets()
            for _ in range(args.children_per_parent):
                if admitted >= n_target or attempts >= max_attempts:
                    break
                attempts += 1
                # choose a difficulty target: spread, or harder for mastered parents
                tc = rng.choice(targets)
                if meta.get("diff_hint") == "harder" and edges:
                    tc = edges[-1] * 1.2  # push toward the hard end
                r = mutate(
                    parent_md, rng,
                    n_edits=args.edits, structure_bias=args.structure_bias, target_cost=tc,
                )
                if r.map_data is None:
                    continue
                band = assign_fs_band(r.fs_cost, edges) if edges else "unknown"
                tier = max(0, DIFFICULTY_BANDS.index(band)) if band in DIFFICULTY_BANDS else 0
                lid = w.add(
                    r.map_data,
                    bucket=str(meta.get("bucket", "MUTANT")),
                    tier=tier,
                    source="mutant",
                    parent=parent_id,
                    fs_cost=r.fs_cost,
                    difficulty_band=band,
                    structure_score=r.structure_score,
                    mechanic=(meta.get("preset") or None),
                )
                if lid is not None:
                    admitted += 1
                    band_hist[band] = band_hist.get(band, 0) + 1
            if attempts and attempts % 200 == 0:
                logger.info("[w%s] admitted %d / %d (attempts=%d, dup=%d, %.1f/s)",
                            writer_id, admitted, n_target, attempts, w.n_dup,
                            admitted / max(time.time() - t0, 1e-6))

    dt = time.time() - t0
    logger.info("DONE writer=%s admitted=%d attempts=%d dup=%d bands=%s (%.1f lvls/s)",
                writer_id, admitted, attempts, w.n_dup, band_hist, admitted / max(dt, 1e-6))
    return {"writer": str(writer_id), "admitted": admitted, "attempts": attempts,
            "dup": w.n_dup, "bands": band_hist, "seconds": dt}


def _produce_star(packed):
    return _produce(*packed)


# --------------------------------------------------------------------- main / dispatch
def main() -> None:
    ap = argparse.ArgumentParser(description="ACCEL mutation producer -> packed pool")
    ap.add_argument("--out", required=True, help="packed pool dir (created if absent)")
    ap.add_argument("--seed", choices=("procgen", "pool"), default="procgen")
    ap.add_argument("--seed-dir", default=None, help="parent pool/benchmark dir (--seed pool)")
    ap.add_argument("--curation-log", default=None,
                    help="learnability_levels.json snapshot for agent-targeted parents")
    ap.add_argument("--n", type=int, default=1000, help="total admitted children wanted")
    ap.add_argument("--children-per-parent", type=int, default=8)
    ap.add_argument("--edits", type=int, default=2, help="edits per mutation")
    ap.add_argument("--structure-bias", type=float, default=0.7)
    ap.add_argument("--seeds-per-preset", type=int, default=60, help="procgen seeds per preset")
    ap.add_argument("--workers", type=int, default=1, help="parallel producer processes")
    ap.add_argument("--writer-id", default="0", help="base writer id (single-worker)")
    ap.add_argument("--shard-size", type=int, default=4000)
    ap.add_argument("--edges-from", default=None,
                    help="labels_index.json with difficulty_edges (for band labels + spread)")
    ap.add_argument("--spread-difficulty", action="store_true",
                    help="randomise the free-space target cost across bands per child")
    ap.add_argument("--frontier-lo", type=float, default=0.3)
    ap.add_argument("--frontier-hi", type=float, default=0.7)
    ap.add_argument("--mastered", type=float, default=0.85)
    ap.add_argument("--max-attempts", type=int, default=None)
    ap.add_argument("--time-budget", type=float, default=0.0, help="seconds; 0 = unbounded")
    ap.add_argument("--seed-rng", type=int, default=0)
    ap.add_argument("--merge", action="store_true", help="compact index segments when done")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.seed == "pool" and not args.seed_dir:
        ap.error("--seed pool requires --seed-dir")

    nw = max(1, args.workers)
    if nw == 1:
        _produce(args, args.writer_id, args.n, 0, 1)
    else:
        import multiprocessing as mp

        # split the target across workers; each owns a distinct writer id + seed slice
        per = [args.n // nw + (1 if i < args.n % nw else 0) for i in range(nw)]
        jobs = [(args, str(i), per[i], i, nw) for i in range(nw)]
        ctx = mp.get_context("spawn")
        with ctx.Pool(nw) as pool:
            results = pool.map(_produce_star, jobs)
        total = sum(r["admitted"] for r in results)
        logger.info("ALL WORKERS DONE: %d workers, %d admitted total", nw, total)

    if args.merge:
        from nclone.levelgen.pool_store import merge_pool

        res = merge_pool(args.out)
        logger.info("merged pool: %s", res)


if __name__ == "__main__":
    main()
