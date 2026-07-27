"""Package the labeled human corpus (+ subset-valid official maps) into a single
``MechanicBenchmarkLoader`` train/test benchmark for the Execution Curriculum
Experiment (see npp-rl ``docs/EXECUTION_CURRICULUM_EXPERIMENT.md``).

This is the *only* data work the experiment needs. It does **not** invoke the
learned level generator -- it reuses the existing levelgen labeling/IO so the
benchmark is a real, completion-proven pool the RL agent learns to *execute*.

What it does
------------
- Reads ``datasets/levelgen/corpus_subset`` (labeled, subset-valid human levels,
  each with a winning demo => a demo-length difficulty band ``d0``..``d4``).
- Reads ``datasets/levelgen/corpus_official`` and keeps only ``subset_ok`` maps
  (1 exit door + 1 switch, entities subset of {mine, gold, ...}); these have no
  demo, so they get difficulty tier 0 ("unknown") and an on-the-fly mechanic tag.
- ``bucket = dominant mechanic tag`` (first tag in the canonical
  :data:`MECHANIC_TAGS` order, so ``mine`` -- present on ~84% of levels -- does
  not swallow the partition; ``wall_jump``/``jump`` levels split out first).
- ``tier = demo-length difficulty band`` (``d0->1 .. d4->5``; official -> 0).
- **Hold out for the ``test`` split:** every subset-valid official level + a
  ``--holdout-frac`` (default 10%) stratified-by-(bucket,tier) random sample of
  human levels. Everything else is ``train``.
- Emits the loader layout: ``<split>/<BUCKET>/T<tier>/<level_id>.pkl`` +
  ``<split>/<split>_metadata.json`` + root ``manifest.json``.

Example::

    python -m scripts.levelgen.pack_benchmark \\
        --out datasets/levelgen/execcur_benchmark

    # tiny smoke benchmark (cap human levels, skip official):
    python -m scripts.levelgen.pack_benchmark \\
        --out /tmp/execcur_smoke --limit 200 --no-official
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from nclone.levelgen.labeling import mechanic_tags, subset_ok
from nclone.levelgen.tokenizer import DIFFICULTY_BANDS, MECHANIC_TAGS

logger = logging.getLogger("levelgen.pack_benchmark")

# Fat-on-ramp tier weighting for the procgen overlay: more easy (t1) levels so
# the learnability frontier is non-empty at PPO cold-start, tapering to t4.
PROCGEN_TIER_WEIGHTS = {1: 2.0, 2: 1.5, 3: 1.0, 4: 1.0}


# --------------------------------------------------------------------------- io
def _load_corpus_records(corpus_dir: Path) -> List[Dict[str, Any]]:
    """Load every record from a sharded corpus directory."""
    index = json.loads((corpus_dir / "ingest_index.json").read_text())
    shards = index.get("shards") or sorted(
        p.name for p in corpus_dir.glob("shard_*.pkl")
    )
    records: List[Dict[str, Any]] = []
    for name in shards:
        blob = pickle.loads((corpus_dir / name).read_bytes())
        records.extend(blob["records"] if isinstance(blob, dict) else blob)
    return records


def _band_to_tier(band: str) -> int:
    """``unknown->0, d0->1, d1->2, ... d4->5`` via the canonical band order."""
    try:
        return DIFFICULTY_BANDS.index(band)
    except ValueError:
        return 0


def _bucket_for_tags(tags: Sequence[str]) -> str:
    """Dominant bucket = first canonical mechanic tag present (uppercased)."""
    for t in MECHANIC_TAGS:
        if t in tags:
            return t.upper()
    return "FLAT"


def _level_dict(
    record: Dict[str, Any],
    bucket: str,
    tier: int,
    split: str,
    *,
    int_id: int,
) -> Dict[str, Any]:
    """Build a ``write_benchmark``-schema level dict from a corpus record."""
    return {
        "level_id": int_id,
        "seed": 0,
        "bucket": bucket,
        "tier": tier,
        "generator": "corpus",
        "preset": record.get("source", "corpus"),
        "map_data": list(record["map_data"]),
        "secondary": [],
        "metadata": {
            "description": f"{record.get('source', 'corpus')} {record.get('src_path', '')}".strip(),
            "split": split,
            "source": record.get("source", "corpus"),
            "orig_level_id": record.get("level_id"),
            "demo_len": int(record.get("demo_len", 0) or 0),
            "mechanic_tags": list((record.get("labels") or {}).get("mechanic_tags", [])),
        },
    }


# --------------------------------------------------------------- procgen overlay
# Chunk size: each parallel task generates this many levels of one (bucket, tier)
# at a contiguous index offset. Small enough to load-balance across cores, large
# enough to amortise the per-worker BenchmarkGenerator + pygame/import setup.
_PROCGEN_CHUNK = 40


def _procgen_chunk_worker(arg):  # pragma: no cover - runs in a subprocess
    """Generate one contiguous index-chunk of a (bucket, tier) cell.

    ``level_id`` is a deterministic hash of ``(split, bucket, tier, index)``, so
    disjoint index ranges across workers never collide and the output is
    reproducible regardless of how the work is sharded.
    """
    bucket, tier, start, n, split, seed_base, check_solvable, enforce_subset = arg
    from scripts.generate_benchmark_dataset import BenchmarkGenerator

    gen = BenchmarkGenerator(
        output_dir=Path("/tmp/_procgen_unused"),
        buckets=[bucket],
        tiers=[tier],
        per_bucket_per_tier_train=0,
        per_bucket_per_tier_test=0,
        seed_base_train=seed_base,
        seed_base_test=seed_base,
        check_solvable=check_solvable,
        enforce_subset_entities=enforce_subset,
    )
    out: List[Dict[str, Any]] = []
    for i in range(start, start + n):
        lv = gen._generate_one(bucket, tier, i, split, seed_base)
        if lv is not None:
            lv["metadata"]["split"] = split
            out.append(lv)
    return out


def generate_procgen_overlay(
    per_cell: int,
    split: str,
    seed_base: int,
    seen_ids: set,
    check_solvable: bool = True,
    workers: Optional[int] = None,
    enforce_subset: bool = True,
) -> List[Dict[str, Any]]:
    """Generate a balanced, solvable-gated procgen overlay over BUCKET_CATALOG.

    Reuses :class:`scripts.generate_benchmark_dataset.BenchmarkGenerator` (preset
    selection + solvability gate) but parallelises generation across cores: each
    (bucket, tier) cell is split into :data:`_PROCGEN_CHUNK`-sized index chunks
    farmed to a process pool. Per-tier counts are weighted by
    :data:`PROCGEN_TIER_WEIGHTS` (fat t1 on-ramp). The 12 generator mechanic
    buckets balance the human corpus's thin cells (CURVE/COMPOSITE/PRECISION/
    SLOPES) — corpus + procgen coexist as a union of bucket names.

    Args:
        per_cell: Base levels per (bucket, tier) before the tier weight.
        split: ``"train"`` or ``"test"`` (stamped into each level's metadata).
        seed_base: Seed offset (use disjoint bases for train vs test).
        seen_ids: Shared set of already-used integer level_ids (mutated).
        check_solvable: Gate each level through the grid reachability check.
        workers: Process-pool size (default ``cpu_count - 2``).
        enforce_subset: Reject any generated level whose entities leave the RL
            training subset ({mine, gold, 1 exit door, 1 exit switch}); on by
            default so the overlay matches the corpus's ``subset_ok`` contract
            (the procedural generators can otherwise emit incidental locked
            doors etc.).

    Returns:
        List of level dicts in the ``write_benchmark`` schema.
    """
    import multiprocessing as mp
    from nclone.map_generation.generator_configs import BUCKET_TIERS, PHASE_1_BUCKETS

    tasks = []
    for bucket in PHASE_1_BUCKETS:
        for tier in BUCKET_TIERS:
            count = int(round(per_cell * PROCGEN_TIER_WEIGHTS.get(tier, 1.0)))
            for start in range(0, count, _PROCGEN_CHUNK):
                n = min(_PROCGEN_CHUNK, count - start)
                tasks.append((bucket, tier, start, n, split, seed_base,
                              check_solvable, enforce_subset))

    n_workers = workers if workers and workers > 0 else max(1, (mp.cpu_count() or 2) - 2)
    logger.info("procgen %s: %d chunks across %d workers (per_cell=%d) ...",
                split, len(tasks), n_workers, per_cell)
    levels: List[Dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        for chunk in pool.imap_unordered(_procgen_chunk_worker, tasks):
            for lv in chunk:
                lid = int(lv["level_id"])
                if lid in seen_ids:
                    continue
                seen_ids.add(lid)
                levels.append(lv)

    # Per-bucket tally for the run log.
    hist = Counter(lv["bucket"] for lv in levels)
    logger.info("procgen %s done: %d levels  %s", split, len(levels), dict(hist))
    return levels


# ----------------------------------------------------------------- benchmark io
def write_multi_benchmark(
    levels_by_split: Dict[str, List[Dict[str, Any]]],
    out_dir: Path,
) -> None:
    """Write a multi-bucket, multi-split MechanicBenchmarkLoader dataset.

    Layout::

        <out>/manifest.json
        <out>/<split>/<split>_metadata.json
        <out>/<split>/<BUCKET>/T<tier>/<level_id>.pkl
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_buckets: set[str] = set()
    all_tiers: set[int] = set()

    for split, levels in levels_by_split.items():
        # bucket -> tier -> [level_id]
        bucket_tiers: Dict[str, Dict[int, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for lv in levels:
            bucket, tier, lid = lv["bucket"], lv["tier"], lv["level_id"]
            d = out_dir / split / bucket / f"T{tier}"
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"{lid}.pkl", "wb") as f:
                pickle.dump(lv, f, protocol=4)
            bucket_tiers[bucket][tier].append(lid)
            all_buckets.add(bucket)
            all_tiers.add(tier)

        split_meta = {
            "split": split,
            "total_levels": len(levels),
            # flat list is what MechanicBenchmarkLoader indexes on
            "level_ids": [lv["level_id"] for lv in levels],
            "buckets": {
                bucket: {
                    "tiers": {
                        str(t): {
                            "level_ids": ids,
                            "path": f"{split}/{bucket}/T{t}",
                            "count": len(ids),
                        }
                        for t, ids in sorted(tiers.items())
                    }
                }
                for bucket, tiers in sorted(bucket_tiers.items())
            },
        }
        (out_dir / split).mkdir(parents=True, exist_ok=True)
        with open(out_dir / split / f"{split}_metadata.json", "w") as f:
            json.dump(split_meta, f, indent=2)

    manifest = {
        "version": 1,
        "source": "scripts.levelgen.pack_benchmark",
        "schema": "benchmark-v1",
        "buckets": sorted(all_buckets),
        "tiers": sorted(all_tiers),
        "splits": sorted(levels_by_split.keys()),
        "generation_info": {"generator": "corpus", "note": "execution-curriculum experiment"},
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


# ----------------------------------------------------------------------- driver
def build(
    corpus_dir: Path,
    official_dir: Optional[Path],
    out_dir: Path,
    holdout_frac: float,
    seed: int,
    limit: Optional[int],
    procgen_per_cell: int = 0,
    procgen_test_per_cell: int = 0,
    procgen_check_solvable: bool = True,
    procgen_workers: Optional[int] = None,
    official_split: str = "test",
) -> Dict[str, Any]:
    rng = random.Random(seed)
    seen_ids: set[int] = set()
    skipped = Counter()

    # --- human corpus (subset-valid, demo-graded) --------------------------
    human = _load_corpus_records(corpus_dir)
    if limit is not None:
        human = human[:limit]
    human_levels: List[Dict[str, Any]] = []
    for rec in human:
        labels = rec.get("labels") or {}
        if not labels.get("subset_ok", subset_ok(rec["map_data"])):
            skipped["human_not_subset"] += 1
            continue
        tags = labels.get("mechanic_tags") or mechanic_tags(rec["map_data"])
        if not tags:
            skipped["human_no_tags"] += 1
            continue
        try:
            int_id = int(rec["level_id"], 16)
        except (TypeError, ValueError):
            int_id = int(rec["level_id"])
        if int_id in seen_ids:
            skipped["human_dup"] += 1
            continue
        seen_ids.add(int_id)
        bucket = _bucket_for_tags(tags)
        tier = _band_to_tier(labels.get("difficulty_band", "unknown"))
        human_levels.append(_level_dict(rec, bucket, tier, "train", int_id=int_id))

    # --- stratified holdout of human levels into test ----------------------
    by_cell: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for lv in human_levels:
        by_cell[(lv["bucket"], lv["tier"])].append(lv)
    test_levels: List[Dict[str, Any]] = []
    train_levels: List[Dict[str, Any]] = []
    for cell, group in by_cell.items():
        rng.shuffle(group)
        n_hold = int(round(len(group) * holdout_frac))
        # guarantee at least one held-out per non-trivial cell so eval has coverage
        if holdout_frac > 0 and n_hold == 0 and len(group) >= 5:
            n_hold = 1
        for lv in group[:n_hold]:
            lv = dict(lv)
            lv["metadata"] = dict(lv["metadata"], split="test")
            test_levels.append(lv)
        train_levels.extend(group[n_hold:])

    # --- official maps (subset-valid only) -> test (default) or train ------
    # ``official_split="train"`` puts every valid official map into the TRAIN
    # split so it can form the FINAL difficulty phase (fold into the ATTRACT
    # bucket via separate_attract --include-official). Default "test" preserves
    # the held-out-generalization behavior.
    n_official = 0
    _off_split = "train" if official_split == "train" else "test"
    _off_target = train_levels if _off_split == "train" else test_levels
    if official_dir is not None and official_dir.exists():
        for rec in _load_corpus_records(official_dir):
            if not subset_ok(rec["map_data"]):
                skipped["official_not_subset"] += 1
                continue
            tags = mechanic_tags(rec["map_data"])
            try:
                int_id = int(rec["level_id"], 16)
            except (TypeError, ValueError):
                int_id = int(rec["level_id"])
            if int_id in seen_ids:
                skipped["official_dup"] += 1
                continue
            seen_ids.add(int_id)
            bucket = _bucket_for_tags(tags)
            lv = _level_dict(rec, bucket, 0, _off_split, int_id=int_id)
            lv["generator"] = "official"
            _off_target.append(lv)
            n_official += 1

    # --- procgen overlay (balanced generator coverage, fat t1 on-ramp) -----
    n_procgen_train = 0
    n_procgen_test = 0
    if procgen_per_cell > 0:
        logger.info("Generating procgen TRAIN overlay (per_cell=%d, weights=%s) ...",
                    procgen_per_cell, PROCGEN_TIER_WEIGHTS)
        pg_train = generate_procgen_overlay(
            procgen_per_cell, "train", seed_base=1_000_000 + seed,
            seen_ids=seen_ids, check_solvable=procgen_check_solvable,
            workers=procgen_workers, enforce_subset=True)
        train_levels.extend(pg_train)
        n_procgen_train = len(pg_train)
        if procgen_test_per_cell > 0:
            logger.info("Generating procgen TEST overlay (per_cell=%d) ...", procgen_test_per_cell)
            pg_test = generate_procgen_overlay(
                procgen_test_per_cell, "test", seed_base=100_000 + seed,
                seen_ids=seen_ids, check_solvable=procgen_check_solvable,
                workers=procgen_workers, enforce_subset=True)
            test_levels.extend(pg_test)
            n_procgen_test = len(pg_test)

    levels_by_split = {"train": train_levels, "test": test_levels}
    write_multi_benchmark(levels_by_split, out_dir)

    # --- summary -----------------------------------------------------------
    def _bucket_hist(levels):
        c = Counter(lv["bucket"] for lv in levels)
        return dict(sorted(c.items(), key=lambda kv: -kv[1]))

    summary = {
        "out_dir": str(out_dir),
        "n_train": len(train_levels),
        "n_test": len(test_levels),
        "official_split": _off_split,
        "n_official": n_official,
        "n_train_corpus": len(train_levels) - n_procgen_train
        - (n_official if _off_split == "train" else 0),
        "n_train_procgen": n_procgen_train,
        "n_test_official": n_official if _off_split == "test" else 0,
        "n_test_human": len(test_levels)
        - (n_official if _off_split == "test" else 0) - n_procgen_test,
        "n_test_procgen": n_procgen_test,
        "train_buckets": _bucket_hist(train_levels),
        "test_buckets": _bucket_hist(test_levels),
        "train_tiers": dict(sorted(Counter(lv["tier"] for lv in train_levels).items())),
        "skipped": dict(skipped),
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="datasets/levelgen/corpus_subset",
                    help="labeled subset-valid human corpus dir")
    ap.add_argument("--official", default="datasets/levelgen/corpus_official",
                    help="official corpus dir (subset-valid maps -> held-out test)")
    ap.add_argument("--official-split", choices=["test", "train"], default="test",
                    help="Which split valid official maps go to. 'train' (with "
                         "separate_attract --include-official) makes them the FINAL "
                         "difficulty phase; 'test' (default) holds them out.")
    ap.add_argument("--no-official", action="store_true",
                    help="skip official maps (e.g. for a tiny smoke benchmark)")
    ap.add_argument("--out", required=True, help="output benchmark root")
    ap.add_argument("--holdout-frac", type=float, default=0.1,
                    help="fraction of human levels held out into the test split")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of human records (smoke testing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procgen-per-cell", type=int, default=0,
                    help="Base procgen levels per (12 buckets x 4 tiers) added to TRAIN, "
                    "tier-weighted by PROCGEN_TIER_WEIGHTS (fat t1 on-ramp). 0 = no overlay. "
                    "~120 -> ~8k balanced overlay.")
    ap.add_argument("--procgen-test-per-cell", type=int, default=0,
                    help="Base procgen levels per cell added to TEST (fresh seeds; "
                    "measures generalization to unseen procgen). 0 = none.")
    ap.add_argument("--no-procgen-solvable", action="store_true",
                    help="Skip the grid reachability gate on procgen (faster, less safe).")
    ap.add_argument("--procgen-workers", type=int, default=None,
                    help="Process-pool size for procgen generation (default cpu_count-2).")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    official_dir = None if args.no_official else Path(args.official)
    summary = build(
        corpus_dir=Path(args.corpus),
        official_dir=official_dir,
        out_dir=Path(args.out),
        holdout_frac=args.holdout_frac,
        seed=args.seed,
        limit=args.limit,
        procgen_per_cell=args.procgen_per_cell,
        procgen_test_per_cell=args.procgen_test_per_cell,
        procgen_check_solvable=not args.no_procgen_solvable,
        procgen_workers=args.procgen_workers,
        official_split=args.official_split,
    )
    logger.info("Benchmark packaged:\n%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
