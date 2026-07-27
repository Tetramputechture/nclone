"""Build a graded procedural-generator SUPPLEMENT for the entity-stage corpus.

Why: the human attract corpus is designed to be non-trivial, so it under-covers
the *easy end* of the free-space difficulty scale (its easiest levels still sit
well above a truly trivial "switch next to door, spawn beside it" layout — see
``docs/LEVEL_POOL_UED_PLAN.md``). This script uses the hand-tuned procgen
generators (``map_generation`` / ``BUCKET_CATALOG``) to manufacture levels across
the difficulty range — especially the trivial d0 band — so the transformer's
entity stage can *learn* the full range and the pool's easy tier is populated.

It emits a drop-in **labeled corpus dir** (shards + ``ingest_index.json`` +
``labels_index.json``) that ``build_dataset --corpus ... corpus_procgen_supplement``
concatenates alongside the human corpus. Difficulty is labelled on the SHARED
free-space scale: pass ``--edges-from <corpus>/labels_index.json`` so a procgen d2
means the same thing as a corpus d2.

Guarantees:
* **Entity-subset pure** — every kept level passes ``subset_ok`` (only toggle
  mines / gold / one exit door+switch), so out-of-subset generators are dropped.
* **Geometrically solvable** — ``inf`` free-space cost (no spawn→switch→door path)
  is rejected; the physics gate still certifies final solvability downstream.
* **Band-balanced** — fills a per-band quota; over-samples the trivial end.

Example::

    python -m scripts.levelgen.build_procgen_supplement \\
        --out datasets/levelgen/corpus_procgen_supplement \\
        --edges-from datasets/levelgen/corpus_subset/labels_index.json \\
        --per-band 600 --workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from nclone.levelgen.free_space_difficulty import (
    assign_band as assign_fs_band,
    fit_edges as fit_fs_edges,
    route_cost as fs_route_cost,
)
from nclone.levelgen.labeling import mechanic_tags, subset_ok
from nclone.levelgen.tokenizer import DIFFICULTY_BANDS, LevelTokenizer
from nclone.map_generation.generator_configs import BUCKET_CATALOG
from nclone.map_generation.generator_factory import GeneratorFactory
from scripts.levelgen.common import LevelRecord, map_hash, write_shards

logger = logging.getLogger("levelgen.procgen_supplement")

_TOK = LevelTokenizer()


def _canonicalize(md):
    """Normalize a procgen ``map_data`` to the tokenizer-canonical byte stream, or
    ``None`` if it is not transformer-ingestible.

    Procgen ``Map.map_data()`` emits a signed ninja-orientation (``-1``) and some
    presets emit maps shorter than ``MIN_MAP_LEN``. We clamp out-of-byte values to
    ``[0, 255]`` (matching the corpus convention at the ninja-orient index) and
    require a byte-EXACT tokenizer round trip — the same gate build_dataset applies
    — so every kept level is guaranteed ingestible by the entity stage.
    """
    md = [v if 0 <= v <= 255 else max(0, min(255, v)) for v in md]
    try:
        tl = _TOK.tokenize(md, cond={})
        if list(_TOK.detokenize(tl)) != md:
            return None
    except Exception:
        return None
    return md

# Layer-1, entity-subset-compatible buckets. The other catalog buckets
# (SWITCH_PUZZLE / LAUNCH_PAD / THWUMP / DRONE / DEATH_BALL) introduce
# out-of-subset entities; subset_ok would drop them anyway, so skip the waste.
_SUBSET_BUCKETS = (
    "FLAT_RUN", "BASIC_JUMP", "WALL_JUMP", "VERTICAL_SHAFT", "MINE_NAVIGATION",
    "SLOPES", "CHAMBER_MAZE", "CURVES", "MULTI_JUMP", "COMPOSITE",
    "PLATFORMING", "PLATFORM_STAIRS", "LOOP_CORRIDOR", "PRECISION",
)


def _enumerate_presets(buckets):
    """[(gen_type, preset)] across the requested buckets' tiers (deduped)."""
    seen, out = set(), []
    for b in buckets:
        for tier_entries in BUCKET_CATALOG.get(b, {}).values():
            for entry in tier_entries:
                gen_type, preset = entry[0], entry[1]
                if (gen_type, preset) not in seen:
                    seen.add((gen_type, preset))
                    out.append((gen_type, preset))
    return out


def _gen_map_data(gen_type, preset, seed):
    """Canonical preset -> tokenizer-ingestible map_data, or ``None``."""
    try:
        m = GeneratorFactory.create_from_preset(gen_type, preset, seed)
        m.generate(seed=seed)
        m.clean_placement()  # lift entities out of solid + flatten buried border tiles
        md = list(m.map_data())
    except Exception:
        return None
    return _canonicalize(md)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a graded procgen difficulty supplement")
    ap.add_argument("--out", required=True, help="output corpus dir")
    ap.add_argument("--edges-from", default=None,
                    help="labels_index.json to read the shared free-space difficulty_edges from "
                         "(omit to self-fit edges on the generated pool)")
    ap.add_argument("--per-band", type=int, default=600, help="target kept levels per d0..d4 band")
    ap.add_argument("--seeds-per-preset", type=int, default=400, help="seeds tried per (gen,preset)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--buckets", nargs="+", default=list(_SUBSET_BUCKETS))
    ap.add_argument("--shard-size", type=int, default=2000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    bands = [b for b in DIFFICULTY_BANDS if b != "unknown"]  # d0..d4
    edges = None
    if args.edges_from:
        edges = json.loads(Path(args.edges_from).read_text()).get("difficulty_edges")
        logger.info("Shared difficulty edges (from %s): %s", args.edges_from, edges)

    presets = _enumerate_presets(args.buckets)
    logger.info("Generating from %d (gen,preset) pairs x %d seeds ...", len(presets), args.seeds_per_preset)

    # 1) Generate a candidate pool (serial; procgen is cheap), dedup + subset-filter.
    seen_hashes = set()
    candidates = []  # list[map_data]
    for gen_type, preset in presets:
        for s in range(args.seeds_per_preset):
            md = _gen_map_data(gen_type, preset, seed=s)
            if md is None or not subset_ok(md):
                continue
            h = map_hash(md)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            candidates.append(md)
    logger.info("Candidate pool: %d unique subset-pure maps", len(candidates))

    # 2) Free-space route cost (parallel) -> shared-scale band. Drop inf (no geo path).
    with Pool(args.workers) as pool:
        costs = pool.map(fs_route_cost, candidates)
    if edges is None:
        edges = fit_fs_edges(costs, n_bands=len(bands) + 1)
        logger.info("Self-fit difficulty edges: %s", edges)

    # 3) Bin to per-band quota (over-sample trivial: process low-cost first).
    by_band = defaultdict(list)
    order = sorted(range(len(candidates)), key=lambda i: (math.inf if not math.isfinite(costs[i]) else costs[i]))
    n_inf = 0
    for i in order:
        c = costs[i]
        if not math.isfinite(c):
            n_inf += 1
            continue
        band = assign_fs_band(c, edges)
        if band == "unknown" or len(by_band[band]) >= args.per_band:
            continue
        by_band[band].append((candidates[i], c))
    logger.info("Binned: %s  (dropped inf/no-geo-path: %d)",
                {b: len(by_band[b]) for b in bands}, n_inf)

    # 4) Emit labeled LevelRecords (source=procgen) -> drop-in corpus dir.
    records, band_hist, tag_hist = [], Counter(), Counter()
    for band in bands:
        for md, cost in by_band[band]:
            tags = mechanic_tags(md)
            records.append(LevelRecord(
                level_id=map_hash(md), source="procgen", map_data=md, demo_len=0,
                labels={
                    "source": "procgen", "subset_ok": True, "mechanic_tags": tags,
                    "difficulty_band": band, "difficulty_metric": "freespace",
                    "fs_cost": float(cost), "demo_len": 0,
                },
            ))
            band_hist[band] += 1
            for t in tags:
                tag_hist[t] += 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("shard_*.pkl"):  # avoid mixing with a prior, larger run
        stale.unlink()
    shard_paths = write_shards(records, out, shard_size=args.shard_size)
    (out / "ingest_index.json").write_text(json.dumps(
        {"n_levels": len(records), "sources": {"procgen": len(records)}, "shards": shard_paths}, indent=2))
    (out / "labels_index.json").write_text(json.dumps({
        "n_levels": len(records), "n_subset_ok": len(records),
        "difficulty_metric": "freespace", "difficulty_edges": edges,
        "sources": {"procgen": len(records)}, "difficulty_bands": dict(band_hist),
        "mechanic_tags": dict(tag_hist),
    }, indent=2))
    logger.info("Wrote %d procgen supplement levels to %s. bands=%s tags=%s",
                len(records), out, dict(band_hist), dict(tag_hist))


if __name__ == "__main__":
    main()
