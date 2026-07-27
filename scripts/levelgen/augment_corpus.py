"""augment_corpus.py — a difficulty-tuned RL level pool from the REAL corpus via
demo-preserving augmentation (horizontal mirror + reachable-area translation).

Why this exists
---------------
Conditional tile generation failed (difficulty is an entity/route property, not a tile
signal — see the ``prepend-conditioning-failed`` finding). The augmented real corpus is
the stronger *primary* level source for the generalized agent's curriculum:

* **Guaranteed solvable** — every level is an isometry of a corpus level whose winning
  demo we *replay to a win on the native sim here* (the gold-standard gate, stronger than
  the geometric contract). Mirror swaps L/R inputs; translation is position-invariant, so
  the (transformed) demo still solves the variant.
* **Subset-valid by construction** — augmentation never adds/removes entities, so exactly
  1 switch/door + ≤64 mines/gold + 1 spawn are preserved.
* **Difficulty-tuned** — difficulty = the original human solve time (``demo_len``), binned
  into bands; the curriculum samples per band. (The geometric free-space ``route_cost`` is
  stored too, as a secondary signal.)
* **Position-varied** — N translations place the same structure across the 42×23 canvas,
  which the position-encoded agent must generalize over.

The contract (``level_contract``) gates the *no-demo* generative path; this demo-bearing
path is gated by replay-to-win instead (the contract's standable/geometric checks are too
strict for real mid-air-spawn levels and would break demos by repairing them).

Usage
-----
    python -m scripts.levelgen.augment_corpus \\
        --corpus datasets/levelgen/corpus_subset \\
        --out datasets/levelgen/pool_augmented \\
        --translations 3 --bands 5 --workers 8 --limit 0
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

from nclone.map_augmentation.attract.attract_augmentor import (
    _HEADER_LEN,
    _SPAWN_X_OFFSET,
    _SPAWN_Y_OFFSET,
    _TILES_LEN,
    _TILES_OFFSET,
    _build_map_data,
    _parse_entities_from_map_bytes,
)
from nclone.map_augmentation.attract.horizontal_mirror import HorizontalMirror
from nclone.map_augmentation.attract.reachable_area_translator import ReachableAreaTranslator

logger = logging.getLogger("levelgen.augment_corpus")


# --------------------------------------------------------------------------- bridge
def map_data_to_attract(map_data: List[int], demo: List[int]) -> Dict:
    """``map_data`` + demo -> AttractData dict (mirror of attract_augmentor._decode)."""
    mb = bytes(map_data)
    return {
        "tiles": list(mb[_TILES_OFFSET : _TILES_OFFSET + _TILES_LEN]),
        "entities": _parse_entities_from_map_bytes(mb),
        "ninja_spawn": (mb[_SPAWN_X_OFFSET], mb[_SPAWN_Y_OFFSET]),
        "header": mb[:_HEADER_LEN],
        "demo_data": list(demo),
    }


# --------------------------------------------------------------------------- gate
def replay_won(map_data: List[int], demo: List[int], slack: int = 120) -> bool:
    """True iff replaying ``demo`` on the native sim reaches a win (with a small slack)."""
    import nclone_native as Nnative
    from nclone.replay.replay_executor import decode_input_to_controls

    sim = Nnative.Simulator.load(list(map_data))
    for b in demo:
        h, j = decode_input_to_controls(int(b))
        sim.tick(int(h), int(j))
        if sim.has_won():
            return True
        if sim.has_died():
            return False
    for _ in range(slack):  # win may land a few frames after the last logged input
        sim.tick(0, 0)
        if sim.has_won():
            return True
    return False


# --------------------------------------------------------------------------- corpus IO
def iter_records(corpus_dir: str):
    for shard in sorted(glob.glob(os.path.join(corpus_dir, "shard_*.pkl"))):
        for r in pickle.load(open(shard, "rb"))["records"]:
            if r.get("demo_inputs") and r.get("demo_len", 0) > 0:
                yield r


def difficulty_edges(demo_lens: List[int], n_bands: int) -> List[float]:
    from nclone.levelgen.free_space_difficulty import fit_edges

    return fit_edges([float(d) for d in demo_lens], n_bands=n_bands)


# --------------------------------------------------------------------------- worker
def _augment_one(args_tuple) -> List[Tuple[List[int], List[int], int, str]]:
    """Produce verified variants of one record.

    Returns a list of ``(map_data, demo, demo_len, op)`` for variants whose demo replays
    to a win. Runs in a worker process (heavy native replays).
    """
    rec, n_translations, seed, verify, include_original, translate_buffer = args_tuple
    md = list(rec["map_data"])
    demo = list(rec["demo_inputs"])
    dlen = int(rec.get("demo_len", len(demo)))
    out: List[Tuple[List[int], List[int], int, str]] = []

    def _accept(new_md, new_demo, op):
        if (not verify) or replay_won(new_md, new_demo):
            out.append((new_md, new_demo, dlen, op))

    if include_original:
        _accept(md, demo, "orig")

    base = map_data_to_attract(md, demo)
    variants = [("hmirror", HorizontalMirror())]
    for i in range(n_translations):
        variants.append((f"translate{i}", ReachableAreaTranslator(seed=seed + i, buffer=translate_buffer)))

    for op, aug in variants:
        try:
            new = aug.augment(base)
            new_bytes = _build_map_data(new)
            if new_bytes is None:
                continue
            _accept(list(new_bytes), list(new["demo_data"]), op)
        except Exception as e:  # pragma: no cover - a bad variant is just skipped
            logger.debug("augment %s failed: %s", op, e)
    return out


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, help="decoded corpus dir (shard_*.pkl)")
    ap.add_argument("--out", required=True, help="output packed-pool dir")
    ap.add_argument("--translations", type=int, default=3, help="random translations per level")
    ap.add_argument("--translate-buffer", type=int, default=2,
                    help="tiles of reachable-bbox margin protected from random fill (higher = "
                         "more demos survive translation, less domain randomization)")
    ap.add_argument("--bands", type=int, default=5, help="difficulty bands over demo_len")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0, help="cap source levels (0 = all)")
    ap.add_argument("--no-verify", action="store_true", help="skip demo replay verification")
    ap.add_argument("--no-original", action="store_true", help="omit the un-augmented level")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=20000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from nclone.levelgen.free_space_difficulty import assign_band, route_cost
    from nclone.levelgen.pool_store import PoolWriter

    records = list(iter_records(args.corpus))
    if args.limit:
        records = records[: args.limit]
    logger.info("loaded %d source levels with demos from %s", len(records), args.corpus)

    edges = difficulty_edges([int(r["demo_len"]) for r in records], args.bands)
    logger.info("difficulty edges (demo_len quantiles, %d bands): %s", args.bands, edges)

    work = [
        (r, args.translations, args.seed + 1000 * i, not args.no_verify,
         not args.no_original, args.translate_buffer)
        for i, r in enumerate(records)
    ]

    n_in = n_out = n_reject = 0
    band_counts: Dict[str, int] = {}
    op_counts: Dict[str, int] = {}

    def _consume(variants, src_id):
        nonlocal n_out
        for (new_md, new_demo, dlen, op) in variants:
            band = assign_band(float(dlen), edges)
            fs = route_cost(new_md)
            fs = None if fs == float("inf") else fs
            lid = writer.add(
                new_md, bucket="AUGMENT", source=f"augment:{op}", parent=src_id,
                fs_cost=fs, difficulty_band=band,
                extra={"demo_len": dlen, "op": op, "demo": new_demo},
            )
            if lid is not None:
                n_out += 1
                band_counts[band] = band_counts.get(band, 0) + 1
                op_counts[op] = op_counts.get(op, 0) + 1

    with PoolWriter(args.out, shard_size=args.shard_size) as writer:
        if args.workers > 1:
            import multiprocessing as mp

            with mp.Pool(args.workers) as pool:
                for rec, variants in zip(records, pool.imap(_augment_one, work, chunksize=8)):
                    n_in += 1
                    expected = (0 if args.no_original else 1) + 1 + args.translations
                    n_reject += max(0, expected - len(variants))
                    _consume(variants, rec.get("level_id"))
                    if n_in % 500 == 0:
                        logger.info("processed %d/%d sources -> %d pooled", n_in, len(records), n_out)
        else:
            for rec, w in zip(records, work):
                n_in += 1
                variants = _augment_one(w)
                expected = (0 if args.no_original else 1) + 1 + args.translations
                n_reject += max(0, expected - len(variants))
                _consume(variants, rec.get("level_id"))

    summary = {
        "sources": n_in, "pooled": n_out, "rejected_variants": n_reject,
        "by_band": band_counts, "by_op": op_counts, "edges": edges, "out": args.out,
    }
    logger.info("DONE: %s", json.dumps(summary, indent=2))
    json.dump(summary, open(os.path.join(args.out, "augment_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
