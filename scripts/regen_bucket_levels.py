#!/usr/bin/env python3
"""Regenerate every level of one bucket of a packed benchmark dataset IN PLACE,
keeping level ids / indices / metadata layout (counts, split files) intact.

Use after a GENERATOR fix changes what a (generator, preset, seed) triple
produces — e.g. the MINE_NAVIGATION ground-mine fix (mines were anchored at
the tile corner without the sim's +1 border correction, floating 48px above
the floor so the agent could walk under them). Each level is regenerated from
its OWN stored generator/preset/seed with the standard generation gates
(exit-door present, spawn present, ``is_map_solvable``, bucket-wide hash
dedup); the level_id and metadata index are preserved so split metadata and
manifests stay valid without re-packing.

Usage::

    python -m scripts.regen_bucket_levels \
        --root datasets/mechanic_benchmark_v6_attract --bucket MINE_NAVIGATION
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from nclone.map_generation.generator_factory import GeneratorFactory
from nclone.map_generation.solvability import is_map_solvable
from scripts.generate_benchmark_dataset import _hash_map_data

MAX_RETRY = 10


def _regen(gen_type: str, preset: str, seed: int, seen: set) -> tuple:
    for attempt in range(MAX_RETRY):
        s = seed + attempt * 1_000_000
        gen = GeneratorFactory.create_from_preset(gen_type, preset, s)
        gen.generate(seed=s)
        md = gen.map_data()
        if md[1156] == 0:
            continue
        if md[1231] == 0 and md[1232] == 0:
            continue
        if not is_map_solvable(md):
            continue
        h = _hash_map_data(md)
        if h in seen:
            continue
        return list(md), s, h
    raise RuntimeError(
        f"no valid {gen_type}:{preset} map after {MAX_RETRY} retries from seed {seed}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)

    all_pkls = []
    seen: set = set()
    for split in ("train", "test"):
        for pkl in sorted((root / split / args.bucket).rglob("*.pkl")):
            with open(pkl, "rb") as f:
                payload = pickle.load(f)
            all_pkls.append((pkl, payload))
    if not all_pkls:
        raise SystemExit(f"no levels under {root}/*/{args.bucket}")

    # Regenerate in ORIGINAL generation order (split, tier, index) so each
    # level re-claims its own hash before any later level can collide with
    # it. Filename (level-id hash) order breaks small parameter spaces: a
    # later-index level regenerates first, steals the hash, and the earlier
    # level burns its retries hunting fresh seeds (stairs_t1 incident).
    all_pkls.sort(key=lambda item: (
        item[0].parts[-4],           # split dir (test/train)
        item[1]["tier"],
        item[1]["metadata"]["index"],
    ))

    changed = identical = kept_old = 0
    for pkl, payload in all_pkls:
        try:
            md, used_seed, new_hash = _regen(
                payload["generator"], payload["preset"], payload["seed"], seen)
        except RuntimeError as e:
            # Exhausted retries (dedup pressure in a tiny parameter space):
            # keep the existing level rather than fail the whole bucket.
            print(f"  KEEP-OLD {pkl.name}: {e}")
            seen.add(payload["metadata"]["map_hash"])
            kept_old += 1
            continue
        seen.add(new_hash)
        if new_hash == payload["metadata"]["map_hash"]:
            identical += 1
            continue
        payload["seed"] = used_seed
        payload["map_data"] = md
        payload["metadata"]["map_hash"] = new_hash
        changed += 1
        if not args.dry_run:
            with open(pkl, "wb") as f:
                pickle.dump(payload, f)

    print(f"{args.bucket}: {changed} levels regenerated "
          f"({identical} byte-identical, {kept_old} kept-old), "
          f"dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
