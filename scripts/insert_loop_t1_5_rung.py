#!/usr/bin/env python3
"""Blend the ``loop_t1_5`` rung into an existing benchmark dataset's
LOOP_CORRIDOR/T2 cell, IN PLACE (V13_NEXT_STEPS Workstream C1).

The v13 train diagnostics show a T1->T2 cliff on LOOP_CORRIDOR (0.898 ->
0.196) that is a ring-SIZE jump — the same shape as WALL_JUMP's 0% shaft
stall, fixed there by an intermediate rung. ``BUCKET_CATALOG`` now blends
``loop_t1_5`` (mid-size rings, single mine) into the T2 cell; this tool
applies the same blend to an ALREADY-PACKED dataset without touching level
ids, counts, or split metadata:

- For every ``<split>/LOOP_CORRIDOR/T2/<level_id>.pkl`` whose ``index`` maps
  to ``loop_t1_5`` under ``_pick_entry`` (even indices), regenerate the map
  with the ``loop_t1_5`` preset using the SAME stored seed (+1M per retry,
  exactly like ``BenchmarkGenerator._generate_one``) and overwrite the
  payload's map_data/generator/preset/metadata. Odd indices stay byte-
  identical (same preset+seed => deterministic same map).
- ``level_id`` / ``index`` / metadata counts are PRESERVED, so
  ``<split>_metadata.json`` and ``manifest.json`` remain valid as-is.
- Every regenerated map passes the same gates as generation: exit-door
  present, spawn present, ``is_map_solvable``, hash-dedup against the whole
  LOOP_CORRIDOR bucket.

Usage::

    python -m scripts.insert_loop_t1_5_rung \
        --root datasets/mechanic_benchmark_v6_attract [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

from nclone.map_generation.generator_configs import get_bucket_entries
from nclone.map_generation.generator_factory import GeneratorFactory
from nclone.map_generation.solvability import is_map_solvable
from scripts.generate_benchmark_dataset import _hash_map_data, _pick_entry

BUCKET = "LOOP_CORRIDOR"
TIER = 2
MAX_RETRY = 10


def _regen(seed: int, gen_type: str, preset: str, seen: set) -> tuple:
    """Regenerate with the generation-time gates; returns (map_data, seed, hash)."""
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
    raise RuntimeError(f"no valid {gen_type}:{preset} map after {MAX_RETRY} "
                       f"retries from seed {seed}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)

    entries = get_bucket_entries(BUCKET, TIER)
    presets = {p for _, p, _ in entries}
    assert "loop_t1_5" in presets, (
        "BUCKET_CATALOG LOOP_CORRIDOR tier 2 does not contain loop_t1_5 — "
        "update generator_configs.py first")

    # Hash-dedup universe: every level currently in the bucket (all tiers,
    # both splits) so a regenerated map can't collide with a kept one.
    seen = set()
    all_pkls = []
    for split in ("train", "test"):
        for pkl in sorted((root / split / BUCKET).rglob("*.pkl")):
            with open(pkl, "rb") as f:
                payload = pickle.load(f)
            all_pkls.append((pkl, payload))
            seen.add(payload["metadata"]["map_hash"])

    changed = kept = 0
    for pkl, payload in all_pkls:
        if payload["tier"] != TIER:
            continue
        idx = payload["metadata"]["index"]
        gen_type, preset = _pick_entry(entries, idx)
        if preset == payload["preset"]:
            kept += 1
            continue
        old_hash = payload["metadata"]["map_hash"]
        md, used_seed, new_hash = _regen(payload["seed"], gen_type, preset, seen)
        seen.discard(old_hash)
        seen.add(new_hash)
        payload["generator"] = gen_type
        payload["preset"] = preset
        payload["seed"] = used_seed
        payload["map_data"] = md
        payload["metadata"]["description"] = (
            f"{BUCKET} tier {TIER} ({gen_type}:{preset})")
        payload["metadata"]["map_hash"] = new_hash
        changed += 1
        if not args.dry_run:
            with open(pkl, "wb") as f:
                pickle.dump(payload, f)
        print(f"  {'DRY ' if args.dry_run else ''}"
              f"{pkl.parent.parent.parent.name}/T{TIER} idx={idx} "
              f"-> {preset} seed={used_seed}")

    print(f"\n{changed} levels re-blended to loop_t1_5, {kept} kept as loop_t2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
