#!/usr/bin/env python
"""Validate (and optionally repair) a mechanic-benchmark dataset so that EVERY
stored level is solvable under the current reachability model.

Generation gates each *seed* with ``is_map_solvable`` (``BenchmarkGenerator
--check_solvable``), but that guarantee is only as current as the reachability
code at generation time. When the jump-aware traversability graph evolves, a
level that passed an older gate can read as unsolvable now -- producing ``inf``
cost-to-go that silently breaks PBRS / goal-distance curriculum for that level
(and floods logs with "position far outside graph bounds"). This tool is the
belt-and-suspenders guarantee: re-audit the *shipped* dataset with the current
``is_map_solvable`` and fail (CI / post-generation) or repair.

Usage:
  # Audit only (exit 1 if any unsolvable) -- use in CI / post-generation:
  python -m scripts.validate_benchmark_dataset --root datasets/mechanic_benchmark_v4

  # Repair in place: regenerate a solvable replacement for each unsolvable level,
  # preserving level_id / bucket / tier / generator / preset:
  python -m scripts.validate_benchmark_dataset --root datasets/mechanic_benchmark_v4 --repair
"""
from __future__ import annotations

import argparse
import collections
import os
import pickle
import sys
from pathlib import Path
from multiprocessing import Pool
from typing import List, Optional, Tuple


def _audit_one(pkl_str: str) -> Tuple[str, str, bool, Optional[str]]:
    """Worker: (bucket/tier key, level_id, solvable, err)."""
    from nclone.map_generation.solvability import is_map_solvable
    import logging

    logging.disable(logging.CRITICAL)
    pkl = Path(pkl_str)
    key = f"{pkl.parts[-3]}/{pkl.parts[-2]}"
    try:
        with open(pkl, "rb") as f:
            md = list(pickle.load(f)["map_data"])
        ok = bool(is_map_solvable(md))
    except Exception as e:  # noqa: BLE001 - report, never crash the sweep
        return (key, pkl.stem, False, f"ERR:{type(e).__name__}")
    return (key, pkl.stem, ok, None)


def audit(root: Path, jobs: int) -> List[Tuple[str, str, str, Optional[str]]]:
    """Return list of (split, bucket/tier, pkl_path, err) for UNSOLVABLE levels."""
    files = [
        str(p)
        for split in ("train", "test")
        for p in sorted((root / split).rglob("*.pkl"))
    ]
    print(f"auditing {len(files)} levels on {jobs} workers ...", flush=True)
    with Pool(jobs) as pool:
        results = pool.map(_audit_one, files, chunksize=8)

    per = collections.Counter()
    unsolv = collections.Counter()
    bad: List[Tuple[str, str, str, Optional[str]]] = []
    file_by_stem = {Path(f).stem + "|" + f"{Path(f).parts[-3]}/{Path(f).parts[-2]}": f for f in files}
    for (key, stem, ok, err), fpath in zip(results, files):
        per[key] += 1
        if not ok:
            unsolv[key] += 1
            split = Path(fpath).parts[-4]
            bad.append((split, key, fpath, err))

    total = len(results)
    print(f"\n=== {total} levels, {len(bad)} UNSOLVABLE ({100*len(bad)/max(1,total):.2f}%) ===")
    if unsolv:
        print("Per bucket/tier unsolvable:")
        for k in sorted(unsolv, key=lambda k: -unsolv[k]):
            print(f"  {k:28s} {unsolv[k]:4d}/{per[k]:4d} ({100*unsolv[k]/per[k]:.0f}%)")
        roll = collections.Counter()
        rollt = collections.Counter()
        for k in per:
            roll[k.split("/")[0]] += per[k]
        for k in unsolv:
            rollt[k.split("/")[0]] += unsolv[k]
        print("Per BUCKET rollup:")
        for b in sorted(rollt, key=lambda b: -rollt[b]):
            print(f"  {b:24s} {rollt[b]:4d}/{roll[b]:4d} ({100*rollt[b]/roll[b]:.0f}%)")
    return bad


def _all_hashes(root: Path) -> set:
    """Every stored map_hash (to keep repairs globally unique like generation)."""
    seen = set()
    for split in ("train", "test"):
        for p in (root / split).rglob("*.pkl"):
            try:
                with open(p, "rb") as f:
                    seen.add(pickle.load(f)["metadata"].get("map_hash"))
            except Exception:  # noqa: BLE001
                continue
    seen.discard(None)
    return seen


def repair(root: Path, bad: List[Tuple[str, str, str, Optional[str]]], max_retry: int) -> int:
    """Regenerate a solvable replacement for each unsolvable level, in place.

    Preserves level_id / bucket / tier / generator / preset / index; only
    map_data + seed + map_hash change. Returns the number NOT repaired.
    """
    from nclone.map_generation.generator_factory import GeneratorFactory
    from nclone.map_generation.solvability import is_map_solvable
    from scripts.generate_benchmark_dataset import _hash_map_data, subset_ok
    import logging

    logging.disable(logging.CRITICAL)
    seen = _all_hashes(root)
    unrepaired = []
    for split, key, fpath, err in bad:
        with open(fpath, "rb") as f:
            payload = pickle.load(f)
        seen.discard(payload["metadata"].get("map_hash"))  # free the old hash
        gen_type = payload["generator"]
        preset = payload["preset"]
        base_seed = int(payload.get("seed", 0))
        fixed = False
        # Reseed forward from the stored seed; the stored seed itself already
        # failed, so start at +1M (the same stride generation uses).
        for attempt in range(1, max_retry + 1):
            seed = base_seed + attempt * 1_000_000
            try:
                mg = GeneratorFactory.create_from_preset(gen_type, preset, seed)
                mg.generate(seed=seed)
                md = mg.map_data()
            except Exception:  # noqa: BLE001
                continue
            if md[1156] == 0:
                continue
            if md[1231] == 0 and md[1232] == 0:
                continue
            if not is_map_solvable(md):
                continue
            if not subset_ok(md):
                continue
            h = _hash_map_data(md)
            if h in seen:
                continue
            # accept
            seen.add(h)
            payload["map_data"] = list(md)
            payload["seed"] = seed
            payload["metadata"]["map_hash"] = h
            payload["metadata"]["repaired"] = True
            with open(fpath, "wb") as f:
                pickle.dump(payload, f)
            fixed = True
            break
        if not fixed:
            unrepaired.append((split, key, fpath))
            seen.add(payload["metadata"].get("map_hash"))  # keep old hash reserved

    print(f"\nrepaired {len(bad) - len(unrepaired)}/{len(bad)}; UNREPAIRED={len(unrepaired)}")
    for split, key, fpath in unrepaired[:20]:
        print(f"  UNREPAIRED {split} {key} {Path(fpath).stem}")
    return len(unrepaired)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dataset root (contains train/ test/)")
    ap.add_argument("--repair", action="store_true", help="regenerate solvable replacements in place")
    ap.add_argument("--max-retry", type=int, default=64, help="reseed attempts per repaired level")
    ap.add_argument("--jobs", type=int, default=min(24, os.cpu_count() or 4))
    args = ap.parse_args()

    root = Path(args.root)
    bad = audit(root, args.jobs)
    if not bad:
        print("\nOK: every level is solvable.")
        return 0
    if not args.repair:
        print("\nFAIL: unsolvable levels present. Re-run with --repair to fix in place.")
        return 1
    remaining = repair(root, bad, args.max_retry)
    # Re-audit to confirm the repair actually closed everything.
    print("\n=== re-audit after repair ===")
    bad2 = audit(root, args.jobs)
    if bad2:
        print(f"\nFAIL: {len(bad2)} levels still unsolvable after repair.")
        return 1
    print("\nOK: dataset fully repaired; every level is solvable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
