#!/usr/bin/env python3
"""Merge a standalone single-bucket dataset into a copy of a full benchmark dataset.

WHY THIS EXISTS -- do NOT just re-run generate_benchmark_dataset.py with
``--buckets NEW_BUCKET --output_dir <existing dataset>``. That writes the level
pickles correctly but then OVERWRITES ``<split>/<split>_metadata.json`` with a
split_meta built from ONLY the buckets it just generated. Since
``BenchmarkLevelSource`` resolves levels from exactly that file, the result is a
dataset whose pickles are all present but whose loader can only see the new
bucket -- every pre-existing bucket silently disappears from training.

This script instead:
  1. copies SRC dataset -> DEST (leaving SRC pristine),
  2. copies the new bucket's level pickles in,
  3. MERGES each split's metadata (adds the bucket entry, extends ``level_ids``,
     fixes ``total_levels``) rather than replacing it,
  4. refreshes manifest.json's bucket counts if present.

    merge_bucket_into_dataset.py --src datasets/mechanic_benchmark_v7_official \
        --bucket-src datasets/mine_field_v2_full --bucket MINE_FIELD \
        --dest datasets/mechanic_benchmark_v8_minefield
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _merge_split(dest: Path, bucket_src: Path, bucket: str, split: str) -> int:
    src_meta_path = bucket_src / split / f"{split}_metadata.json"
    dst_meta_path = dest / split / f"{split}_metadata.json"
    if not src_meta_path.exists():
        print(f"  [{split}] no source metadata, skipping")
        return 0
    if not dst_meta_path.exists():
        print(f"  [{split}] ERROR: destination metadata missing", file=sys.stderr)
        return 0

    src_meta = json.loads(src_meta_path.read_text())
    dst_meta = json.loads(dst_meta_path.read_text())

    if bucket in dst_meta.get("buckets", {}):
        print(f"  [{split}] ERROR: {bucket} already present in destination; refusing "
              f"to merge twice", file=sys.stderr)
        return 0

    # 1. level pickles
    n_files = 0
    for tier_dir in sorted((bucket_src / split / bucket).glob("T*")):
        out_dir = dest / split / bucket / tier_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for pkl in tier_dir.glob("*.pkl"):
            shutil.copy2(pkl, out_dir / pkl.name)
            n_files += 1

    # 2. metadata merge -- additive, never a replace
    # level_ids live per TIER inside the bucket entry, not on the bucket itself:
    #   buckets[B] = {"total": n, "tiers": {"1": {"count":…, "level_ids": [...]}}}
    if bucket in src_meta.get("buckets", {}):
        new_ids = [lid
                   for tier in src_meta["buckets"][bucket].get("tiers", {}).values()
                   for lid in tier.get("level_ids", [])]
    else:
        new_ids = list(src_meta.get("level_ids", []))
    existing = set(dst_meta.get("level_ids", []))
    clash = existing & set(new_ids)
    if clash:
        print(f"  [{split}] ERROR: {len(clash)} level_id collisions (e.g. "
              f"{sorted(clash)[:3]}) -- bucket seed offsets overlap", file=sys.stderr)
        return 0

    dst_meta["buckets"][bucket] = src_meta["buckets"][bucket]
    dst_meta["level_ids"] = list(dst_meta.get("level_ids", [])) + new_ids
    dst_meta["total_levels"] = len(dst_meta["level_ids"])
    dst_meta_path.write_text(json.dumps(dst_meta, indent=2))
    print(f"  [{split}] +{n_files} pickles, +{len(new_ids)} level_ids "
          f"-> total {dst_meta['total_levels']}")
    return n_files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="existing full dataset (left untouched)")
    ap.add_argument("--bucket-src", required=True, help="standalone single-bucket dataset")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--dest", required=True, help="new dataset dir to create")
    ap.add_argument("--force", action="store_true", help="overwrite dest if it exists")
    args = ap.parse_args()

    src, bsrc, dest = Path(args.src), Path(args.bucket_src), Path(args.dest)
    for p in (src, bsrc):
        if not p.is_dir():
            print(f"missing: {p}", file=sys.stderr)
            return 2
    if dest.exists():
        if not args.force:
            print(f"dest exists: {dest} (use --force)", file=sys.stderr)
            return 2
        shutil.rmtree(dest)

    print(f"copying {src} -> {dest}")
    shutil.copytree(src, dest)

    total = 0
    for split in ("train", "test"):
        total += _merge_split(dest, bsrc, args.bucket, split)
    if total == 0:
        print("nothing merged", file=sys.stderr)
        return 1

    # manifest.json is descriptive; refresh its per-bucket counts if it has them.
    man = dest / "manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        counts = {}
        for split in ("train", "test"):
            mp = dest / split / f"{split}_metadata.json"
            if mp.exists():
                counts[split] = json.loads(mp.read_text())["total_levels"]
        m["merged_bucket"] = args.bucket
        m["split_totals_after_merge"] = counts
        man.write_text(json.dumps(m, indent=2))
        print(f"manifest updated: {counts}")
    print(f"done: {total} level files merged into {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
