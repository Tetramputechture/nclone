#!/usr/bin/env python3
"""Inject a single hand-authored level (default: ``test_maps/clincher``) into a
mechanic-benchmark dataset as its own bucket (default ``CLINCHER``).

This makes the level addressable by the bucket-gated curriculum
(``CurriculumPhaseCallback`` admits levels *by bucket name*), so a curriculum
preset can end on a terminal phase that trains only on this level while the
critic (warmed across the procedural mechanic buckets) drives BSRS shaping.

The dataset layout written here matches ``generate_benchmark_dataset.py``:

    <root>/manifest.json
    <root>/<split>/<split>_metadata.json
    <root>/<split>/<BUCKET>/T<tier>/<level_id>.pkl

Idempotent: re-running removes any prior copy of the bucket before re-adding
it, so the manifest/metadata counts stay correct.

Example::

    python -m scripts.inject_clincher_bucket \
        --dataset-root ./datasets/mechanic_full_v1 \
        --map nclone/test_maps/clincher
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, List


def _load_map_data(map_path: Path) -> List[int]:
    """Read a raw N++ map file into the list-of-ints ``map_data`` contract.

    Mirrors ``NPlayHeadless.load_map``: ``map_data = [int(b) for b in bytes]``.
    """
    return [int(b) for b in map_path.read_bytes()]


def _split_present(root: Path, split: str) -> bool:
    return (root / split / f"{split}_metadata.json").exists()


def _remove_existing_bucket(root: Path, split: str, bucket: str, meta: Dict) -> None:
    """Drop any prior copy of ``bucket`` from disk + the in-memory metadata."""
    bucket_dir = root / split / bucket
    if bucket_dir.exists():
        shutil.rmtree(bucket_dir)
    prior = meta.get("buckets", {}).pop(bucket, None)
    if prior:
        prior_ids = {
            lid
            for tier in prior.get("tiers", {}).values()
            for lid in tier.get("level_ids", [])
        }
        meta["level_ids"] = [
            lid for lid in meta.get("level_ids", []) if lid not in prior_ids
        ]
        meta["total_levels"] = len(meta["level_ids"])


def inject(
    root: Path,
    map_path: Path,
    bucket: str,
    tier: int,
    level_id: int,
    splits: List[str],
    generator: str,
    preset: str,
) -> None:
    map_data = _load_map_data(map_path)
    print(f"Loaded {map_path} -> {len(map_data)} map_data bytes")

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    for split in splits:
        if not _split_present(root, split):
            print(f"  [{split}] no split metadata; skipping")
            continue
        meta_path = root / split / f"{split}_metadata.json"
        meta = json.loads(meta_path.read_text())

        _remove_existing_bucket(root, split, bucket, meta)

        # --- write the pickle payload (same schema as the generator) ---
        rel_dir = f"{split}/{bucket}/T{tier}"
        pkl_dir = root / split / bucket / f"T{tier}"
        pkl_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "level_id": int(level_id),
            "seed": int(level_id),
            "bucket": bucket,
            "tier": int(tier),
            "generator": generator,
            "preset": preset,
            "map_data": map_data,
            "secondary": [],
            "metadata": {
                "description": f"{bucket} hand-authored ({map_path.name})",
                "split": split,
                "source": "hand_authored",
                "map_path": str(map_path),
            },
        }
        with (pkl_dir / f"{level_id}.pkl").open("wb") as f:
            pickle.dump(payload, f)

        # --- register in per-split metadata ---
        meta.setdefault("buckets", {})[bucket] = {
            "tiers": {
                str(tier): {
                    "count": 1,
                    "level_ids": [int(level_id)],
                    "path": rel_dir,
                }
            }
        }
        meta.setdefault("level_ids", []).append(int(level_id))
        meta["total_levels"] = len(meta["level_ids"])
        meta_path.write_text(json.dumps(meta, indent=2))

        # --- register in the top-level manifest ---
        if bucket not in manifest["buckets"]:
            manifest["buckets"].append(bucket)
        sp = manifest["splits"].setdefault(
            split, {"total_levels": 0, "per_bucket": {}}
        )
        sp["per_bucket"][bucket] = 1
        sp["total_levels"] = sum(sp["per_bucket"].values())
        print(f"  [{split}] wrote {rel_dir}/{level_id}.pkl; "
              f"split total -> {sp['total_levels']}")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Updated {manifest_path}; buckets now {manifest['buckets']}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--map", type=Path, default=Path("nclone/test_maps/clincher"))
    p.add_argument("--bucket", default="CLINCHER")
    p.add_argument("--tier", type=int, default=1)
    p.add_argument(
        "--level-id",
        type=int,
        default=900000000000000001,
        help="Stable level_id for the train copy; test gets +1.",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Splits to inject into (each gets its own level_id).",
    )
    p.add_argument("--generator", default="hand_authored")
    p.add_argument("--preset", default="clincher")
    args = p.parse_args()

    if not args.map.exists():
        raise SystemExit(f"Map not found: {args.map}")

    # Each split gets a distinct level_id so the loader's rglob is unambiguous.
    for offset, split in enumerate(args.splits):
        inject(
            root=args.dataset_root,
            map_path=args.map,
            bucket=args.bucket,
            tier=args.tier,
            level_id=args.level_id + offset,
            splits=[split],
            generator=args.generator,
            preset=args.preset,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
