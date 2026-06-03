"""Generate the mechanic-primary benchmark dataset.

This script produces a procedural, mechanic-bucket-primary benchmark
dataset for training, analysis, and BSRS-based curriculum learning.
It is additive to and independent of the legacy
``nclone.map_generation.generate_test_suite_maps`` script:

- Taxonomy is defined by ``BUCKET_CATALOG`` in
  ``nclone/map_generation/generator_configs.py`` (mechanic -> tier ->
  generator presets), not by ``CATEGORIES``.
- Output format is compatible with ``TestSuiteLoader`` / ``EnvMapLoader``:
  individual pickle files per level, a per-split metadata JSON, and a
  top-level ``manifest.json`` that captures the full catalog snapshot
  for reproducibility.
- Validation is deliberately light (hash dedup only) to keep generation
  fast.

Usage
-----
Generate both train and test splits::

    python -m scripts.generate_benchmark_dataset

Smaller smoke run::

    python -m scripts.generate_benchmark_dataset \\
        --output_dir ./datasets/mechanic_benchmark_smoke \\
        --per_bucket_per_tier_train 2 --per_bucket_per_tier_test 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Support running both as module (-m scripts.generate_benchmark_dataset)
# and as a direct script. The repo uses nclone as the installed package.
from nclone.map_generation.generator_configs import (
    BUCKET_CATALOG,
    BUCKET_TIERS,
    GENERATOR_PRESETS,
    PHASE_1_BUCKETS,
    get_bucket_entries,
    validate_bucket_catalog,
)
from nclone.map_generation.generator_factory import GeneratorFactory


DEFAULT_OUTPUT_DIR = "./datasets/mechanic_benchmark"
DEFAULT_SEED_BASE_TRAIN = 1_000_000
DEFAULT_SEED_BASE_TEST = 100_000
DEFAULT_PER_BUCKET_PER_TIER_TRAIN = 25
DEFAULT_PER_BUCKET_PER_TIER_TEST = 10
MAX_RETRY = 10
BUCKET_STRIDE = 10_000_000  # bucket_index * stride = seed offset between buckets
TIER_STRIDE = 1_000_000  # tier * stride = seed offset between tiers


def _bucket_index(bucket: str) -> int:
    """Deterministic 0-based index for each bucket (position in catalog)."""
    buckets = list(BUCKET_CATALOG.keys())
    return buckets.index(bucket)


def _compute_level_id(bucket: str, tier: int, index: int, split: str) -> int:
    """Compute a stable integer level ID used by ``LevelPool`` and caches.

    The ID is a 63-bit truncation of a SHA-1 over
    ``(split, bucket, tier, index)``. Collisions across splits are
    astronomically unlikely (<2^-31 for all plausible dataset sizes).
    """
    payload = f"{split}:{bucket}:{tier}:{index}".encode("utf-8")
    digest = hashlib.sha1(payload).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _compute_seed(bucket: str, tier: int, index: int, seed_base: int) -> int:
    return seed_base + _bucket_index(bucket) * BUCKET_STRIDE + tier * TIER_STRIDE + index


def _pick_entry(
    entries: List[Tuple[str, str, int]], level_index: int
) -> Tuple[str, str]:
    """Round-robin across entries by weight. Deterministic."""
    expanded: List[Tuple[str, str]] = []
    for gen_type, preset, weight in entries:
        expanded.extend([(gen_type, preset)] * max(1, int(weight)))
    return expanded[level_index % len(expanded)]


def _serialize_preset_config(generator_type: str, preset: str) -> Dict[str, Any]:
    """Return a JSON-safe dict snapshot of the preset config."""
    config = GENERATOR_PRESETS[generator_type][preset]
    if is_dataclass(config):
        return asdict(config)
    return {"repr": repr(config)}


def _hash_map_data(map_data: List[int]) -> str:
    """Stable hex hash of the map_data list for dedup + manifest."""
    h = hashlib.sha1()
    # Encode once into bytes; map data values fit in int16/int32 generally.
    h.update(
        b",".join(str(int(v)).encode("ascii") for v in map_data)
    )
    return h.hexdigest()


class BenchmarkGenerator:
    """Top-level orchestrator for mechanic-bucket dataset generation."""

    def __init__(
        self,
        output_dir: Path,
        buckets: List[str],
        tiers: List[int],
        per_bucket_per_tier_train: int,
        per_bucket_per_tier_test: int,
        seed_base_train: int,
        seed_base_test: int,
        max_retry: int = MAX_RETRY,
        check_solvable: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.buckets = buckets
        self.tiers = tiers
        self.per_train = per_bucket_per_tier_train
        self.per_test = per_bucket_per_tier_test
        self.seed_base_train = seed_base_train
        self.seed_base_test = seed_base_test
        self.max_retry = max_retry
        self.check_solvable = check_solvable

        self._seen_hashes: set = set()
        self._duplicates = 0
        self._skipped: List[Dict[str, Any]] = []

    def _generate_one(
        self,
        bucket: str,
        tier: int,
        index: int,
        split: str,
        seed_base: int,
    ) -> Optional[Dict[str, Any]]:
        entries = get_bucket_entries(bucket, tier)
        if not entries:
            return None

        gen_type, preset = _pick_entry(entries, index)
        level_id = _compute_level_id(bucket, tier, index, split)

        for attempt in range(self.max_retry):
            seed = _compute_seed(bucket, tier, index, seed_base) + attempt * 1_000_000
            map_gen = GeneratorFactory.create_from_preset(gen_type, preset, seed)
            map_gen.generate(seed=seed)
            map_data = map_gen.map_data()

            if map_data[1156] == 0:
                # No exit door => broken generation; try next seed.
                continue
            ninja_x = map_data[1231]
            ninja_y = map_data[1232]
            if ninja_x == 0 and ninja_y == 0:
                continue

            # Optional reachability gate: reject maps where the switch or exit
            # is not reachable from the ninja spawn (uses the grid-based
            # traversability graph, not the heavy lattice precompute). Off by
            # default; the procedural generators are solvable-by-construction.
            if self.check_solvable:
                from nclone.map_generation.solvability import is_map_solvable

                if not is_map_solvable(map_data):
                    continue

            map_hash = _hash_map_data(map_data)
            if map_hash in self._seen_hashes:
                self._duplicates += 1
                continue

            self._seen_hashes.add(map_hash)
            return {
                "level_id": level_id,
                "seed": seed,
                "bucket": bucket,
                "tier": tier,
                "generator": gen_type,
                "preset": preset,
                "map_data": list(map_data),
                "secondary": [],
                "metadata": {
                    "description": f"{bucket} tier {tier} ({gen_type}:{preset})",
                    "split": split,
                    "source": "procedural",
                    "map_hash": map_hash,
                    "index": index,
                },
            }

        self._skipped.append(
            {"bucket": bucket, "tier": tier, "index": index, "split": split}
        )
        return None

    def _generate_split(
        self, split: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[int, List[int]]]]:
        """Generate all levels for one split. Returns (levels, layout).

        ``layout`` is ``{bucket: {tier: [level_index, ...]}}`` preserving
        the on-disk ordering so downstream consumers can reconstruct it
        without re-hashing.
        """
        count = self.per_train if split == "train" else self.per_test
        seed_base = self.seed_base_train if split == "train" else self.seed_base_test

        levels: List[Dict[str, Any]] = []
        layout: Dict[str, Dict[int, List[int]]] = {}

        for bucket in self.buckets:
            layout.setdefault(bucket, {})
            for tier in self.tiers:
                entries = get_bucket_entries(bucket, tier)
                if not entries:
                    layout[bucket][tier] = []
                    continue

                tier_level_ids: List[int] = []
                for i in range(count):
                    level = self._generate_one(bucket, tier, i, split, seed_base)
                    if level is None:
                        continue
                    levels.append(level)
                    tier_level_ids.append(level["level_id"])
                layout[bucket][tier] = tier_level_ids

                print(
                    f"  [{split}] {bucket}/T{tier}: generated {len(tier_level_ids)}/{count}"
                )

        return levels, layout

    def _write_split(
        self,
        split: str,
        levels: List[Dict[str, Any]],
        layout: Dict[str, Dict[int, List[int]]],
    ) -> None:
        split_dir = self.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for level in levels:
            bucket = level["bucket"]
            tier = level["tier"]
            tier_dir = split_dir / bucket / f"T{tier}"
            tier_dir.mkdir(parents=True, exist_ok=True)
            level_file = tier_dir / f"{level['level_id']}.pkl"
            with open(level_file, "wb") as f:
                pickle.dump(level, f)

        split_meta: Dict[str, Any] = {
            "split": split,
            "total_levels": len(levels),
            "buckets": {},
            # Flat ordered list used by env-side loaders that just want a
            # sequential schedule of level_ids.
            "level_ids": [level["level_id"] for level in levels],
        }
        for bucket, per_tier in layout.items():
            split_meta["buckets"][bucket] = {
                "total": sum(len(ids) for ids in per_tier.values()),
                "tiers": {
                    str(tier): {
                        "count": len(per_tier.get(tier, [])),
                        "level_ids": per_tier.get(tier, []),
                        "path": f"{split}/{bucket}/T{tier}",
                    }
                    for tier in self.tiers
                },
            }

        meta_file = split_dir / f"{split}_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(split_meta, f, indent=2)

    def _write_manifest(
        self,
        split_summaries: Dict[str, Dict[str, Any]],
    ) -> None:
        catalog_snapshot: Dict[str, Any] = {}
        for bucket, tiers in BUCKET_CATALOG.items():
            catalog_snapshot[bucket] = {
                str(tier): [
                    {
                        "generator": gen_type,
                        "preset": preset,
                        "weight": weight,
                        "config": _serialize_preset_config(gen_type, preset),
                    }
                    for gen_type, preset, weight in entries
                ]
                for tier, entries in tiers.items()
            }

        manifest = {
            "version": 1,
            "source": "scripts/generate_benchmark_dataset.py",
            "schema": "mechanic_primary_v1",
            "buckets": self.buckets,
            "tiers": self.tiers,
            "seed_bases": {
                "train": self.seed_base_train,
                "test": self.seed_base_test,
            },
            "counts_per_bucket_per_tier": {
                "train": self.per_train,
                "test": self.per_test,
            },
            "splits": split_summaries,
            "catalog": catalog_snapshot,
            "generation_info": {
                "duplicates_retried": self._duplicates,
                "skipped": self._skipped,
                "unique_hashes": len(self._seen_hashes),
            },
        }

        manifest_file = self.output_dir / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def run(self, mode: str) -> int:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Mechanic-bucket benchmark dataset ===")
        print(f"Output: {self.output_dir.resolve()}")
        print(f"Buckets: {self.buckets}")
        print(f"Tiers: {self.tiers}")
        print(f"Mode: {mode}")

        catalog_errors = validate_bucket_catalog()
        if catalog_errors:
            print("BUCKET_CATALOG validation errors:", file=sys.stderr)
            for err in catalog_errors:
                print(f"  - {err}", file=sys.stderr)
            return 2

        split_summaries: Dict[str, Dict[str, Any]] = {}

        splits = []
        if mode in ("train", "both"):
            splits.append("train")
        if mode in ("test", "both"):
            splits.append("test")

        for split in splits:
            print(f"\n--- Generating split: {split} ---")
            levels, layout = self._generate_split(split)
            self._write_split(split, levels, layout)
            split_summaries[split] = {
                "total_levels": len(levels),
                "per_bucket": {
                    bucket: sum(len(ids) for ids in per_tier.values())
                    for bucket, per_tier in layout.items()
                },
            }

        self._write_manifest(split_summaries)

        print("\n=== Generation complete ===")
        for split, summary in split_summaries.items():
            print(f"  {split}: {summary['total_levels']} levels")
        if self._skipped:
            print(
                f"  Skipped {len(self._skipped)} levels after {self.max_retry} retries"
            )
        print(f"  Duplicates retried: {self._duplicates}")
        return 0


def _parse_buckets(arg: str) -> List[str]:
    raw = [b.strip() for b in arg.split(",") if b.strip()]
    bad = [b for b in raw if b not in BUCKET_CATALOG]
    if bad:
        raise argparse.ArgumentTypeError(
            f"Unknown bucket(s): {bad}. Available: {list(BUCKET_CATALOG.keys())}"
        )
    return raw


def _parse_tiers(arg: str) -> List[int]:
    raw = [int(t.strip()) for t in arg.split(",") if t.strip()]
    bad = [t for t in raw if t not in BUCKET_TIERS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"Unknown tier(s): {bad}. Valid tiers: {BUCKET_TIERS}"
        )
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the mechanic-primary benchmark dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--mode", type=str, choices=["train", "test", "both"], default="both",
        help="Which split(s) to emit (default: both)",
    )
    parser.add_argument(
        "--per_bucket_per_tier_train", type=int,
        default=DEFAULT_PER_BUCKET_PER_TIER_TRAIN,
        help=f"Train levels per bucket-tier (default: {DEFAULT_PER_BUCKET_PER_TIER_TRAIN})",
    )
    parser.add_argument(
        "--per_bucket_per_tier_test", type=int,
        default=DEFAULT_PER_BUCKET_PER_TIER_TEST,
        help=f"Test levels per bucket-tier (default: {DEFAULT_PER_BUCKET_PER_TIER_TEST})",
    )
    parser.add_argument(
        "--buckets", type=_parse_buckets, default=PHASE_1_BUCKETS,
        help="Comma-separated bucket names to generate (default: Phase-1 four)",
    )
    parser.add_argument(
        "--tiers", type=_parse_tiers, default=BUCKET_TIERS,
        help="Comma-separated tier numbers (default: 1,2,3,4)",
    )
    parser.add_argument(
        "--seed_base_train", type=int, default=DEFAULT_SEED_BASE_TRAIN,
        help=f"Training seed base (default: {DEFAULT_SEED_BASE_TRAIN})",
    )
    parser.add_argument(
        "--seed_base_test", type=int, default=DEFAULT_SEED_BASE_TEST,
        help=f"Test seed base (default: {DEFAULT_SEED_BASE_TEST})",
    )
    parser.add_argument(
        "--max_retry", type=int, default=MAX_RETRY,
        help=f"Max dedup retries per level (default: {MAX_RETRY})",
    )
    parser.add_argument(
        "--check_solvable", action="store_true",
        help="Reject maps whose switch/exit is unreachable from spawn via the "
        "grid reachability graph (slower; reseeds within --max_retry). The "
        "procedural generators are solvable-by-construction, so this is a "
        "safety net -- recommended when adding/tuning generators.",
    )
    args = parser.parse_args()

    generator = BenchmarkGenerator(
        output_dir=Path(args.output_dir),
        buckets=list(args.buckets),
        tiers=list(args.tiers),
        per_bucket_per_tier_train=args.per_bucket_per_tier_train,
        per_bucket_per_tier_test=args.per_bucket_per_tier_test,
        seed_base_train=args.seed_base_train,
        seed_base_test=args.seed_base_test,
        max_retry=args.max_retry,
        check_solvable=args.check_solvable,
    )
    return generator.run(args.mode)


if __name__ == "__main__":
    sys.exit(main())
