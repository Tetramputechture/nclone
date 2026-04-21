"""End-to-end + validity tests for the mechanic-primary benchmark pipeline.

Covers:

1. ``validate_bucket_catalog()`` -- every non-empty bucket/tier entry in
   ``BUCKET_CATALOG`` points to a real ``(generator, preset)`` pair
   defined in ``GENERATOR_PRESETS`` and has a positive weight.
2. Script smoke test -- invoking ``BenchmarkGenerator.run`` with a tiny
   per-bucket-per-tier count writes the expected on-disk layout
   (manifest.json, train/test metadata, per-level pickles) and the
   pickles round-trip into dicts that carry the keys downstream
   consumers (``npp_rl.benchmarks.mechanic_buckets.procedural_levels``,
   ``nclone.evaluation.test_suite_loader.TestSuiteLoader``-compatible
   loaders) rely on.
3. Layout invariants -- per-split metadata ``level_ids`` are unique and
   match the union of pickle filenames, and manifest split summaries
   mirror the metadata totals.

These tests deliberately avoid exercising the N++ simulator: they
validate data-pipeline correctness only, which keeps the whole suite
fast (< 1s) and non-flaky on CI.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from nclone.map_generation.generator_configs import (
    BUCKET_CATALOG,
    PHASE_1_BUCKETS,
    validate_bucket_catalog,
)
from scripts.generate_benchmark_dataset import (
    BenchmarkGenerator,
    _compute_level_id,
    _compute_seed,
)


EXPECTED_LEVEL_KEYS = {
    "level_id",
    "seed",
    "bucket",
    "tier",
    "generator",
    "preset",
    "map_data",
    "metadata",
}
MAP_DATA_MIN_LEN = 1235  # header + tile grid + entity section start


def test_bucket_catalog_is_self_consistent():
    """Every non-empty ``BUCKET_CATALOG`` entry must resolve cleanly."""
    errors = validate_bucket_catalog()
    assert not errors, "BUCKET_CATALOG has invalid entries:\n  - " + "\n  - ".join(
        errors
    )


def test_all_phase1_buckets_are_catalog_keys():
    """``PHASE_1_BUCKETS`` must be a subset of ``BUCKET_CATALOG``."""
    missing = [b for b in PHASE_1_BUCKETS if b not in BUCKET_CATALOG]
    assert not missing, f"PHASE_1_BUCKETS not in catalog: {missing}"

    for bucket in PHASE_1_BUCKETS:
        tiers = BUCKET_CATALOG[bucket]
        assert tiers, f"Phase-1 bucket {bucket!r} has no tier entries"
        for tier, entries in tiers.items():
            assert entries, f"{bucket}/T{tier} has no catalog entries"


def test_level_id_stable_and_split_disambiguated():
    """``_compute_level_id`` is deterministic and varies across splits."""
    a1 = _compute_level_id("BASIC_JUMP", 1, 0, "train")
    a2 = _compute_level_id("BASIC_JUMP", 1, 0, "train")
    assert a1 == a2, "level_id not deterministic"

    b = _compute_level_id("BASIC_JUMP", 1, 0, "test")
    assert a1 != b, "train and test share the same level_id"


def test_seed_layout_is_unique_within_split():
    """Seed offsets across (bucket, tier, index) within one split are unique."""
    seen = set()
    for bucket in PHASE_1_BUCKETS:
        for tier in (1, 2, 3, 4):
            for index in range(3):
                seed = _compute_seed(bucket, tier, index, seed_base=1_000_000)
                assert seed not in seen, (
                    f"duplicate seed {seed} for {bucket}/T{tier}#{index}"
                )
                seen.add(seed)


@pytest.fixture(scope="module")
def smoke_dataset(tmp_path_factory):
    """Generate a tiny dataset once per test session for smoke-test reuse."""
    out_dir = tmp_path_factory.mktemp("benchmark_smoke")
    generator = BenchmarkGenerator(
        output_dir=Path(out_dir),
        buckets=list(PHASE_1_BUCKETS),
        tiers=[1, 2],
        per_bucket_per_tier_train=2,
        per_bucket_per_tier_test=1,
        seed_base_train=1_000_000,
        seed_base_test=100_000,
    )
    rc = generator.run("both")
    assert rc == 0, f"BenchmarkGenerator.run returned non-zero: {rc}"
    return Path(out_dir)


def test_smoke_writes_manifest_and_split_metadata(smoke_dataset):
    manifest_file = smoke_dataset / "manifest.json"
    assert manifest_file.exists(), "manifest.json missing"
    manifest = json.loads(manifest_file.read_text())

    assert manifest["schema"] == "mechanic_primary_v1"
    assert manifest["buckets"] == list(PHASE_1_BUCKETS)
    assert manifest["tiers"] == [1, 2]
    assert set(manifest["splits"].keys()) == {"train", "test"}
    assert manifest["counts_per_bucket_per_tier"] == {"train": 2, "test": 1}

    for split in ("train", "test"):
        meta_file = smoke_dataset / split / f"{split}_metadata.json"
        assert meta_file.exists(), f"{split}_metadata.json missing"
        meta = json.loads(meta_file.read_text())
        assert meta["split"] == split
        assert meta["total_levels"] == manifest["splits"][split]["total_levels"]
        assert meta["total_levels"] > 0, f"{split} produced zero levels"


def test_smoke_level_pickles_round_trip(smoke_dataset):
    manifest = json.loads((smoke_dataset / "manifest.json").read_text())

    for split in ("train", "test"):
        meta = json.loads(
            (smoke_dataset / split / f"{split}_metadata.json").read_text()
        )
        pickle_paths = sorted((smoke_dataset / split).rglob("*.pkl"))
        assert len(pickle_paths) == meta["total_levels"], (
            f"{split}: pickle count {len(pickle_paths)} != metadata total "
            f"{meta['total_levels']}"
        )

        level_ids_on_disk = []
        for p in pickle_paths:
            with open(p, "rb") as f:
                level = pickle.load(f)

            missing = EXPECTED_LEVEL_KEYS - set(level.keys())
            assert not missing, f"{p}: missing keys {missing}"

            assert isinstance(level["map_data"], list)
            assert len(level["map_data"]) >= MAP_DATA_MIN_LEN
            assert level["bucket"] in PHASE_1_BUCKETS
            assert level["tier"] in (1, 2)
            assert p.stem == str(level["level_id"]), (
                f"pickle filename {p.name} != level_id {level['level_id']}"
            )
            assert level["metadata"]["split"] == split
            level_ids_on_disk.append(level["level_id"])

        assert sorted(level_ids_on_disk) == sorted(meta["level_ids"]), (
            f"{split}: on-disk level_ids differ from metadata level_ids"
        )
        assert len(set(level_ids_on_disk)) == len(level_ids_on_disk), (
            f"{split}: duplicate level_ids on disk"
        )

    assert manifest["generation_info"]["unique_hashes"] > 0


def test_smoke_layout_is_by_bucket_then_tier(smoke_dataset):
    """Each level must live at ``<split>/<bucket>/T<tier>/<level_id>.pkl``."""
    for split in ("train", "test"):
        for p in (smoke_dataset / split).rglob("*.pkl"):
            parts = p.relative_to(smoke_dataset).parts
            assert len(parts) == 4, f"unexpected layout: {parts}"
            split_dir, bucket, tier_dir, fname = parts
            assert split_dir == split
            assert bucket in PHASE_1_BUCKETS
            assert tier_dir.startswith("T") and tier_dir[1:].isdigit()
            assert fname.endswith(".pkl")


def test_mechanic_benchmark_loader_reads_smoke_dataset(smoke_dataset):
    """``MechanicBenchmarkLoader`` matches on-disk layout (used by test_environment)."""
    from nclone.evaluation.mechanic_benchmark_loader import MechanicBenchmarkLoader

    loader = MechanicBenchmarkLoader(smoke_dataset, splits=("train", "test"))
    assert len(loader) > 0
    payload = loader.load_payload(0)
    assert "map_data" in payload and len(payload["map_data"]) >= MAP_DATA_MIN_LEN
    lines = MechanicBenchmarkLoader.overlay_lines(payload, 0, len(loader))
    assert any("[benchmark]" in line for line in lines)
    assert payload.get("bucket") in PHASE_1_BUCKETS
