"""Phase-2 tests: composable templates round-trip through the dataset pipeline,
and legacy records stay byte-identical (label fields are additive).
"""

import scripts.generate_benchmark_dataset as gbd


def _make_gen():
    return gbd.BenchmarkGenerator(
        output_dir="/tmp/_compose_test_unused",
        buckets=["FLAT_RUN"],
        tiers=[1],
        per_bucket_per_tier_train=1,
        per_bucket_per_tier_test=1,
        seed_base_train=1_000_000,
        seed_base_test=100_000,
    )


def test_legacy_record_has_empty_secondary_and_no_label_metadata():
    gen = _make_gen()
    rec = gen._generate_one("CURVES", 1, 0, "train", 1_000_000)
    assert rec is not None
    assert rec["secondary"] == []
    assert "mechanics" not in rec["metadata"]
    assert "declared_difficulty" not in rec["metadata"]
    assert "components" not in rec["metadata"]
    # legacy metadata keys are exactly the original set
    assert set(rec["metadata"]) == {
        "description", "split", "source", "map_hash", "index"
    }


def test_template_record_is_enriched(monkeypatch):
    entry = [("template_mixed_traverse_2x3", "default", 1)]
    monkeypatch.setattr(gbd, "get_bucket_entries", lambda b, t: entry)
    gen = _make_gen()
    rec = gen._generate_one("COMPOSITE", 2, 0, "train", 1_000_000)
    assert rec is not None
    assert rec["generator"] == "template_mixed_traverse_2x3"
    assert rec["secondary"], "expected non-empty fine-tag secondary list"
    assert "mechanics" in rec["metadata"] and rec["metadata"]["mechanics"]
    assert "declared_difficulty" in rec["metadata"]
    assert "components" in rec["metadata"]


def test_template_record_map_is_valid_and_solvable():
    from nclone.map_generation.generator_factory import GeneratorFactory
    from nclone.map_generation.solvability import is_map_solvable

    g = GeneratorFactory.create_from_preset("template_flat_intro_2x3", "default", seed=5)
    g.generate(seed=5)
    md = g.map_data()
    assert md[1156] == 1  # exactly one exit door
    assert is_map_solvable(md)


def test_bucket_catalog_still_validates():
    # validate_bucket_catalog returns a list of problems; the shipped catalog
    # must stay clean after registering the composable template generators.
    problems = gbd.validate_bucket_catalog()
    assert problems == [], f"catalog problems: {problems}"
