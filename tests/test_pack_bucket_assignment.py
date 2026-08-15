"""Guards on how a multi-tagged level gets named with a single bucket.

Human levels carry ~3 mechanic tags each, so the choice of *which* tag names the
level decides the whole mechanic axis of the curriculum. The original rule --
first tag in a fixed canonical order -- silently collapsed that axis once the
corpus widened beyond the 5-entity subset.
"""

from __future__ import annotations

from scripts.levelgen.pack_benchmark import _bucket_for_tags

# Approximate fire rates measured over the human corpus under the native entity
# profile. `mine` is near-universal and therefore says almost nothing about a
# level; `composite` and `curve` are rare and therefore say a lot.
BASE_RATES = {
    "mine": 0.85,
    "wall_jump": 0.56,
    "precision": 0.36,
    "slope": 0.31,
    "jump": 0.29,
    "curve": 0.12,
    "flat": 0.08,
    "composite": 0.078,
}


class TestRarestWins:
    def test_picks_the_least_common_tag(self):
        assert _bucket_for_tags(["mine", "wall_jump", "curve"], BASE_RATES) == "CURVE"

    def test_near_universal_tag_loses_to_anything_else(self):
        """`mine` fires on 85% of levels, so it is the least informative label."""
        assert _bucket_for_tags(["mine", "slope"], BASE_RATES) == "SLOPE"

    def test_mine_still_wins_when_it_is_the_only_tag(self):
        assert _bucket_for_tags(["mine"], BASE_RATES) == "MINE"

    def test_precision_is_reachable(self):
        """Regression: `precision` sits last in the canonical order, so the old
        rule could never assign it despite it firing on ~36% of levels."""
        assert _bucket_for_tags(["mine", "wall_jump", "precision"], BASE_RATES) == "PRECISION"

    def test_composite_is_reachable(self):
        assert _bucket_for_tags(["mine", "jump", "composite"], BASE_RATES) == "COMPOSITE"

    def test_is_deterministic_under_ties(self):
        """Equal rates must not let dict/set ordering pick the bucket."""
        rates = {"jump": 0.5, "slope": 0.5}
        first = _bucket_for_tags(["jump", "slope"], rates)
        assert first == _bucket_for_tags(["slope", "jump"], rates)
        assert first == "JUMP"  # name tie-break

    def test_unknown_tag_is_treated_as_rarest(self):
        """A tag absent from the rates has fired 0 times, so it is maximally
        informative -- not silently dropped."""
        assert _bucket_for_tags(["mine", "brand_new"], BASE_RATES) == "BRAND_NEW"


class TestCanonicalFallback:
    def test_no_rates_reproduces_the_old_order(self):
        assert _bucket_for_tags(["mine", "wall_jump", "curve"]) == "WALL_JUMP"
        assert _bucket_for_tags(["mine", "precision"]) == "MINE"

    def test_empty_tags_defaults_to_flat(self):
        assert _bucket_for_tags([]) == "FLAT"
        assert _bucket_for_tags([], BASE_RATES) == "FLAT"
