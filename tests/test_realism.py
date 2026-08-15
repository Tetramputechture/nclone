"""Guards on the human-manifold realism model.

The model's job is to catch procedural levels drifting away from the maps humans
build. These tests pin the two directions that matter: real maps score as real,
and the specific degeneracies our generators exhibit score as unreal.
"""

from __future__ import annotations

import glob
import pickle

import pytest

np = pytest.importorskip("numpy")

from nclone.levelgen.realism import (  # noqa: E402
    CALIBRATION_PATH,
    FEATURE_KEYS,
    level_features,
    load_default,
)

GRID_W, GRID_H = 42, 23

pytestmark = pytest.mark.skipif(
    not CALIBRATION_PATH.exists(),
    reason="realism calibration not generated (scripts/levelgen/calibrate_realism.py)",
)


def _map_data(grid, entities=()):
    md = [0] * 184
    for row in grid:
        md.extend(row)
    md.extend([0] * (1235 - len(md)))
    for t, x, y in entities:
        md.extend([t, x, y, 0, 0])
    md.extend([0] * 5)
    return md


def _uniform(v):
    return [[v] * GRID_W for _ in range(GRID_H)]


class TestFeatures:
    def test_feature_vector_shape_and_order(self):
        f = level_features(_map_data(_uniform(0)))
        assert len(f) == len(FEATURE_KEYS)

    def test_solid_frac_endpoints(self):
        assert level_features(_map_data(_uniform(0)))[0] == 0.0
        assert level_features(_map_data(_uniform(1)))[0] == 1.0

    def test_uniform_grid_is_maximally_repetitive(self):
        f = level_features(_map_data(_uniform(1)))
        assert f[FEATURE_KEYS.index("row_repeat")] == 1.0
        assert f[FEATURE_KEYS.index("col_repeat")] == 1.0

    def test_partial_entropy_ignores_trivial_tiles(self):
        """Empty and full tiles are not palette choices, so they must not count."""
        f = level_features(_map_data(_uniform(0)))
        assert f[FEATURE_KEYS.index("partial_entropy")] == 0.0

    def test_uniform_tile_spray_saturates_entropy(self):
        """The signature failure: every slope id used equally.

        Humans pick a few ids and repeat them to build a continuous surface;
        random spray approaches log2(36) = 5.17 bits.
        """
        rng = np.random.default_rng(0)
        grid = [[int(rng.integers(2, 38)) for _ in range(GRID_W)] for _ in range(GRID_H)]
        f = level_features(_map_data(grid))
        assert f[FEATURE_KEYS.index("partial_entropy")] > 4.5

    def test_entity_counts_stop_at_terminator(self):
        f = level_features(_map_data(_uniform(0), entities=[(1, 10, 10), (2, 12, 12)]))
        assert f[FEATURE_KEYS.index("log_entities")] == pytest.approx(np.log1p(2))
        assert f[FEATURE_KEYS.index("log_gold")] == pytest.approx(np.log1p(1))


class TestScoring:
    def test_score_is_deterministic(self):
        md = _map_data(_uniform(0), entities=[(3, 10, 10), (4, 20, 20)])
        m = load_default()
        assert m.score(md) == m.score(md)

    def test_degenerate_levels_are_rejected(self):
        """An empty room and a solid block are both off-manifold."""
        m = load_default()
        for grid in (_uniform(0), _uniform(1)):
            md = _map_data(grid, entities=[(3, 10, 10), (4, 20, 20)])
            assert not m.is_realistic(md)

    def test_explain_names_the_offending_features(self):
        m = load_default()
        md = _map_data(_uniform(1), entities=[(3, 10, 10), (4, 20, 20)])
        rep = m.explain(md)
        assert rep["realistic"] is False
        assert set(rep["deviation"]) == set(FEATURE_KEYS)
        worst = max(rep["deviation"], key=lambda k: abs(rep["deviation"][k]))
        # A fully-solid, entity-free map is unhuman on geometry and population;
        # either is a legitimate headline, but it must not point at something else.
        assert worst in {"solid_frac", "row_repeat", "col_repeat",
                         "log_entities", "log_gold"}


class TestAgainstRealCorpora:
    """The model is only useful if genuine maps clear it."""

    def _official(self, limit=120):
        shards = sorted(glob.glob("datasets/levelgen/corpus_official/shard_*.pkl"))
        if not shards:
            pytest.skip("official corpus not present")
        out = []
        for s in shards:
            with open(s, "rb") as fh:
                for rec in pickle.load(fh)["records"]:
                    out.append(rec["map_data"])
                    if len(out) >= limit:
                        return out
        return out

    def test_official_campaign_maps_mostly_pass(self):
        m = load_default()
        mds = self._official()
        rate = sum(m.is_realistic(md) for md in mds) / len(mds)
        # Threshold is the held-out human p99, so ~99% is the expectation; allow
        # slack for the campaign being a small, stylistically distinct slice.
        assert rate > 0.90, f"only {rate:.1%} of official maps scored as realistic"

    def test_official_maps_score_better_than_a_solid_block(self):
        m = load_default()
        block = _map_data(_uniform(1), entities=[(3, 10, 10), (4, 20, 20)])
        worst_real = max(m.score(md) for md in self._official(60))
        assert worst_real < m.score(block)
