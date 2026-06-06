"""Unit coverage for levelgen labeling + eval metric helpers (no corpus needed)."""

from __future__ import annotations

import numpy as np

from nclone.levelgen.labeling import (
    assign_band,
    difficulty_edges,
    entity_counts,
    mechanic_tags,
    subset_ok,
)
from nclone.levelgen.tokenizer import Entity, LevelTokenizer, NUM_TILES


def _make_map(tiles, entities, spawn=(86, 42, 0)):
    header = bytes(184)
    meta = bytearray(85)
    meta[81], meta[82], meta[83] = spawn[0], spawn[1], spawn[2]
    tail = max(100, 5 * len(entities))
    return LevelTokenizer._assemble(header, tiles, bytes(meta), entities, tail)


def test_subset_ok_and_counts():
    tiles = [0] * NUM_TILES
    ents = [Entity(3, 90, 48, 0, 0), Entity(4, 30, 60, 0, 0), Entity(1, 69, 21, 0, 0)]
    md = _make_map(tiles, ents)
    assert subset_ok(md)
    assert entity_counts(md) == {3: 1, 4: 1, 1: 1}
    # add an unsupported entity -> not subset
    md2 = _make_map(tiles, ents + [Entity(14, 50, 50, 0, 0)])
    assert not subset_ok(md2)
    # two doors -> not subset
    md3 = _make_map(tiles, ents + [Entity(3, 50, 50, 0, 0)])
    assert not subset_ok(md3)


def test_mechanic_tags_detect_mine_and_curve():
    tiles = [0] * NUM_TILES
    # put a curve tile in the interior
    tiles[5 * 42 + 10] = 12  # curve
    ents = [Entity(3, 90, 48, 0, 0), Entity(4, 30, 60, 0, 0)] + [
        Entity(1, 10 + i, 20, 0, 0) for i in range(10)
    ]
    md = _make_map(tiles, ents)
    tags = mechanic_tags(md)
    assert "mine" in tags
    assert "curve" in tags
    assert "precision" in tags  # >=8 mines


def test_difficulty_banding_monotonic():
    edges = difficulty_edges(list(range(1, 1001)), n_bands=5)
    assert len(edges) == 4
    assert assign_band(1, edges) == "d0"
    assert assign_band(1000, edges) == "d4"
    assert assign_band(0, edges) == "unknown"
    # monotonic non-decreasing band index
    bands = [int(assign_band(v, edges)[1:]) for v in range(1, 1001)]
    assert bands == sorted(bands)


def test_eval_kl_zero_on_identical():
    from scripts.levelgen.eval_generator import unigram_dist, patch2x2_dist, _kl

    tiles = [np.random.randint(0, 5, size=966).astype(np.uint8) for _ in range(20)]
    assert _kl(unigram_dist(tiles), unigram_dist(tiles)) < 1e-6
    assert _kl(patch2x2_dist(tiles), patch2x2_dist(tiles)) < 1e-6
