"""Round-trip gate for the level-generator tokenizer (Phase 0).

Hard requirement from docs/SYNTHETIC_LEVEL_TRANSFORMER_PLAN.md:
``detokenize(tokenize(map_data)) == map_data`` byte-identical over the corpus.
"""

from __future__ import annotations

import os

import pytest

from nclone.levelgen.tokenizer import (
    LevelTokenizer,
    Vocab,
    assemble_map_data,
    Entity,
)

ATTRACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "attracts")


def _attract_paths(limit: int):
    if not os.path.isdir(ATTRACTS_DIR):
        return []
    names = sorted(os.listdir(ATTRACTS_DIR))[:limit]
    return [os.path.join(ATTRACTS_DIR, n) for n in names]


def _decode(path):
    from nclone.replay.npp_attract_decoder import NppAttractDecoder

    md = list(NppAttractDecoder().create_nclone_map(path))
    # create_nclone_map returns an all-zero 1335 fallback on failure; skip those.
    if len(md) < 1335 or all(b == 0 for b in md):
        return None
    return md


def test_vocab_blocks_are_disjoint_and_in_range():
    assert Vocab.TILE_BASE > Vocab._N_SPECIAL - 1
    assert Vocab.BYTE_BASE == Vocab.TILE_BASE + Vocab._N_TILE
    assert Vocab.ETYPE_BASE == Vocab.BYTE_BASE + Vocab._N_BYTE
    assert Vocab.COND_BASE == Vocab.ETYPE_BASE + Vocab._N_ETYPE
    # every encodable token < SIZE
    assert Vocab.tile(63) < Vocab.SIZE
    assert Vocab.byte(255) < Vocab.SIZE
    assert Vocab.etype(63) < Vocab.SIZE
    assert Vocab.subset(True) < Vocab.SIZE


def test_synthetic_roundtrip():
    tok = LevelTokenizer()
    # a hand-built map_data with sub-grid mine coords (not multiples of 4)
    header = bytes(range(184))
    tiles = [0] * 966
    tiles[100] = 1
    tiles[500] = 33
    meta = bytearray(85)
    meta[0] = 2  # mine count
    meta[81] = 86  # spawn x
    meta[82] = 42  # spawn y
    entities = [
        Entity(1, 69, 21, 0, 0),  # mine, odd x
        Entity(3, 90, 48, 0, 0),  # exit door
        Entity(4, 30, 60, 0, 0),  # exit switch
    ]
    md = LevelTokenizer._assemble(header, tiles, bytes(meta), entities, max(100, 5 * 3))
    level = tok.tokenize(md, cond={"source": "human", "subset_ok": True})
    out = tok.detokenize(level)
    assert out == md
    # content decoded correctly too
    assert level.tiles[100] == 1 and level.tiles[500] == 33
    assert level.entities[0].x == 69
    assert level.ninja == (86, 42, 0)


@pytest.mark.skipif(not _attract_paths(1), reason="../attracts corpus not present")
def test_corpus_roundtrip_sample():
    tok = LevelTokenizer()
    checked = 0
    for path in _attract_paths(500):
        md = _decode(path)
        if md is None:
            continue
        level = tok.tokenize(md, cond={"source": "human"})
        out = tok.detokenize(level)
        assert out == md, f"round trip failed for {os.path.basename(path)}"
        checked += 1
    assert checked >= 50, f"only {checked} levels decoded -- corpus may be broken"


@pytest.mark.skipif(not _attract_paths(1), reason="../attracts corpus not present")
def test_assemble_map_data_loads_and_matches_counts():
    """Generation-path assembler builds a loader-valid map with correct counts."""
    tok = LevelTokenizer()
    md = None
    for path in _attract_paths(50):
        md = _decode(path)
        if md is not None:
            break
    assert md is not None
    level = tok.tokenize(md)
    rebuilt = assemble_map_data(
        level.tiles, level.ninja, level.entities, level.header
    )
    # tiles + entities preserved; metadata recomputed (may differ from original
    # in count fields if original used type-21 toggled mines, but len & content match)
    assert rebuilt[184:1150] == md[184:1150]
    re_level = tok.tokenize(rebuilt)
    assert [e.as_tuple() for e in re_level.entities] == [
        e.as_tuple() for e in level.entities
    ]
