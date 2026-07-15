"""Packed pool store tests: round-trip, parallel-writer union, dedup, loader adapter."""

from __future__ import annotations

import pytest

from nclone.levelgen.pool_store import (
    PoolWriter,
    PackedPool,
    PackedBenchmarkLoader,
    is_packed_pool,
    merge_pool,
    content_id,
    content_hash,
    tile_simhash,
    hamming,
)


def _fake_map_data(tile_fill_cols=0, n_entities=2, tag=0):
    """A minimal valid (bytes 0..255) map_data with a switch+door (subset-pure-ish)."""
    md = [0] * 1335
    # tiles 184..1149: a border + optional interior columns to vary content
    W, H = 42, 23
    tiles = [0] * (W * H)
    for x in range(W):
        tiles[x] = 1
        tiles[x + (H - 1) * W] = 1
    for y in range(H):
        tiles[y * W] = 1
        tiles[W - 1 + y * W] = 1
    for c in range(tile_fill_cols):
        gx = 4 + c
        for gy in range(3, 18):
            tiles[gx + gy * W] = 1
    md[184:1150] = tiles
    md[1156] = 1  # exit door count
    md[1231], md[1232], md[1233] = 10, 10, 1  # spawn + orient (valid bytes)
    # entities at 1235: exit door (3) + switch (4) + tag mines
    ents = [3, 30, 30, 0, 0, 4, 20, 20, 0, 0]
    for k in range(tag):
        ents += [1, 40 + k, 40, 0, 0]
    md[1235 : 1235 + len(ents)] = ents
    return md


def test_roundtrip_and_id(tmp_path):
    w = PoolWriter(tmp_path, writer_id="0")
    md = _fake_map_data(tile_fill_cols=3, tag=2)
    lid = w.add(md, source="generated", bucket="X", tier=1, difficulty_band="d1")
    w.close()
    assert lid == content_id(md)
    assert is_packed_pool(tmp_path)
    pool = PackedPool(tmp_path)
    assert len(pool) == 1
    assert pool.map_data(lid) == md
    rec = pool.record(lid)
    assert rec["n_mines"] == 2 and rec["band"] == "d1" and rec["source"] == "generated"


def test_dedup_within_writer(tmp_path):
    w = PoolWriter(tmp_path, writer_id="0")
    md = _fake_map_data(tile_fill_cols=2)
    assert w.add(md) is not None
    assert w.add(md) is None  # exact duplicate rejected
    w.close()
    assert len(PackedPool(tmp_path)) == 1
    assert w.n_dup == 1


def test_parallel_writers_union_and_cross_dedup(tmp_path):
    shared = _fake_map_data(tile_fill_cols=5)
    for wid in ("0", "1"):
        w = PoolWriter(tmp_path, writer_id=wid)
        w.add(shared)  # both write the SAME level
        w.add(_fake_map_data(tile_fill_cols=int(wid) + 6))  # plus a unique one each
        w.close()
    pool = PackedPool(tmp_path)
    # shared deduped across writers -> 1 + 2 unique = 3
    assert len(pool) == 3


def test_near_dup_filter(tmp_path):
    w = PoolWriter(tmp_path, writer_id="0")
    a = _fake_map_data(tile_fill_cols=5)
    b = _fake_map_data(tile_fill_cols=5, tag=1)  # same tiles, one extra entity
    w.add(a)
    w.add(b)
    w.close()
    assert len(PackedPool(tmp_path)) == 2  # exact dedup keeps both
    # identical tile geometry -> identical simhash -> near-dup collapses to 1
    assert len(PackedPool(tmp_path, near_dup_bits=0)) == 1


def test_loader_adapter_contract(tmp_path):
    w = PoolWriter(tmp_path, writer_id="0")
    w.add(_fake_map_data(tile_fill_cols=3), bucket="MUTANT", tier=2, source="mutant", parent=123)
    w.close()
    ld = PackedBenchmarkLoader(tmp_path)
    assert len(ld) == 1
    p = ld.load_payload(0)
    assert set(["level_id", "map_data", "bucket", "tier", "metadata"]).issubset(p.keys())
    assert p["bucket"] == "MUTANT"
    assert p["metadata"]["parent"] == 123
    assert ld.bucket_name_for_index(0) == "MUTANT"


def test_merge_compacts_segments(tmp_path):
    for wid in ("0", "1", "2"):
        w = PoolWriter(tmp_path, writer_id=wid)
        w.add(_fake_map_data(tile_fill_cols=int(wid) + 2))
        w.close()
    n_before = len(PackedPool(tmp_path))
    res = merge_pool(tmp_path)
    assert res["kept"] == n_before
    assert list(tmp_path.glob("index_*.jsonl")) == [tmp_path / "index_0.jsonl"]
    assert len(PackedPool(tmp_path)) == n_before


def test_rejects_out_of_range_bytes(tmp_path):
    w = PoolWriter(tmp_path, writer_id="0")
    md = _fake_map_data()
    md[1233] = -1  # stray ninja orientation from an ungated map
    with pytest.raises(ValueError):
        w.add(md)
    w.close()


def test_simhash_hamming_self_zero():
    md = _fake_map_data(tile_fill_cols=4)
    sig = tile_simhash(md[184:1150])
    assert hamming(sig, sig) == 0
    assert content_hash(md) == content_hash(md)
