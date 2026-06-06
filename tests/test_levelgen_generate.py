"""Generator-stage tests: benchmark write/load round-trip + solvability gate.

The gate + loader tests need the native sim and (for the gate) a real solvable map,
so they are guarded. The write/load test is pure and always runs.
"""

from __future__ import annotations

import os

import pytest

from nclone.levelgen.tokenizer import Entity, assemble_map_data

ATTRACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "attracts")


def test_benchmark_write_load_roundtrip(tmp_path):
    """A generated level round-trips through write_benchmark -> MechanicBenchmarkLoader."""
    from scripts.levelgen.generate_levels import write_benchmark
    from scripts.levelgen.common import map_hash_int

    header = bytes(184)
    tiles = [0] * 966
    ents = [Entity(4, 30, 60, 0, 0), Entity(3, 90, 48, 0, 0), Entity(1, 69, 21, 0, 0)]
    md = assemble_map_data(tiles, (86, 42, 0), ents, header)
    lid = map_hash_int(md)
    level = {
        "level_id": lid,
        "seed": 0,
        "bucket": "GENERATED",
        "tier": 2,
        "generator": "transformer",
        "preset": "mine_d2",
        "map_data": md,
        "secondary": [],
        "metadata": {"description": "t", "split": "train", "source": "generated", "map_hash": lid, "index": 0},
    }
    write_benchmark([level], tmp_path, "train", "GENERATED")

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "train" / "train_metadata.json").exists()

    from nclone.evaluation.mechanic_benchmark_loader import MechanicBenchmarkLoader

    loader = MechanicBenchmarkLoader(tmp_path, splits=("train",))
    assert len(loader) == 1
    assert loader.get_level_ids() == [lid]
    assert loader.get_map_data(0) == md
    assert loader.bucket_name_for_index(0) == "GENERATED"


@pytest.mark.skipif(not os.path.isdir(ATTRACTS_DIR), reason="corpus not present")
def test_gate_accepts_solvable_rejects_degenerate():
    import contextlib
    import io

    from nclone.replay.npp_attract_decoder import NppAttractDecoder
    from nclone.levelgen.labeling import subset_ok
    from scripts.levelgen.generate_levels import gate

    dec = NppAttractDecoder()
    solvable_md = None
    for n in sorted(os.listdir(ATTRACTS_DIR))[:200]:
        with contextlib.redirect_stdout(io.StringIO()):
            md = list(dec.create_nclone_map(os.path.join(ATTRACTS_DIR, n)))
        if len(md) >= 1335 and not all(b == 0 for b in md) and subset_ok(md):
            solvable_md = md
            break
    assert solvable_md is not None
    d = gate(solvable_md)
    assert d is not None and d > 0  # solvable -> finite d*

    # degenerate all-zero map -> unsolvable
    assert gate([0] * 1335) is None
