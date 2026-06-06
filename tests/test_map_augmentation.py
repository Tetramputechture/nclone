"""Tests for physics-preserving map augmentation (mirror) and the dataset tool."""

from __future__ import annotations

import glob
import pickle
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from nclone.map_augmentation import (
    MIRROR_AXIS_UNITS,
    iter_augmented,
    mirror_map_data,
    mirror_map_horizontally,
)
from nclone.map_generation.map import Map

REPO = Path(__file__).resolve().parents[1]
OFFICIAL_S = REPO / "nclone" / "maps" / "official" / "S"
# Prefer the current int-coordinate dataset; the *_smoke pkls predate the
# int-coordinate fix and carry float coords (on which 176 - x drifts 1 ULP).
PROCGEN = REPO / "datasets" / "mechanic_benchmark"


def _int_procgen_maps(limit: int = 12):
    out = []
    for p in sorted(glob.glob(str(PROCGEN / "**" / "*.pkl"), recursive=True)):
        md = pickle.load(open(p, "rb"))["map_data"]
        if all(float(v).is_integer() for v in md):
            out.append(md)
        if len(out) >= limit:
            break
    return out


def _official_maps(limit: int = 12):
    paths = sorted(glob.glob(str(OFFICIAL_S / "*")))[:limit]
    out = []
    for p in paths:
        with open(p, "rb") as f:
            out.append((p, [int(b) for b in f.read()]))
    return out


def _entity_types(md):
    return Counter(md[i] for i in range(1235, len(md) - 4, 5))


def test_axis_constant():
    # 44 tiles (42 interior + 2 border) * 4 units/tile.
    assert MIRROR_AXIS_UNITS == 176


@pytest.mark.parametrize("path,md", _official_maps())
def test_double_mirror_is_identity_official(path, md):
    assert mirror_map_data(mirror_map_data(md)) == md


def test_double_mirror_identity_procgen():
    maps = _int_procgen_maps()
    if not maps:
        pytest.skip("no int-coordinate procgen dataset present")
    for md in maps:
        assert mirror_map_data(mirror_map_data(md)) == md


@pytest.mark.parametrize("path,md", _official_maps())
def test_mirror_preserves_structure(path, md):
    m = mirror_map_data(md)
    # Same length, same exit-door header count, same entity-type multiset.
    assert len(m) == len(md)
    assert m[1156] == md[1156]
    assert _entity_types(m) == _entity_types(md)
    # x coordinates are reflected about the axis.
    assert m[1231] == MIRROR_AXIS_UNITS - md[1231]


@pytest.mark.parametrize("path,md", _official_maps())
def test_mirror_preserves_solvability(path, md):
    from nclone.map_generation.solvability import is_map_solvable

    assert is_map_solvable(md) == is_map_solvable(mirror_map_data(md))


def test_iter_augmented_dedup_symmetric():
    # Construct a genuinely left-right symmetric map: empty tiles, ninja on the
    # mirror axis with no facing, no entities -> mirror == original.
    sym = Map().map_data()
    sym[1231] = MIRROR_AXIS_UNITS // 2  # ninja x on the axis
    sym[1233] = 0  # no facing orientation to flip
    sym = sym[:1235]  # drop any entity records
    assert mirror_map_data(sym) == sym
    assert list(iter_augmented(sym, ("mirror_h",), dedup=True)) == []
    # Without dedup it still yields the (identical) variant.
    assert len(list(iter_augmented(sym, ("mirror_h",), dedup=False))) == 1


def test_mirror_map_horizontally_wrapper_roundtrip():
    maps = _int_procgen_maps(1)
    if not maps:
        pytest.skip("no int-coordinate procgen dataset present")
    md = maps[0]
    # The Map wrapper canonicalises (Map carries no file header), so compare
    # against the canonical round-trip rather than the raw bytes.
    canon = Map.from_map_data(md).map_data()
    m = Map.from_map_data(md)
    once = mirror_map_horizontally(m)
    twice = mirror_map_horizontally(once)
    assert twice.map_data() == canon  # involutive on the canonical form
    assert once.map_data() == mirror_map_data(canon)  # matches the primitive


def test_augment_dataset_script_procgen(tmp_path):
    if not list(PROCGEN.glob("**/*.pkl")):
        pytest.skip("no smoke procgen dataset present")
    out = tmp_path / "aug"
    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.augment_dataset",
            "--procgen", str(PROCGEN),
            "--output_dir", str(out),
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    variants = list(out.glob("**/*.pkl"))
    assert variants, "expected augmented pkls to be written"
    rec = pickle.load(open(variants[0], "rb"))
    assert rec["metadata"]["source"] == "augmented"
    assert rec["metadata"]["augmentation"] == "mirror_h"
    assert "parent_level_id" in rec["metadata"]
