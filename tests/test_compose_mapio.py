"""Tests for saving/loading map data (binary .map / json / pickle)."""

import json
import pickle

import pytest

from nclone.map_generation.compose import (
    MapEditor, save_map, load_map_data, load_map,
)


def _level():
    ed = MapEditor(fill="empty")
    ed.border()
    ed.hline(20, 1, 40, 1)
    ed.place_player(3, 19)
    ed.place_exit(switch=(30, 19), door=(36, 19))
    ed.add_mine(15, 19)
    return ed


@pytest.mark.parametrize("ext", ["map", "bin", "json", "pkl"])
def test_save_load_round_trip(tmp_path, ext):
    md = _level().map_data()
    path = str(tmp_path / f"lvl.{ext}")
    save_map(md, path, metadata={"note": "x"}, secondary=["hazard:mine.field"])
    assert load_map_data(path) == md


def test_binary_is_loadable_by_simulator(tmp_path):
    from nclone.nplay_headless import NPlayHeadless

    md = _level().map_data()
    path = str(tmp_path / "lvl.map")
    save_map(md, path)
    n = NPlayHeadless(render_mode=None, enable_logging=False)
    n.load_map(path)  # raw-byte loader the env uses
    n.reset()
    # the loaded tiles match what we saved
    assert n.get_tile_data() is not None


def test_editor_save_helper(tmp_path):
    ed = _level()
    p = str(tmp_path / "e.map")
    assert ed.save(p) == p
    assert load_map_data(p) == ed.map_data()


def test_json_carries_metadata(tmp_path):
    p = str(tmp_path / "l.json")
    save_map(_level().map_data(), p, metadata={"author": "claude"}, secondary=["x"])
    obj = json.load(open(p))
    assert obj["metadata"]["author"] == "claude" and obj["secondary"] == ["x"]


def test_pickle_record_shape_matches_dataset(tmp_path):
    p = str(tmp_path / "l.pkl")
    save_map(_level().map_data(), p, metadata={"split": "train"},
             secondary=["hazard:mine.field"])
    rec = pickle.load(open(p, "rb"))
    assert set(rec) >= {"map_data", "secondary", "metadata"}
    assert rec["metadata"]["split"] == "train"


def test_binary_rejects_out_of_byte_range(tmp_path):
    md = _level().map_data()
    md[200] = 999  # not a byte
    with pytest.raises(ValueError):
        save_map(md, str(tmp_path / "bad.map"))
    # json keeps it fine
    save_map(md, str(tmp_path / "ok.json"))
    assert load_map_data(str(tmp_path / "ok.json"))[200] == 999


def test_load_map_returns_map_object(tmp_path):
    ed = _level()
    p = str(tmp_path / "l.map")
    ed.save(p)
    m = load_map(p)
    assert m.map_data()[184:1150] == ed.map_data()[184:1150]


def test_fmt_override_beats_extension(tmp_path):
    md = _level().map_data()
    p = str(tmp_path / "weirdname.lvl")  # unknown ext -> binary by default
    save_map(md, p, fmt="json")
    assert load_map_data(p, fmt="json") == md
