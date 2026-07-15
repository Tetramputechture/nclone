"""Tests for the invalid-value diagnostic capture (gating / dedupe / cap)."""
from __future__ import annotations

import json
import os

import nclone.gym_environment.invalid_value_capture as cap


def _reset_state():
    cap._seen.clear()
    cap._count = 0


def test_disabled_when_no_dump_dir(tmp_path, monkeypatch):
    _reset_state()
    monkeypatch.delenv("NCLONE_INVALID_DUMP_DIR", raising=False)
    monkeypatch.delenv("NCLONE_GRAPH_CACHE_DIR", raising=False)
    cap.capture_invalid_value("oob", [0] * 1300, {"x": 1})
    # nothing written anywhere (no dir resolved)
    assert cap._count == 0


def test_writes_json_and_dedupes(tmp_path, monkeypatch):
    _reset_state()
    monkeypatch.setenv("NCLONE_INVALID_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("NCLONE_INVALID_DUMP_RENDER", "0")
    md = list(range(1300))
    cap.capture_invalid_value("oob_graph_distance", md, {"ninja_pos": [1, 2]})
    cap.capture_invalid_value("oob_graph_distance", md, {"ninja_pos": [1, 2]})  # dup
    jsons = list(tmp_path.glob("*.json"))
    assert len(jsons) == 1, "duplicate (reason, map) must not write a second dump"
    payload = json.loads(jsons[0].read_text())
    assert payload["reason"] == "oob_graph_distance"
    assert payload["map_data"] == md
    assert payload["context"]["ninja_pos"] == [1, 2]


def test_distinct_levels_each_dump(tmp_path, monkeypatch):
    _reset_state()
    monkeypatch.setenv("NCLONE_INVALID_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("NCLONE_INVALID_DUMP_RENDER", "0")
    cap.capture_invalid_value("oob", list(range(1300)), {})
    cap.capture_invalid_value("oob", list(range(1, 1301)), {})  # different hash
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_cap_enforced(tmp_path, monkeypatch):
    _reset_state()
    monkeypatch.setenv("NCLONE_INVALID_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("NCLONE_INVALID_DUMP_RENDER", "0")
    monkeypatch.setenv("NCLONE_INVALID_DUMP_CAP", "3")
    for i in range(10):
        cap.capture_invalid_value("oob", list(range(i, i + 1300)), {})
    assert len(list(tmp_path.glob("*.json"))) == 3


def test_never_raises_on_bad_input(tmp_path, monkeypatch):
    _reset_state()
    monkeypatch.setenv("NCLONE_INVALID_DUMP_DIR", str(tmp_path))
    # map_data None -> no-op, no raise
    cap.capture_invalid_value("oob", None, {"x": 1})
    # non-serializable context coerced via _jsonable, must not raise
    cap.capture_invalid_value("oob", [0] * 1300, {"obj": object()})
