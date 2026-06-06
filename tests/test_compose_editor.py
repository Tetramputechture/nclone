"""Tests for the from-scratch MapEditor + check_level validator."""

import pytest

from nclone.map_generation.compose import MapEditor, check_level
from nclone.constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


def _playable():
    """A minimal hand-built, solvable level."""
    ed = MapEditor(fill="empty")
    ed.border()
    ed.hline(20, 1, 40, tile=1)
    ed.place_player(3, 19)
    ed.place_exit(switch=(30, 19), door=(36, 19))
    return ed


# --- single + batch tile ops --------------------------------------------------
def test_single_and_batch_tile_ops_and_bounds():
    ed = MapEditor(fill="empty")
    ed.set_tile(5, 5, 1)
    assert ed.get_tile(5, 5) == 1
    ed.set_tile(-1, 0, 1)  # out of bounds -> skipped, no error
    ed.set_tile(MAP_TILE_WIDTH, 0, 1)
    ed.fill_rect(2, 2, 4, 4, 1)
    assert all(ed.get_tile(x, y) == 1 for x in (2, 3, 4) for y in (2, 3, 4))
    ed.clear_rect(3, 3, 3, 3)
    assert ed.get_tile(3, 3) == 0


def test_shapes_disk_ring_line_arc():
    ed = MapEditor(fill="empty")
    ed.disk(10, 10, 3, 1)
    assert ed.get_tile(10, 10) == 1 and ed.get_tile(10, 13) == 1
    ed.ring(20, 10, 4, 1)
    assert ed.get_tile(20, 10) == 0  # ring is hollow
    assert ed.get_tile(24, 10) == 1
    ed.line(0, 0, 5, 5, 1)  # clamps the (0,0) write inside bounds
    assert ed.get_tile(5, 5) == 1


# --- entities + idempotency ---------------------------------------------------
def test_place_player_and_exit_are_idempotent():
    ed = _playable()
    ed.place_player(4, 19)  # overwrite
    ed.place_exit(switch=(31, 19), door=(37, 19))  # overwrite
    rep = ed.check()
    assert rep["counts"]["player"] == 1
    assert rep["counts"]["exit_door"] == 1 and rep["counts"]["exit_switch"] == 1
    ents = {e["name"]: e["tile"] for e in rep["entities"]}
    assert ents["ninja"] == [4, 19]
    assert ents["exit_door"] == [37, 19] and ents["exit_switch"] == [31, 19]


def test_batch_mines_and_purity():
    ed = _playable()
    ed.add_mines([(10, 19), (15, 19)])
    types = {e["type"] for e in ed.describe()["entities"]}
    assert types <= {0, 1, 21, 3, 4}


# --- ASCII-art pictorial drawing ---------------------------------------------
def test_draw_ascii_paints_tiles_and_entities():
    art = """
    #####
    #P..E
    #..S.
    #####
    """
    legend = {"#": "solid", ".": "empty", "P": "player", "E": "exit_door", "S": "exit_switch"}
    ed = MapEditor(fill="empty")
    ed.draw_ascii("\n".join(l.strip() for l in art.strip("\n").splitlines()),
                  legend, origin=(0, 0))
    rep = ed.check()
    assert rep["counts"]["player"] == 1
    assert rep["counts"]["exit_door"] == 1 and rep["counts"]["exit_switch"] == 1
    ents = {e["name"]: e["tile"] for e in rep["entities"]}
    assert ents["ninja"] == [1, 1]
    assert ents["exit_door"] == [4, 1]
    assert ents["exit_switch"] == [3, 2]


# --- validator (the "test that it's possible" loop) ---------------------------
def test_check_passes_for_playable_level():
    rep = _playable().check()
    assert rep["ok"] is True and rep["solvable"] is True and rep["errors"] == []


def test_check_reports_missing_exit():
    ed = MapEditor(fill="empty")
    ed.border()
    ed.hline(20, 1, 40, 1)
    ed.place_player(3, 19)
    rep = ed.check()
    assert rep["ok"] is False
    assert any("exit door" in e for e in rep["errors"])
    assert any("exit switch" in e for e in rep["errors"])


def test_check_flags_player_buried_in_solid():
    ed = MapEditor(fill="solid")  # everything solid
    ed.place_player(10, 10)
    ed.place_exit(switch=(20, 10), door=(25, 10))
    rep = ed.check()
    assert rep["ok"] is False
    assert any("inside a solid tile" in e for e in rep["errors"])


def test_check_unsolvable_when_goal_walled_off():
    ed = _playable()
    # wall off the right side so the switch/door are unreachable
    ed.vline(25, 1, 21, 1)
    rep = ed.check()
    assert rep["solvable"] is False and rep["ok"] is False


def test_check_warns_mine_on_spawn():
    ed = _playable()
    ed.add_mine(3, 19)  # right on the spawn
    rep = ed.check()
    assert any("adjacent to the spawn" in w for w in rep["warnings"])


# --- sub-tile entity placement ------------------------------------------------
def _mine_px(ed):
    return [p["px"] for p in ed.check()["positions"] if p["name"] == "toggle_mine"]


def test_add_mine_centers_in_tile_by_default():
    # tile 21 spans px 504..528; centre is 516 (corner anchor stays at 504)
    ed = _playable()
    ed.add_mine(21, 9)
    ed.add_mine(21, 9, anchor="corner")
    pxs = _mine_px(ed)
    assert [516, 228] in pxs and [504, 216] in pxs


def test_middle_anchor_constant_is_map_center():
    from nclone.map_generation.compose import MIDDLE
    ed = _playable()
    ed.add_mine(*MIDDLE)  # MIDDLE = (21.0, 11.5) -> +0.5 bias -> tile (21,11) area
    assert _mine_px(ed)  # placed, byte-safe, no exception


def test_fill_mines_makes_a_cluster_not_dots():
    ed = _playable()
    ed.mine_disk(10, 10, 1.0, spacing_px=8)
    assert ed.check()["counts"]["mine"] >= 6  # a dense blob, not 1-2 stray mines


def test_check_warns_on_stacked_mines():
    ed = _playable()
    ed.add_mine(12, 10)
    ed.add_mine(12, 10)  # same spot -> stacked
    assert any("stacked" in w for w in ed.check()["warnings"])


# --- reproducibility ----------------------------------------------------------
def test_editor_reproducible_with_scatter():
    def build():
        ed = _playable()
        ed.scatter_mines(5, 19, 25, 19, n=4)
        return ed.map_data()

    assert build() == build()


def test_smiley_face_is_buildable_and_solvable():
    ed = MapEditor(fill="empty")
    ed.border()
    cx, cy = 21, 9
    ed.ring(cx, cy, 7, 1, thickness=1.5)
    ed.disk(cx - 3, cy - 2, 1, 1)
    ed.disk(cx + 3, cy - 2, 1, 1)
    ed.arc(cx, cy, 4, 25, 155, 1)
    ed.hline(20, 1, 40, 1)
    ed.place_player(3, 19)
    ed.place_exit(switch=(33, 19), door=(38, 19))
    rep = ed.check()
    assert rep["ok"] is True and rep["solvable"] is True
