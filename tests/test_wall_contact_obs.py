"""P1-4 / P2-7 regression: wall-contact + wall_normal obs features are live,
and buffer normalization distinguishes "just armed" from "inactive".

``wall_count`` is never incremented during simulation in either backend (tile
wall contacts fold into ceiling_count), so the game_state wall-contact feature
(index 12) and wall_normal (index 16) were permanently -1 / 0.0. Both are now
gated on the ninja's live ``walled`` flag, whose ``wall_normal`` companion is
recomputed by post_collision whenever walled is true.

Feature indices in the 23-dim base ninja state (both backends):
  7-9  jump/floor/wall buffers, 11 floor contact, 12 wall contact,
  16 wall_normal, 17 walled.
"""

from __future__ import annotations

import pytest

from nclone.map_generation.map import Map
from nclone.native_headless import NativePlayHeadless
from nclone.nplay_headless import NPlayHeadless

WALL_X = 20  # grid column of the wall the ninja runs into


def _wall_box_md() -> list:
    """Flat corridor ending in a wall column at x=WALL_X."""
    m = Map()
    for x in range(2, WALL_X + 1):
        m.set_tile(x, 12, 1)  # floor
    for y in range(6, 13):
        m.set_tile(WALL_X, y, 1)  # wall at the right end
    m.set_ninja_spawn(4, 11)
    m.add_entity(3, 6, 11, 0, 0, switch_x=8, switch_y=11)  # exit + switch (unused)
    return [int(v) for v in m.map_data()]


def _make_headless(backend: str):
    if backend == "native":
        nh = NativePlayHeadless()
    else:
        nh = NPlayHeadless(enable_rendering=False)
    nh.load_map_from_map_data(_wall_box_md())
    return nh


def _run_until_walled(nh, max_frames: int = 400):
    for _ in range(max_frames):
        nh.tick(1, 0)  # hold right, into the wall
        if nh.sim.ninja.walled:
            return True
    return False


@pytest.mark.parametrize("backend", ["python", "native"])
def test_wall_contact_and_normal_live_when_walled(backend):
    nh = _make_headless(backend)
    assert _run_until_walled(nh), f"[{backend}] ninja never became walled"

    state = nh.get_ninja_state()
    ninja = nh.sim.ninja
    assert ninja.walled
    # Wall contact (12) reads 1.0 while walled.
    assert state[12] == pytest.approx(1.0), (
        f"[{backend}] wall-contact feature dead while walled: {state[12]}"
    )
    # Walled flag (17) agrees.
    assert state[17] == pytest.approx(1.0)
    # wall_normal (16) is +-1 with the correct sign: the wall is on the
    # ninja's RIGHT, and wall_normal points FROM the wall TO the ninja => -1.
    assert state[16] == pytest.approx(float(ninja.wall_normal))
    assert state[16] == pytest.approx(-1.0), (
        f"[{backend}] wall_normal sign wrong: {state[16]}"
    )


@pytest.mark.parametrize("backend", ["python", "native"])
def test_wall_slide_exposes_wall_normal(backend):
    """While actually wall-SLIDING (state 5), the features stay live."""
    nh = _make_headless(backend)
    assert _run_until_walled(nh)
    # Jump at the wall, keep holding right: the ninja rises, then falls
    # pressed against the wall -> wall-slide state 5.
    nh.tick(1, 1)
    found_slide = False
    for _ in range(120):
        nh.tick(1, 0)
        if nh.sim.ninja.state == 5:
            found_slide = True
            break
    assert found_slide, f"[{backend}] never entered wall-slide state"

    state = nh.get_ninja_state()
    assert state[12] == pytest.approx(1.0)
    assert state[16] == pytest.approx(-1.0)


@pytest.mark.parametrize("backend", ["python", "native"])
def test_wall_features_zero_when_airborne_away_from_wall(backend):
    nh = _make_headless(backend)
    # Jump straight up from spawn (nowhere near the wall).
    nh.tick(0, 1)
    for _ in range(3):
        nh.tick(0, 1)
    ninja = nh.sim.ninja
    assert ninja.airborn and not ninja.walled
    state = nh.get_ninja_state()
    assert state[12] == pytest.approx(-1.0), (
        f"[{backend}] wall-contact should read -1 airborne away from walls"
    )
    assert state[16] == pytest.approx(0.0), (
        f"[{backend}] wall_normal should read 0 airborne away from walls"
    )


def test_buffer_normalization_distinguishes_just_armed_from_inactive():
    """P2-7: (b+1)/(window+1) mapping — b=0 (ACTIVE, just armed) must not
    alias b=-1 (inactive) at -1.0. Direct unit test on the python backend
    (the native backend uses the identical formula; cross-backend game_state
    parity is covered by test_native_backend_parity)."""
    nh = _make_headless("python")
    ninja = nh.sim.ninja

    ninja.jump_buffer = -1
    assert nh.get_ninja_state()[7] == pytest.approx(-1.0)

    ninja.jump_buffer = 0  # just armed: distinct from inactive
    v0 = nh.get_ninja_state()[7]
    assert v0 == pytest.approx((1.0 / 6.0) * 2 - 1)
    assert v0 > -1.0

    ninja.jump_buffer = 5  # window edge maps to the top of the range
    assert nh.get_ninja_state()[7] == pytest.approx(1.0)

    # Same mapping for floor (8) and wall (9) buffers.
    ninja.floor_buffer = 0
    ninja.wall_buffer = 0
    state = nh.get_ninja_state()
    assert state[8] == pytest.approx((1.0 / 6.0) * 2 - 1)
    assert state[9] == pytest.approx((1.0 / 6.0) * 2 - 1)
