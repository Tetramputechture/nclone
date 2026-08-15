"""The Python reference threat-TTL field must equal the Rust one.

`gym_environment/threat_ttl_reference.py` is a SECOND implementation of a field
that already exists in `nclone-native/src/threat_field.rs`. It exists only
because the dev tool renders through the pure-Python backend (the native sim has
no renderer), so without it the overlay could never be seen on the tool where it
matters.

Two implementations of the same thing drift. When this one drifts, the failure is
particularly nasty: the overlay keeps rendering, looks plausible, and no longer
shows what the policy receives -- so it is believed while being wrong. These
tests load the same map into both backends, step them identically, and compare
the fields cell for cell.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nclone.gym_environment.threat_ttl_reference import (
    CH_CERTAIN_SDF,
    CH_CERTAIN_TTL,
    CH_REACTIVE_SDF,
    CH_REACTIVE_TTL,
    N_CHANNELS,
    OUT_HEIGHT,
    OUT_WIDTH,
    compute_threat_ttl_reference,
)

_OFFICIAL = Path(__file__).resolve().parents[1] / "maps" / "official"
MAX_FRAMES, STRIDE = 64, 4

# Spanning the corpus: drone-free, dense patrol, and both reactive classes.
_CASES = [
    "000 the basics",
    "589 escape is at hand",
    "109 out of the micro",
    "044 one got loose",  # death balls
    "166 tutorial flashback",  # chaser drones
    # The non-drone hazards, so parity covers the kinematic + reactive sources
    # added for them and not just the patrol-drone path.
    "027 rear guard",  # rocket + floor guard (the guard latches -> tier C)
    "019 abstract factory",  # thwumps
    "124 you can't always depend on the kindness of lasers",  # shove thwumps
    "004 pyramid party",  # turret -- exercises the beam segment path
    "009 abstract steeplechase",  # dual lasers
]


def _map_path(name):
    if not _OFFICIAL.is_dir():
        pytest.skip("official map corpus not present")
    hits = list(_OFFICIAL.rglob(name))
    if not hits:
        pytest.skip(f"map {name!r} not in corpus")
    return str(hits[0])


def _backends(path):
    """The same level loaded into both simulators."""
    from nclone.native_headless import NativePlayHeadless
    from nclone.nplay_headless import NPlayHeadless

    native = NativePlayHeadless()
    native.load_map(path)
    python = NPlayHeadless(enable_rendering=False)
    python.load_map(path)
    return native, python


def _rust_field(native):
    raw = native._native_sim.threat_ttl(MAX_FRAMES, STRIDE)
    return np.frombuffer(raw, dtype=np.float32).reshape(N_CHANNELS, OUT_HEIGHT, OUT_WIDTH)


@pytest.mark.parametrize("map_name", _CASES)
def test_reference_matches_the_rust_engine(map_name):
    path = _map_path(map_name)
    native, python = _backends(path)

    # Step both with identical input so their entity states stay in lockstep.
    for step in range(24):
        native.tick(0, 0)
        python.tick(0, 0)

        rust = _rust_field(native)
        ref = compute_threat_ttl_reference(python, MAX_FRAMES, STRIDE)

        assert ref.shape == rust.shape
        for ch, label in (
            (CH_CERTAIN_SDF, "certain SDF"),
            (CH_CERTAIN_TTL, "certain TTL"),
            (CH_REACTIVE_SDF, "reactive SDF"),
            (CH_REACTIVE_TTL, "reactive TTL"),
        ):
            # float32 vs float64 intermediate arithmetic, so allow a tick of
            # rounding -- but nothing that could move a cell across the lethal
            # boundary or shift a TTL by a whole sample (1/16 of the window).
            np.testing.assert_allclose(
                ref[ch],
                rust[ch],
                atol=1e-5,
                err_msg=(
                    f"{map_name} step {step}: {label} channel diverged between the "
                    f"Python reference (dev-tool overlay) and the Rust engine "
                    f"(what the policy sees)"
                ),
            )


def test_reference_reports_the_same_semantics_as_the_engine():
    """Spot-check the invariants, not just equality, so a bug shared by BOTH
    implementations still shows up."""
    native, python = _backends(_map_path("589 escape is at hand"))
    for _ in range(20):
        native.tick(0, 0)
        python.tick(0, 0)
    ref = compute_threat_ttl_reference(python, MAX_FRAMES, STRIDE)

    assert ref.dtype == np.float32
    assert ref.min() >= -1.0 and ref.max() <= 1.0
    # A cell lethal NOW must read 0.0 in TTL and <= 0.0 in the SDF: inside the
    # kill radius the signed distance is non-positive by construction.
    lethal_now = ref[CH_CERTAIN_TTL] == 0.0
    assert lethal_now.any(), "expected some cells lethal now on a dense patrol map"
    assert np.all(ref[CH_CERTAIN_SDF][lethal_now] <= 0.0)
    # And the converse: anywhere the SDF says strictly outside the kill disc,
    # TTL must not claim lethal-now.
    outside = ref[CH_CERTAIN_SDF] > 0.0
    assert not np.any(ref[CH_CERTAIN_TTL][outside] == 0.0)


def test_drone_free_level_is_all_safe():
    native, python = _backends(_map_path("000 the basics"))
    native.tick(0, 0)
    python.tick(0, 0)
    ref = compute_threat_ttl_reference(python, MAX_FRAMES, STRIDE)
    assert np.all(ref == 1.0)
    np.testing.assert_array_equal(ref, _rust_field(native))


def test_field_cells_map_to_the_right_world_coordinates():
    """A drone's own cell must be the cell that reads lethal-now.

    Entity positions are offset from the tile grid: the playable area starts one
    24px border tile in, so field cell (0, 0) is world (24, 24), NOT (0, 0). An
    overlay (or any consumer) that indexes from the window origin lands two cells
    off -- close enough to look plausible, and wrong precisely where a hazard
    field must not be. This pins the mapping from both directions.
    """
    from nclone.gym_environment.threat_ttl_reference import BLOCK_PX, PLAYFIELD_OFFSET_PX

    native, python = _backends(_map_path("589 escape is at hand"))
    for _ in range(20):
        native.tick(0, 0)
        python.tick(0, 0)
    ttl = compute_threat_ttl_reference(python, MAX_FRAMES, STRIDE)[CH_CERTAIN_TTL]

    zap, mini = python.get_drone_positions()
    assert zap or mini, "expected drones on this map"
    for px, py in list(zap) + list(mini):
        col = int((px - PLAYFIELD_OFFSET_PX) // BLOCK_PX)
        row = int((py - PLAYFIELD_OFFSET_PX) // BLOCK_PX)
        if not (0 <= col < OUT_WIDTH and 0 <= row < OUT_HEIGHT):
            continue
        assert ttl[row, col] == 0.0, (
            f"drone at world ({px:.1f}, {py:.1f}) maps to cell ({row}, {col}) "
            f"but that cell reads TTL {ttl[row, col]} instead of lethal-now. "
            f"The grid origin is almost certainly being treated as (0, 0) "
            f"rather than ({PLAYFIELD_OFFSET_PX}, {PLAYFIELD_OFFSET_PX})."
        )


def test_grid_covers_exactly_the_playable_area():
    """84x46 cells of 12px starting at 24px must span the 1008x552 playfield."""
    from nclone.gym_environment.threat_ttl_reference import BLOCK_PX, PLAYFIELD_OFFSET_PX

    assert OUT_WIDTH * BLOCK_PX == 1008.0
    assert OUT_HEIGHT * BLOCK_PX == 552.0
    assert PLAYFIELD_OFFSET_PX == 24.0
