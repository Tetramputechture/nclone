"""v26 W6a: static fall-lethality field (``fall_lethality_map`` obs key).

Impact deaths are ~40% of v25's losses and are committed at TAKEOFF — once the
ninja is falling, no action recovers — so the existing impact signals
(``lethal_landing_tau``, the landing PBRS/aux terms) all fire too late. This
field is the static geometry answer: per 6px cell, how survivable is a fall from
here, given the ACTUAL normal of the floor below.

These tests pin: (1) the physics matches ``ninja.py``'s kill rule including the
normal projection, (2) the field is emitted and shaped when the flags are on,
(3) the fine crop co-registers with the global field, (4) a real pit reads lethal
while flat ground reads safe.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pytest

from nclone.constants.physics_constants import (
    GRAVITY_FALL,
    MAX_SURVIVABLE_IMPACT,
    TILE_PIXEL_SIZE,
)
from nclone.gym_environment.constants import (
    EGO_CROP_FALL_FINE_SIZE,
    HAZARD_MAP_HEIGHT,
    HAZARD_MAP_WIDTH,
    HAZARD_MAP_OBS_HEIGHT,
    HAZARD_MAP_OBS_WIDTH,
)
from nclone.gym_environment.config import EnvironmentConfig, ReachabilityConfig
from nclone.gym_environment.fall_lethality import (
    BORDER_CELLS,
    CELL_PX,
    compute_fall_lethality_map,
    crop_fall_lethality,
    free_fall_speed,
    lethal_drop_px,
    survivable_threshold,
)
from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.map_generation.map_jump_platforms import MapJumpPlatforms


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
def test_flat_floor_is_the_least_forgiving_surface():
    """The headline correction from V26 §1.8: a naive MAX_SURVIVABLE_IMPACT = 6
    reading gives ~11.3 tiles, but the threshold drops by 4/3*|n_y|, so a flat
    floor kills from ~6.8 tiles — under a third of a 23-tile level."""
    flat = lethal_drop_px(1.0)
    assert survivable_threshold(1.0) == pytest.approx(4.6667, abs=1e-3)
    assert flat / TILE_PIXEL_SIZE == pytest.approx(6.81, abs=0.05)
    # Every non-flat surface is strictly MORE forgiving: the threshold rises AND
    # the projected impact speed falls.
    for ny in (0.891, 0.707, 0.454):
        assert lethal_drop_px(ny) > flat


def test_threshold_matches_the_sim_kill_rule():
    for ny in (0.0, 0.454, 0.707, 0.891, 1.0):
        assert survivable_threshold(ny) == pytest.approx(
            MAX_SURVIVABLE_IMPACT - 4 / 3 * abs(ny)
        )


def test_free_fall_speed_saturates_at_terminal_velocity():
    assert free_fall_speed(0.0) == 0.0
    assert free_fall_speed(163.3) == pytest.approx(4.667, abs=0.01)
    assert free_fall_speed(1e9) == pytest.approx(12.0)


def test_a_vertical_wall_is_never_lethal_to_land_on():
    """|n_y| = 0 projects a straight-down fall to zero impact velocity, so the
    threshold can never be exceeded no matter the height."""
    assert lethal_drop_px(0.0) == float("inf")


# ---------------------------------------------------------------------------
# Field construction
# ---------------------------------------------------------------------------
def _flat_floor_tiles(row: int, cols=range(1, 41)):
    """tile_dict with one solid row; tile type 1 is full solid."""
    return {(c, row): 1 for c in cols}


def _cell_of(world_x, world_y):
    return (
        int(world_x / CELL_PX) - BORDER_CELLS,
        int(world_y / CELL_PX) - BORDER_CELLS,
    )


def test_field_marks_a_deep_drop_lethal_and_a_shallow_one_safe():
    floor_row = 20
    tiles = _flat_floor_tiles(floor_row)
    field = compute_fall_lethality_map(
        tiles, HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT
    )
    assert field.shape == (2, HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT)

    floor_top_y = floor_row * TILE_PIXEL_SIZE
    x = 20 * TILE_PIXEL_SIZE

    # 2 tiles above the floor: well under the ~6.8-tile flat-floor limit.
    cx, cy = _cell_of(x, floor_top_y - 2 * TILE_PIXEL_SIZE)
    assert field[0, cx, cy] > 0.0, "a 2-tile drop must read survivable"

    # 10 tiles above: comfortably past it.
    cx, cy = _cell_of(x, floor_top_y - 10 * TILE_PIXEL_SIZE)
    assert field[0, cx, cy] < 0.0, "a 10-tile drop onto flat floor must read lethal"


def test_field_sign_flips_near_the_computed_threshold():
    floor_row = 20
    field = compute_fall_lethality_map(
        _flat_floor_tiles(floor_row), HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT
    )
    x = 20 * TILE_PIXEL_SIZE
    floor_top_y = floor_row * TILE_PIXEL_SIZE

    # Walk up the column and find where the margin changes sign.
    flip_drop = None
    prev = None
    for drop_px in range(4, 400, 3):
        cx, cy = _cell_of(x, floor_top_y - drop_px)
        if not (0 <= cy < HAZARD_MAP_HEIGHT):
            break
        m = field[0, cx, cy]
        if prev is not None and prev > 0.0 >= m:
            flip_drop = drop_px
            break
        prev = m

    assert flip_drop is not None, "the field must contain a survivable/lethal boundary"
    # Within ~2 cells of the analytic flat-floor limit (the field samples the
    # ninja center at 6px cells, and the floor sample sits a ninja-radius above
    # the tile edge).
    assert abs(flip_drop - lethal_drop_px(1.0)) < 4 * CELL_PX + TILE_PIXEL_SIZE


def test_the_cell_a_ninja_stands_in_reads_no_drop():
    """Regression: the 6px cell CENTER sits a pixel below the surface sample the
    ninja's center occupies while standing. Without a one-cell snap tolerance the
    downward search skips past its own floor and reports the drop to the NEXT
    floor down — i.e. "you are 8 tiles above a fatal landing" while the ninja is
    standing still on solid ground. Found live on MapDropLanding."""
    env = _make_env(expose_fall_lethality_map=True)
    md = list(MapJumpPlatforms(seed=3).generate(seed=3).map_data())
    env.reset(options={"map_data": md, "map_name": "jp3", "new_level": True})
    field = env._get_fall_lethality_map()

    nx, ny = env.nplay_headless.ninja_position()
    cx, cy = _cell_of(nx, ny)
    assert field[1, cx, cy] == pytest.approx(0.0, abs=1e-3), (
        "a grounded ninja's own cell must report no drop"
    )
    assert field[0, cx, cy] > 0.0


def test_cells_with_nothing_below_read_maximally_lethal():
    """A bottomless column has no landing surface at all; the default must be
    the dangerous reading, not the safe one."""
    field = compute_fall_lethality_map(
        {}, HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT
    )
    assert np.all(field[0] == -1.0)
    assert np.all(field[1] == 1.0)


def test_drop_channel_is_monotone_going_up_a_column():
    floor_row = 20
    field = compute_fall_lethality_map(
        _flat_floor_tiles(floor_row), HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT
    )
    x = 20 * TILE_PIXEL_SIZE
    floor_top_y = floor_row * TILE_PIXEL_SIZE
    drops = []
    for tiles_up in range(1, 10):
        cx, cy = _cell_of(x, floor_top_y - tiles_up * TILE_PIXEL_SIZE)
        drops.append(float(field[1, cx, cy]))
    assert all(b >= a for a, b in zip(drops, drops[1:])), drops


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------
def test_crop_co_registers_with_the_global_field():
    field = compute_fall_lethality_map(
        _flat_floor_tiles(20), HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT
    )
    nx, ny = 20 * TILE_PIXEL_SIZE + 3.0, 12 * TILE_PIXEL_SIZE + 3.0
    size = EGO_CROP_FALL_FINE_SIZE
    crop = crop_fall_lethality(field, nx, ny, size)
    assert crop.shape == (2, size, size)

    cx, cy = _cell_of(nx, ny)
    half = size // 2
    assert crop[0, half, half] == pytest.approx(field[0, cx, cy])
    assert crop[1, half, half] == pytest.approx(field[1, cx, cy])


def test_crop_pads_offgrid_as_solid_ground():
    """Off-grid is the level border — a wall, not a pit. Padding with the
    lethal default would make every edge-hugging state read as danger."""
    field = compute_fall_lethality_map(
        _flat_floor_tiles(20), HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT
    )
    crop = crop_fall_lethality(field, 0.0, 0.0, EGO_CROP_FALL_FINE_SIZE)
    assert crop[0, 0, 0] == 1.0
    assert crop[1, 0, 0] == 0.0


# ---------------------------------------------------------------------------
# Environment wiring
# ---------------------------------------------------------------------------
def _make_env(**reach):
    cfg = EnvironmentConfig.for_training(
        frame_skip=1,
        sim_backend="python",
        reachability=ReachabilityConfig(**reach),
    )
    return NppEnvironment(config=cfg)


def test_env_emits_both_fall_keys_when_enabled():
    env = _make_env(
        expose_fall_lethality_map=True, expose_ego_crop_fall_fine=True
    )
    md = list(MapJumpPlatforms(seed=3).generate(seed=3).map_data())
    obs, _ = env.reset(
        options={"map_data": md, "map_name": "jp3", "new_level": True}
    )

    # The global field is POOLED for the policy (HAZARD_MAP_OBS_*); the fine ego
    # crop still slices the full-resolution 6px field.
    assert obs["fall_lethality_map"].shape == (
        2, HAZARD_MAP_OBS_WIDTH, HAZARD_MAP_OBS_HEIGHT,
    )
    assert obs["ego_crop_fall_fine"].shape == (
        2, EGO_CROP_FALL_FINE_SIZE, EGO_CROP_FALL_FINE_SIZE,
    )
    for key in ("fall_lethality_map", "ego_crop_fall_fine"):
        space = env.observation_space.spaces[key]
        assert space.shape == obs[key].shape
        assert space.contains(obs[key]), key


def test_env_omits_fall_keys_by_default():
    env = _make_env()
    md = list(MapJumpPlatforms(seed=3).generate(seed=3).map_data())
    obs, _ = env.reset(
        options={"map_data": md, "map_name": "jp3", "new_level": True}
    )
    assert "fall_lethality_map" not in obs
    assert "ego_crop_fall_fine" not in obs


def test_field_is_cached_per_level_and_rebuilt_on_level_change():
    env = _make_env(expose_fall_lethality_map=True)
    md_a = list(MapJumpPlatforms(seed=3).generate(seed=3).map_data())
    env.reset(options={"map_data": md_a, "map_name": "jp3", "new_level": True})
    first = env._get_fall_lethality_map()
    assert env._get_fall_lethality_map() is first, "same level must reuse the field"

    md_b = list(MapJumpPlatforms(seed=9).generate(seed=9).map_data())
    env.reset(options={"map_data": md_b, "map_name": "jp9", "new_level": True})
    assert env._get_fall_lethality_map() is not first, (
        "a new level must rebuild the field — a stale field describes the "
        "PREVIOUS level's pits"
    )


def test_a_real_generated_level_has_both_safe_and_lethal_cells():
    """Sanity that the field is not degenerate on real geometry."""
    env = _make_env(expose_fall_lethality_map=True)
    md = list(MapJumpPlatforms(seed=3).generate(seed=3).map_data())
    obs, _ = env.reset(
        options={"map_data": md, "map_name": "jp3", "new_level": True}
    )
    field = obs["fall_lethality_map"]
    assert (field[0] > 0).any(), "no survivable cell anywhere"
    assert (field[0] < 0).any(), "no lethal cell anywhere on a platforms map"
    assert np.isfinite(field).all()
    assert field.min() >= -1.0 and field.max() <= 1.0
