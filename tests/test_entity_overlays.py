"""NPP-38: entity_overlays shrunk to 5 channels (toggle-mines scope).

Verifies the post-rescope channel layout, that dropped launch_pad / bounce_block /
thwump channels are truly gone, and that mine density semantics are preserved
against the pre-rescope 8-channel reference on a mines-only fixture.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from nclone.constants.physics_constants import (
    MAP_TILE_HEIGHT,
    MAP_TILE_WIDTH,
    TILE_PIXEL_SIZE,
)
from nclone.gym_environment.entity_overlay_utils import (
    ENTITY_OVERLAY_CH_EXIT,
    ENTITY_OVERLAY_CH_MINE_DEADLY,
    ENTITY_OVERLAY_CH_MINE_SAFE,
    ENTITY_OVERLAY_CH_NINJA,
    ENTITY_OVERLAY_CH_SWITCH,
    ENTITY_OVERLAY_CHANNELS,
    MINE_DENSITY_NORMALIZER,
    MINE_KILL_ZONE_RADIUS_PX,
    MINE_STATE_DEADLY,
    classify_mines_by_state,
    compute_entity_overlays,
)


def _mine(x: float, y: float, state: int = MINE_STATE_DEADLY) -> SimpleNamespace:
    return SimpleNamespace(xpos=float(x), ypos=float(y), state=int(state))


def _px_from_inner_tile(tx: int, ty: int) -> tuple[float, float]:
    return (
        (tx + 1) * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE / 2.0,
        (ty + 1) * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE / 2.0,
    )


def test_overlay_shape_dtype_range():
    arr = compute_entity_overlays(
        ninja_pos=(100.0, 100.0),
        switch_pos=(200.0, 200.0),
        switch_activated=False,
        exit_pos=(300.0, 300.0),
        deadly_mines=[],
        safe_mines=[],
    )
    assert arr.shape == (ENTITY_OVERLAY_CHANNELS, MAP_TILE_WIDTH, MAP_TILE_HEIGHT)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0


def test_overlay_has_exactly_5_channels():
    arr = compute_entity_overlays(
        ninja_pos=(100.0, 100.0),
        switch_pos=None,
        switch_activated=False,
        exit_pos=None,
        deadly_mines=[],
        safe_mines=[],
    )
    assert arr.shape[0] == 5
    assert ENTITY_OVERLAY_CHANNELS == 5


def test_ninja_bilinear_splat_sums_to_one_on_center():
    tx, ty = 10, 5
    px, py = _px_from_inner_tile(tx, ty)
    arr = compute_entity_overlays(
        ninja_pos=(px, py),
        switch_pos=None,
        switch_activated=False,
        exit_pos=None,
        deadly_mines=[],
        safe_mines=[],
    )
    total = float(arr[ENTITY_OVERLAY_CH_NINJA].sum())
    assert math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-5)


def test_switch_suppressed_when_activated():
    arr_inactive = compute_entity_overlays(
        ninja_pos=(100.0, 100.0),
        switch_pos=(200.0, 200.0),
        switch_activated=False,
        exit_pos=None,
        deadly_mines=[],
        safe_mines=[],
    )
    arr_active = compute_entity_overlays(
        ninja_pos=(100.0, 100.0),
        switch_pos=(200.0, 200.0),
        switch_activated=True,
        exit_pos=None,
        deadly_mines=[],
        safe_mines=[],
    )
    assert arr_inactive[ENTITY_OVERLAY_CH_SWITCH].sum() > 0.0
    assert arr_active[ENTITY_OVERLAY_CH_SWITCH].sum() == 0.0


def test_exit_channel_populated():
    arr = compute_entity_overlays(
        ninja_pos=(50.0, 50.0),
        switch_pos=None,
        switch_activated=True,
        exit_pos=(400.0, 300.0),
        deadly_mines=[],
        safe_mines=[],
    )
    assert arr[ENTITY_OVERLAY_CH_EXIT].sum() == 1.0


def test_mine_density_matches_legacy_reference():
    """The mine density channels must match the pre-NPP-38 reference exactly."""
    deadly = [_mine(x, 100.0) for x in (60.0, 120.0, 180.0)]
    safe = [_mine(x, 180.0) for x in (80.0, 160.0)]

    arr = compute_entity_overlays(
        ninja_pos=(100.0, 100.0),
        switch_pos=(400.0, 400.0),
        switch_activated=False,
        exit_pos=(500.0, 500.0),
        deadly_mines=deadly,
        safe_mines=safe,
    )

    expected_deadly = _reference_mine_density(deadly)
    expected_safe = _reference_mine_density(safe)

    np.testing.assert_allclose(
        arr[ENTITY_OVERLAY_CH_MINE_DEADLY], expected_deadly, atol=1e-6
    )
    np.testing.assert_allclose(
        arr[ENTITY_OVERLAY_CH_MINE_SAFE], expected_safe, atol=1e-6
    )


def _reference_mine_density(mines) -> np.ndarray:
    """Independent reference for mine density channels."""
    out = np.zeros((MAP_TILE_WIDTH, MAP_TILE_HEIGHT), dtype=np.float32)
    for tx in range(MAP_TILE_WIDTH):
        for ty in range(MAP_TILE_HEIGHT):
            tx_min = (tx + 1) * TILE_PIXEL_SIZE
            tx_max = tx_min + TILE_PIXEL_SIZE
            ty_min = (ty + 1) * TILE_PIXEL_SIZE
            ty_max = ty_min + TILE_PIXEL_SIZE
            count = 0
            for m in mines:
                cx = max(tx_min, min(m.xpos, tx_max - 1e-9))
                cy = max(ty_min, min(m.ypos, ty_max - 1e-9))
                dx = m.xpos - cx
                dy = m.ypos - cy
                if dx * dx + dy * dy <= MINE_KILL_ZONE_RADIUS_PX ** 2:
                    count += 1
            out[tx, ty] = min(1.0, count / MINE_DENSITY_NORMALIZER)
    return out


def test_classify_mines_by_state_splits_on_live_state():
    """Mines must be classified by live `state`, not by entity type.

    Regression for the overlay/hazard divergence: a type-1 TOGGLE_MINE that has
    transitioned to STATE_UNTOGGLED (state != 0) must appear in the safe
    bucket, not the deadly bucket, and vice versa for a type-21
    TOGGLE_MINE_TOGGLED that has become deadly.
    """
    t1_deadly = SimpleNamespace(xpos=10.0, ypos=10.0, state=0, entity_type=1)
    t1_safe = SimpleNamespace(xpos=20.0, ypos=20.0, state=1, entity_type=1)
    t21_deadly = SimpleNamespace(xpos=30.0, ypos=30.0, state=0, entity_type=21)
    t21_toggling = SimpleNamespace(xpos=40.0, ypos=40.0, state=2, entity_type=21)

    deadly, safe = classify_mines_by_state(
        [t1_deadly, t1_safe, t21_deadly, t21_toggling]
    )

    assert t1_deadly in deadly
    assert t21_deadly in deadly
    assert t1_safe in safe
    assert t21_toggling in safe
    assert len(deadly) == 2
    assert len(safe) == 2


def test_overlay_classification_agrees_with_hazard_state_filter():
    """End-to-end: deadly density must include only state==0 mines, regardless of type.

    The hazard map filters `state == 0` across the union of toggle_mines and
    toggled_mines; the overlay must do the same via `classify_mines_by_state`.
    """
    mines = [
        SimpleNamespace(xpos=100.0, ypos=100.0, state=0, entity_type=1),
        SimpleNamespace(xpos=160.0, ypos=100.0, state=1, entity_type=1),
        SimpleNamespace(xpos=220.0, ypos=100.0, state=0, entity_type=21),
        SimpleNamespace(xpos=280.0, ypos=100.0, state=2, entity_type=21),
    ]
    deadly, safe = classify_mines_by_state(mines)

    arr = compute_entity_overlays(
        ninja_pos=(50.0, 50.0),
        switch_pos=None,
        switch_activated=True,
        exit_pos=None,
        deadly_mines=deadly,
        safe_mines=safe,
    )

    expected_deadly = _reference_mine_density(deadly)
    expected_safe = _reference_mine_density(safe)
    np.testing.assert_allclose(
        arr[ENTITY_OVERLAY_CH_MINE_DEADLY], expected_deadly, atol=1e-6
    )
    np.testing.assert_allclose(
        arr[ENTITY_OVERLAY_CH_MINE_SAFE], expected_safe, atol=1e-6
    )
    assert float(arr[ENTITY_OVERLAY_CH_MINE_DEADLY].sum()) > 0.0
    assert float(arr[ENTITY_OVERLAY_CH_MINE_SAFE].sum()) > 0.0


def test_no_launch_pad_signature_in_call():
    """Sanity: the new signature does not accept dropped-entity kwargs."""
    with pytest.raises(TypeError):
        compute_entity_overlays(
            ninja_pos=(100.0, 100.0),
            switch_pos=None,
            switch_activated=False,
            exit_pos=None,
            deadly_mines=[],
            safe_mines=[],
            launch_pads=[],
        )
    with pytest.raises(TypeError):
        compute_entity_overlays(
            ninja_pos=(100.0, 100.0),
            switch_pos=None,
            switch_activated=False,
            exit_pos=None,
            deadly_mines=[],
            safe_mines=[],
            bounce_blocks=[],
        )
