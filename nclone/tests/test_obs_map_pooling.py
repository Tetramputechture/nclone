"""Pooling contract for the downsampled global map observations.

``hazard_map`` and ``fall_lethality_map`` are emitted to the policy at half the
resolution they are computed at. The property that must hold is CONSERVATISM: a
coarse cell may never look safer than the safest reading it summarizes, or the
agent would be told a mine/drop is not there.
"""

import numpy as np
import pytest

from nclone.gym_environment.constants import (
    HAZARD_MAP_CHANNELS,
    HAZARD_MAP_HEIGHT,
    HAZARD_MAP_OBS_DOWNSAMPLE,
    HAZARD_MAP_OBS_HEIGHT,
    HAZARD_MAP_OBS_WIDTH,
    HAZARD_MAP_WIDTH,
)
from nclone.gym_environment.fall_lethality import FALL_LETHALITY_CHANNELS
from nclone.gym_environment.obs_map_pooling import pool_channels


def test_obs_dims_divide_the_fine_grid_evenly():
    assert HAZARD_MAP_WIDTH % HAZARD_MAP_OBS_DOWNSAMPLE == 0
    assert HAZARD_MAP_HEIGHT % HAZARD_MAP_OBS_DOWNSAMPLE == 0
    assert HAZARD_MAP_OBS_WIDTH == HAZARD_MAP_WIDTH // HAZARD_MAP_OBS_DOWNSAMPLE
    assert HAZARD_MAP_OBS_HEIGHT == HAZARD_MAP_HEIGHT // HAZARD_MAP_OBS_DOWNSAMPLE


def test_pool_channels_shape_and_reduction():
    full = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    pooled = pool_channels(full, 2, ("min", "max"))
    assert pooled.shape == (2, 2, 2)
    assert pooled.dtype == np.float32
    # Block (0,0) of ch0 covers full[0, :2, :2] = [[0,1],[4,5]] -> min 0
    assert pooled[0, 0, 0] == full[0, :2, :2].min()
    assert pooled[1, 0, 0] == full[1, :2, :2].max()


def test_downsample_of_one_is_identity():
    full = np.random.RandomState(0).randn(2, 6, 6).astype(np.float32)
    np.testing.assert_array_equal(pool_channels(full, 1, ("min", "max")), full)


def test_hazard_pooling_never_hides_a_mine():
    """A single deep-negative SDF cell must survive into the coarse cell."""
    rng = np.random.RandomState(1)
    full = rng.uniform(0.5, 1.0, size=(HAZARD_MAP_CHANNELS, 8, 8)).astype(np.float32)
    full[0, 3, 5] = -1.0  # a mine's kill zone, one fine cell wide

    pooled = pool_channels(full, 2, ("min",) * HAZARD_MAP_CHANNELS)

    assert pooled[0, 3 // 2, 5 // 2] == -1.0
    # And conservatism in general: pooled <= every fine cell it covers.
    for ch in range(HAZARD_MAP_CHANNELS):
        for i in range(pooled.shape[1]):
            for j in range(pooled.shape[2]):
                block = full[ch, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
                assert pooled[ch, i, j] == pytest.approx(block.min())


def test_fall_pooling_keeps_worst_case_per_channel():
    """ch0 (survivable margin) min-pools; ch1 (drop distance) max-pools."""
    full = np.zeros((FALL_LETHALITY_CHANNELS, 4, 4), dtype=np.float32)
    full[0].fill(1.0)
    full[0, 1, 1] = -0.8  # a lethal landing hidden in an otherwise safe block
    full[1].fill(0.1)
    full[1, 1, 1] = 0.9  # ...which is also the longest drop

    pooled = pool_channels(full, 2, ("min", "max"))

    assert pooled[0, 0, 0] == pytest.approx(-0.8), "lethal margin was averaged away"
    assert pooled[1, 0, 0] == pytest.approx(0.9), "long drop was averaged away"


@pytest.mark.parametrize("modes", [("min", "min"), ("min", "max")])
def test_pooling_commutes_with_the_horizontal_mirror(modes):
    """M(pool(x)) == pool(M(x)) at the real map dimensions.

    The mirror-augmentation tiers synthesise the mirrored observation instead of
    letting the simulator produce it, so a reduction that did not commute with
    the flip would desynchronise the synthesised map from the real one — with no
    shape mismatch to catch it. Holds because the block size divides the width.
    """
    rng = np.random.RandomState(0)
    full = rng.randn(2, HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT).astype(np.float32)

    pooled_then_flipped = pool_channels(full, HAZARD_MAP_OBS_DOWNSAMPLE, modes)[
        :, ::-1, :
    ]
    flipped_then_pooled = pool_channels(
        np.ascontiguousarray(full[:, ::-1, :]), HAZARD_MAP_OBS_DOWNSAMPLE, modes
    )

    np.testing.assert_array_equal(pooled_then_flipped, flipped_then_pooled)


def test_downsample_divides_the_width_so_the_mirror_stays_exact():
    """The precondition the commutation proof rests on."""
    assert HAZARD_MAP_WIDTH % HAZARD_MAP_OBS_DOWNSAMPLE == 0


def test_modes_must_cover_every_channel():
    full = np.zeros((3, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="one reduction per channel"):
        pool_channels(full, 2, ("min", "max"))
    with pytest.raises(ValueError, match="unknown pooling mode"):
        pool_channels(full, 2, ("min", "max", "mean"))
