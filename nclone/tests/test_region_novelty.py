"""v26 W4: topological ROOM novelty for the maze failure mode.

CHAMBER_MAZE T3 is at 0.71 over 3027-frame episodes and is the only bucket that
times out. Every intrinsic available today is STATE novelty, which saturates in a
maze — once every corridor has been walked the bonus is uniformly near zero, yet
the agent still does not know which room to try next.

These tests pin the decomposition (rooms only appear once the corridors are cut,
which is why raw connected components are not enough), the per-episode reset, and
the reward/telemetry wiring.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pytest

from nclone.gym_environment.config import EnvironmentConfig, ReachabilityConfig
from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.gym_environment.region_novelty import label_regions
from nclone.map_generation.map_maze import MazeGenerator


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------
def test_none_and_empty_masks_are_handled():
    assert label_regions(None) == (None, 0)
    labels, n = label_regions(np.zeros((10, 10), dtype=bool))
    assert labels is None and n == 0


def test_two_chambers_joined_by_a_thin_corridor_split_into_two_rooms():
    """The headline requirement. Raw connected components would call this ONE
    region — the corridor joins them — which is why the erosion step exists."""
    mask = np.zeros((24, 12), dtype=bool)
    mask[2:9, 2:10] = True     # left chamber
    mask[15:22, 2:10] = True   # right chamber
    mask[9:15, 5] = True       # 1-cell-wide corridor between them

    labels, n = label_regions(mask)
    assert n == 2, n
    # Both chambers keep distinct ids...
    assert labels[5, 5] != labels[18, 5]
    # ...and the corridor is absorbed rather than left unassigned.
    assert labels[12, 5] in (labels[5, 5], labels[18, 5])


def test_every_reachable_cell_gets_a_room_and_no_unreachable_one_does():
    mask = np.zeros((24, 12), dtype=bool)
    mask[2:9, 2:10] = True
    mask[15:22, 2:10] = True
    mask[9:15, 5] = True

    labels, n = label_regions(mask)
    assert (labels[mask] > 0).all(), "a reachable cell with no room is a hole"
    assert (labels[~mask] == 0).all(), "an unreachable cell must stay unlabelled"
    assert set(np.unique(labels[mask])) == set(range(1, n + 1)), "ids must be dense"


def test_a_single_open_chamber_is_one_room():
    mask = np.zeros((20, 20), dtype=bool)
    mask[3:17, 3:17] = True
    labels, n = label_regions(mask)
    assert n == 1
    assert (labels[mask] == 1).all()


def test_a_pure_corridor_level_falls_back_to_raw_components():
    """Erosion dissolves a level with no chamber wider than one cell. The
    partition must still be defined — it just has fewer, larger rooms."""
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:18, 5] = True       # one long 1-cell corridor
    labels, n = label_regions(mask)
    assert n == 1
    assert (labels[mask] > 0).all()


def test_disconnected_areas_are_separate_rooms():
    mask = np.zeros((24, 12), dtype=bool)
    mask[2:9, 2:10] = True
    mask[15:22, 2:10] = True   # no corridor at all
    _, n = label_regions(mask)
    assert n == 2


def test_erosion_noise_does_not_inflate_the_room_count():
    """A 3x3 nub erodes to a single cell; counting it as a room would make the
    coverage denominator meaningless."""
    mask = np.zeros((30, 14), dtype=bool)
    mask[2:12, 2:12] = True    # a real chamber
    mask[20:23, 4:7] = True    # a nub, disconnected
    _, n = label_regions(mask)
    assert n == 1, n


# ---------------------------------------------------------------------------
# Environment wiring
# ---------------------------------------------------------------------------
def _make_env(coef=0.0):
    cfg = EnvironmentConfig.for_training(
        frame_skip=1,
        sim_backend="python",
        region_novelty_coef=coef,
        reachability=ReachabilityConfig(),
    )
    return NppEnvironment(config=cfg)


def _maze_map(seed=1):
    return list(MazeGenerator(seed=seed).generate(seed=seed).map_data())


def test_disabled_by_default_costs_nothing():
    env = _make_env(coef=0.0)
    env.reset(options={"map_data": _maze_map(), "map_name": "cm", "new_level": True})
    for _ in range(30):
        _, _, term, trunc, info = env.step(3)
        assert info["region_bonus"] == 0.0
        assert not info["region_new"]
        if term or trunc:
            break


def test_first_entry_pays_once_per_episode():
    env = _make_env(coef=0.05)
    env.reset(options={"map_data": _maze_map(), "map_name": "cm", "new_level": True})

    seen_ids = set()
    payments = 0
    for _ in range(400):
        _, _, term, trunc, info = env.step(np.random.randint(0, 6))
        if info["region_new"]:
            payments += 1
            assert info["region_bonus"] == pytest.approx(0.05)
        else:
            assert info["region_bonus"] == 0.0
        if term or trunc:
            break
    # The spawn room alone guarantees at least one discovery.
    assert payments >= 1
    # Never more discoveries than there are rooms.
    assert payments <= max(1, info["region_count"])
    del seen_ids


def test_coverage_is_monotone_within_an_episode_and_resets_across():
    env = _make_env(coef=0.05)
    env.reset(options={"map_data": _maze_map(), "map_name": "cm", "new_level": True})
    prev = 0.0
    last = 0.0
    for _ in range(300):
        _, _, term, trunc, info = env.step(np.random.randint(0, 6))
        cov = info["region_coverage"]
        assert cov >= prev - 1e-9, (prev, cov)
        assert 0.0 <= cov <= 1.0
        prev = cov
        last = cov
        if term or trunc:
            break
    assert last > 0.0, "the spawn room must at least register"

    env.reset(options={"map_data": _maze_map(), "map_name": "cm", "new_level": True})
    _, _, _, _, info = env.step(0)
    assert info["region_coverage"] <= 1.0 / max(1, info["region_count"]) + 1e-9, (
        "room discovery must be strictly EPISODIC — a room is novel again next "
        "episode even though the partition is a per-level constant"
    )


def test_partition_is_cached_per_level():
    env = _make_env(coef=0.05)
    env.reset(options={"map_data": _maze_map(1), "map_name": "cm1", "new_level": True})
    env.step(0)
    first = env._region_labels()[0]
    assert env._region_labels()[0] is first, "same level must reuse the partition"
