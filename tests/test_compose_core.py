"""Phase-1 smoke tests for the composable 2D region-grid map system.

Guards the core invariants the legacy benchmark pipeline relies on:
reproducibility, entity purity, exactly one exit door + switch, tile-accuracy
(the offset-correct entity decode), label rollup validity, and solvability.
"""

import pytest

from nclone.map_generation import compose
from nclone.map_generation.compose import (
    describe_level,
    assert_tile_accurate,
    labels as labels_mod,
)

TEMPLATES = ["flat_intro_2x3", "mixed_traverse_2x3"]
ALLOWED_ENTITY_TYPES = {0, 1, 21, 3, 4}  # ninja, mines, exit door/switch


def _entities(map_data):
    return describe_level(map_data)["entities"]


def test_labels_vocabulary_rolls_up_to_real_buckets():
    assert labels_mod.validate_labels() == []


@pytest.mark.parametrize("template", TEMPLATES)
@pytest.mark.parametrize("seed", [0, 7, 42])
def test_reproducible(template, seed):
    a, _ = compose.compose(template, seed=seed)
    b, _ = compose.compose(template, seed=seed)
    assert a.map_data() == b.map_data()


@pytest.mark.parametrize("template", TEMPLATES)
@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_entity_purity_and_single_exit(template, seed):
    m, _ = compose.compose(template, seed=seed)
    ents = _entities(m.map_data())
    types = {e["type"] for e in ents}
    assert types <= ALLOWED_ENTITY_TYPES, f"{template} seed={seed} types={types}"
    doors = [e for e in ents if e["type"] == 3]
    switches = [e for e in ents if e["type"] == 4]
    assert len(doors) == 1 and len(switches) == 1


@pytest.mark.parametrize("template", TEMPLATES)
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_tile_accurate(template, seed):
    m, _ = compose.compose(template, seed=seed)
    assert_tile_accurate(m.map_data())


def test_describe_level_entity_coords_are_offset_correct():
    """A known spawn grid coordinate must round-trip through serialize+decode.

    This is the regression guard for the ``//6`` / missing-offset bug.
    """
    from nclone.map_generation.map import Map

    m = Map(seed=0)
    m.set_ninja_spawn(7, 10, orientation=1)
    m.add_entity(3, 16, 14, 0, 0, 12, 14)  # door at (16,14), switch at (12,14)
    m.add_entity(1, 9, 13)  # mine, no offset
    ents = {e["name"]: e["tile"] for e in describe_level(m.map_data())["entities"]}
    assert ents["ninja"] == [7, 10]
    assert ents["exit_door"] == [16, 14]
    assert ents["exit_switch"] == [12, 14]
    assert ents["toggle_mine"] == [9, 13]


@pytest.mark.parametrize("template", TEMPLATES)
@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_primary_bucket_is_a_real_bucket(template, seed):
    from nclone.map_generation.generator_configs import BUCKET_CATALOG

    _, lab = compose.compose(template, seed=seed)
    assert lab.primary_bucket in BUCKET_CATALOG


@pytest.mark.parametrize("template", TEMPLATES)
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_solvable_by_construction(template, seed):
    from nclone.map_generation.solvability import is_map_solvable

    m, _ = compose.compose(template, seed=seed)
    assert is_map_solvable(m.map_data())
