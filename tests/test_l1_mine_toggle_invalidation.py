"""L1.5: Toggle mine state change forces path cache rebuild on same level_id."""

import numpy as np

from nclone.constants.entity_types import EntityType
from nclone.graph.level_data import LevelData
from nclone.graph.reachability.path_distance_calculator import (
    CachedPathDistanceCalculator,
)


def _simple_adjacency():
    return {
        (0, 0): [((12, 0), 12.0)],
        (12, 0): [((24, 0), 12.0)],
        (24, 0): [],
    }


def test_build_level_cache_rebuilds_when_mine_state_changes():
    adj = _simple_adjacency()
    base = adj
    mine = {
        "type": EntityType.TOGGLE_MINE,
        "x": 18,
        "y": 0,
        "state": 1,
    }
    ld = LevelData(
        start_position=(24, 24),
        tiles=np.zeros((23, 42), dtype=np.int32),
        entities=[mine],
        switch_states={},
        level_id="unit_test_mine_toggle",
    )

    calc = CachedPathDistanceCalculator(max_cache_size=50)
    r1 = calc.build_level_cache(ld, adj, base, None)
    assert r1 is True

    mine["state"] = 0
    r2 = calc.build_level_cache(ld, adj, base, None)
    assert r2 is True

    mine["state"] = 0
    r3 = calc.build_level_cache(ld, adj, base, None)
    assert r3 is False
