"""L1.5: Level cache build stays within spec budget on a small graph."""

import time

import numpy as np

from nclone.graph.level_data import LevelData
from nclone.graph.reachability.path_distance_calculator import (
    CachedPathDistanceCalculator,
)


def test_build_level_cache_under_200ms_small_level():
    adj = {
        (0, 0): [((12, 0), 12.0)],
        (12, 0): [((24, 0), 12.0)],
        (24, 0): [((36, 0), 12.0)],
        (36, 0): [],
    }
    ld = LevelData(
        start_position=(24, 24),
        tiles=np.zeros((23, 42), dtype=np.int32),
        entities=[],
        switch_states={},
        level_id="unit_test_timing",
    )
    calc = CachedPathDistanceCalculator(max_cache_size=50)
    t0 = time.perf_counter()
    calc.build_level_cache(ld, adj, adj, None)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 200.0, f"build_level_cache took {elapsed_ms:.1f}ms"
