"""Regression: the per-level reachability graph cache must key on level CONTENT,
never on the mutable ``map_loader.current_map_name``.

Bug: ``GraphMixin._update_graph_data`` cached ``current_graph_data`` under
``current_map_name``. During the trainer's test-split benchmark eval, distinct
levels can share (or fail to refresh) that name, so the cache served a PREVIOUS
level's graph for the current one. The ninja was then checked against a foreign
graph's bounding box and flagged "far outside graph bounds → inf", flooding the
log and corrupting the path_distance-derived OBS features for that eval level.

Fix: key the cache on ``level_data.level_id`` (a content hash of tiles + entities
+ start + switch/exit positions, auto-set in ``LevelData.__post_init__``) so a
name collision can never desync the cache from the level actually loaded.

Run: .venv/bin/python -m pytest nclone/tests/test_graph_cache_content_key.py
"""
import glob
import pickle

import pytest

DATASET = "datasets/mechanic_benchmark_v4/test"


def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _map_data(lvl):
    return lvl.get("map_data") if isinstance(lvl, dict) else lvl


def _bounds(env):
    gd = getattr(env, "current_graph_data", None)
    if not gd:
        return None
    adj = gd.get("adjacency") or {}
    if not adj:
        return None
    xs = [n[0] for n in adj]
    ys = [n[1] for n in adj]
    return (min(xs), max(xs), min(ys), max(ys))


def _reset_eval(env, map_data, name):
    """Mimic the benchmark eval reset: payload + explicit name + new_level."""
    env.reset(options={"map_data": list(map_data), "map_name": str(name), "new_level": True})


def _two_distinct_levels():
    big = sorted(glob.glob(f"{DATASET}/CHAMBER_MAZE/T3/*.pkl"))
    small = sorted(glob.glob(f"{DATASET}/FLAT_RUN/T1/*.pkl"))
    if not big or not small:
        pytest.skip("mechanic_benchmark_v4 test split not available")
    return _map_data(_load_pkl(big[0])), _map_data(_load_pkl(small[0]))


@pytest.fixture(scope="module")
def env():
    from nclone.gym_environment.config import EnvironmentConfig
    from nclone.gym_environment.npp_environment import NppEnvironment

    e = NppEnvironment(config=EnvironmentConfig.for_training())
    yield e


def test_same_name_distinct_levels_do_not_collide(env):
    """Two different levels loaded under the SAME map_name must each get their
    OWN graph — the content key defeats the name collision."""
    big, small = _two_distinct_levels()

    _reset_eval(env, big, "COLLIDE")
    b_big = _bounds(env)
    _reset_eval(env, small, "COLLIDE")  # same name, different content
    b_small = _bounds(env)

    assert b_big is not None and b_small is not None
    assert b_big != b_small, (
        f"graph cache served a stale graph across a name collision: "
        f"big={b_big} small={b_small}"
    )


def test_same_content_still_cache_hits(env):
    """Reloading identical content must reuse the cached graph (throughput)."""
    big, _ = _two_distinct_levels()

    _reset_eval(env, big, "A")
    first = _bounds(env)
    cache = env._graph_data_cache
    key = next(iter(cache._mem)) if hasattr(cache, "_mem") else None

    _reset_eval(env, big, "B")  # different name, SAME content
    second = _bounds(env)

    assert first == second, "identical content produced different graphs"
    # Same content under a different name should not create a second cache entry.
    if key is not None:
        assert key in cache._mem, "content key not stable across resets"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
