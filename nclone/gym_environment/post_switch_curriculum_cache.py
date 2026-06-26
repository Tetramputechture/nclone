"""Lazy, disk-cached per-level POST-SWITCH start-state curricula.

Multi-level training (a benchmark bucket of many maps) can't precompute a
:class:`StartStateCurriculum` for every level up front — generation is an RRT
explore that costs seconds per level. Instead we build a level's post-switch
curriculum the first time a worker encounters that level and cache it both
in-process (LRU) and on disk (shared across workers / runs), exactly like the
free-space cost-to-go cache (:mod:`nclone.value_function.free_space_distance`).

The disk cache key is the map_data hash + the generation params, so a param
change can't return a stale pool. Reuses ``NCLONE_GRAPH_CACHE_DIR`` (a
``post_switch`` subdir) when set so the one env var enables the whole training
cache stack; otherwise a node-local ``/tmp`` dir.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import OrderedDict
from typing import List, Optional

import numpy as np

from .start_state_curriculum import StartStateCurriculum

logger = logging.getLogger(__name__)

# ps2: generate_post_switch now harvests the exit-guided RRT tree as graduated
# return-leg start states (near-exit..full-leg), not just switch-position states.
_CACHE_VERSION = "ps2"
# Bounded in-process LRU so a worker that revisits the same handful of levels
# doesn't re-read/regenerate. Small: curricula are a few hundred prefixes each.
_MEM_CACHE: "OrderedDict[str, StartStateCurriculum]" = OrderedDict()
_MEM_CACHE_MAX = 32


def _cache_dir() -> Optional[str]:
    d = os.environ.get("NCLONE_POST_SWITCH_CACHE_DIR")
    if not d:
        graph = os.environ.get("NCLONE_GRAPH_CACHE_DIR")
        d = (
            os.path.join(graph, "post_switch")
            if graph
            else "/tmp/nclone_post_switch_cache"
        )
    try:
        os.makedirs(d, exist_ok=True)
        return d
    except OSError:
        return None


def _cache_key(map_data: List[int], n_states: int, max_iter: int) -> str:
    h = hashlib.md5()
    h.update(_CACHE_VERSION.encode())
    h.update(repr((int(n_states), int(max_iter))).encode())
    h.update(np.asarray([int(v) for v in map_data], dtype=np.int64).tobytes())
    return h.hexdigest()


def get_or_build(
    map_data: List[int],
    *,
    n_states: int = 300,
    max_iter: int = 120000,
    seed: int = 0,
) -> StartStateCurriculum:
    """Return the post-switch curriculum for ``map_data`` (mem -> disk -> build).

    Always returns a :class:`StartStateCurriculum`; an empty one (``len == 0``)
    if no switch-consuming prefixes were found, so callers can treat "no
    curriculum" as a benign no-op."""
    key = _cache_key(map_data, n_states, max_iter)

    cur = _MEM_CACHE.get(key)
    if cur is not None:
        _MEM_CACHE.move_to_end(key)
        return cur

    cdir = _cache_dir()
    path = os.path.join(cdir, f"{key}.json") if cdir else None
    if path and os.path.isfile(path):
        try:
            cur = StartStateCurriculum.load(path)
            _remember(key, cur)
            return cur
        except Exception as exc:  # noqa: BLE001 - corrupt cache -> rebuild
            logger.debug(f"[post_switch] failed to load cache {path}: {exc}")

    try:
        cur = StartStateCurriculum.generate_post_switch(
            map_data,
            n_states=n_states,
            max_iter=max_iter,
            seed=seed,
            log=lambda m: logger.debug(f"[post_switch] {m}"),
        )
    except Exception as exc:  # noqa: BLE001 - never break the reset path
        logger.warning(f"[post_switch] generation failed: {exc}")
        cur = StartStateCurriculum(goal_kind="exit")

    if path:
        try:
            cur.save(path)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[post_switch] failed to save cache {path}: {exc}")
    _remember(key, cur)
    return cur


def _remember(key: str, cur: StartStateCurriculum) -> None:
    _MEM_CACHE[key] = cur
    _MEM_CACHE.move_to_end(key)
    while len(_MEM_CACHE) > _MEM_CACHE_MAX:
        _MEM_CACHE.popitem(last=False)
