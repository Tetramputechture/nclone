"""Lazy, disk-cached per-level PRE-SWITCH start-state curricula (v16).

The reverse-curriculum sibling of :mod:`post_switch_curriculum_cache`, but for the
PRE-switch leg: mid-route states on the way to the exit SWITCH, tagged by
free-space cost-to-go so a difficulty schedule can start the ninja near the switch
and walk the start back toward spawn. Targets the PRECISION bucket, whose p=0.06
success is an EXECUTION wall (0.81 mine-death) that pure RL from spawn can't climb —
seeding the hard final approach lets the agent practise the threading directly.

Generation is an RRT explore (seconds per level), so a level's curriculum is built
the first time a worker meets it and cached in-process (LRU) + on disk (shared
across workers/runs), keyed by map_data hash + generation params. Reuses
``NCLONE_GRAPH_CACHE_DIR`` (a ``pre_switch`` subdir) when set.
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

_CACHE_VERSION = "pre1"
_MEM_CACHE: "OrderedDict[str, StartStateCurriculum]" = OrderedDict()
_MEM_CACHE_MAX = 32


def _cache_dir() -> Optional[str]:
    d = os.environ.get("NCLONE_PRE_SWITCH_CACHE_DIR")
    if not d:
        graph = os.environ.get("NCLONE_GRAPH_CACHE_DIR")
        d = (
            os.path.join(graph, "pre_switch")
            if graph
            else "/tmp/nclone_pre_switch_cache"
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
    n_states: int = 400,
    max_iter: int = 200000,
    seed: int = 0,
) -> StartStateCurriculum:
    """Return the pre-switch (goal=switch) curriculum for ``map_data``
    (mem -> disk -> build). Always returns a :class:`StartStateCurriculum`; an
    empty one (``len == 0``) when the RRT found no reachable mid-route states, so
    callers treat "no curriculum" as a benign no-op."""
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
            logger.debug(f"[pre_switch] failed to load cache {path}: {exc}")

    try:
        cur = StartStateCurriculum.generate(
            map_data,
            goal_kind="switch",
            n_states=n_states,
            max_iter=max_iter,
            seed=seed,
            log=lambda m: logger.debug(f"[pre_switch] {m}"),
        )
    except Exception as exc:  # noqa: BLE001 - never break the reset path
        logger.warning(f"[pre_switch] generation failed: {exc}")
        cur = StartStateCurriculum(goal_kind="switch")

    if path:
        try:
            cur.save(path)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[pre_switch] failed to save cache {path}: {exc}")
    _remember(key, cur)
    return cur


def _remember(key: str, cur: StartStateCurriculum) -> None:
    _MEM_CACHE[key] = cur
    _MEM_CACHE.move_to_end(key)
    while len(_MEM_CACHE) > _MEM_CACHE_MAX:
        _MEM_CACHE.popitem(last=False)
