"""Which locked doors a level actually requires — cached, and free of the potential.

Why this is its own module
--------------------------
``obs["necessary_doors_open"]`` drives two things: the locked-door milestone
reward (``R_LOCKED_DOOR_SWITCH``) and the HWM sub-goal phase.  Until now the
only producer of that count was a *potential* — ``DoorAwareFreeSpaceField`` or
``LatticeDistanceSource`` — so asking "is this door necessary?" meant also
building a phase-legged cost field.  That coupling is expensive and, worse, it
is a confound:

* The door-legged **potential** was A/B'd and LOST (−0.126 rho, CI
  [−0.203, −0.055]; worst-case build 3,288 ms).  Turning it off is what
  gamma-5c did, and that silently zeroed the necessity count too.
* The milestone **reward** has never been enabled at all.

So the two were never separable, and the one that lost dragged the one that was
never tried down with it.  This module makes the count available on its own.

What it costs
-------------
Necessity is O(n) oracle configurations in the number of *openable* doors (see
``door_dependency``), where a configuration is one ~0.07 s free-space flood.
Two things keep that off the reset path:

1. **Doors are rare and cheap to rule out.** 77% of official maps carry no
   locked door at all, and every procedural bucket except the door bucket
   carries none.  ``parse_entities`` answers that from the raw map bytes in
   microseconds, with no field build.
2. **The answer is static per map.**  Necessity depends on geometry, not on the
   episode, so it is computed once and cached — an in-process LRU over a shared
   on-disk pickle, the same two-tier scheme ``free_space_distance`` uses.  The
   payload is a tuple of small ints, so the disk tier is bytes, not megabytes.

Direction of error
------------------
Inherited from the solver and deliberate: over-including a door costs one
spurious milestone; under-including one means the agent opens a door it had to
open and is paid nothing.  On failure this returns an EMPTY order with
``solvable=False`` and a reason — callers must treat that as "no milestone
available", never as "the level is impossible" (the unsolvable verdict is
measured wrong at least 8.6% of the time; see ``door_aware_field``).
"""

from __future__ import annotations

import hashlib
import os
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

from ..constants.entity_types import EntityType


#: Bump when the solver's necessity semantics change (not when the *legs* do).
DOOR_NECESSITY_VERSION = "dn1"

_MEM_CACHE: "OrderedDict[str, DoorNecessity]" = OrderedDict()
_MEM_CAP = int(os.environ.get("NCLONE_DOOR_NECESSITY_MEM_CAP", "1024"))

#: Process-wide count of LIVE solves (both cache tiers missed).  Lets a caller
#: assert that a training run is reading the cache rather than paying for it.
_LIVE_SOLVES = 0


def live_solve_count() -> int:
    """Number of necessity solves in this process that missed both caches."""
    return _LIVE_SOLVES


@dataclass(frozen=True)
class DoorNecessity:
    """Answer for one level.

    ``order`` is the necessary doors in a dependency-respecting open order,
    indexed into the level's locked doors *in simulator order* — the same
    indexing ``FreeSpaceDistanceSource.locked_doors`` and
    ``ReachabilityMixin._lattice_open_door_indices`` use, so a caller can count
    opened-and-necessary by set membership.
    """

    order: Tuple[int, ...] = ()
    n_doors: int = 0
    solvable: bool = True
    reason: str = ""
    build_ms: float = 0.0
    cache_tier: str = "none"

    @property
    def n_necessary(self) -> int:
        return len(self.order)

    @property
    def has_doors(self) -> bool:
        return self.n_doors > 0


#: The level carries no locked door at all — the common case, and the one that
#: must cost nothing.  Shared instance; the dataclass is frozen.
_NO_DOORS = DoorNecessity(cache_tier="no_doors")


def count_locked_doors(map_data: Sequence[int]) -> int:
    """Locked doors on the level, from the raw map bytes.

    Deliberately does not construct a simulator or a field: this is the guard
    that keeps the 77% of levels with no doors off the solver path entirely.
    """
    from ..levelgen.labeling import parse_entities

    try:
        return sum(
            1 for e in parse_entities(map_data)
            if int(e[0]) == int(EntityType.LOCKED_DOOR)
        )
    except Exception:
        return 0


def _cache_dir() -> Optional[str]:
    """Shared on-disk cache dir, or None to disable the disk tier.

    Rides the same env var as the free-space cache (its own subdir) so one
    setting enables both and a deploy cannot half-configure them.
    """
    d = os.environ.get("NCLONE_DOOR_NECESSITY_CACHE_DIR")
    if not d:
        base = (
            os.environ.get("NCLONE_FREESPACE_CACHE_DIR")
            or os.environ.get("NCLONE_GRAPH_CACHE_DIR")
        )
        d = (
            os.path.join(base, "door_necessity") if base
            else "/tmp/nclone_door_necessity_cache"
        )
    try:
        os.makedirs(d, exist_ok=True)
        return d
    except OSError:
        return None


def _cache_key(map_data: Sequence[int], params: Tuple) -> str:
    import numpy as np

    from .free_space_distance import _native_fingerprint

    h = hashlib.md5()
    h.update(DOOR_NECESSITY_VERSION.encode())
    h.update(_native_fingerprint().encode())
    h.update(repr(params).encode())
    h.update(np.asarray([int(v) for v in map_data], dtype=np.int64).tobytes())
    return h.hexdigest()


def _mem_put(key: str, value: "DoorNecessity") -> None:
    # Stored with ``cache_tier="mem"`` so a later read reports how IT was served,
    # not how the entry was originally produced. A cache hit that still says
    # "build" makes a warm run look like it is paying for every level.
    _MEM_CACHE[key] = replace(value, cache_tier="mem")
    _MEM_CACHE.move_to_end(key)
    while len(_MEM_CACHE) > _MEM_CAP:
        _MEM_CACHE.popitem(last=False)


def necessary_doors(
    map_data: Sequence[int],
    *,
    name: Optional[str] = None,
    mine_hard: bool = True,
    physics_cost: bool = True,
    one_way: Optional[bool] = True,
    mine_split_topology: bool = True,
    use_cache: bool = True,
) -> DoorNecessity:
    """Necessary locked doors for ``map_data``, cached.

    The solver keywords must match the field the *rest* of the run uses, because
    necessity is a statement about a topology: a door that is necessary on the
    mine-hard mask need not be on the mine-soft one.  They are folded into the
    cache key, so a mismatch is a miss rather than a stale answer.

    Never raises: any failure degrades to "no necessary doors" with a reason.
    """
    global _LIVE_SOLVES

    n_doors = count_locked_doors(map_data)
    if n_doors == 0:
        return _NO_DOORS

    params = (
        bool(mine_hard), bool(physics_cost),
        None if one_way is None else bool(one_way),
        bool(mine_split_topology),
    )
    key = _cache_key(map_data, params) if use_cache else None

    if key is not None:
        hit = _MEM_CACHE.get(key)
        if hit is not None:
            _MEM_CACHE.move_to_end(key)
            return hit

    cache_dir = _cache_dir() if key is not None else None
    path = os.path.join(cache_dir, key + ".pkl") if cache_dir else None
    if path and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            got = DoorNecessity(
                order=tuple(int(i) for i in payload["order"]),
                n_doors=int(payload["n_doors"]),
                solvable=bool(payload["solvable"]),
                reason=str(payload.get("reason", "")),
                cache_tier="disk",
            )
            _mem_put(key, got)
            return got
        except Exception:
            pass  # torn / stale entry -> re-solve

    t0 = time.perf_counter()
    _LIVE_SOLVES += 1
    order: List[int] = []
    solvable, reason = True, ""
    try:
        from .free_space_door_plan import solve

        plan, _oracle = solve(
            map_data,
            mine_hard=mine_hard,
            physics_cost=physics_cost,
            one_way=one_way,
            mine_split_topology=mine_split_topology,
            name=name,
            compute_legs=False,
        )
        if plan is None:
            reason = "no plan (no doors, no goals, or field unavailable)"
        elif not plan.solvable:
            solvable, reason = False, plan.reason or "unsolvable"
        else:
            order = [int(i) for i in plan.necessary_order]
    except Exception as exc:
        solvable = False
        reason = f"solver raised: {type(exc).__name__}: {exc}"

    result = DoorNecessity(
        order=tuple(order),
        n_doors=n_doors,
        solvable=solvable,
        reason=reason,
        build_ms=(time.perf_counter() - t0) * 1000.0,
        cache_tier="build",
    )
    if key is not None:
        _mem_put(key, result)
    if path:
        try:  # atomic write: concurrent workers must never read a partial file
            tmp = f"{path}.tmp.{os.getpid()}"
            with open(tmp, "wb") as f:
                pickle.dump(
                    {
                        "order": list(result.order),
                        "n_doors": result.n_doors,
                        "solvable": result.solvable,
                        "reason": result.reason,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            os.replace(tmp, path)
        except Exception:
            pass
    return result


def clear_caches() -> None:
    """Drop the in-process tier (tests; the disk tier is keyed and safe to keep)."""
    _MEM_CACHE.clear()


__all__ = [
    "DOOR_NECESSITY_VERSION",
    "DoorNecessity",
    "clear_caches",
    "count_locked_doors",
    "live_solve_count",
    "necessary_doors",
]
