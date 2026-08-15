"""Production global navigation potential.

The global potential deliberately separates *where to go* from *how to execute*.
Its primary source is the cheap, goal-seeded physics-cost flood, made
configuration-aware for locked doors and directional for one-way platforms.  A
surface lattice is consulted only as a cache-only fallback when the primary
field cannot price the spawn; runtime resets never start an unbounded lattice
enumeration.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Sequence

import numpy as np

from .door_aware_field import DoorAwareFreeSpaceField
from .free_space_distance import FreeSpaceDistanceSource, live_build_count


GLOBAL_POTENTIAL_VERSION = "global1"
GLOBAL_MINE_MASK_VERSION = "occupancy14-route20-v1"


class GlobalPotentialSource:
    """Dense phase-aware cost-to-completion for generalized training.

    The public methods intentionally match the subset of ``DistanceSource`` used
    by ``ReachabilityMixin``.  ``spawn_finite`` commits the source for the whole
    episode: primary global field when possible, cached lattice fallback
    otherwise.  This prevents scale/source switching mid-episode.
    """

    def __init__(
        self,
        map_data,
        *,
        name: Optional[str] = None,
        one_way: bool = True,
        door_aware: bool = True,
        lattice_fallback: bool = True,
        lattice_max_nodes: int = 18_000,
    ) -> None:
        self.map_data = [int(v) for v in map_data]
        self.name = name
        self.one_way = bool(one_way)
        self.door_aware = bool(door_aware)
        self.lattice_fallback = bool(lattice_fallback)
        self.lattice_max_nodes = int(lattice_max_nodes)

        self.available = False
        self.source_kind = "unavailable"
        self.fallback_reason: Optional[str] = None
        self.necessary_order = []
        self._primary = None
        self._lattice = None
        self._active = None

        # Construction telemetry. ``cache_hit`` counts LIVE free-space builds
        # across the whole construction (base plus every door leg), not the last
        # field's tier: a door-aware source whose base was cached but whose legs
        # were all computed would otherwise report a hit while costing seconds.
        # ``build_ms`` is the wall-clock the caller actually paid, which is the
        # number that decides whether a reset misses its vec-env deadline.
        self.build_ms: float = 0.0
        self.cache_hit: bool = False
        _t0 = time.perf_counter()
        _builds0 = live_build_count()

        try:
            if self.door_aware:
                primary = DoorAwareFreeSpaceField(
                    self.map_data,
                    mine_hard=True,
                    physics_cost=True,
                    one_way=self.one_way,
                    mine_split_topology=True,
                    name=name,
                )
                self.necessary_order = list(
                    getattr(primary, "necessary_order", ()) or ()
                )
                has_locked = bool(
                    getattr(getattr(primary, "_base", None), "locked_doors", ())
                )
                self.source_kind = (
                    "global_door_aware"
                    if getattr(primary, "door_aware", False)
                    else ("global_optimistic" if has_locked else "global")
                )
                self.fallback_reason = getattr(primary, "fallback_reason", None)
            else:
                primary = FreeSpaceDistanceSource(
                    self.map_data,
                    name=name,
                    mine_hard=True,
                    physics_cost=True,
                    one_way=self.one_way,
                    mine_split_topology=True,
                    doors="open",
                )
                self.source_kind = "global"
            if getattr(primary, "available", False):
                self._primary = primary
                self._active = primary
                self.available = True
        except Exception as exc:  # degrade to the cache-only fallback
            self.fallback_reason = (
                f"primary build failed: {type(exc).__name__}: {exc}"
            )
        finally:
            self.build_ms = (time.perf_counter() - _t0) * 1000.0
            self.cache_hit = live_build_count() == _builds0

    @property
    def provenance(self) -> dict:
        return {
            "version": GLOBAL_POTENTIAL_VERSION,
            "mine_mask_version": GLOBAL_MINE_MASK_VERSION,
            "source": self.source_kind,
            "one_way": self.one_way,
            "door_aware": self.door_aware,
            "lattice_fallback": self.lattice_fallback,
            "fallback_reason": self.fallback_reason,
            "build_ms": self.build_ms,
            "cache_hit": self.cache_hit,
        }

    def _primary_spawn_finite(self, x: float, y: float) -> bool:
        src = self._primary
        if src is None:
            return False
        try:
            return bool(src.spawn_finite(float(x), float(y)))
        except Exception:
            try:
                c = src.cost_to_completion(float(x), float(y), False, ())
                return bool(np.isfinite(c))
            except Exception:
                return False

    def _load_cached_lattice(self):
        """Load a matching lattice only when its artifact already exists.

        Checking the exact ``LatticeAudit`` cache key preserves all planner
        parameter/version semantics.  A miss is final for this reset: the
        offline global-potential precompute owns expensive lattice fallback
        construction and its timeout budget.
        """
        if self._lattice is not None:
            return self._lattice
        if not self.lattice_fallback:
            return None
        try:
            from .tools.lattice_audit import LatticeAudit

            audit = LatticeAudit(
                map_data=self.map_data,
                name=self.name,
                workers=1,
                max_nodes=self.lattice_max_nodes,
            )
            if not os.path.exists(audit._cache_path()):
                self.fallback_reason = "global severed; lattice cache miss"
                return None

            from .lattice_distance import LatticeDistanceSource

            lattice = LatticeDistanceSource(
                self.map_data,
                name=self.name,
                workers=1,
                max_nodes=self.lattice_max_nodes,
                use_cache=True,
                airborne_field=True,
                prefer_field=True,
            )
            if getattr(lattice, "available", False):
                self._lattice = lattice
                return lattice
        except Exception as exc:
            self.fallback_reason = (
                f"cached lattice fallback failed: {type(exc).__name__}: {exc}"
            )
        return None

    def spawn_finite(self, spawn_x: float, spawn_y: float) -> bool:
        if self._primary_spawn_finite(spawn_x, spawn_y):
            self._active = self._primary
            self.available = True
            return True
        lattice = self._load_cached_lattice()
        if lattice is not None and lattice.spawn_finite(spawn_x, spawn_y):
            self._active = lattice
            self.source_kind = "lattice_fallback"
            self.available = True
            return True
        self._active = self._primary
        return False

    def door_phase(
        self, switch_activated: bool, open_door_indices=None
    ) -> Optional[int]:
        src = self._active
        fn = getattr(src, "door_phase", None)
        if fn is None:
            return None
        try:
            return fn(switch_activated, open_door_indices)
        except TypeError:
            return fn(open_door_indices)

    def cost_to_completion(
        self,
        x: float,
        y: float,
        switch_activated: bool,
        open_door_indices: Optional[Sequence[int]] = None,
    ) -> float:
        src = self._active
        if src is None:
            return float("inf")
        if not hasattr(src, "cost_to_completion"):
            try:
                return float(
                    src.cost_to_go(
                        x, y, switch_activated,
                        open_door_indices=open_door_indices,
                    )
                )
            except TypeError:
                return float(src.cost_to_go(x, y, switch_activated))
        try:
            return float(
                src.cost_to_completion(
                    x, y, switch_activated, open_door_indices
                )
            )
        except TypeError:
            return float(src.cost_to_completion(x, y, switch_activated))

    def cost_to_go(
        self,
        x: float,
        y: float,
        switch_activated: bool,
        open_door_indices: Optional[Sequence[int]] = None,
    ) -> float:
        src = self._active
        if src is None:
            return float("inf")
        try:
            return float(
                src.cost_to_go(x, y, switch_activated, open_door_indices)
            )
        except TypeError:
            return float(src.cost_to_go(x, y, switch_activated))

    def nav_feature(
        self,
        x: float,
        y: float,
        switch_activated: bool,
        open_door_indices: Optional[Sequence[int]] = None,
    ):
        src = self._active
        if src is None:
            return np.zeros(4, dtype=np.float32)
        try:
            return src.nav_feature(
                x, y, switch_activated,
                open_door_indices=open_door_indices,
            )
        except TypeError:
            return src.nav_feature(x, y, switch_activated)
