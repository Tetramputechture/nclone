"""Hot in-memory lookup wrappers around :class:`TileCacheStore`.

Thin sugar used by :mod:`edge_enumeration` to keep the three-phase
pipeline readable.  All functions return ``None`` on cache miss so the
caller can populate via the builder.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from .signature import hash_signature
from .store import LandingEntry, LaunchEntry, TileCacheStore


def launch_lookup(
    store: TileCacheStore,
    signature: Sequence[int],
    surface_type: int,
    subcell: Tuple[int, int],
    vel_bin_in: int,
    input_class_id: int,
) -> Optional[LaunchEntry]:
    """Return the cached launch entry, or ``None`` if not yet populated."""
    sig_hash = int(hash_signature(signature))
    key_extra = (
        int(surface_type),
        int(subcell[0]),
        int(subcell[1]),
        int(vel_bin_in),
        int(input_class_id),
    )
    return store.get_launch(sig_hash, key_extra)


def landing_lookup(
    store: TileCacheStore,
    signature: Sequence[int],
    entry_subcell: Tuple[int, int],
    entry_vx_bin_air: int,
    entry_vy_bin_air: int,
) -> Optional[LandingEntry]:
    """Return the cached landing entry, or ``None`` if not yet populated."""
    sig_hash = int(hash_signature(signature))
    key_extra = (
        int(entry_subcell[0]),
        int(entry_subcell[1]),
        int(entry_vx_bin_air),
        int(entry_vy_bin_air),
    )
    return store.get_landing(sig_hash, key_extra)


__all__ = ["launch_lookup", "landing_lookup"]
