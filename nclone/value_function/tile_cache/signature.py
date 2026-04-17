"""3x3 tile-neighborhood signature extraction and hashing.

A "signature" is the tuple of 9 tile types (``uint8``) in the 3x3 window
centered on a given cell, ordered row-major::

    (t_nw, t_n, t_ne,
     t_w,  t_c, t_e,
     t_sw, t_s, t_se)

This tuple is packed into a single ``uint64`` (each tile fits in 7 bits,
9 tiles fit in 63 bits) that serves as the cache key.

N++ uses 38 distinct tile types (0..37), so the theoretical signature
space is ``38**9 ~= 1.7e14``.  In practice tile corpora produce far
fewer unique signatures (~50k-200k across the benchmark pool), which
is why the cache is populated lazily on observation rather than
pre-enumerated.

See ``nclone/tile_definitions.py`` for the enumeration of tile types.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

# Cache keys use uint64 so we can stash the full 9-tile packing plus
# future normalization flags without collisions.
SIG_HASH_DTYPE = np.uint64

# Number of bits reserved per tile in the packed hash.  7 bits allows
# any tile value in [0, 127], well above the current max of 37.
_BITS_PER_TILE = 7
_TILE_MASK = (1 << _BITS_PER_TILE) - 1
_NUM_TILES = 9  # 3x3

# Out-of-bounds cells are treated as fully-solid (tile type 1, matching
# the simulator's border behaviour in ``loader.rs``).
OOB_TILE = 1


def extract_3x3_signature(
    tile_grid: np.ndarray,
    col: int,
    row: int,
) -> np.ndarray:
    """Extract the 3x3 tile-type window centered at ``(col, row)``.

    Args:
        tile_grid: 2D ``uint8`` tile array of shape ``(cols, rows)`` using
            the same column-major layout as ``SurfaceGraph.tile_dict``
            (indexed as ``tile_grid[col, row]``).
        col: Center column (0-based).
        row: Center row (0-based).

    Returns:
        A length-9 ``uint8`` numpy array in row-major order (NW, N, NE,
        W, C, E, SW, S, SE).  Cells outside ``tile_grid`` are reported
        as ``OOB_TILE`` (solid wall).
    """
    cols, rows = tile_grid.shape
    out = np.empty(_NUM_TILES, dtype=np.uint8)
    k = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            cc = col + dc
            rr = row + dr
            if 0 <= cc < cols and 0 <= rr < rows:
                out[k] = tile_grid[cc, rr]
            else:
                out[k] = OOB_TILE
            k += 1
    return out


def pack_signature(sig: Sequence[int]) -> int:
    """Pack 9 tile ids into a single ``uint64`` cache key.

    Packing order matches ``extract_3x3_signature`` output.  Each tile
    occupies ``_BITS_PER_TILE`` bits; tile 0 is placed in the lowest-
    order bits so shifting right by ``sig_hash >> (7 * 8)`` yields the
    center tile for shard bucketing.

    Returns:
        Python ``int`` (can be cast to ``uint64`` via
        ``np.uint64(pack_signature(sig))``).
    """
    if len(sig) != _NUM_TILES:
        raise ValueError(
            f"expected 3x3 signature of length {_NUM_TILES}, got {len(sig)}"
        )
    packed = 0
    for i, t in enumerate(sig):
        if not 0 <= int(t) <= _TILE_MASK:
            raise ValueError(
                f"tile id {t} at position {i} exceeds {_BITS_PER_TILE}-bit range"
            )
        packed |= (int(t) & _TILE_MASK) << (i * _BITS_PER_TILE)
    return packed


def unpack_signature(packed: int) -> Tuple[int, ...]:
    """Inverse of :func:`pack_signature`.  Returns a length-9 tuple."""
    p = int(packed)
    return tuple((p >> (i * _BITS_PER_TILE)) & _TILE_MASK for i in range(_NUM_TILES))


def hash_signature(sig: Sequence[int]) -> np.uint64:
    """Pack a 3x3 signature into the ``uint64`` hash used as the cache key.

    Small convenience wrapper around :func:`pack_signature` that returns
    a numpy scalar of the cache's native dtype, ready for insertion into
    a shard key array.
    """
    return SIG_HASH_DTYPE(pack_signature(sig))


def shard_id_from_hash(sig_hash: int, num_shards: int) -> int:
    """Map a signature hash to its shard bucket.

    Shards are keyed by the top bits of the hash (the highest-index
    tile in the row-major packing, typically the SE corner), which
    empirically distributes evenly across real tile corpora.
    """
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    return int(sig_hash >> (_BITS_PER_TILE * (_NUM_TILES - 1))) % num_shards


def tile_grid_from_map_data(map_data: Sequence[int]) -> np.ndarray:
    """Build a ``(cols, rows)`` tile-grid from raw map-file bytes.

    Mirrors the simulator's ``load_tiles`` routine (``loader.rs``):
    tiles at raw offsets ``184..184+42*23`` are placed at
    ``tile_grid[col+1, row+1]`` in a ``44x25`` grid; the outer border
    is filled with tile type 1 (solid).

    Args:
        map_data: Flat integer list of at least 1150 entries (full map
            bytes as returned by ``[int(b) for b in f.read()]``).

    Returns:
        ``uint8`` numpy array of shape ``(44, 25)``.
    """
    from ...constants import (
        FULL_MAP_WIDTH,
        FULL_MAP_HEIGHT,
        MAP_TILE_WIDTH,
        MAP_TILE_HEIGHT,
    )

    grid = np.ones((FULL_MAP_WIDTH, FULL_MAP_HEIGHT), dtype=np.uint8)
    for x in range(MAP_TILE_WIDTH):
        for y in range(MAP_TILE_HEIGHT):
            grid[x + 1, y + 1] = int(map_data[184 + x + y * MAP_TILE_WIDTH]) & 0xFF
    return grid


def signature_frequencies(
    tile_grid: np.ndarray,
    interior_only: bool = True,
) -> Dict[int, int]:
    """Compute a histogram of 3x3 signatures present in a ``tile_grid``.

    Useful for corpus-driven analysis / cache-coverage audits.

    Args:
        tile_grid: ``uint8`` tile array from :func:`tile_grid_from_map_data`.
        interior_only: When ``True``, skips the outer border row and
            column (which always see out-of-bounds neighbours).

    Returns:
        Dict mapping packed signature hash -> count.
    """
    cols, rows = tile_grid.shape
    col_range = range(1, cols - 1) if interior_only else range(cols)
    row_range = range(1, rows - 1) if interior_only else range(rows)
    counts: Dict[int, int] = {}
    for c in col_range:
        for r in row_range:
            sig = extract_3x3_signature(tile_grid, c, r)
            h = pack_signature(sig)
            counts[h] = counts.get(h, 0) + 1
    return counts
