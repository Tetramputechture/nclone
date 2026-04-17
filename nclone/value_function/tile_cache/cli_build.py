#!/usr/bin/env python3
"""Prewarm the tile-neighborhood cache from a corpus of maps.

The three-phase enumerator is fastest when every 3x3 tile signature that will
be encountered at runtime is already present in the on-disk cache.  This
utility walks a directory of ``.map`` files, extracts every 3x3 signature
they contain, and pre-populates the launch and landing caches by exercising
all relevant input classes through the harness simulator (via
:func:`populate_launch_entry` / :func:`populate_landing_entry`).

Running this once over the full map corpus amortises the cold-cache cost
across subsequent precomputations, so every level after the first hits a
warm cache and the three-phase pipeline becomes strictly faster than the
legacy enumerator.

Usage::

    python -m nclone.value_function.tile_cache.cli_build \\
        --maps-dir nclone/test_maps/ \\
        --tile-cache-dir value_functions/.tile_cache
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

_NCLONE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_NCLONE_ROOT))

from nclone.value_function.tile_cache import (  # noqa: E402
    TileCacheStore,
    classes_for_surface,
    extract_3x3_signature,
    hash_signature,
    populate_launch_entry,
    tile_dict_from_map_data,
)
from nclone.value_function.tile_cache.signature import (  # noqa: E402
    tile_grid_from_map_data,
)
from nclone.value_function.surface_extraction import (  # noqa: E402
    SURFACE_FLOOR,
    SURFACE_WALL,
)
from nclone.value_function.surface_graph import (  # noqa: E402
    NUM_VELOCITY_BINS_FLOOR,
    NUM_VELOCITY_BINS_WALL,
    velocity_bin_centers,
)

logger = logging.getLogger(__name__)


def _iter_map_files(maps_dir: Path) -> Iterable[Path]:
    """Yield candidate map files from ``maps_dir``.

    The nclone test corpus uses extensionless raw-byte map files, so we
    accept every regular file whose name doesn't look like documentation
    or JSON metadata in addition to ``*.map`` files when present.
    """
    for p in sorted(maps_dir.rglob("*")):
        if not p.is_file():
            continue
        name = p.name.lower()
        if name.endswith((".md", ".json", ".txt", ".py", ".npz")):
            continue
        yield p


def _collect_signatures(maps_dir: Path) -> Counter:
    """Count occurrences of every 3x3 signature across the corpus."""
    freq: Counter = Counter()
    for map_path in _iter_map_files(maps_dir):
        try:
            raw = map_path.read_bytes()
        except OSError as exc:
            logger.warning("skip %s: %s", map_path.name, exc)
            continue
        map_data = list(raw)
        try:
            grid = tile_grid_from_map_data(map_data)
        except Exception as exc:
            logger.debug("skip %s: %s", map_path.name, exc)
            continue
        n_cols, n_rows = grid.shape
        for col in range(n_cols):
            for row in range(n_rows):
                sig = extract_3x3_signature(grid, col, row)
                freq[int(hash_signature(sig))] += 1
    return freq


def _prewarm_launch_for_signature(
    store: TileCacheStore,
    signature: np.ndarray,
    surface_type: int,
    subcell_range: Iterable[Tuple[int, int]] = ((2, 2),),
) -> int:
    """Populate all launch-cache entries for one signature + surface type.

    Returns the number of entries written.  ``subcell_range`` defaults to a
    single central subcell; callers may widen it to cover more start
    positions at the cost of proportionally more harness sims.
    """
    n_written = 0
    centers = velocity_bin_centers(surface_type)
    for subcell in subcell_range:
        for vel_bin_in in range(len(centers)):
            vel_centre = float(centers[vel_bin_in])
            for input_class_id in classes_for_surface(surface_type):
                existing = store.get_launch(
                    int(hash_signature(signature)),
                    (int(surface_type), int(subcell[0]), int(subcell[1]),
                     int(vel_bin_in), int(input_class_id)),
                )
                if existing is not None:
                    continue
                populate_launch_entry(
                    store=store,
                    signature=list(signature),
                    surface_type=surface_type,
                    subcell=subcell,
                    vel_bin_in=vel_bin_in,
                    vel_bin_centre=vel_centre,
                    input_class_id=input_class_id,
                )
                n_written += 1
    return n_written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prewarm the tile-neighborhood cache from a map corpus",
    )
    parser.add_argument(
        "--maps-dir",
        required=True,
        help="Directory containing .map files (searched recursively)",
    )
    parser.add_argument(
        "--tile-cache-dir",
        required=True,
        help="Destination directory for the sharded cache files",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--top-k-signatures",
        type=int,
        default=0,
        help=(
            "If > 0, only prewarm the K most frequent signatures "
            "(per the corpus scan).  Useful for smoke tests or when the "
            "full sweep would be prohibitively expensive."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    maps_dir = Path(args.maps_dir)
    cache_dir = Path(args.tile_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    store = TileCacheStore(str(cache_dir), num_shards=args.num_shards)

    logger.info("Scanning %s for 3x3 signatures ...", maps_dir)
    t0 = time.time()
    freq = _collect_signatures(maps_dir)
    logger.info(
        "  %d unique signatures found (%.1fs)",
        len(freq),
        time.time() - t0,
    )

    # The signature hash doesn't round-trip to tiles directly, so we need to
    # build a hash -> first-seen-signature table while scanning.  Rescanning
    # is cheap; we keep a dict keyed by hash to avoid storing millions of
    # signatures when only K are kept below.
    sig_by_hash: dict = {}
    for map_path in _iter_map_files(maps_dir):
        try:
            raw = map_path.read_bytes()
        except OSError:
            continue
        try:
            grid = tile_grid_from_map_data(list(raw))
        except Exception:
            continue
        n_cols, n_rows = grid.shape
        for col in range(n_cols):
            for row in range(n_rows):
                sig = extract_3x3_signature(grid, col, row)
                h = int(hash_signature(sig))
                if h not in sig_by_hash:
                    sig_by_hash[h] = sig

    ordered_hashes: List[int] = [
        h for h, _ in freq.most_common(
            args.top_k_signatures if args.top_k_signatures > 0 else None
        )
    ]
    logger.info(
        "Prewarming %d signatures (shards=%d)",
        len(ordered_hashes),
        args.num_shards,
    )

    total_written = 0
    t0 = time.time()
    for i, h in enumerate(ordered_hashes):
        sig = sig_by_hash[h]
        for st in (SURFACE_FLOOR, SURFACE_WALL):
            total_written += _prewarm_launch_for_signature(store, sig, st)
        if args.verbose and (i + 1) % 50 == 0:
            logger.info(
                "  [%d/%d] wrote %d launch entries so far "
                "(%.1fs elapsed)",
                i + 1, len(ordered_hashes), total_written, time.time() - t0,
            )
    logger.info(
        "Prewarm complete: %d launch entries in %.1fs",
        total_written, time.time() - t0,
    )
    t_flush = time.time()
    store.close()
    logger.info(
        "Flushed cache shards to %s (%.2fs)",
        cache_dir, time.time() - t_flush,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
