"""Ingest expert demonstrations into arc caches for PBRS precomputation.

Parses N++ attract-format `.replay` files, simulates them through the native
Rust physics engine with ``tick_tiles_only`` (no entity side-effects), and
returns per-frame state arrays that ``ArcCache.ingest_demonstration`` can
consume.

Exported API
------------
simulate_replay_to_frames(replay_path, map_data) -> List[Tuple]
    Simulate a single replay file; return [(x,y,vx,vy,state_code), ...].

ingest_demos_into_caches(replay_paths, map_data, arc_cache_switch,
                          arc_cache_exit, d_star_switch, d_star_exit, graph)
    Batch-ingest all replays into both phase caches.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Airborne state codes from edge_enumeration (3=jumping, 4=falling)
_AIRBORNE_STATES = {3, 4}


def simulate_replay_to_frames(
    replay_path: str,
    map_data: List[int],
) -> List[Tuple[float, float, float, float, int]]:
    """Simulate a single `.replay` file and return per-frame states.

    Uses ``tick_tiles_only`` so entity states are not mutated during
    simulation — identical to the edge-enumeration approach.

    Args:
        replay_path: Path to an N++ attract-format ``.replay`` file.
        map_data: Level map bytes (list[int]) for the same level.

    Returns:
        List of ``(x, y, vx, vy, state_code)`` tuples, one per simulated
        frame until win/death/timeout (max 20 000 frames).

    Raises:
        ImportError: If nclone_native is not available.
        ValueError: If the replay file cannot be parsed.
    """
    from nclone_native import Simulator as NativeSimulator
    from ..replay.npp_attract_decoder import NppAttractDecoder

    decoder = NppAttractDecoder()
    decoded = decoder.decode_npp_attract_file(replay_path)
    inputs: List[int] = decoded.get("demo_data", [])
    if not inputs:
        raise ValueError(f"No demo inputs found in {replay_path}")

    # Convert 0–7 encoded inputs to (hor_input, jump_input) pairs
    # N++ input byte: bits 0-1 = horizontal (-1, 0, 1), bit 2 = jump
    # Encoding from input_decoder.py / convert_actions.py:
    #   0 = no input,  1 = right,   2 = left,   3 = right+jump
    #   4 = left+jump, 5 = jump,    6 = right (alt), 7 = left (alt)
    # The mapping below follows the standard N++ attract format decoding.
    _INPUT_TABLE: List[Tuple[int, int]] = [
        (0, 0),   # 0: no input
        (1, 0),   # 1: right
        (-1, 0),  # 2: left
        (1, 1),   # 3: right + jump
        (-1, 1),  # 4: left + jump
        (0, 1),   # 5: jump only
        (1, 0),   # 6: right (alt)
        (-1, 0),  # 7: left (alt)
    ]

    sim = NativeSimulator.load(map_data)

    frames: List[Tuple[float, float, float, float, int]] = []
    MAX_FRAMES = 20_000

    for raw_input in inputs[:MAX_FRAMES]:
        hor, jmp = _INPUT_TABLE[int(raw_input) & 7]
        sim.tick_tiles_only(hor, jmp)

        pos = sim.get_ninja_pos()
        vel = sim.get_ninja_velocity()
        sc = int(sim.get_ninja_state_code())
        frames.append((pos[0], pos[1], vel[0], vel[1], sc))

        # Stop on win (state 8) or death (6, 7)
        if sc in (6, 7, 8):
            break

    return frames


def ingest_demos_into_caches(
    replay_paths: Sequence[str],
    map_data: List[int],
    arc_cache_switch,
    arc_cache_exit,
    d_star_switch: np.ndarray,
    d_star_exit: np.ndarray,
    graph,
) -> Tuple[int, int]:
    """Ingest all replays into both phase arc caches.

    Args:
        replay_paths: Paths to ``.replay`` files for this level.
        map_data: Level map bytes.
        arc_cache_switch: ArcCache for phase-0 (switch not collected).
        arc_cache_exit: ArcCache for phase-1 (exit).
        d_star_switch: Cost-to-go array for switch goal.
        d_star_exit: Cost-to-go array for exit goal.
        graph: SurfaceGraph for node lookup.

    Returns:
        (n_switch_insertions, n_exit_insertions) totals across all replays.
    """
    total_sw = 0
    total_ex = 0

    for rp in replay_paths:
        rp = str(rp)
        try:
            frames = simulate_replay_to_frames(rp, map_data)
        except Exception as e:
            logger.warning(f"  Skipping demo {os.path.basename(rp)}: {e}")
            continue

        n_air = sum(1 for _, _, _, _, sc in frames if sc in _AIRBORNE_STATES)
        if n_air == 0:
            logger.debug(f"  {os.path.basename(rp)}: no airborne frames, skipping")
            continue

        sw = arc_cache_switch.ingest_demonstration(frames, d_star_switch, graph)
        ex = arc_cache_exit.ingest_demonstration(frames, d_star_exit, graph)
        total_sw += sw
        total_ex += ex
        logger.info(
            f"  {os.path.basename(rp)}: {len(frames)} frames, "
            f"{n_air} airborne → +{sw} switch cells, +{ex} exit cells"
        )

    return total_sw, total_ex


def find_replay_files(demo_dir: str) -> List[str]:
    """Recursively find all ``.replay`` files under ``demo_dir``.

    Args:
        demo_dir: Directory to search.

    Returns:
        Sorted list of absolute paths to ``.replay`` files.
    """
    paths: List[str] = []
    for root, _dirs, files in os.walk(demo_dir):
        for fname in files:
            if fname.endswith(".replay"):
                paths.append(os.path.join(root, fname))
    return sorted(paths)
