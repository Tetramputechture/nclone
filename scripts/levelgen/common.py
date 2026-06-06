"""Shared helpers for the level-generator pipeline scripts.

Defines the on-disk shard format used between ingest -> label -> build_dataset.
A shard is a pickle holding ``{"records": [LevelRecord, ...]}``; a ``records.json``
index at the corpus root lists shard files + per-level metadata (no map_data) for
quick scanning.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class LevelRecord:
    """One ingested level. ``map_data`` is a list[int] (single bytes 0..255)."""

    level_id: str  # stable hash of map_data
    source: str  # human | official | procgen | generated
    map_data: List[int]
    demo_inputs: List[int] = field(default_factory=list)  # winning replay inputs (0..7)
    demo_len: int = 0
    src_path: str = ""
    labels: Dict[str, object] = field(default_factory=dict)


def map_hash(map_data: List[int]) -> str:
    """Stable content hash of a map (ignores demo)."""
    return hashlib.sha1(bytes(map_data)).hexdigest()[:16]


def map_hash_int(map_data: List[int]) -> int:
    """Stable 63-bit integer content hash (benchmark level_id convention)."""
    return int(hashlib.sha1(bytes(map_data)).hexdigest(), 16) & ((1 << 63) - 1)


def write_shards(
    records: List[LevelRecord], out_dir: Path, shard_size: int = 2000, prefix: str = "shard"
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for i in range(0, len(records), shard_size):
        chunk = records[i : i + shard_size]
        name = f"{prefix}_{i // shard_size:04d}.pkl"
        with open(out_dir / name, "wb") as f:
            pickle.dump({"records": [asdict(r) for r in chunk]}, f, protocol=4)
        paths.append(name)
    return paths


def iter_records(corpus_dir: Path, shard_paths: Optional[List[str]] = None):
    """Yield LevelRecord from every shard under ``corpus_dir``."""
    corpus_dir = Path(corpus_dir)
    if shard_paths is None:
        shard_paths = sorted(p.name for p in corpus_dir.glob("shard_*.pkl"))
    for name in shard_paths:
        with open(corpus_dir / name, "rb") as f:
            blob = pickle.load(f)
        for d in blob["records"]:
            yield LevelRecord(**d)
