"""Packed binary pool store — compact, streamable, parallel-write level storage (B3/B4).

Millions of per-level ``.pkl`` files do not scale (inode pressure, O(N) directory
walks, slow ``refresh``). This module stores a level pool as:

* **binary shards** ``shard_<writer>_<k>.bin`` — raw ``map_data`` byte arrays
  concatenated back-to-back (``map_data`` *is* bytes; no pickle overhead);
* a **JSONL index** ``index_<writer>.jsonl`` — one line per level with the byte
  ``(shard, offset, length)`` locator plus all pool metadata (provenance, free-space
  difficulty, structure descriptor, entity counts, a near-dup signature).

Design points:

* **Embarrassingly parallel writes.** Each worker owns its own ``index_<writer>.jsonl``
  + shard files, so N processes/machines write the same pool dir with **no locking**
  (``docs/SYNTHETIC_LEVELGEN_NEXT_STEPS.md`` B1). The reader unions all index segments.
* **Exact + near dedup (B4).** Exact dedup by content hash; near-dup via a 64-bit
  tile simhash so millions of mutants don't collapse to a few modes (the procgen-noise
  lesson).
* **Drop-in for the RL pool.** :class:`PackedBenchmarkLoader` exposes the
  ``MechanicBenchmarkLoader`` interface, so ``GeneratedLevelSource`` consumes a packed
  pool unchanged.

The reader keeps only the (small) index in memory and ``seek``s the shards on demand,
so a multi-GB pool streams without loading map_data eagerly.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("levelgen.pool_store")

POOL_MANIFEST = "pool_manifest.json"
_INDEX_GLOB = "index_*.jsonl"


# --------------------------------------------------------------------------- hashing
def content_hash(map_data: Sequence[int]) -> str:
    return hashlib.sha1(bytes(map_data)).hexdigest()[:16]


def content_id(map_data: Sequence[int]) -> int:
    """Stable 63-bit integer id (matches ``scripts.levelgen.common.map_hash_int``)."""
    return int(hashlib.sha1(bytes(map_data)).hexdigest(), 16) & ((1 << 63) - 1)


def tile_simhash(tiles: Sequence[int]) -> int:
    """64-bit simhash over 2×2 tile-occupancy patches (near-dup detection).

    Two levels with small Hamming distance between their simhashes have nearly the same
    coarse geometry. Robust to a few tile/entity edits (so a mutant ≠ near-dup of its
    parent only once the edit is structural enough).
    """
    from .structure import GRID_W, GRID_H

    bits = [0] * 64
    for by in range(0, GRID_H - 1, 2):
        for bx in range(0, GRID_W - 1, 2):
            patt = 0
            for k, (dx, dy) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
                gx, gy = bx + dx, by + dy
                if gx < GRID_W and gy < GRID_H and tiles[gx + gy * GRID_W] != 0:
                    patt |= 1 << k
            feat = (bx << 12) | (by << 4) | patt
            h = (feat * 0x9E3779B1) & 0xFFFFFFFFFFFFFFFF  # cheap 64-bit mix
            h ^= h >> 31
            for b in range(64):
                bits[b] += 1 if (h >> b) & 1 else -1
    sig = 0
    for b in range(64):
        if bits[b] > 0:
            sig |= 1 << b
    return sig


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------- writer
class PoolWriter:
    """Append levels to a packed pool. One writer per process (lock-free parallel)."""

    def __init__(
        self,
        pool_dir: str | Path,
        writer_id: str = "0",
        *,
        shard_size: int = 4000,
        dedup: bool = True,
    ) -> None:
        self.dir = Path(pool_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.writer_id = str(writer_id)
        self.shard_size = int(shard_size)
        self.dedup = dedup
        self._seen: set[str] = set()
        self._n_in_shard = 0
        self._shard_k = 0
        self._bin = None
        self._idx = open(self.dir / f"index_{self.writer_id}.jsonl", "a", encoding="utf-8")
        self._open_shard()
        self.n_written = 0
        self.n_dup = 0
        self._ensure_manifest()

    def _ensure_manifest(self) -> None:
        mp = self.dir / POOL_MANIFEST
        if not mp.is_file():
            mp.write_text(json.dumps({"version": 1, "schema": "packed-pool-v1"}, indent=2))

    def _shard_name(self) -> str:
        return f"shard_{self.writer_id}_{self._shard_k:05d}.bin"

    def _open_shard(self) -> None:
        if self._bin is not None:
            self._bin.close()
        self._cur_shard = self._shard_name()
        path = self.dir / self._cur_shard
        # append so a re-run of the same writer_id continues its shard cleanly
        self._bin = open(path, "ab")
        self._offset = path.stat().st_size
        self._n_in_shard = 0

    def add(
        self,
        map_data: Sequence[int],
        *,
        bucket: str = "GENERATED",
        tier: int = 0,
        source: str = "generated",
        parent: Optional[int] = None,
        fs_cost: Optional[float] = None,
        difficulty_band: str = "unknown",
        structure_score: Optional[float] = None,
        mechanic: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """Write one level; returns its ``level_id`` or ``None`` if a duplicate.

        Computes structure score / entity counts / simhash if not supplied. Validates
        bytes are 0..255 (catches a stray ninja-orientation ``-1`` from an ungated map).
        """
        md = list(map_data)
        if any(not (0 <= b <= 255) for b in md):
            raise ValueError("map_data has bytes outside 0..255 (gate the level first)")
        h = content_hash(md)
        if self.dedup and h in self._seen:
            self.n_dup += 1
            return None
        self._seen.add(h)

        from .labeling import entity_counts
        from .structure import describe_map_data, structure_bin
        from .tokenizer import TILE_START, TILE_END

        if structure_score is None:
            structure_score = describe_map_data(md).structure_score
        counts = entity_counts(md)
        n_mines = counts.get(1, 0) + counts.get(21, 0)
        n_gold = counts.get(2, 0)
        sig = tile_simhash(md[TILE_START:TILE_END])

        if self._n_in_shard >= self.shard_size:
            self._shard_k += 1
            self._open_shard()

        blob = bytes(md)
        off = self._offset
        self._bin.write(blob)
        self._offset += len(blob)
        self._n_in_shard += 1

        lid = content_id(md)
        rec = {
            "id": lid,
            "h": h,
            "shard": self._cur_shard,
            "off": off,
            "len": len(blob),
            "bucket": bucket,
            "tier": int(tier),
            "source": source,
            "parent": parent,
            "fs_cost": (None if fs_cost is None else float(fs_cost)),
            "band": difficulty_band,
            "struct": round(float(structure_score), 4),
            "struct_bin": structure_bin(float(structure_score)),
            "mech": mechanic,
            "n_mines": int(n_mines),
            "n_gold": int(n_gold),
            "sig": sig,
        }
        if extra:
            rec.update(extra)
        self._idx.write(json.dumps(rec) + "\n")
        self.n_written += 1
        return lid

    def flush(self) -> None:
        if self._bin:
            self._bin.flush()
        self._idx.flush()

    def close(self) -> None:
        self.flush()
        if self._bin:
            self._bin.close()
            self._bin = None
        self._idx.close()

    def __enter__(self) -> "PoolWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- reader
class PackedPool:
    """Read-only view over a packed pool dir; unions all writer index segments."""

    def __init__(
        self,
        pool_dir: str | Path,
        *,
        dedup: bool = True,
        near_dup_bits: Optional[int] = None,
    ) -> None:
        self.dir = Path(pool_dir)
        self._records: List[Dict[str, Any]] = []
        self._by_id: Dict[int, Dict[str, Any]] = {}
        self._fh: Dict[str, Any] = {}
        self._load(dedup=dedup, near_dup_bits=near_dup_bits)

    def _load(self, *, dedup: bool, near_dup_bits: Optional[int]) -> None:
        seen_hash: set[str] = set()
        sigs: List[int] = []
        for idx_path in sorted(self.dir.glob(_INDEX_GLOB)):
            with open(idx_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if dedup and rec["h"] in seen_hash:
                        continue
                    if near_dup_bits is not None:
                        s = rec.get("sig", 0)
                        if any(hamming(s, t) <= near_dup_bits for t in sigs):
                            continue
                        sigs.append(s)
                    seen_hash.add(rec["h"])
                    self._records.append(rec)
                    self._by_id[int(rec["id"])] = rec

    def __len__(self) -> int:
        return len(self._records)

    def level_ids(self) -> List[int]:
        return [int(r["id"]) for r in self._records]

    def records(self) -> List[Dict[str, Any]]:
        return list(self._records)

    def record(self, level_id: int) -> Dict[str, Any]:
        return self._by_id[int(level_id)]

    def _read_blob(self, rec: Dict[str, Any]) -> bytes:
        shard = rec["shard"]
        fh = self._fh.get(shard)
        if fh is None:
            fh = open(self.dir / shard, "rb")
            self._fh[shard] = fh
        fh.seek(rec["off"])
        return fh.read(rec["len"])

    def map_data(self, level_id: int) -> List[int]:
        return list(self._read_blob(self.record(level_id)))

    def map_data_at(self, index: int) -> List[int]:
        return list(self._read_blob(self._records[index]))

    def stats(self) -> Dict[str, Any]:
        from collections import Counter

        srcs = Counter(r.get("source") for r in self._records)
        bands = Counter(r.get("band") for r in self._records)
        sbins = Counter(r.get("struct_bin") for r in self._records)
        return {
            "n_levels": len(self._records),
            "by_source": dict(srcs),
            "by_band": dict(bands),
            "by_struct_bin": dict(sbins),
            "n_shards": len({r["shard"] for r in self._records}),
        }

    def close(self) -> None:
        for fh in self._fh.values():
            fh.close()
        self._fh.clear()


def is_packed_pool(path: str | Path) -> bool:
    return (Path(path) / POOL_MANIFEST).is_file()


# ----------------------------------------------- MechanicBenchmarkLoader-compatible
class PackedBenchmarkLoader:
    """Drop-in for ``MechanicBenchmarkLoader`` backed by a :class:`PackedPool`.

    Exposes the subset of the interface ``GeneratedLevelSource`` / training tooling
    uses: ``__len__``, ``get_level_ids``, ``load_payload(index)``, ``get_map_data``,
    ``bucket_name_for_index``. Index order is the pool's record order.
    """

    def __init__(self, dataset_root: str | Path, splits: Sequence[str] = ("train",)) -> None:
        self.root = Path(dataset_root)
        self.splits = tuple(splits)
        self._pool = PackedPool(self.root)
        self._records = self._pool.records()

    def __len__(self) -> int:
        return len(self._records)

    def get_level_ids(self) -> List[int]:
        return [int(r["id"]) for r in self._records]

    def get_map_data(self, index: int) -> List[int]:
        return self._pool.map_data_at(index)

    def bucket_name_for_index(self, index: int) -> str:
        return str(self._records[index].get("bucket", "GENERATED"))

    def load_payload(self, index: int) -> Dict[str, Any]:
        rec = self._records[index]
        return {
            "level_id": int(rec["id"]),
            "map_data": self._pool.map_data_at(index),
            "bucket": rec.get("bucket", "GENERATED"),
            "tier": int(rec.get("tier", 0)),
            "generator": rec.get("source", "generated"),
            "preset": f"{rec.get('mech', 'any')}_{rec.get('band', 'unknown')}",
            "metadata": {
                "source": rec.get("source", "generated"),
                "parent": rec.get("parent"),
                "fs_cost": rec.get("fs_cost"),
                "difficulty_band": rec.get("band", "unknown"),
                "structure_score": rec.get("struct"),
                "mechanic": rec.get("mech"),
                "n_mines": rec.get("n_mines"),
                "n_gold": rec.get("n_gold"),
            },
        }

    def close(self) -> None:
        self._pool.close()


def merge_pool(pool_dir: str | Path, *, near_dup_bits: Optional[int] = None) -> Dict[str, Any]:
    """Compact all index segments into one deduped ``index_merged.jsonl`` (B4).

    Reads with exact (+ optional near-dup) dedup, rewrites a single merged index, and
    removes the per-writer segments. Shards are left in place (records still point into
    them). Returns a summary. Run offline between generation rounds.
    """
    pool = PackedPool(pool_dir, dedup=True, near_dup_bits=near_dup_bits)
    kept = pool.records()
    out = Path(pool_dir) / "index_merged.jsonl.tmp"
    with open(out, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")
    # remove old segments, then move merged into place
    removed = 0
    for idx_path in Path(pool_dir).glob(_INDEX_GLOB):
        idx_path.unlink()
        removed += 1
    out.rename(Path(pool_dir) / "index_0.jsonl")
    pool.close()
    return {"kept": len(kept), "removed_segments": removed}


__all__ = [
    "PoolWriter",
    "PackedPool",
    "PackedBenchmarkLoader",
    "is_packed_pool",
    "merge_pool",
    "content_hash",
    "content_id",
    "tile_simhash",
    "hamming",
    "POOL_MANIFEST",
]
