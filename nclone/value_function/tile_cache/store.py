"""Sharded on-disk persistent store for the tile cache.

The store is backed by compressed ``.npz`` files, one per shard.  Each
shard holds all entries whose 3x3 signature hash falls in its shard
bucket (see :func:`signature.shard_id_from_hash`).  Entries come in two
flavours — launch and landing — and share a single shard per hash so
disk and memory locality are preserved when looking up back-to-back
launch and landing queries for the same signature.

Shard file format
-----------------

A shard is an ``.npz`` containing::

    # ---- header (scalar arrays) ----
    header_physics_version       : uint32
    header_bin_schedule_version  : uint32
    header_primitive_set_version : uint32
    header_harness_variant_version : uint32
    header_shard_id              : uint32
    header_num_shards            : uint32
    header_format_version        : uint32

    # ---- launch entries ----
    launch_keys                  : uint64 [N_L]  — packed signature hash
    launch_key_extra             : uint32 [N_L, 5]
        # columns: (surface_type, start_subcell_x, start_subcell_y,
        #          vel_bin_in, input_class)
    launch_values                : int16 [N_L, 12]  — see ``LaunchEntry``

    # ---- landing entries ----
    landing_keys                 : uint64 [N_D]
    landing_key_extra            : uint32 [N_D, 4]
        # columns: (entry_subcell_x, entry_subcell_y,
        #          entry_vx_bin_air, entry_vy_bin_air)
    landing_values               : int16 [N_D, 6]  — see ``LandingEntry``

Version fields on the header are compared on load; mismatched shards
are ignored (NOT deleted) so old caches can be kept around for
comparison studies.

Thread / process safety
-----------------------
Writes use an advisory ``fcntl`` lock file (``<shard>.lock``) so
multiple builder processes can populate the cache concurrently with
read-modify-write semantics.  Readers open the shard read-only.

The store instances are designed to be used as either:
  * A long-lived singleton on the main process (lazy shard load,
    write-back on flush) for cold-cache level precomputes.
  * A per-worker read-through cache for MP-parallel edge enumeration,
    flushed back to disk between phases.
"""

from __future__ import annotations

import os
import logging
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    import fcntl  # POSIX only
    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None
    _HAVE_FCNTL = False

from .signature import shard_id_from_hash

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Versioning — bump these when the corresponding invariant changes.
# ---------------------------------------------------------------------------
#: Native simulator physics version.  Bump on any change to physics
#: constants, integration order, or collision resolution.
PHYSICS_VERSION: int = 1
#: Velocity-bin schedule version.  Pulled dynamically from
#: ``surface_graph.BIN_SCHEDULE_VERSION`` so that editing the schedule in
#: one place invalidates stale caches automatically.
from ..surface_graph import BIN_SCHEDULE_VERSION as _SG_BIN_SCHEDULE_VERSION  # noqa: E402
BIN_SCHEDULE_VERSION: int = int(_SG_BIN_SCHEDULE_VERSION)
#: Primitive-sweep version — bump when ``RUN_DURATIONS`` /
#: ``JUMP_HOLD_DURATIONS`` / ``H_INPUTS`` lists change.
PRIMITIVE_SET_VERSION: int = 1
#: Harness variant version — bump when the 5x5 harness layout or spawn
#: convention changes.
HARNESS_VARIANT_VERSION: int = 1
#: Shard file format version (structural changes only).
_FORMAT_VERSION: int = 1
#: Total number of shards.  Sharding key is the top 7 bits of the
#: signature hash (SE-corner tile), so 128 shards distributes evenly.
DEFAULT_NUM_SHARDS: int = 128


# ---------------------------------------------------------------------------
# Dataclasses for one launch / landing record
# ---------------------------------------------------------------------------

@dataclass
class LaunchEntry:
    """One row of the launch cache.

    Encodes either (a) a full in-window edge terminating at a surface
    inside the 3x3 window (``escaped=0``) or (b) a handoff state at the
    moment the ninja's bounding box exited the window (``escaped=1``),
    after which the ballistic propagator takes over.
    """
    escaped: int        # 0 = stayed in window, 1 = exited window
    died: int           # 1 if ninja died during launch phase
    frames: int         # frames used in launch phase

    # Valid when escaped == 0 (full in-window landing)
    land_cell_dx: int = 0    # -1..+1 cell offset inside 3x3
    land_cell_dy: int = 0
    land_subcell_x: int = 0  # 0..23 within the landing cell (1 px granularity)
    land_subcell_y: int = 0
    land_surface_type: int = 0
    land_vel_bin: int = 0

    # Valid when escaped == 1 (airborne exit)
    exit_pos_dx: int = 0     # subpixel relative to launch pos (quantised to 1 px)
    exit_pos_dy: int = 0
    exit_vx_q: int = 0       # velocity quantised to 0.01 px/frame
    exit_vy_q: int = 0
    hold_remaining: int = 0  # residual jump-hold frames


@dataclass
class LandingEntry:
    """One row of the landing cache.

    Maps an airborne entry state (at the moment the ballistic sweep
    first hits a solid tile cell) to the resolved surface node.
    """
    landing_cell_dx: int       # -1..+1 cell offset inside 3x3 of the sig
    landing_cell_dy: int
    landing_subcell_x: int     # 0..23 within the landing cell
    landing_subcell_y: int
    landing_surface_type: int  # SURFACE_FLOOR / SURFACE_WALL (/ SLIDE / LAUNCH_PAD)
    landing_vel_bin: int       # index into the non-uniform schedule for that surface
    contact_frame_offset: int  # extra frames burned during collision resolution


# Record widths in int cells (must match the encode/decode below).
_LAUNCH_VALUE_COLS: int = 14
_LAUNCH_KEY_EXTRA_COLS: int = 5
_LANDING_VALUE_COLS: int = 7
_LANDING_KEY_EXTRA_COLS: int = 4


def _pack_launch_value(e: LaunchEntry) -> np.ndarray:
    return np.array([
        e.escaped, e.died, e.frames,
        e.land_cell_dx, e.land_cell_dy,
        e.land_subcell_x, e.land_subcell_y,
        e.land_surface_type, e.land_vel_bin,
        e.exit_pos_dx, e.exit_pos_dy,
        e.exit_vx_q, e.exit_vy_q,
        e.hold_remaining,
    ], dtype=np.int32)


def _unpack_launch_value(row: np.ndarray) -> LaunchEntry:
    return LaunchEntry(
        escaped=int(row[0]),
        died=int(row[1]),
        frames=int(row[2]),
        land_cell_dx=int(row[3]),
        land_cell_dy=int(row[4]),
        land_subcell_x=int(row[5]),
        land_subcell_y=int(row[6]),
        land_surface_type=int(row[7]),
        land_vel_bin=int(row[8]),
        exit_pos_dx=int(row[9]),
        exit_pos_dy=int(row[10]),
        exit_vx_q=int(row[11]),
        exit_vy_q=int(row[12]),
        hold_remaining=int(row[13]) if len(row) > 13 else 0,
    )


def _pack_landing_value(e: LandingEntry) -> np.ndarray:
    return np.array([
        e.landing_cell_dx, e.landing_cell_dy,
        e.landing_subcell_x, e.landing_subcell_y,
        e.landing_surface_type, e.landing_vel_bin,
        e.contact_frame_offset,
    ], dtype=np.int32)


def _unpack_landing_value(row: np.ndarray) -> LandingEntry:
    return LandingEntry(
        landing_cell_dx=int(row[0]),
        landing_cell_dy=int(row[1]),
        landing_subcell_x=int(row[2]),
        landing_subcell_y=int(row[3]),
        landing_surface_type=int(row[4]),
        landing_vel_bin=int(row[5]),
        contact_frame_offset=int(row[6]),
    )


# ---------------------------------------------------------------------------
# Shard header / on-disk IO
# ---------------------------------------------------------------------------

@dataclass
class ShardHeader:
    """Metadata baked into each shard file."""
    physics_version: int = PHYSICS_VERSION
    bin_schedule_version: int = BIN_SCHEDULE_VERSION
    primitive_set_version: int = PRIMITIVE_SET_VERSION
    harness_variant_version: int = HARNESS_VARIANT_VERSION
    shard_id: int = 0
    num_shards: int = DEFAULT_NUM_SHARDS
    format_version: int = _FORMAT_VERSION

    def matches(self, other: "ShardHeader") -> bool:
        """True if two headers are compatible for merge / lookup."""
        return (
            self.physics_version == other.physics_version
            and self.bin_schedule_version == other.bin_schedule_version
            and self.primitive_set_version == other.primitive_set_version
            and self.harness_variant_version == other.harness_variant_version
            and self.format_version == other.format_version
        )

    def to_arrays(self) -> Dict[str, np.ndarray]:
        return {
            "header_physics_version": np.asarray(self.physics_version, np.uint32),
            "header_bin_schedule_version": np.asarray(
                self.bin_schedule_version, np.uint32
            ),
            "header_primitive_set_version": np.asarray(
                self.primitive_set_version, np.uint32
            ),
            "header_harness_variant_version": np.asarray(
                self.harness_variant_version, np.uint32
            ),
            "header_shard_id": np.asarray(self.shard_id, np.uint32),
            "header_num_shards": np.asarray(self.num_shards, np.uint32),
            "header_format_version": np.asarray(self.format_version, np.uint32),
        }

    @classmethod
    def from_arrays(cls, arr: Dict[str, np.ndarray]) -> "ShardHeader":
        return cls(
            physics_version=int(arr["header_physics_version"]),
            bin_schedule_version=int(arr["header_bin_schedule_version"]),
            primitive_set_version=int(arr["header_primitive_set_version"]),
            harness_variant_version=int(arr["header_harness_variant_version"]),
            shard_id=int(arr["header_shard_id"]),
            num_shards=int(arr["header_num_shards"]),
            format_version=int(arr["header_format_version"]),
        )


# ---------------------------------------------------------------------------
# In-memory shard representation
# ---------------------------------------------------------------------------

@dataclass
class _Shard:
    """A single shard loaded into memory."""
    header: ShardHeader
    # Launch entries: per-hash bucket of (key_extra_tuple -> value_row)
    launch: Dict[int, Dict[Tuple[int, ...], np.ndarray]] = field(default_factory=dict)
    landing: Dict[int, Dict[Tuple[int, ...], np.ndarray]] = field(default_factory=dict)
    dirty: bool = False

    def n_launch(self) -> int:
        return sum(len(b) for b in self.launch.values())

    def n_landing(self) -> int:
        return sum(len(b) for b in self.landing.values())


# ---------------------------------------------------------------------------
# Main store
# ---------------------------------------------------------------------------

class TileCacheStore:
    """Persistent launch + landing tile cache.

    The store is lazy: shards are loaded into memory on first access
    and flushed to disk on :meth:`flush` (or ``__exit__``).  Writers
    acquire an advisory file lock per shard to coordinate with other
    processes.

    The in-memory representation is optimised for O(1) lookup -- each
    shard maps ``sig_hash -> key_extra -> value_row`` so callers can
    check cache membership with a single dictionary lookup.
    """

    def __init__(
        self,
        cache_dir: os.PathLike,
        num_shards: int = DEFAULT_NUM_SHARDS,
        header_template: Optional[ShardHeader] = None,
        create: bool = True,
    ) -> None:
        self._dir = Path(cache_dir)
        self._num_shards = int(num_shards)
        self._header_template = header_template or ShardHeader(
            num_shards=self._num_shards,
        )
        self._shards: Dict[int, _Shard] = {}
        self._lock = threading.RLock()
        if create:
            self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _shard_path(self, shard_id: int) -> Path:
        return self._dir / f"shard_{shard_id:04d}.npz"

    def _shard_lockpath(self, shard_id: int) -> Path:
        return self._dir / f"shard_{shard_id:04d}.lock"

    # ------------------------------------------------------------------
    # Locking
    # ------------------------------------------------------------------
    @contextmanager
    def _lock_shard(self, shard_id: int) -> Iterator[None]:
        """Acquire an advisory lock for one shard during read-modify-write.

        On POSIX this uses ``fcntl.flock``; on platforms without
        ``fcntl`` it falls back to process-local locking only.
        """
        lock_path = self._shard_lockpath(shard_id)
        if _HAVE_FCNTL:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
        else:  # pragma: no cover
            with self._lock:
                yield

    # ------------------------------------------------------------------
    # Shard IO
    # ------------------------------------------------------------------
    def _load_shard(self, shard_id: int) -> _Shard:
        """Load a shard from disk, or return a fresh empty one."""
        path = self._shard_path(shard_id)
        if not path.exists():
            header = ShardHeader(
                **{**self._header_template.__dict__, "shard_id": shard_id,
                   "num_shards": self._num_shards}
            )
            return _Shard(header=header)

        with np.load(path, allow_pickle=False) as npz:
            try:
                header = ShardHeader.from_arrays(dict(npz.items()))
            except KeyError as err:  # pragma: no cover
                logger.warning("shard %s missing header field %s; discarding",
                               path, err)
                header = ShardHeader(shard_id=shard_id,
                                     num_shards=self._num_shards)
                return _Shard(header=header)

            if not header.matches(self._header_template):
                logger.info(
                    "shard %s header mismatch (physics=%d/%d bins=%d/%d "
                    "prims=%d/%d harness=%d/%d); ignoring",
                    path,
                    header.physics_version, self._header_template.physics_version,
                    header.bin_schedule_version,
                    self._header_template.bin_schedule_version,
                    header.primitive_set_version,
                    self._header_template.primitive_set_version,
                    header.harness_variant_version,
                    self._header_template.harness_variant_version,
                )
                # Return an empty shard stamped with the current header
                header = ShardHeader(
                    **{**self._header_template.__dict__, "shard_id": shard_id,
                       "num_shards": self._num_shards}
                )
                return _Shard(header=header)

            shard = _Shard(header=header)
            launch_keys = npz["launch_keys"]
            launch_key_extra = npz["launch_key_extra"]
            launch_values = npz["launch_values"]
            for i in range(launch_keys.shape[0]):
                h = int(launch_keys[i])
                extra = tuple(int(v) for v in launch_key_extra[i])
                shard.launch.setdefault(h, {})[extra] = launch_values[i].astype(np.int32)

            landing_keys = npz["landing_keys"]
            landing_key_extra = npz["landing_key_extra"]
            landing_values = npz["landing_values"]
            for i in range(landing_keys.shape[0]):
                h = int(landing_keys[i])
                extra = tuple(int(v) for v in landing_key_extra[i])
                shard.landing.setdefault(h, {})[extra] = (
                    landing_values[i].astype(np.int32)
                )
            return shard

    def _save_shard(self, shard: _Shard) -> None:
        """Atomic save of a shard via temp-file + rename."""
        path = self._shard_path(shard.header.shard_id)

        # Flatten dicts into contiguous arrays
        l_keys: List[int] = []
        l_extras: List[Tuple[int, ...]] = []
        l_values: List[np.ndarray] = []
        for h, bucket in shard.launch.items():
            for extra, row in bucket.items():
                l_keys.append(h)
                l_extras.append(extra)
                l_values.append(row)

        d_keys: List[int] = []
        d_extras: List[Tuple[int, ...]] = []
        d_values: List[np.ndarray] = []
        for h, bucket in shard.landing.items():
            for extra, row in bucket.items():
                d_keys.append(h)
                d_extras.append(extra)
                d_values.append(row)

        arrays = dict(shard.header.to_arrays())
        arrays["launch_keys"] = np.asarray(l_keys, dtype=np.uint64)
        arrays["launch_key_extra"] = (
            np.asarray(l_extras, dtype=np.int32).reshape(-1, _LAUNCH_KEY_EXTRA_COLS)
            if l_extras else np.zeros((0, _LAUNCH_KEY_EXTRA_COLS), dtype=np.int32)
        )
        arrays["launch_values"] = (
            np.stack(l_values, axis=0).astype(np.int32)
            if l_values else np.zeros((0, _LAUNCH_VALUE_COLS), dtype=np.int32)
        )
        arrays["landing_keys"] = np.asarray(d_keys, dtype=np.uint64)
        arrays["landing_key_extra"] = (
            np.asarray(d_extras, dtype=np.int32).reshape(-1, _LANDING_KEY_EXTRA_COLS)
            if d_extras else np.zeros((0, _LANDING_KEY_EXTRA_COLS), dtype=np.int32)
        )
        arrays["landing_values"] = (
            np.stack(d_values, axis=0).astype(np.int32)
            if d_values else np.zeros((0, _LANDING_VALUE_COLS), dtype=np.int32)
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=str(path.parent), delete=False, suffix=".tmp"
        ) as tmp:
            np.savez_compressed(tmp, **arrays)
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)
        shard.dirty = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_shard(self, shard_id: int) -> _Shard:
        """Return the in-memory shard, loading it from disk on first use."""
        with self._lock:
            s = self._shards.get(shard_id)
            if s is None:
                s = self._load_shard(shard_id)
                self._shards[shard_id] = s
            return s

    def shard_for_hash(self, sig_hash: int) -> _Shard:
        sid = shard_id_from_hash(int(sig_hash), self._num_shards)
        return self.get_shard(sid)

    # ---- launch ----
    def get_launch(
        self,
        sig_hash: int,
        key_extra: Tuple[int, ...],
    ) -> Optional[LaunchEntry]:
        """Return the cached launch entry or ``None`` on miss."""
        shard = self.shard_for_hash(sig_hash)
        bucket = shard.launch.get(int(sig_hash))
        if bucket is None:
            return None
        row = bucket.get(key_extra)
        if row is None:
            return None
        return _unpack_launch_value(row)

    def put_launch(
        self,
        sig_hash: int,
        key_extra: Tuple[int, ...],
        entry: LaunchEntry,
    ) -> None:
        if len(key_extra) != _LAUNCH_KEY_EXTRA_COLS:
            raise ValueError(
                f"launch key_extra must have {_LAUNCH_KEY_EXTRA_COLS} components"
            )
        shard = self.shard_for_hash(sig_hash)
        shard.launch.setdefault(int(sig_hash), {})[key_extra] = _pack_launch_value(entry)
        shard.dirty = True

    # ---- landing ----
    def get_landing(
        self,
        sig_hash: int,
        key_extra: Tuple[int, ...],
    ) -> Optional[LandingEntry]:
        shard = self.shard_for_hash(sig_hash)
        bucket = shard.landing.get(int(sig_hash))
        if bucket is None:
            return None
        row = bucket.get(key_extra)
        if row is None:
            return None
        return _unpack_landing_value(row)

    def put_landing(
        self,
        sig_hash: int,
        key_extra: Tuple[int, ...],
        entry: LandingEntry,
    ) -> None:
        if len(key_extra) != _LANDING_KEY_EXTRA_COLS:
            raise ValueError(
                f"landing key_extra must have {_LANDING_KEY_EXTRA_COLS} components"
            )
        shard = self.shard_for_hash(sig_hash)
        shard.landing.setdefault(int(sig_hash), {})[key_extra] = _pack_landing_value(
            entry
        )
        shard.dirty = True

    # ------------------------------------------------------------------
    # Bulk ops
    # ------------------------------------------------------------------
    def flush(self) -> None:
        """Persist every dirty shard to disk with per-shard file locking."""
        with self._lock:
            for sid, shard in list(self._shards.items()):
                if not shard.dirty:
                    continue
                with self._lock_shard(sid):
                    # Re-read from disk under lock to merge concurrent writers
                    disk_shard = self._load_shard(sid)
                    if disk_shard.header.matches(shard.header):
                        # Merge: local entries win (they were computed here).
                        for h, bucket in shard.launch.items():
                            disk_bucket = disk_shard.launch.setdefault(h, {})
                            disk_bucket.update(bucket)
                        for h, bucket in shard.landing.items():
                            disk_bucket = disk_shard.landing.setdefault(h, {})
                            disk_bucket.update(bucket)
                    disk_shard.header = shard.header
                    self._save_shard(disk_shard)
                shard.dirty = False

    def close(self) -> None:
        self.flush()
        self._shards.clear()

    def __enter__(self) -> "TileCacheStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, int]:
        """Return total entry counts across all loaded shards."""
        n_launch = sum(s.n_launch() for s in self._shards.values())
        n_landing = sum(s.n_landing() for s in self._shards.values())
        n_loaded = len(self._shards)
        return {
            "loaded_shards": n_loaded,
            "launch_entries": n_launch,
            "landing_entries": n_landing,
        }
