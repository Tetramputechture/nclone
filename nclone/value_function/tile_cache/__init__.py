"""Persistent 3x3 tile-neighborhood cache for edge enumeration.

The tile cache accelerates per-level precomputation by memoizing the
tile-dependent portions of every arc — the launch phase and the landing
phase — keyed by the 3x3 tile signature at the source / impact cell plus
the relevant launch/entry quantization.

Only the locally tile-dependent physics is cached.  The tile-independent
ballistic flight through open space is computed analytically by
``ballistic.propagate_and_sweep`` on-the-fly; this keeps the cache size
bounded and avoids a combinatorial explosion for long open jumps.

Subpackage layout:
    - ``signature``  — 3x3 extraction + 64-bit hashing.
    - ``harness``    — tiny 5x5 maps used as isolated sim harnesses.
    - ``store``      — sharded ``.npz`` on-disk store with version headers.
    - ``builder``    — (added later) populates the cache via harness sims.
    - ``lookup``     — (added later) hot in-memory lookup path.
    - ``ballistic``  — (added later) analytical ballistic propagator.

The cache is versioned on every shard via ``physics_version``,
``bin_schedule_version``, ``primitive_set_version``, and
``harness_variant_version`` so schedule / physics changes can invalidate
old shards without deleting them.
"""

from .signature import (
    SIG_HASH_DTYPE,
    extract_3x3_signature,
    hash_signature,
    pack_signature,
    unpack_signature,
)
from .harness import (
    HARNESS_SIZE,
    HARNESS_WINDOW,
    HARNESS_BUFFER,
    HARNESS_CENTER_COL,
    HARNESS_CENTER_ROW,
    build_launch_harness_map,
    build_landing_harness_map,
    launch_cell_origin_px,
)
from .store import (
    PHYSICS_VERSION,
    BIN_SCHEDULE_VERSION,
    PRIMITIVE_SET_VERSION,
    HARNESS_VARIANT_VERSION,
    LaunchEntry,
    LandingEntry,
    ShardHeader,
    TileCacheStore,
)
from .ballistic import (
    BallisticResult,
    Died,
    Landed,
    Timeout,
    propagate_and_sweep,
    tile_dict_from_map_data,
    tile_dict_is_solid,
)
from .lookup import landing_lookup, launch_lookup
from .builder import (
    AIR_VX_BIN_CENTERS,
    AIR_VY_BIN_CENTERS,
    H_INPUTS,
    InputClass,
    JUMP_HOLD_DURATIONS,
    LaunchSimResult,
    MAX_HARNESS_FRAMES,
    POST_RELEASE_INPUTS,
    RUN_DURATIONS,
    SUBCELL_BINS,
    classes_for_surface,
    decode_input_class,
    populate_landing_entry,
    populate_launch_entry,
    quantise_air_velocity,
    quantise_subcell,
    subcell_centre,
    sweep_launch_signature,
)

__all__ = [
    "SIG_HASH_DTYPE",
    "extract_3x3_signature",
    "hash_signature",
    "pack_signature",
    "unpack_signature",
    "HARNESS_SIZE",
    "HARNESS_WINDOW",
    "HARNESS_BUFFER",
    "HARNESS_CENTER_COL",
    "HARNESS_CENTER_ROW",
    "build_launch_harness_map",
    "build_landing_harness_map",
    "launch_cell_origin_px",
    "PHYSICS_VERSION",
    "BIN_SCHEDULE_VERSION",
    "PRIMITIVE_SET_VERSION",
    "HARNESS_VARIANT_VERSION",
    "LaunchEntry",
    "LandingEntry",
    "ShardHeader",
    "TileCacheStore",
    "BallisticResult",
    "Died",
    "Landed",
    "Timeout",
    "propagate_and_sweep",
    "tile_dict_from_map_data",
    "tile_dict_is_solid",
    "landing_lookup",
    "launch_lookup",
    "AIR_VX_BIN_CENTERS",
    "AIR_VY_BIN_CENTERS",
    "H_INPUTS",
    "InputClass",
    "JUMP_HOLD_DURATIONS",
    "LaunchSimResult",
    "MAX_HARNESS_FRAMES",
    "POST_RELEASE_INPUTS",
    "RUN_DURATIONS",
    "SUBCELL_BINS",
    "classes_for_surface",
    "decode_input_class",
    "populate_landing_entry",
    "populate_launch_entry",
    "quantise_air_velocity",
    "quantise_subcell",
    "subcell_centre",
    "sweep_launch_signature",
]
