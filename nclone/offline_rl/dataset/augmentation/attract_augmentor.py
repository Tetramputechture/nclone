"""Core augmentor: decode → augment → encode → save.

AttractAugmentor reads an npp_attract file (or a .replay file), applies one
or more BaseAugmentation transforms, then writes the result as a .replay file
that TransitionExtractor can consume without modification.

Map-data byte layout (used to reconstruct augmented map_data)
-------------------------------------------------------------
Offset   Length  Content
0        184     Header (preserved verbatim from original)
184      966     Tile data: 42*23 bytes, index = x + y*42
1150     80      Gap / entity-count block
1150     2       Entity count slot 0 (type-1 mines)
1152     2       Entity count slot 1 (type-3 exits)
1154     2       Entity count slot 2 (type-2 gold)
1156     2       Entity count slot 3 (type-4 exit-switches)
1158     2       Unknown (always 1)
...      ...     Reserved / unknown
1231     1       Ninja spawn x (map-data units)
1232     1       Ninja spawn y (map-data units)
1233     ...     Remaining entity-section bytes (padding, then entities at 1235)
1235     5*N     Entities (type, x, y, orientation, mode) – terminated by type==0
"""

from __future__ import annotations

import logging
import struct
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

from ....replay.npp_attract_decoder import NppAttractDecoder
from ....replay.gameplay_recorder import CompactReplay
from .base_augmentation import AttractData, BaseAugmentation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Map-data layout constants
# ---------------------------------------------------------------------------

_HEADER_OFFSET: int = 0
_HEADER_LEN: int = 184

_TILES_OFFSET: int = 184
_TILES_LEN: int = 966  # 42 * 23

# Entity-section starts immediately after tiles.
_ENTITY_SECTION_OFFSET: int = _TILES_OFFSET + _TILES_LEN  # 1150

# Within the entity section (relative to _ENTITY_SECTION_OFFSET):
_REL_COUNT_MINE: int = 0    # type-1 mines
_REL_COUNT_EXIT: int = 2    # type-3 exits
_REL_COUNT_GOLD: int = 4    # type-2 gold
_REL_COUNT_ESWITCH: int = 6 # type-4 exit-switches
_REL_UNKNOWN: int = 8       # always 1

_SPAWN_X_OFFSET: int = 1231  # absolute byte offset in map_data
_SPAWN_Y_OFFSET: int = 1232

_ENTITIES_START_OFFSET: int = 1235  # absolute byte offset in map_data
_ENTITY_BYTES: int = 5              # type, x, y, orientation, mode

# Minimum entity section size (85 bytes to reach first entity slot).
_MIN_ENTITY_SECTION: int = 185


# ---------------------------------------------------------------------------
# Augmentor
# ---------------------------------------------------------------------------


class AttractAugmentor:
    """Decode → augment → encode pipeline for npp_attract files.

    Args:
        augmentations: Ordered list of BaseAugmentation instances to apply.
    """

    def __init__(self, augmentations: Sequence[BaseAugmentation]) -> None:
        if not augmentations:
            raise ValueError("At least one augmentation must be provided.")
        self._augmentations = list(augmentations)
        self._decoder = NppAttractDecoder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def augment_file(
        self,
        source_path: str,
        output_dir: str,
        *,
        overwrite: bool = False,
    ) -> List[Path]:
        """Augment a single source file and write results to *output_dir*.

        Each augmentation produces exactly one output file whose name is
        ``<stem>_<augmentation.name>.replay``.

        Args:
            source_path: Path to an npp_attract file (no extension) or a
                         .replay file.
            output_dir:  Directory in which to write augmented .replay files.
            overwrite:   If False (default), skip files that already exist.

        Returns:
            List of Paths to successfully written output files (may be empty
            if all outputs already existed and *overwrite* is False).
        """
        src = Path(source_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        data = self._decode(src)
        if data is None:
            return []

        written: List[Path] = []
        for aug in self._augmentations:
            out_path = out_dir / f"{src.stem}_{aug.name}.replay"
            if out_path.exists() and not overwrite:
                logger.debug("Skipping existing file: %s", out_path)
                written.append(out_path)
                continue

            try:
                aug_data = aug.augment(data)
            except Exception:
                logger.exception(
                    "Augmentation %s failed for %s", aug.name, src
                )
                continue

            map_bytes = _build_map_data(aug_data)
            if map_bytes is None:
                logger.warning(
                    "Could not build map_data for %s/%s – skipping",
                    src.stem,
                    aug.name,
                )
                continue

            replay = CompactReplay(
                episode_id=out_path.stem,
                map_data=map_bytes,
                input_sequence=aug_data["demo_data"],
                level_id=None,
                start_time=datetime.now(),
                end_time=datetime.now(),
                success=True,
            )
            out_path.write_bytes(replay.to_binary())
            logger.debug("Wrote %s", out_path)
            written.append(out_path)

        return written

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode(self, src: Path) -> Optional[AttractData]:
        """Decode src to AttractData, returning None on failure."""
        try:
            if src.suffix == ".replay":
                data = src.read_bytes()
                replay = CompactReplay.from_binary(data, episode_id=src.stem)
                if not replay.success:
                    logger.warning(
                        "%s is not a successful replay – skipping", src
                    )
                    return None
                map_bytes = bytes(replay.map_data)
                return {
                    "tiles": list(
                        map_bytes[_TILES_OFFSET : _TILES_OFFSET + _TILES_LEN]
                    ),
                    "entities": _parse_entities_from_map_bytes(map_bytes),
                    "ninja_spawn": (
                        map_bytes[_SPAWN_X_OFFSET],
                        map_bytes[_SPAWN_Y_OFFSET],
                    ),
                    "header": map_bytes[:_HEADER_LEN],
                    "demo_data": replay.input_sequence,
                }
            else:
                # npp_attract format (no extension)
                decoded = self._decoder.decode_npp_attract_file(str(src))
                if not decoded.get("demo_data"):
                    logger.warning(
                        "No demo data in %s – skipping", src
                    )
                    return None
                return decoded
        except Exception:
            logger.exception("Failed to decode %s", src)
            return None


# ---------------------------------------------------------------------------
# Map-data construction
# ---------------------------------------------------------------------------


def _build_map_data(data: AttractData) -> Optional[bytes]:
    """Construct a map_data byte string from decoded (possibly augmented) data.

    The resulting bytes match the layout expected by nclone_native.Simulator
    and the TransitionExtractor.

    Args:
        data: AttractData with keys tiles, entities, ninja_spawn, header.

    Returns:
        Bytes suitable for passing to CompactReplay, or None on error.
    """
    try:
        header: bytes = bytes(data["header"])
        tiles: List[int] = data["tiles"]
        entities: List[dict] = data["entities"]
        spawn_x, spawn_y = data["ninja_spawn"]

        if len(tiles) != _TILES_LEN:
            raise ValueError(
                f"Expected {_TILES_LEN} tiles, got {len(tiles)}"
            )
        if len(header) != _HEADER_LEN:
            raise ValueError(
                f"Expected {_HEADER_LEN} header bytes, got {len(header)}"
            )

        # Entity section: needs to cover at least up to the first entity slot
        # (5 bytes per entity starting at relative offset 85).
        num_entities = len(entities)
        entity_data_size = num_entities * _ENTITY_BYTES
        # Relative offset 85 = _ENTITIES_START_OFFSET - _ENTITY_SECTION_OFFSET
        entity_rel_start = _ENTITIES_START_OFFSET - _ENTITY_SECTION_OFFSET  # 85
        section_size = max(
            _MIN_ENTITY_SECTION,
            entity_rel_start + entity_data_size + _ENTITY_BYTES,  # +5 for terminator
        )
        entity_section = bytearray(section_size)

        # Entity type counts.
        counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for ent in entities:
            et = ent["type"]
            if et in counts:
                counts[et] += 1
            elif et == 21:  # TOGGLE_MINE_TOGGLED counts as type-1
                counts[1] += 1

        entity_section[_REL_COUNT_MINE] = counts[1]
        entity_section[_REL_COUNT_EXIT] = counts[3]
        entity_section[_REL_COUNT_GOLD] = counts[2]
        entity_section[_REL_COUNT_ESWITCH] = counts[4]
        entity_section[_REL_UNKNOWN] = 1

        # Ninja spawn (relative offsets 81, 82).
        rel_spawn_x = _SPAWN_X_OFFSET - _ENTITY_SECTION_OFFSET  # 81
        rel_spawn_y = _SPAWN_Y_OFFSET - _ENTITY_SECTION_OFFSET  # 82
        entity_section[rel_spawn_x] = int(spawn_x) & 0xFF
        entity_section[rel_spawn_y] = int(spawn_y) & 0xFF

        # Entity records (relative offset 85 onwards).
        idx = entity_rel_start
        for ent in entities:
            if idx + _ENTITY_BYTES - 1 >= section_size:
                logger.warning(
                    "_build_map_data: entity section overflow at idx=%d", idx
                )
                break
            entity_section[idx + 0] = int(ent["type"]) & 0xFF
            entity_section[idx + 1] = int(ent["x"]) & 0xFF
            entity_section[idx + 2] = int(ent["y"]) & 0xFF
            entity_section[idx + 3] = int(ent.get("orientation", 0)) & 0xFF
            entity_section[idx + 4] = int(ent.get("mode", 0)) & 0xFF
            idx += _ENTITY_BYTES

        return header + bytes(tiles) + bytes(entity_section)

    except Exception:
        logger.exception("_build_map_data failed")
        return None


def _parse_entities_from_map_bytes(map_bytes: bytes) -> List[dict]:
    """Parse entity records from raw map_data bytes.

    Mirrors the logic in NppAttractDecoder._parse_entities but operates on
    already-assembled map_data (used when the source is a .replay file).
    """
    entities: List[dict] = []
    idx = _ENTITIES_START_OFFSET
    while idx + _ENTITY_BYTES - 1 < len(map_bytes):
        entity_type = map_bytes[idx]
        if entity_type == 0:
            break
        entities.append(
            {
                "type": entity_type,
                "x": map_bytes[idx + 1],
                "y": map_bytes[idx + 2],
                "orientation": map_bytes[idx + 3],
                "mode": map_bytes[idx + 4],
            }
        )
        idx += _ENTITY_BYTES
    return entities
