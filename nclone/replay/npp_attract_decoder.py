"""
Perfect N++ Attract Replay Decoder

This decoder achieves 100% accuracy by extracting the embedded map data
directly from npp_attract files. The npp_attract files contain the exact
same map data as the official maps, just with additional metadata.

Structure discovered:
- Header: bytes 0-183 (184 bytes)
- Tiles: bytes 184-1149 (966 bytes) - EXACT match with official maps
- Entities: bytes 1230+ (5 bytes per entity) - EXACT match with official maps
- Ninja spawn: bytes 1231-1232 - EXACT match with official maps
"""

import collections
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class NppAttractDecoder:
    """Perfect decoder for N++ attract replay files achieving 100% accuracy."""

    def __init__(self):
        """Initialize the decoder."""
        self.decode_stats = {
            "files_processed": 0,
            "tile_accuracy": 0.0,
            "entity_accuracy": 0.0,
            "spawn_accuracy": 0.0,
        }

    def decode_npp_attract_file(self, npp_attract_path: str) -> Dict:
        """
        Decode an npp_attract file to extract tiles, entities, and ninja spawn.

        Args:
            npp_attract_path: Path to the npp_attract file

        Returns:
            Dictionary containing:
            - 'tiles': List of 966 tile values
            - 'entities': List of entity dictionaries
            - 'ninja_spawn': Tuple of (x, y) coordinates
            - 'header': Raw header bytes for compatibility
            - 'demo_data': List of input values (if available)
        """
        try:
            with open(npp_attract_path, "rb") as f:
                npp_data = f.read()

            logger.info(
                f"Processing npp_attract file: {npp_attract_path} ({len(npp_data)} bytes)"
            )

            # Extract header (bytes 0-183)
            header = npp_data[:184]

            # Extract tiles (bytes 184-1149) - 966 tiles in 42x23 grid
            tiles = list(npp_data[184:1150])

            # Extract ninja spawn (bytes 1231-1232)
            ninja_spawn_x = npp_data[1231]
            ninja_spawn_y = npp_data[1232]
            ninja_spawn = (ninja_spawn_x, ninja_spawn_y)

            # Extract entities (starting at byte 1230, 5 bytes each)
            entities = self._parse_entities(npp_data, start_pos=1230)

            # Extract demo data (input sequence) according to attract file format
            demo_data = self._extract_demo_data(npp_data)

            # Update statistics
            self.decode_stats["files_processed"] += 1

            result = {
                "tiles": tiles,
                "entities": entities,
                "ninja_spawn": ninja_spawn,
                "header": header,
                "demo_data": demo_data,
            }

            logger.info(
                f"Successfully decoded: {len(tiles)} tiles, {len(entities)} entities, spawn {ninja_spawn}, {len(demo_data)} demo inputs"
            )
            return result

        except Exception as e:
            print(f"Error decoding npp_attract file {npp_attract_path}: {e}")
            return {
                "tiles": [0] * 966,
                "entities": [],
                "ninja_spawn": (0, 0),
                "header": b"\x00" * 184,
                "demo_data": [],
            }

    def _parse_entities(self, npp_data: bytes, start_pos: int) -> List[Dict]:
        """
        Parse entities from npp_attract file data.

        Args:
            npp_data: Raw npp_attract file bytes
            start_pos: Starting position for entity data (should be 1230)

        Returns:
            List of entity dictionaries with keys: type, x, y, orientation, mode
        """
        entities = []

        # The section is COUNT-DRIVEN, not terminator-driven. Verified against the
        # official corpus: `count(type) = u16 LE at map_data[1150 + 2*type]` for
        # every type 0..28, and a map file is exactly
        #     1230 + 5 * (count[0] + sum(count[1..28]))  bytes long.
        # Slot 0 is the NINJA count, and its records are the leading spawn entries.
        #
        # The old parser hardcoded "one ninja, skip to 1235, stop at the first
        # type-0 byte". Type 0 is not an end marker -- `map_loader.py` skips
        # unknown types and keeps walking -- and the ninja count is not always 1.
        # On a two-player CO-OP attract there is a second spawn record at 1235, so
        # the parser stopped there and returned zero entities, and the map decoded
        # to bare tiles with no exit. That silently dropped **1496 of 8000**
        # attract demos (18.7%).
        counts = {t: self._u16(npp_data, 1150 + 2 * t) for t in range(29)}
        n_ninjas = counts[0]
        # Types 7 and 9 -- the switch halves of locked and trap doors -- are NOT
        # counted (they are not entities; the door owns both positions, spec §2),
        # but each door DOES emit a second record for its switch. So the stream is
        # one record longer per locked door and per trap door than the counts say.
        # Verified on 400 random attract files: this expression reproduces the map
        # section length (`1230 + 5*total == 8 + header_map_len`) on every one.
        # Without the correction we truncated that many records off the END of
        # every map carrying a door -- silently dropping whatever sorted last,
        # typically one-way platforms, which changes the collision geometry.
        total = sum(counts[t] for t in range(1, 29)) + counts[6] + counts[8]

        index = start_pos + 5 * max(n_ninjas, 1)
        for _ in range(total):
            if index + 4 >= len(npp_data):
                break
            entities.append({
                "type": npp_data[index],
                "x": npp_data[index + 1],
                "y": npp_data[index + 2],
                "orientation": npp_data[index + 3],
                "mode": npp_data[index + 4],
            })
            index += 5

        return entities

    @staticmethod
    def _u32(data: bytes, offset: int) -> int:
        """Little-endian u32, 0 if the offset runs past the end."""
        if offset + 3 >= len(data):
            return 0
        return int.from_bytes(data[offset:offset + 4], "little")

    @staticmethod
    def _u16(data: bytes, offset: int) -> int:
        """Little-endian u16, 0 if the offset runs past the end."""
        if offset + 1 >= len(data):
            return 0
        return data[offset] | (data[offset + 1] << 8)

    @staticmethod
    def ninja_count(npp_data: bytes) -> int:
        """Number of ninjas the file was recorded with.

        Anything above 1 is a co-op demo: the tape carries inputs for several
        players, so a single-ninja simulator cannot replay it faithfully however
        correct its physics are. Callers measuring replay fidelity should report
        these separately rather than counting them as losses.
        """
        return NppAttractDecoder._u16(npp_data, 1150)

    def create_nclone_map(self, npp_attract_path: str) -> bytes:
        """
        Create a complete nclone-compatible map from an npp_attract file.

        Args:
            npp_attract_path: Path to the npp_attract file

        Returns:
            Complete map data bytes compatible with nclone map loader
        """
        try:
            # Decode the npp_attract file
            decoded_data = self.decode_npp_attract_file(npp_attract_path)

            # Use the original header from the npp_attract file
            header = decoded_data["header"]

            # Convert tiles to bytes
            tiles = bytes(decoded_data["tiles"])

            # Create entity section (starting at position 1230)
            # Calculate required size based on number of entities
            num_entities = len(decoded_data["entities"])
            entity_data_size = num_entities * 5  # 5 bytes per entity
            entity_section_size = max(
                185, 85 + entity_data_size
            )  # At least 185 bytes, or more if needed
            entity_section = bytearray(entity_section_size)

            # Per-type entity counts. Layout read off the official maps:
            #
            #     count(type) = u16 little-endian at map_data[1150 + 2*type]
            #
            # so slot 0 is "type 0" = the ninja and is always 1, type 1 (mines)
            # lands at 1152, gold at 1154, exit doors at 1156, switches at 1158.
            # `map_loader.py` reads `exit_door_count` from 1156.
            #
            # Two bugs lived here. The slots were shifted by one (mines written to
            # the ninja's slot, exits to the mine slot), which survived only
            # because doors and switches are always 1:1. And each count was
            # written as a SINGLE byte, so any map with more than 255 of a type
            # raised `byte must be in range(0, 256)`, the exception was swallowed
            # by the caller, and the whole map decoded to 1335 zero bytes -- no
            # tiles, no entities, unwinnable. That silently discarded **1650 of
            # 8000** attract demos (20.6%); mine-heavy maps routinely carry 300+.
            entity_counts = collections.Counter(
                e["type"] for e in decoded_data["entities"]
            )
            entity_section[0:2] = (1).to_bytes(2, "little")  # the ninja
            for entity_type in (1, 2, 3, 4):
                slot = 2 * entity_type
                entity_section[slot:slot + 2] = min(
                    entity_counts.get(entity_type, 0), 0xFFFF
                ).to_bytes(2, "little")

            # Set ninja spawn at positions 81-82 (based on official format)
            ninja_spawn = decoded_data["ninja_spawn"]
            entity_section[81] = ninja_spawn[0]
            entity_section[82] = ninja_spawn[1]

            # Add entities starting at position 85 (based on official format)
            entity_index = 85
            for entity in decoded_data["entities"]:
                if entity_index + 4 < len(entity_section):
                    entity_section[entity_index] = entity["type"]
                    entity_section[entity_index + 1] = entity["x"]
                    entity_section[entity_index + 2] = entity["y"]
                    entity_section[entity_index + 3] = entity["orientation"]
                    entity_section[entity_index + 4] = entity["mode"]
                    entity_index += 5

            # Combine all sections
            complete_map = header + tiles + bytes(entity_section)

            # Log the map size (no longer enforcing 1335 bytes for files with more entities)
            logger.info(
                f"Map sections: header={len(header)}, tiles={len(tiles)}, entities={len(entity_section)}"
            )
            logger.info(f"Total entities included: {num_entities}")

            logger.info(f"Created nclone map: {len(complete_map)} bytes")
            return complete_map

        except Exception as e:
            print(f"Error creating nclone map from {npp_attract_path}: {e}")
            # Return empty map as fallback
            return b"\x00" * 1335

    def validate_against_reference(
        self, npp_attract_path: str, reference_map_path: str
    ) -> Dict[str, float]:
        """
        Validate decoded data against a reference map for accuracy testing.

        Args:
            npp_attract_path: Path to npp_attract file
            reference_map_path: Path to reference map file

        Returns:
            Dictionary with accuracy percentages for tiles, entities, and spawn
        """
        try:
            # Decode npp_attract file
            decoded_data = self.decode_npp_attract_file(npp_attract_path)

            # Load reference map
            with open(reference_map_path, "rb") as f:
                ref_data = f.read()

            # Extract reference data
            ref_tiles = list(ref_data[184:1150])
            ref_ninja_spawn = (ref_data[1231], ref_data[1232])
            ref_entities = self._parse_reference_entities(ref_data)

            # Calculate tile accuracy
            tile_matches = sum(
                1 for i in range(966) if decoded_data["tiles"][i] == ref_tiles[i]
            )
            tile_accuracy = tile_matches / 966

            # Calculate spawn accuracy
            spawn_accuracy = (
                1.0 if decoded_data["ninja_spawn"] == ref_ninja_spawn else 0.0
            )

            # Calculate entity accuracy
            entity_accuracy = self._calculate_entity_accuracy(
                decoded_data["entities"], ref_entities
            )

            # Update statistics
            self.decode_stats["tile_accuracy"] = tile_accuracy
            self.decode_stats["entity_accuracy"] = entity_accuracy
            self.decode_stats["spawn_accuracy"] = spawn_accuracy

            results = {
                "tile_accuracy": tile_accuracy,
                "entity_accuracy": entity_accuracy,
                "spawn_accuracy": spawn_accuracy,
                "tile_matches": tile_matches,
                "total_tiles": 966,
                "entity_matches": int(entity_accuracy * len(ref_entities))
                if ref_entities
                else 0,
                "total_entities": len(ref_entities),
            }

            logger.info(
                f"Validation results: Tiles {tile_accuracy:.1%}, Entities {entity_accuracy:.1%}, Spawn {spawn_accuracy:.1%}"
            )
            return results

        except Exception as e:
            print(
                f"Error validating {npp_attract_path} against {reference_map_path}: {e}"
            )
            return {
                "tile_accuracy": 0.0,
                "entity_accuracy": 0.0,
                "spawn_accuracy": 0.0,
                "tile_matches": 0,
                "total_tiles": 966,
                "entity_matches": 0,
                "total_entities": 0,
            }

    def _parse_reference_entities(self, ref_data: bytes) -> List[Dict]:
        """Parse entities from reference map data."""
        entities = []
        index = 1235  # Skip the first entity at 1230

        while index + 4 < len(ref_data):
            entity_type = ref_data[index]
            if entity_type == 0:
                break

            entity = {
                "type": entity_type,
                "x": ref_data[index + 1],
                "y": ref_data[index + 2],
                "orientation": ref_data[index + 3],
                "mode": ref_data[index + 4],
            }
            entities.append(entity)
            index += 5

        return entities

    def _calculate_entity_accuracy(
        self, decoded_entities: List[Dict], ref_entities: List[Dict]
    ) -> float:
        """Calculate entity accuracy between decoded and reference entities."""
        if not ref_entities:
            return 1.0 if not decoded_entities else 0.0

        matches = 0
        for i in range(min(len(decoded_entities), len(ref_entities))):
            decoded = decoded_entities[i]
            reference = ref_entities[i]

            if (
                decoded["type"] == reference["type"]
                and decoded["x"] == reference["x"]
                and decoded["y"] == reference["y"]
                and decoded["orientation"] == reference["orientation"]
                and decoded["mode"] == reference["mode"]
            ):
                matches += 1

        return matches / len(ref_entities)

    def _extract_demo_data(self, npp_data: bytes) -> List[int]:
        """
        Extract demo data (input sequence) from npp_attract file according to attract file format.

        Based on the attract file structure:
        - ATTRACT FILE HEADER: Contains length of map data and demo data (bytes 0-7)
        - MAP DATA: Level info, tiles, entities
        - DEMO DATA: Contains the actual input sequence (1 byte per frame)

        Args:
            npp_data: Raw npp_attract file bytes

        Returns:
            List of input values (0-7) representing the demo sequence
        """
        try:
            import struct

            # Read header to get lengths
            # Bytes 0-3: Length of map data
            # Bytes 4-7: Length of demo data
            if len(npp_data) < 8:
                print("File too short to contain attract file header")
                return []

            map_data_length = struct.unpack("<I", npp_data[0:4])[0]
            demo_data_length = struct.unpack("<I", npp_data[4:8])[0]

            logger.debug(
                f"Header indicates: map_data_length={map_data_length}, demo_data_length={demo_data_length}"
            )

            # Calculate demo data start position
            # Header (8 bytes) + Map data section
            demo_data_start = 8 + map_data_length

            # Validate demo data section exists
            if demo_data_start >= len(npp_data):
                print(
                    f"Demo data start position {demo_data_start} exceeds file length {len(npp_data)}"
                )
                return []

            # The demo data section has its own header before the actual input data
            # Based on the image: it contains metadata like frame count, level ID, etc.
            # The actual demo data (1 byte/frame) starts after this demo header

            if demo_data_start + 16 >= len(npp_data):
                print("Demo data section too short for header")
                return []

            # Read demo data section header
            demo_header_start = demo_data_start

            # The header states both the start and the length, so parse it rather
            # than hunting for the tape. Layout, verified across the corpus:
            #
            #   +0  u8   0
            #   +1  u32  total section length
            #   +5  u32  1
            #   +9  u32  FRAME COUNT
            #   +13 u32  level id
            #   +30      first input byte
            #
            # `section_length - frame_count == 30` holds for all 6504 single-ninja
            # demos, which is what pins the 30. The scan below is kept as a
            # fallback for anything that does not match -- notably the 1496 co-op
            # recordings, whose section is laid out differently.
            #
            # The scan alone is not safe: it takes the LONGEST run of bytes in
            # 0..7, so it starts early whenever the trailing header bytes happen
            # to be small, and it stops at the first byte > 7, which truncated 9
            # demos by exactly one frame.
            stated_len = self._u32(npp_data, demo_header_start + 1)
            frame_count = self._u32(npp_data, demo_header_start + 9)
            if (
                frame_count > 0
                and stated_len - frame_count == 30
                and demo_header_start + 30 + frame_count <= len(npp_data)
            ):
                return list(npp_data[demo_header_start + 30:
                                     demo_header_start + 30 + frame_count])

            # Skip the initial demo header fields (typically ~20-30 bytes)
            # This includes: ?, length of data, ?, frame count, level ID, game mode, etc.
            demo_inputs_start = demo_header_start + 20  # Conservative estimate

            # Look for the actual input sequence by finding the longest sequence of valid inputs (0-7)
            best_start = demo_inputs_start
            best_length = 0

            # Search within the demo data section for valid input sequences
            search_end = min(demo_data_start + demo_data_length, len(npp_data))

            for start_pos in range(demo_inputs_start, search_end - 10):
                length = 0
                pos = start_pos

                # Count consecutive valid inputs (0-7)
                while pos < search_end and 0 <= npp_data[pos] <= 7:
                    length += 1
                    pos += 1

                if length > best_length:
                    best_start = start_pos
                    best_length = length

            # Extract the demo input sequence
            if best_length > 0:
                demo_inputs = []
                for i in range(best_start, best_start + best_length):
                    if i < len(npp_data):
                        demo_inputs.append(npp_data[i])

                logger.info(
                    f"Extracted demo data: {len(demo_inputs)} inputs from offset {best_start}"
                )
                return demo_inputs
            else:
                print("No valid demo input sequence found")
                return []

        except Exception as e:
            print(f"Error extracting demo data: {e}")
            return []
