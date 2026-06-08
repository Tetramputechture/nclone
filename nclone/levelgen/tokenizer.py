"""``map_data`` <-> token-sequence tokenizer for the level generator.

Design (Phase 0 of ``docs/SYNTHETIC_LEVEL_TRANSFORMER_PLAN.md``):

The model never sees the 184-byte header nor the entity-count / ninja-spawn metadata
block (``map_data[1150:1235]``) -- those are structural and recovered deterministically.
The model sees *content* tokens: the 42x23 tile grid, the ninja spawn, and the entity
records. Conditioning tokens (source / difficulty band / mechanic tags / subset flag)
are prepended.

Byte-exact round trip is the hard Phase-0 gate. We achieve it by carrying the header and
the metadata region verbatim on the :class:`TokenizedLevel` and rebuilding only the tile
region (184:1150) and entity region (1235:) from tokens. The metadata region already holds
the ninja spawn (1231..1233) and entity counts, so reconstruction is exact regardless of
how those tokens are emitted.

Important correction to the plan's "entities are grid-aligned" note: real corpus entities
(e.g. mines) sit at *unit* resolution that is **not** a multiple of 4 (a mine at x=69 has
no integer grid coordinate). So entity coordinates are tokenized at full unit resolution
(0..255), not as grid cells -- otherwise the round trip is lossy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

# --- map_data byte layout (single byte per int) -----------------------------
HEADER_LEN = 184
TILE_START = 184
TILE_END = 1150  # exclusive; 42 * 23 = 966 tiles
NUM_TILES = TILE_END - TILE_START  # 966
# Number of real tile ids the models embed (0..37). The decoder/corpus can emit a
# handful of higher, degenerate ids (observed up to 50) -- build_dataset drops any
# level whose tile region exceeds this so the tile embedding never indexes OOB.
N_TILE_VALUES = 38
META_START = 1150  # entity counts + ninja spawn + padding
META_END = 1235  # exclusive; entity records begin here
ENT_START = 1235
ENT_STRIDE = 5  # [type, x, y, orient, mode]
MIN_MAP_LEN = 1335  # entity section padded to >= 185 bytes (85 + >=100)

MapData = Union[bytes, bytearray, Sequence[int]]

# --- conditioning vocabularies ----------------------------------------------
SOURCES: Tuple[str, ...] = ("human", "official", "procgen", "generated")
# 6 difficulty bands by demo frame-length (edges calibrated in label_corpus.py);
# index 0 is reserved for "unknown / unlabelled".
DIFFICULTY_BANDS: Tuple[str, ...] = (
    "unknown",
    "d0",
    "d1",
    "d2",
    "d3",
    "d4",
)
# Mechanic tag domains -- mirror the compose registry domains.
MECHANIC_TAGS: Tuple[str, ...] = (
    "flat",
    "jump",
    "wall_jump",
    "mine",
    "slope",
    "curve",
    "composite",
    "precision",
)
SUBSET_FLAGS: Tuple[str, ...] = ("subset_no", "subset_ok")


class Vocab:
    """Flat token vocabulary partitioned into contiguous blocks.

    Block layout (offsets are computed, never hard-coded elsewhere):
      special | tile values | byte values | entity types | conditioning
    """

    # special tokens
    PAD = 0
    BOS = 1
    EOS = 2
    TILE_SEP = 3
    NINJA = 4
    ENT_SEP = 5
    ENT = 6  # entity-record marker
    _N_SPECIAL = 16  # leave headroom

    # block sizes
    _N_TILE = 64  # tile types 0..33 (corpus max ~37); headroom to 63
    _N_BYTE = 256  # coordinate / orient / mode bytes 0..255
    _N_ETYPE = 64  # entity type ids 0..40; headroom to 63

    TILE_BASE = _N_SPECIAL
    BYTE_BASE = TILE_BASE + _N_TILE
    ETYPE_BASE = BYTE_BASE + _N_BYTE
    COND_BASE = ETYPE_BASE + _N_ETYPE

    _N_COND = len(SOURCES) + len(DIFFICULTY_BANDS) + len(MECHANIC_TAGS) + len(SUBSET_FLAGS)
    SIZE = COND_BASE + _N_COND

    # conditioning sub-block offsets (within the COND block)
    _COND_SOURCE = COND_BASE
    _COND_DIFF = _COND_SOURCE + len(SOURCES)
    _COND_MECH = _COND_DIFF + len(DIFFICULTY_BANDS)
    _COND_SUBSET = _COND_MECH + len(MECHANIC_TAGS)

    # --- encoders -----------------------------------------------------------
    @classmethod
    def tile(cls, v: int) -> int:
        if not 0 <= v < cls._N_TILE:
            raise ValueError(f"tile value {v} out of vocab range [0,{cls._N_TILE})")
        return cls.TILE_BASE + v

    @classmethod
    def byte(cls, v: int) -> int:
        if not 0 <= v < cls._N_BYTE:
            raise ValueError(f"byte value {v} out of range [0,256)")
        return cls.BYTE_BASE + v

    @classmethod
    def etype(cls, t: int) -> int:
        if not 0 <= t < cls._N_ETYPE:
            raise ValueError(f"entity type {t} out of vocab range [0,{cls._N_ETYPE})")
        return cls.ETYPE_BASE + t

    @classmethod
    def source(cls, name: str) -> int:
        return cls._COND_SOURCE + SOURCES.index(name)

    @classmethod
    def difficulty(cls, band: str) -> int:
        return cls._COND_DIFF + DIFFICULTY_BANDS.index(band)

    @classmethod
    def mechanic(cls, tag: str) -> int:
        return cls._COND_MECH + MECHANIC_TAGS.index(tag)

    @classmethod
    def subset(cls, ok: bool) -> int:
        return cls._COND_SUBSET + (1 if ok else 0)

    # --- decoders -----------------------------------------------------------
    @classmethod
    def is_tile(cls, tok: int) -> bool:
        return cls.TILE_BASE <= tok < cls.BYTE_BASE

    @classmethod
    def is_byte(cls, tok: int) -> bool:
        return cls.BYTE_BASE <= tok < cls.ETYPE_BASE

    @classmethod
    def is_etype(cls, tok: int) -> bool:
        return cls.ETYPE_BASE <= tok < cls.COND_BASE

    @classmethod
    def tile_value(cls, tok: int) -> int:
        return tok - cls.TILE_BASE

    @classmethod
    def byte_value(cls, tok: int) -> int:
        return tok - cls.BYTE_BASE

    @classmethod
    def etype_value(cls, tok: int) -> int:
        return tok - cls.ETYPE_BASE


@dataclass
class Entity:
    type: int
    x: int
    y: int
    orient: int
    mode: int

    def as_tuple(self) -> Tuple[int, int, int, int, int]:
        return (self.type, self.x, self.y, self.orient, self.mode)


@dataclass
class TokenizedLevel:
    """A tokenized level plus the structural bytes needed for an exact round trip."""

    tokens: List[int]
    header: bytes  # map_data[0:184], verbatim
    meta: bytes  # map_data[1150:1235], verbatim (counts + spawn + pad)
    tail_len: int  # len(map_data) - 1235 (entity records + zero padding)
    # parsed content (also recoverable from tokens; kept for convenience)
    tiles: List[int] = field(default_factory=list)
    ninja: Tuple[int, int, int] = (0, 0, 0)
    entities: List[Entity] = field(default_factory=list)
    # conditioning labels (not part of the round trip)
    cond: Dict[str, object] = field(default_factory=dict)


class LevelTokenizer:
    """Stateless map_data <-> token translator. See module docstring."""

    def __init__(self) -> None:
        self.vocab = Vocab

    # ------------------------------------------------------------------ encode
    def tokenize(
        self,
        map_data: MapData,
        cond: Optional[Dict[str, object]] = None,
        include_cond: bool = True,
        build_tokens: bool = True,
    ) -> TokenizedLevel:
        """Parse a ``map_data`` into a :class:`TokenizedLevel`.

        ``build_tokens=False`` skips constructing the flat ``tokens`` stream and
        returns only the parsed content (tiles / ninja / entities + structural
        bytes). Use it when you only need the parsed fields (e.g. ``build_dataset``)
        -- it avoids the cost of the token build and, importantly, the
        ``Vocab.etype``/``Vocab.tile`` range checks that would otherwise raise on a
        degenerate corpus level instead of letting the caller drop it.
        """
        md = list(map_data)
        if len(md) < MIN_MAP_LEN:
            raise ValueError(f"map_data too short: {len(md)} < {MIN_MAP_LEN}")
        if any(not (0 <= b <= 255) for b in md):
            raise ValueError("map_data contains values outside 0..255")

        header = bytes(md[0:HEADER_LEN])
        tiles = md[TILE_START:TILE_END]
        meta = bytes(md[META_START:META_END])
        tail_len = len(md) - ENT_START

        ninja = (md[1231], md[1232], md[1233])
        entities = self._parse_entities(md)

        cond = dict(cond or {})
        toks: List[int] = []
        if build_tokens:
            toks.append(Vocab.BOS)
            if include_cond:
                toks.extend(self._cond_tokens(cond))

            toks.append(Vocab.TILE_SEP)
            toks.extend(Vocab.tile(t) for t in tiles)

            toks.append(Vocab.NINJA)
            toks.extend(Vocab.byte(b) for b in ninja)

            toks.append(Vocab.ENT_SEP)
            for e in entities:
                toks.append(Vocab.ENT)
                toks.append(Vocab.etype(e.type))
                toks.append(Vocab.byte(e.x))
                toks.append(Vocab.byte(e.y))
                toks.append(Vocab.byte(e.orient))
                toks.append(Vocab.byte(e.mode))
            toks.append(Vocab.EOS)

        return TokenizedLevel(
            tokens=toks,
            header=header,
            meta=meta,
            tail_len=tail_len,
            tiles=tiles,
            ninja=ninja,
            entities=entities,
            cond=cond,
        )

    def _cond_tokens(self, cond: Dict[str, object]) -> List[int]:
        out: List[int] = []
        src = cond.get("source")
        if src in SOURCES:
            out.append(Vocab.source(src))
        band = cond.get("difficulty_band")
        if band in DIFFICULTY_BANDS:
            out.append(Vocab.difficulty(band))
        for tag in cond.get("mechanic_tags", []) or []:
            if tag in MECHANIC_TAGS:
                out.append(Vocab.mechanic(tag))
        if "subset_ok" in cond:
            out.append(Vocab.subset(bool(cond["subset_ok"])))
        return out

    @staticmethod
    def _parse_entities(md: List[int]) -> List[Entity]:
        ents: List[Entity] = []
        i = ENT_START
        n = len(md)
        while i + ENT_STRIDE <= n:
            t = md[i]
            if t == 0:  # terminator / zero padding
                break
            ents.append(Entity(t, md[i + 1], md[i + 2], md[i + 3], md[i + 4]))
            i += ENT_STRIDE
        return ents

    # ------------------------------------------------------------------ decode
    def detokenize(self, level: TokenizedLevel) -> List[int]:
        """Reconstruct map_data byte-for-byte from a :class:`TokenizedLevel`.

        Uses the carried ``header`` and ``meta`` verbatim; rebuilds the tile region
        and entity region from the token stream.
        """
        tiles, ninja, entities = self._decode_content(level.tokens)
        return self._assemble(level.header, tiles, level.meta, entities, level.tail_len)

    @staticmethod
    def _assemble(
        header: bytes,
        tiles: List[int],
        meta: bytes,
        entities: List[Entity],
        tail_len: int,
    ) -> List[int]:
        if len(tiles) != NUM_TILES:
            raise ValueError(f"expected {NUM_TILES} tiles, got {len(tiles)}")
        if len(header) != HEADER_LEN:
            raise ValueError(f"header must be {HEADER_LEN} bytes")
        if len(meta) != (META_END - META_START):
            raise ValueError(f"meta must be {META_END - META_START} bytes")
        out: List[int] = list(header) + list(tiles) + list(meta)
        ent_bytes: List[int] = []
        for e in entities:
            ent_bytes.extend(e.as_tuple())
        if len(ent_bytes) > tail_len:
            raise ValueError("entity bytes exceed tail_len")
        ent_bytes.extend([0] * (tail_len - len(ent_bytes)))
        out.extend(ent_bytes)
        return out

    def _decode_content(
        self, tokens: Sequence[int]
    ) -> Tuple[List[int], Tuple[int, int, int], List[Entity]]:
        tiles: List[int] = []
        ninja = (0, 0, 0)
        entities: List[Entity] = []
        i = 0
        n = len(tokens)
        # tiles
        while i < n and tokens[i] != Vocab.TILE_SEP:
            i += 1
        i += 1  # skip TILE_SEP
        while i < n and Vocab.is_tile(tokens[i]):
            tiles.append(Vocab.tile_value(tokens[i]))
            i += 1
        # ninja
        if i < n and tokens[i] == Vocab.NINJA:
            i += 1
            nb = []
            while i < n and Vocab.is_byte(tokens[i]) and len(nb) < 3:
                nb.append(Vocab.byte_value(tokens[i]))
                i += 1
            if len(nb) == 3:
                ninja = (nb[0], nb[1], nb[2])
        # entities
        if i < n and tokens[i] == Vocab.ENT_SEP:
            i += 1
        while i < n and tokens[i] == Vocab.ENT:
            i += 1
            vals = []
            while i < n and (Vocab.is_byte(tokens[i]) or Vocab.is_etype(tokens[i])) and len(vals) < 5:
                tok = tokens[i]
                vals.append(Vocab.etype_value(tok) if Vocab.is_etype(tok) else Vocab.byte_value(tok))
                i += 1
            if len(vals) == 5:
                entities.append(Entity(*vals))
        return tiles, ninja, entities


# ---------------------------------------------------------------- generation
# meta-region (map_data[1150:1235]) byte offsets, relative to META_START.
_META_COUNT_MINE = 0  # type 1
_META_COUNT_EXIT = 2  # type 3
_META_COUNT_GOLD = 4  # type 2
_META_COUNT_ESWITCH = 6  # type 4
_META_UNKNOWN = 8  # always 1
_META_SPAWN_X = 81  # abs 1231
_META_SPAWN_Y = 82  # abs 1232
_META_SPAWN_ORIENT = 83  # abs 1233
_META_LEN = META_END - META_START  # 85


def assemble_map_data(
    tiles: Sequence[int],
    ninja: Tuple[int, int, int],
    entities: Sequence[Entity],
    template_header: bytes,
) -> List[int]:
    """Build a fresh, loader-valid ``map_data`` for a *generated* level.

    Unlike :meth:`LevelTokenizer.detokenize` (which carries the original ``meta``
    block verbatim for an exact round trip), this recomputes the entity-count and
    ninja-spawn metadata block from scratch, matching the layout that
    ``NppAttractDecoder.create_nclone_map`` emits, so the level loads in the sim.
    """
    if len(tiles) != NUM_TILES:
        raise ValueError(f"expected {NUM_TILES} tiles, got {len(tiles)}")
    if len(template_header) != HEADER_LEN:
        raise ValueError(f"template_header must be {HEADER_LEN} bytes")

    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for e in entities:
        if e.type in counts:
            counts[e.type] += 1

    meta = bytearray(_META_LEN)
    meta[_META_COUNT_MINE] = min(counts[1], 255)
    meta[_META_COUNT_EXIT] = min(counts[3], 255)
    meta[_META_COUNT_GOLD] = min(counts[2], 255)
    meta[_META_COUNT_ESWITCH] = min(counts[4], 255)
    meta[_META_UNKNOWN] = 1
    meta[_META_SPAWN_X] = ninja[0] & 0xFF
    meta[_META_SPAWN_Y] = ninja[1] & 0xFF
    meta[_META_SPAWN_ORIENT] = ninja[2] & 0xFF

    n = len(entities)
    tail_len = max(100, ENT_STRIDE * n)  # entity section padded to >= 185 bytes
    return LevelTokenizer._assemble(
        bytes(template_header), list(tiles), bytes(meta), list(entities), tail_len
    )


__all__ = [
    "LevelTokenizer",
    "TokenizedLevel",
    "Vocab",
    "Entity",
    "assemble_map_data",
    "SOURCES",
    "DIFFICULTY_BANDS",
    "MECHANIC_TAGS",
    "SUBSET_FLAGS",
    "NUM_TILES",
    "N_TILE_VALUES",
]
