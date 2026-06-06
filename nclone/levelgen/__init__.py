"""Synthetic level generation (transformer level generator).

See ``docs/SYNTHETIC_LEVEL_TRANSFORMER_PLAN.md``. This package owns the tokenizer
(``map_data`` <-> tokens, byte-exact round trip), the conditioning vocabulary, and
the tile/entity transformer models. Corpus ingestion, dataset building, training,
generation and evaluation live under ``scripts/levelgen/``.
"""

from .tokenizer import (
    LevelTokenizer,
    TokenizedLevel,
    Vocab,
    DIFFICULTY_BANDS,
    MECHANIC_TAGS,
    SOURCES,
)

__all__ = [
    "LevelTokenizer",
    "TokenizedLevel",
    "Vocab",
    "DIFFICULTY_BANDS",
    "MECHANIC_TAGS",
    "SOURCES",
]
