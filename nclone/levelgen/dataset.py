"""Torch datasets over the build_dataset HDF5 (tile stage + entity stage).

Kept light: HDF5 is opened lazily per-worker. The tile stage consumes the raw
42x23 tile-id grid plus a conditioning vector; the entity stage consumes the tile
grid (as cross-attention context) plus the flat entity token stream produced by the
shared :class:`~nclone.levelgen.tokenizer.LevelTokenizer` vocabulary.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:  # torch is optional for the non-model pipeline stages
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore

from .tokenizer import (
    DIFFICULTY_BANDS,
    MECHANIC_TAGS,
    SOURCES,
    Entity,
    Vocab,
)

GRID_H, GRID_W = 23, 42
NUM_TILES = GRID_H * GRID_W


class _H5Base(Dataset):
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self._h = None
        import h5py

        with h5py.File(h5_path, "r") as h:
            self.n = int(h.attrs["n_levels"])

    def _file(self):
        if self._h is None:
            import h5py

            self._h = h5py.File(self.h5_path, "r")
        return self._h

    def __len__(self):
        return self.n

    def _cond_vec(self, h, i) -> Dict[str, "torch.Tensor"]:
        return {
            "source": torch.tensor(int(h["cond_source"][i]), dtype=torch.long),
            "difficulty": torch.tensor(int(h["cond_diff"][i]), dtype=torch.long),
            "subset": torch.tensor(int(h["cond_subset"][i]), dtype=torch.long),
            "mech": torch.tensor(np.array(h["cond_mech"][i]), dtype=torch.float32),
        }


class TileDataset(_H5Base):
    """Yields {tiles: (966,) long, cond: {...}} for the tile transformer."""

    def __getitem__(self, i):
        h = self._file()
        tiles = torch.tensor(np.array(h["tiles"][i]), dtype=torch.long)
        return {"tiles": tiles, "cond": self._cond_vec(h, i)}


class EntityDataset(_H5Base):
    """Yields tile context + entity token stream for the entity transformer.

    ``ent_tokens`` is the flat token sequence ``[ENT_SEP](ENT etype x y orient mode)*[EOS]``
    using the shared vocabulary, so the decoder output detokenizes directly.
    """

    def __getitem__(self, i):
        h = self._file()
        tiles = torch.tensor(np.array(h["tiles"][i]), dtype=torch.long)
        ninja = np.array(h["ninja"][i])
        lo, hi = int(h["ent_ptr"][i]), int(h["ent_ptr"][i + 1])
        recs = np.array(h["ent_data"][lo:hi]).reshape(-1, 5)
        toks: List[int] = [Vocab.NINJA, Vocab.byte(int(ninja[0])), Vocab.byte(int(ninja[1])),
                           Vocab.byte(int(ninja[2])), Vocab.ENT_SEP]
        for r in recs:
            toks.append(Vocab.ENT)
            toks.append(Vocab.etype(int(r[0])))
            toks.append(Vocab.byte(int(r[1])))
            toks.append(Vocab.byte(int(r[2])))
            toks.append(Vocab.byte(int(r[3])))
            toks.append(Vocab.byte(int(r[4])))
        toks.append(Vocab.EOS)
        return {
            "tiles": tiles,
            "ent_tokens": torch.tensor(toks, dtype=torch.long),
            "cond": self._cond_vec(h, i),
        }


def collate_entity(batch):
    """Pad entity token streams to the batch max (PAD=Vocab.PAD)."""
    tiles = torch.stack([b["tiles"] for b in batch])
    maxlen = max(b["ent_tokens"].numel() for b in batch)
    toks = torch.full((len(batch), maxlen), Vocab.PAD, dtype=torch.long)
    mask = torch.zeros((len(batch), maxlen), dtype=torch.bool)
    for j, b in enumerate(batch):
        L = b["ent_tokens"].numel()
        toks[j, :L] = b["ent_tokens"]
        mask[j, :L] = True
    cond = {
        k: torch.stack([b["cond"][k] for b in batch]) for k in batch[0]["cond"]
    }
    return {"tiles": tiles, "ent_tokens": toks, "ent_mask": mask, "cond": cond}


__all__ = ["TileDataset", "EntityDataset", "collate_entity", "GRID_H", "GRID_W", "NUM_TILES"]
