"""Phase 0, step 4: turn a labeled shard corpus into a compact HDF5 token dataset.

Storage (lazy, lzf-compressed, following the offline-RL f16/lzf pattern):
  tiles       (N, 966) uint8    -- tile-grid ids 0..37
  ninja       (N, 3)   uint8    -- spawn x, y, orient bytes
  ent_data    (M, 5)   uint8    -- flattened entity records (type,x,y,orient,mode)
  ent_ptr     (N+1,)   int64    -- CSR offsets into ent_data per level
  cond_source (N,)     uint8    -- index into tokenizer.SOURCES
  cond_diff   (N,)     uint8    -- index into tokenizer.DIFFICULTY_BANDS
  cond_subset (N,)     uint8    -- 0/1
  cond_mech   (N, T)   uint8    -- multi-hot over tokenizer.MECHANIC_TAGS
  demo_len    (N,)     int32

Augmentation: horizontal mirror (pure, involutive ``mirror_map_data``) is the free
2x for the generator corpus -- it preserves difficulty (same demo) and chirality-swaps
slope/curve tiles. (ReachableAreaTranslator is demo-coupled and lives in the offline
attract-augmentor pipeline; excluded here to keep map-only building dependency-light.)

Example::

    python -m scripts.levelgen.build_dataset --corpus datasets/levelgen/corpus \\
        --out datasets/levelgen/tiles.h5 --subset-only --mirror
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import numpy as np

from nclone.levelgen.tokenizer import (
    DIFFICULTY_BANDS,
    LevelTokenizer,
    MECHANIC_TAGS,
    SOURCES,
)
from nclone.levelgen.labeling import assign_band, mechanic_tags, subset_ok
from nclone.map_augmentation.transforms import mirror_map_data
from scripts.levelgen.common import iter_records

logger = logging.getLogger("levelgen.build")


def _encode(map_data, labels, tok: LevelTokenizer):
    lvl = tok.tokenize(map_data, include_cond=False)
    tiles = np.array(lvl.tiles, dtype=np.uint8)
    ninja = np.array(lvl.ninja, dtype=np.uint8)
    ents = np.array([e.as_tuple() for e in lvl.entities], dtype=np.uint8).reshape(-1, 5)
    src = SOURCES.index(labels["source"]) if labels["source"] in SOURCES else 0
    diff = DIFFICULTY_BANDS.index(labels["difficulty_band"])
    subset = 1 if labels["subset_ok"] else 0
    mech = np.zeros(len(MECHANIC_TAGS), dtype=np.uint8)
    for t in labels["mechanic_tags"]:
        if t in MECHANIC_TAGS:
            mech[MECHANIC_TAGS.index(t)] = 1
    return tiles, ninja, ents, src, diff, subset, mech, int(labels.get("demo_len", 0))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build HDF5 token dataset from labeled corpus")
    ap.add_argument("--corpus", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--subset-only", action="store_true", help="keep only subset_ok levels")
    ap.add_argument("--mirror", action="store_true", help="add horizontal-mirror variants")
    ap.add_argument("--bands", type=int, default=5)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    corpus = Path(args.corpus)
    edges = None
    idx_path = corpus / "labels_index.json"
    if idx_path.exists():
        edges = json.loads(idx_path.read_text()).get("difficulty_edges")

    tok = LevelTokenizer()
    tiles_l, ninja_l, ent_l, src_l, diff_l, sub_l, mech_l, dl_l = [], [], [], [], [], [], [], []
    ent_ptr: List[int] = [0]
    n_in = n_kept = n_aug = 0

    def add(map_data, labels):
        nonlocal n_kept
        t, nj, e, s, d, su, m, dl = _encode(map_data, labels, tok)
        tiles_l.append(t)
        ninja_l.append(nj)
        ent_l.append(e)
        ent_ptr.append(ent_ptr[-1] + len(e))
        src_l.append(s)
        diff_l.append(d)
        sub_l.append(su)
        mech_l.append(m)
        dl_l.append(dl)
        n_kept += 1

    for r in iter_records(corpus):
        n_in += 1
        labels = r.labels or {
            "source": r.source,
            "subset_ok": subset_ok(r.map_data),
            "mechanic_tags": mechanic_tags(r.map_data),
            "difficulty_band": assign_band(r.demo_len, edges or []),
            "demo_len": r.demo_len,
        }
        if args.subset_only and not labels["subset_ok"]:
            continue
        add(r.map_data, labels)
        if args.mirror:
            mirrored = mirror_map_data(list(r.map_data))
            mlabels = dict(labels)
            mlabels["mechanic_tags"] = mechanic_tags(mirrored)  # chirality may shift tags
            mlabels["subset_ok"] = subset_ok(mirrored)
            add(mirrored, mlabels)
            n_aug += 1

    if n_kept == 0:
        raise SystemExit("no levels kept (try without --subset-only)")

    tiles = np.stack(tiles_l)
    ninja = np.stack(ninja_l)
    ent_data = (
        np.concatenate(ent_l, axis=0) if any(len(e) for e in ent_l) else np.zeros((0, 5), np.uint8)
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    import h5py

    with h5py.File(out, "w") as h:
        h.create_dataset("tiles", data=tiles, compression="lzf")
        h.create_dataset("ninja", data=ninja, compression="lzf")
        h.create_dataset("ent_data", data=ent_data, compression="lzf")
        h.create_dataset("ent_ptr", data=np.array(ent_ptr, np.int64))
        h.create_dataset("cond_source", data=np.array(src_l, np.uint8))
        h.create_dataset("cond_diff", data=np.array(diff_l, np.uint8))
        h.create_dataset("cond_subset", data=np.array(sub_l, np.uint8))
        h.create_dataset("cond_mech", data=np.stack(mech_l))
        h.create_dataset("demo_len", data=np.array(dl_l, np.int32))
        h.attrs["n_levels"] = n_kept
        h.attrs["sources"] = json.dumps(list(SOURCES))
        h.attrs["difficulty_bands"] = json.dumps(list(DIFFICULTY_BANDS))
        h.attrs["mechanic_tags"] = json.dumps(list(MECHANIC_TAGS))
        h.attrs["difficulty_edges"] = json.dumps(edges or [])
    logger.info(
        "Wrote %s: %d levels (in=%d, mirror_aug=%d, entities=%d)",
        out,
        n_kept,
        n_in,
        n_aug,
        len(ent_data),
    )


if __name__ == "__main__":
    main()
