"""Phase 0, step 1: ingest any source into a uniform shard corpus.

Sources:
  * ``--attracts``  : a directory of npp_attract files (raw 250k, or the symlink
    dir produced by ``filter_attracts``). Decoded via ``NppAttractDecoder``; the
    winning demo inputs are kept alongside ``map_data``.
  * ``--official``  : ``nclone/maps/official`` raw binary maps, kept iff native-safe
    (``official_native_safe_filter``). No demos.

Levels are deduped by ``map_data`` hash and written as pickle shards (see
``scripts/levelgen/common.py``). Degenerate decodes (the all-zero 1335 fallback
that ``create_nclone_map`` returns on failure) are dropped.

Example::

    python -m scripts.levelgen.ingest_corpus \\
        --attracts /path/to/attracts_subset --workers 16 \\
        --out datasets/levelgen/corpus
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional

from scripts.levelgen.common import LevelRecord, map_hash, write_shards

logger = logging.getLogger("levelgen.ingest")

# The attract decoder logs one INFO line per file; mute it for bulk ingest.
logging.getLogger("nclone.replay.npp_attract_decoder").setLevel(logging.WARNING)


def _decode_attract(path: str) -> Optional[LevelRecord]:
    from nclone.replay.npp_attract_decoder import NppAttractDecoder

    dec = NppAttractDecoder()
    try:
        # create_nclone_map prints on failure; silence it in workers.
        with contextlib.redirect_stdout(io.StringIO()):
            md = list(dec.create_nclone_map(path))
            decoded = dec.decode_npp_attract_file(path)
    except Exception:
        return None
    if len(md) < 1335 or all(b == 0 for b in md):
        return None
    demo = list(decoded.get("demo_data", []))
    return LevelRecord(
        level_id=map_hash(md),
        source="human",
        map_data=md,
        demo_inputs=demo,
        demo_len=len(demo),
        src_path=os.path.basename(path),
    )


def _ingest_attracts(attracts_dir: str, workers: int, limit: Optional[int]) -> List[LevelRecord]:
    files = sorted(
        str(p) for p in Path(attracts_dir).iterdir() if p.is_file() and not p.name.startswith(".")
    )
    if limit:
        files = files[:limit]
    logger.info("Decoding %d attract files with %d workers...", len(files), workers)
    out: List[LevelRecord] = []
    with Pool(processes=workers) as pool:
        for rec in pool.imap_unordered(_decode_attract, files, chunksize=64):
            if rec is not None:
                out.append(rec)
    logger.info("Decoded %d/%d attract levels", len(out), len(files))
    return out


def _ingest_official(official_dir: str, dirs: Optional[List[str]]) -> List[LevelRecord]:
    from nclone.maps.official_native_safe_filter import classify_map_data

    root = Path(official_dir)
    subdirs = dirs or [d.name for d in root.iterdir() if d.is_dir()] or [None]
    out: List[LevelRecord] = []
    scanned = 0
    for sub in subdirs:
        base = root / sub if sub else root
        if not base.is_dir():
            continue
        for p in sorted(base.iterdir()):
            if not p.is_file():
                continue
            scanned += 1
            md = list(p.read_bytes())
            if len(md) < 1335:
                continue
            try:
                ok, _reason, _hist = classify_map_data(md)
            except Exception:
                continue
            if not ok:
                continue
            out.append(
                LevelRecord(
                    level_id=map_hash(md),
                    source="official",
                    map_data=md,
                    demo_inputs=[],
                    demo_len=0,
                    src_path=str(p.relative_to(root)),
                )
            )
    logger.info("Kept %d/%d native-safe official maps", len(out), scanned)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest a level source into shard corpus")
    ap.add_argument("--attracts", type=str, default=None, help="attract files dir (raw or filtered)")
    ap.add_argument("--official", type=str, default=None, help="official maps root")
    ap.add_argument(
        "--official-dirs", type=str, nargs="*", default=None, help="official subdirs (e.g. S S2 SI)"
    )
    ap.add_argument("--out", type=str, required=True, help="output corpus directory")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="cap attract files (debug)")
    ap.add_argument("--shard-size", type=int, default=2000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    records: List[LevelRecord] = []
    if args.attracts:
        records += _ingest_attracts(args.attracts, args.workers, args.limit)
    if args.official:
        records += _ingest_official(args.official, args.official_dirs)
    if not records:
        raise SystemExit("no sources given (use --attracts and/or --official)")

    # Dedup by map hash (keeps the first, which prefers attract demos by arg order).
    seen = set()
    deduped: List[LevelRecord] = []
    for r in records:
        if r.level_id in seen:
            continue
        seen.add(r.level_id)
        deduped.append(r)
    logger.info("Deduped %d -> %d unique levels", len(records), len(deduped))

    out_dir = Path(args.out)
    shard_paths = write_shards(deduped, out_dir, shard_size=args.shard_size)
    index = {
        "n_levels": len(deduped),
        "shards": shard_paths,
        "sources": {s: sum(1 for r in deduped if r.source == s) for s in {r.source for r in deduped}},
        "with_demo": sum(1 for r in deduped if r.demo_len > 0),
    }
    with open(out_dir / "ingest_index.json", "w") as f:
        json.dump(index, f, indent=2)
    logger.info("Wrote %d shards to %s: %s", len(shard_paths), out_dir, index["sources"])


if __name__ == "__main__":
    main()
