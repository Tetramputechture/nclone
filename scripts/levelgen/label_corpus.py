"""Phase 0, step 3: attach labels to an ingested shard corpus.

Per level: ``source`` (already set at ingest), ``subset_ok`` flag, ``mechanic_tags``,
and a ``difficulty_band`` derived from the winning-demo frame length (quantile bands
fit over the corpus). Rewrites shards in place (adds to ``record.labels``) and writes
a ``labels_index.json`` with the fitted band edges + tag/source histograms.

Example::

    python -m scripts.levelgen.label_corpus --corpus datasets/levelgen/corpus
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from collections import Counter
from pathlib import Path

from nclone.levelgen.labeling import (
    assign_band,
    difficulty_edges,
    mechanic_tags,
    subset_ok,
)
from scripts.levelgen.common import iter_records

logger = logging.getLogger("levelgen.label")


def main() -> None:
    ap = argparse.ArgumentParser(description="Label an ingested level corpus")
    ap.add_argument("--corpus", type=str, required=True)
    ap.add_argument("--bands", type=int, default=5, help="number of difficulty bands")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    corpus = Path(args.corpus)
    records = list(iter_records(corpus))
    logger.info("Loaded %d records", len(records))

    # Fit difficulty band edges over demo lengths (human levels with a demo).
    edges = difficulty_edges([r.demo_len for r in records], n_bands=args.bands)
    logger.info("Difficulty band edges (frames): %s", edges)

    src_hist: Counter = Counter()
    band_hist: Counter = Counter()
    tag_hist: Counter = Counter()
    n_subset = 0
    for r in records:
        ok = subset_ok(r.map_data)
        tags = mechanic_tags(r.map_data)
        band = assign_band(r.demo_len, edges)
        r.labels = {
            "source": r.source,
            "subset_ok": ok,
            "mechanic_tags": tags,
            "difficulty_band": band,
            "demo_len": r.demo_len,
        }
        n_subset += int(ok)
        src_hist[r.source] += 1
        band_hist[band] += 1
        for t in tags:
            tag_hist[t] += 1

    # rewrite shards in place
    from dataclasses import asdict

    by_shard = {}
    shard_paths = sorted(p.name for p in corpus.glob("shard_*.pkl"))
    # re-iterate shard by shard to preserve sharding
    idx = 0
    for name in shard_paths:
        with open(corpus / name, "rb") as f:
            blob = pickle.load(f)
        n = len(blob["records"])
        for j in range(n):
            blob["records"][j]["labels"] = records[idx + j].labels
        idx += n
        with open(corpus / name, "wb") as f:
            pickle.dump(blob, f, protocol=4)
        by_shard[name] = n

    index = {
        "n_levels": len(records),
        "n_subset_ok": n_subset,
        "difficulty_edges": edges,
        "sources": dict(src_hist),
        "difficulty_bands": dict(band_hist),
        "mechanic_tags": dict(tag_hist),
    }
    with open(corpus / "labels_index.json", "w") as f:
        json.dump(index, f, indent=2)
    logger.info(
        "Labeled %d levels (%d subset_ok). bands=%s tags=%s",
        len(records),
        n_subset,
        dict(band_hist),
        dict(tag_hist),
    )


if __name__ == "__main__":
    main()
