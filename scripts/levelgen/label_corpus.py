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


def _dstar_calibration(records, sample_per_band: int, timeout: float) -> dict:
    """Gate a stratified sample and report median d* per difficulty band.

    Validates the demo-length proxy: if the proxy is good, median d* should rise
    monotonically d0 -> d4. (Diagnostic only; does not auto-adjust the edges.)
    """
    import statistics
    from collections import defaultdict
    from scripts.levelgen.generate_levels import gate_with_timeout

    by_band = defaultdict(list)
    for r in records:
        b = (r.labels or {}).get("difficulty_band", "unknown")
        if b != "unknown":
            by_band[b].append(r)
    out = {}
    for band in sorted(by_band):
        sample = by_band[band][:sample_per_band]
        ds = [d for d in (gate_with_timeout(list(r.map_data), timeout) for r in sample) if d is not None]
        out[band] = {"n": len(ds), "median_dstar": float(statistics.median(ds)) if ds else None}
        logger.info("d* calibration %s: n=%d median=%s", band, len(ds), out[band]["median_dstar"])
    meds = [out[b]["median_dstar"] for b in sorted(out) if out[b]["median_dstar"] is not None]
    out["monotonic"] = all(a <= b for a, b in zip(meds, meds[1:])) if len(meds) > 1 else None
    logger.info("d* band ordering monotonic in proxy: %s", out["monotonic"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Label an ingested level corpus")
    ap.add_argument("--corpus", type=str, required=True)
    ap.add_argument("--bands", type=int, default=5, help="number of difficulty bands")
    ap.add_argument("--dstar-sample", type=int, default=0,
                    help="gate N levels/band to validate the demo-length proxy vs real d* (slow; 0=off)")
    ap.add_argument("--dstar-timeout", type=float, default=30.0)
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
    if args.dstar_sample > 0:
        index["dstar_calibration"] = _dstar_calibration(records, args.dstar_sample, args.dstar_timeout)
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
