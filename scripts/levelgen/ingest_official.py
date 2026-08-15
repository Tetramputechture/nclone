"""Ingest the official N++ maps into a labeled corpus, at a chosen entity scope.

Why this exists
---------------
`pack_benchmark` takes an ``--entity-profile`` and gates every level on it, so
widening the training scope looks like a one-flag change. It is not: the corpus
it reads from was itself filtered at INGEST time, and the existing
``datasets/levelgen/corpus_official`` holds 475 records that all pass even the
5-type ``minimal`` profile. Every level carrying a rocket, thwump, drone or
bounce block was dropped before the profile knob ever saw it, so turning the knob
up admits nothing new.

That is the trap this script closes. It re-ingests straight from the map files at
whatever profile is asked for, so the pack-time gate becomes the *narrowing* step
it is supposed to be rather than a no-op over an already-narrowed pool.

Usage
-----
    python -m scripts.levelgen.ingest_official \\
        --official-root nclone/maps/official \\
        --entity-profile phase1 \\
        --out datasets/levelgen/corpus_official_phase1

Then pack as usual, e.g.::

    python -m scripts.levelgen.pack_benchmark \\
        --official datasets/levelgen/corpus_official_phase1 \\
        --entity-profile phase1 --out datasets/<benchmark>
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from collections import Counter
from pathlib import Path

from nclone.constants.entity_types import ENTITY_PROFILES
from nclone.levelgen.labeling import entity_counts, subset_ok
from nclone.levelgen.pool_store import content_hash

logger = logging.getLogger("levelgen.ingest_official")

#: The difficulty-ordered families that make up the S* target set. A map's
#: parent directory names its family and the filename's leading number is the
#: game's OWN difficulty order, which is the ordering the curriculum should use
#: rather than any metric we invent.
_S_FAMILIES = ("S", "S2", "SI", "SL", "SL2", "SS", "SS2")


def _difficulty_index(path: Path) -> int:
    """Leading integer of the filename, or -1. This is the game's own ordering."""
    head = path.name.split(" ", 1)[0]
    try:
        return int(head)
    except ValueError:
        return -1


def ingest(
    official_root: Path,
    out_dir: Path,
    entity_profile: str,
    limit: int | None = None,
) -> dict:
    allowed = ENTITY_PROFILES[entity_profile]
    rejected = Counter()
    n_seen = 0
    n_dup = 0
    seen_hashes: set[str] = set()
    records: list[dict] = []

    for path in sorted(official_root.rglob("*")):
        if not path.is_file():
            continue
        try:
            raw = path.read_bytes()
            counts = entity_counts(raw)
        except Exception:
            rejected["unparseable"] += 1
            continue
        n_seen += 1
        map_data = list(raw)
        out_of_scope = sorted(set(counts) - set(allowed) - {0})
        if out_of_scope:
            for t in out_of_scope:
                rejected[f"type_{t}"] += 1
            continue
        if not subset_ok(map_data, allowed=allowed):
            # In scope by TYPE but still rejected -- `subset_ok` also requires
            # exactly one exit door and one switch. Counted separately so the
            # summary never silently loses maps: a bare written-vs-seen gap is
            # exactly the kind of quiet shortfall this whole exercise is about.
            rejected["exit_pair"] += 1
            continue
        h = content_hash(map_data)
        if h in seen_hashes:
            n_dup += 1
            continue
        seen_hashes.add(h)
        family = path.parent.name
        records.append(
            {
                "level_id": h,
                "map_data": map_data,
                "source": "official",
                "src_path": str(path),
                "demo_inputs": None,
                "demo_len": 0,
                "labels": {
                    "family": family,
                    "name": path.name,
                    "difficulty_index": _difficulty_index(path),
                    "is_s_star": family in _S_FAMILIES,
                    "entity_profile": entity_profile,
                },
            }
        )
        if limit is not None and len(records) >= limit:
            break

    # Emit the format `pack_benchmark` reads: an `ingest_index.json` naming
    # pickled `shard_*.pkl` files. NOT the PoolWriter (.bin + .jsonl) layout --
    # the two are not interchangeable, and writing the wrong one produces a
    # corpus that packs to silently zero levels.
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = "shard_0000.pkl"
    (out_dir / shard).write_bytes(pickle.dumps({"records": records}))
    (out_dir / "ingest_index.json").write_text(
        json.dumps(
            {
                "n_levels": len(records),
                "shards": [shard],
                "sources": {"official": len(records)},
                "with_demo": 0,
            },
            indent=2,
        )
    )
    n_written = len(records)

    summary = {
        "official_root": str(official_root),
        "entity_profile": entity_profile,
        "allowed_types": sorted(int(t) for t in allowed),
        "n_seen": n_seen,
        "n_written": n_written,
        "n_duplicates": n_dup,
        "rejected": dict(rejected.most_common()),
    }
    (out_dir / "ingest_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--official-root", default="nclone/maps/official")
    ap.add_argument(
        "--entity-profile",
        choices=sorted(ENTITY_PROFILES),
        default="phase1",
        help="Entity types a map may contain and still be ingested. The phase "
             "profiles follow docs/ENTITY_INTEGRATION_PLAN.md §9.",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    summary = ingest(
        Path(args.official_root), Path(args.out), args.entity_profile, args.limit
    )
    logger.info(
        "ingested %d/%d maps at profile %r (%d duplicates)",
        summary["n_written"], summary["n_seen"], args.entity_profile,
        summary["n_duplicates"],
    )
    if summary["rejected"]:
        logger.info("out-of-scope rejections: %s", summary["rejected"])


if __name__ == "__main__":
    main()
