"""Reorganize a packed MechanicBenchmark so the attract/corpus levels are a
SEPARATE difficulty-sorted block AFTER the controlled basic-mechanic (procgen)
buckets.

Motivation: the packed benchmark mixes two very different sources in the same
mechanic buckets/tiers:
  * PROCGEN levels (generator != 'corpus'): controlled, single-mechanic levels
    produced by the map_* generators (flat_run, basic_jump, slopes, ...). These
    are the genuine "teach one basic mechanic" content.
  * CORPUS / ATTRACT levels (generator == 'corpus'): real designed N++ levels
    decoded from human/attract demos (often official maps) — intricate, with
    mines/slopes/gold in the playable path. These are NOT basic-movement content
    and should come LATER, sorted by difficulty.

The benchmark loader derives bucket+tier from the ``<BUCKET>/T<tier>/`` path, so
this routes every level by its ``generator`` field:
  * procgen  -> keep its mechanic bucket + tier (the basic-mechanic block)
  * corpus   -> a single ``ATTRACT`` bucket, RE-TIERED by demo length (a proven
                execution-difficulty proxy) into uniform global quantiles, so the
                attract block is one clean difficulty progression.

This also cleanly splits the MIXED ``WALL_JUMP`` bucket (procgen wall-jumps stay a
basic mechanic; corpus wall-jumps move to ATTRACT) by the same generator test.

Usage::

    python -m scripts.levelgen.separate_attract \
        --in datasets/levelgen/generalist_benchmark \
        --out datasets/levelgen/generalist_curriculum_v2 \
        --attract-tiers 6 [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ATTRACT_BUCKET = "ATTRACT"


def _scan(in_root: Path, split: str, include_official: bool = False) -> List[dict]:
    rows = []
    for pkl in (in_root / split).rglob("*.pkl"):
        try:
            d = pickle.load(open(pkl, "rb"))
        except Exception:
            continue
        gen = d.get("generator")
        is_official = (gen == "official")
        # ``include_official`` folds valid official maps into the ATTRACT bucket
        # alongside the human/attract corpus (the FINAL difficulty phase). Official
        # maps carry no demo, so they are tiered separately (top tier) below.
        is_corpus = (gen == "corpus") or (include_official and is_official)
        demo_len = (d.get("metadata") or {}).get("demo_len")
        rows.append(dict(
            level_id=int(d["level_id"]), payload=d, pkl=pkl,
            bucket=str(d.get("bucket") or pkl.parent.parent.name),
            tier=int(d.get("tier", 1)),
            is_corpus=is_corpus,
            is_official=is_official,
            demo_len=float(demo_len) if demo_len else None,
        ))
    return rows


def _attract_tiers(rows: List[dict], n_tiers: int) -> Dict[int, int]:
    """Global demo-length quantile tiers (1..n_tiers) for the corpus levels.
    Levels missing a demo length fall to the median band."""
    corp = [r for r in rows if r["is_corpus"]]
    # Only demo-carrying corpus levels define the demo-length quantile edges;
    # official maps (no demo) are placed at the TOP tier as the hardest content.
    dl = np.array([r["demo_len"] for r in corp
                   if r["demo_len"] is not None and not r.get("is_official")])
    edges = (np.quantile(dl, np.linspace(0, 1, n_tiers + 1)[1:-1]) if dl.size
             else np.array([]))
    out = {}
    for r in corp:
        if r.get("is_official"):
            out[r["level_id"]] = n_tiers  # official = final difficulty (top tier)
        elif r["demo_len"] is None:
            out[r["level_id"]] = (n_tiers + 1) // 2
        else:
            out[r["level_id"]] = int(np.searchsorted(edges, r["demo_len"], side="right")) + 1
    return out


def _process(in_root: Path, n_tiers: int, include_official: bool = False):
    res = {}
    for split in ("train", "test"):
        rows = _scan(in_root, split, include_official=include_official)
        atiers = _attract_tiers(rows, n_tiers)
        res[split] = dict(rows=rows, atiers=atiers)
    return res


def _report(processed, n_tiers):
    for split, blk in processed.items():
        rows, atiers = blk["rows"], blk["atiers"]
        proc = [r for r in rows if not r["is_corpus"]]
        corp = [r for r in rows if r["is_corpus"]]
        print(f"\n===== {split}: {len(proc)} procgen (basic mechanics) | "
              f"{len(corp)} attract =====")
        pc = Counter(r["bucket"] for r in proc)
        print("  procgen buckets:", dict(sorted(pc.items())))
        at = Counter(atiers[r["level_id"]] for r in corp)
        dl_by = defaultdict(list)
        for r in corp:
            dl_by[atiers[r["level_id"]]].append(r["demo_len"] or 0)
        print(f"  ATTRACT tiers (n/level, median demo_len — must increase):")
        for t in range(1, n_tiers + 1):
            v = dl_by.get(t, [])
            print(f"    T{t}: n={at.get(t,0):4d}  median_demo_len={np.median(v):.0f}" if v
                  else f"    T{t}: n=0")


def _write(in_root, out_root, processed, n_tiers):
    if out_root.exists():
        raise SystemExit(f"refusing to overwrite {out_root}")
    out_root.mkdir(parents=True)
    manifest = json.load(open(in_root / "manifest.json"))
    manifest["reorganized"] = {"by": "separate_attract", "attract_tiers": n_tiers,
                               "attract_bucket": ATTRACT_BUCKET}
    json.dump(manifest, open(out_root / "manifest.json", "w"), indent=2)

    for split, blk in processed.items():
        rows, atiers = blk["rows"], blk["atiers"]
        buckets_meta: Dict[str, Dict] = {}
        all_ids: List[int] = []
        for r in rows:
            if r["is_corpus"]:
                bucket, tier = ATTRACT_BUCKET, atiers[r["level_id"]]
            else:
                bucket, tier = r["bucket"], r["tier"]
            dst = out_root / split / bucket / f"T{tier}"
            dst.mkdir(parents=True, exist_ok=True)
            pl = r["payload"]
            pl["tier"] = tier
            if r["is_corpus"]:
                pl["bucket"] = ATTRACT_BUCKET
            pickle.dump(pl, open(dst / f"{r['level_id']}.pkl", "wb"))
            all_ids.append(r["level_id"])
            tb = buckets_meta.setdefault(bucket, {}).setdefault("tiers", {})
            row = tb.setdefault(str(tier), {"level_ids": [], "path": f"{split}/{bucket}/T{tier}", "count": 0})
            row["level_ids"].append(r["level_id"])
            row["count"] += 1
        meta = dict(split=split, total_levels=len(all_ids), level_ids=all_ids, buckets=buckets_meta)
        json.dump(meta, open(out_root / split / f"{split}_metadata.json", "w"))
        print(f"  wrote {split}: {len(all_ids)} levels")
    print(f"\nReorganized curriculum at: {out_root}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_root", required=True)
    p.add_argument("--out", dest="out_root", default=None)
    p.add_argument("--attract-tiers", type=int, default=6)
    p.add_argument("--include-official", action="store_true",
                   help="Also fold generator=='official' maps into the ATTRACT "
                        "bucket (at the top tier) so valid official maps join the "
                        "final difficulty phase. Requires pack_benchmark "
                        "--official-split train so they are in the train split.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    in_root = Path(args.in_root).expanduser().resolve()
    processed = _process(in_root, args.attract_tiers,
                         include_official=args.include_official)
    _report(processed, args.attract_tiers)
    if args.dry_run or not args.out_root:
        print("\n(dry-run: nothing written)")
        return 0
    _write(in_root, Path(args.out_root).expanduser().resolve(), processed, args.attract_tiers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
