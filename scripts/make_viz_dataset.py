#!/usr/bin/env python3
"""Build a throwaway benchmark-layout dataset from raw generator presets, for viz.

Motivation: to eyeball a generator you either step the live browser one level at a
time, or you need levels in a dataset so ``test_environment.py --benchmark-montage``
can contact-sheet them. ``--export-frame`` does NOT work for this -- with
``--test-generators`` it renders something other than the browser's current map
(verified: two different presets exported byte-identical PNGs while the generators
themselves produced clearly different levels).

This writes the minimal structure the montage loader needs:

    <out>/train/<BUCKET>/T<n>/<level_id>.pkl
    <out>/train/train_metadata.json
    <out>/manifest.json

    make_viz_dataset.py --spec mine_maze:medium --spec mine_maze:complex \
        --spec mine_maze:heavy --n 12 --out /tmp/mm_viz
    python -m nclone.test_environment --benchmark-montage \
        --benchmark-split train --test-dataset-path /tmp/mm_viz --montage-output /tmp/mm_sheet

NOT a training dataset: no dedup, no solvability gate, no tier semantics -- levels
are binned by ``--spec`` order purely so the sheets group sensibly.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", action="append", required=True,
                    help="generator:preset (repeatable)")
    ap.add_argument("--n", type=int, default=12, help="levels per spec")
    ap.add_argument("--seed0", type=int, default=10001)
    ap.add_argument("--bucket", default="VIZ")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from nclone.map_generation.generator_factory import GeneratorFactory
    from nclone.map_generation.solvability import is_map_solvable

    out = Path(args.out)
    split_dir = out / "train"
    per_tier: dict[int, list[int]] = {}
    total = 0

    for ti, spec in enumerate(args.spec, start=1):
        gen_type, preset = spec.split(":", 1)
        tier_dir = split_dir / args.bucket / f"T{ti}"
        tier_dir.mkdir(parents=True, exist_ok=True)
        ids: list[int] = []
        solvable = 0
        for k in range(args.n):
            seed = args.seed0 + k
            try:
                g = GeneratorFactory.create_from_preset(gen_type, preset, seed)
                g.generate(seed=seed)
                md = list(g.map_data())
            except Exception as exc:
                print(f"  {spec} seed={seed}: FAILED {type(exc).__name__}: {exc}")
                continue
            if is_map_solvable(md):
                solvable += 1
            lid = ti * 1_000_000 + k
            ids.append(lid)
            with open(tier_dir / f"{lid}.pkl", "wb") as f:
                pickle.dump({"level_id": lid, "bucket": args.bucket, "tier": ti,
                             "map_data": md, "spec": spec, "seed": seed}, f)
            total += 1
        per_tier[ti] = ids
        print(f"T{ti} {spec:24s} {len(ids):3d} levels, graph-solvable {solvable}/{len(ids)}")

    meta = {
        "split": "train",
        "total_levels": total,
        "buckets": {args.bucket: {
            "total": total,
            "tiers": {str(t): {"count": len(v), "level_ids": v,
                               "path": f"train/{args.bucket}/T{t}"}
                      for t, v in per_tier.items()},
        }},
        "level_ids": [i for v in per_tier.values() for i in v],
    }
    (split_dir / "train_metadata.json").write_text(json.dumps(meta, indent=2))
    (out / "manifest.json").write_text(json.dumps(
        {"viz_only": True, "specs": args.spec, "total": total}, indent=2))
    print(f"\nwrote {total} levels -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
