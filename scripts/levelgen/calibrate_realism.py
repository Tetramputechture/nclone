#!/usr/bin/env python3
"""Fit the human-manifold realism model and report how our generators score.

Splits the human corpus into a reference half and a held-out half, fits the
robust feature scaling on the reference, and sets the acceptance threshold at the
**held-out** p99. Calibrating on held-out humans rather than on our own output is
the whole point: the threshold answers "how far off-manifold does a real level
ever get", so any rejection of our generators is a measurement, not a target.

Writes ``nclone/levelgen/realism_calibration.npz``.

Usage:
  python -m scripts.levelgen.calibrate_realism \
      --corpus datasets/levelgen/corpus_full --stride 8 \
      --benchmark datasets/mechanic_benchmark_v7_official/train
"""
from __future__ import annotations

import argparse
import glob
import pickle
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nclone.levelgen.realism import (  # noqa: E402
    CALIBRATION_PATH,
    FEATURE_KEYS,
    RealismModel,
    level_features,
)


def load_corpus(corpus: str, stride: int) -> list:
    shards = sorted(glob.glob(f"{corpus}/shard_*.pkl"))
    if not shards:
        raise SystemExit(f"no shards under {corpus}")
    feats = []
    for f in shards[::stride]:
        with open(f, "rb") as fh:
            for rec in pickle.load(fh)["records"]:
                feats.append(level_features(rec["map_data"]))
    return feats


def robust_scale(ref: np.ndarray):
    """Median centre + IQR-ish spread.

    Robust rather than mean/std because the corpus tails hold troll and art
    levels; a couple of 1988-entity maps would otherwise set the scale.
    """
    center = np.median(ref, axis=0)
    spread = np.quantile(np.abs(ref - center), 0.75, axis=0)
    spread[spread < 1e-6] = 1e-6
    return center, spread


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="datasets/levelgen/corpus_full")
    ap.add_argument("--stride", type=int, default=8,
                    help="Use every Nth shard (the corpus is ~107 shards).")
    ap.add_argument("--reference-size", type=int, default=8000,
                    help="Human levels kept in the shipped model.")
    ap.add_argument("--holdout-size", type=int, default=4000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--quantile", type=float, default=0.99,
                    help="Held-out human quantile that becomes the threshold.")
    ap.add_argument("--benchmark", default=None,
                    help="Optional benchmark dataset root; scores each bucket.")
    ap.add_argument("--out", default=str(CALIBRATION_PATH))
    ap.add_argument("--seed", type=int, default=20260802)
    args = ap.parse_args()

    print(f"Loading human corpus from {args.corpus} (stride {args.stride}) ...")
    feats = load_corpus(args.corpus, args.stride)
    print(f"  {len(feats)} levels, {len(FEATURE_KEYS)} features")

    random.Random(args.seed).shuffle(feats)
    need = args.reference_size + args.holdout_size
    if len(feats) < need:
        raise SystemExit(f"need >= {need} levels, corpus sample gave {len(feats)}")

    ref_raw = np.asarray(feats[: args.reference_size], dtype="float64")
    held_raw = np.asarray(
        feats[args.reference_size : args.reference_size + args.holdout_size],
        dtype="float64",
    )

    center, scale = robust_scale(ref_raw)
    reference = (ref_raw - center) / scale

    model = RealismModel(center, scale, reference, threshold=float("inf"), k=args.k)
    held_scores = np.asarray([model.score_features(f) for f in held_raw])
    threshold = float(np.quantile(held_scores, args.quantile))
    model.threshold = threshold

    print("\nheld-out human scores: "
          f"p50={np.quantile(held_scores, .5):.2f} "
          f"p90={np.quantile(held_scores, .9):.2f} "
          f"p99={np.quantile(held_scores, .99):.2f}")
    print(f"--> threshold = {threshold:.3f} (held-out p{args.quantile * 100:g})")

    np.savez_compressed(
        args.out,
        center=center,
        scale=scale,
        reference=reference,
        threshold=threshold,
        k=args.k,
        feature_keys=np.array(FEATURE_KEYS),
    )
    size_kb = Path(args.out).stat().st_size / 1024
    print(f"wrote {args.out} ({size_kb:.0f} KB)")

    if args.benchmark:
        print(f"\n{'bucket':28s} {'n':>5s} {'p50':>7s} {'p90':>7s} {'rejected':>9s}")
        print("-" * 60)
        buckets: dict = {}
        for p in glob.glob(f"{args.benchmark}/*/*/*.pkl"):
            with open(p, "rb") as fh:
                d = pickle.load(fh)
            buckets.setdefault(d["bucket"], []).append(d["map_data"])
        for b in sorted(buckets):
            mds = buckets[b][:300]
            s = np.asarray([model.score(m) for m in mds])
            print(f"{b:28s} {len(s):5d} {np.quantile(s, .5):7.2f} "
                  f"{np.quantile(s, .9):7.2f} "
                  f"{100 * (s > threshold).mean():8.0f}%")
        print("\nBuckets built from real maps (ATTRACT / OFFICIAL) should pass at "
              "roughly the held-out human rate; synthetic buckets rejecting near "
              "100% is the drift this model exists to measure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
