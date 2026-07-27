"""viz_pool.py — render a difficulty-stratified visualization subset of a level pool.

Samples N levels per difficulty band from a packed pool (e.g. the output of
``augment_corpus.py``) and renders each to a labelled schematic (solid terrain + the
player / mines / gold / exit switch+door at their true positions). Also assembles one
per-band contact sheet so you can eyeball whether the bands actually look
easy→hard and the levels are sane.

Usage
-----
    python -m scripts.levelgen.viz_pool --pool datasets/levelgen/pool_augmented \\
        --out viz/pool_augmented --per-band 12 --seed 0
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
from collections import defaultdict
from typing import Dict, List

logger = logging.getLogger("levelgen.viz_pool")

_BAND_ORDER = ["d0", "d1", "d2", "d3", "d4", "unknown"]


def _read_index(pool_dir: str) -> List[dict]:
    recs: List[dict] = []
    for f in sorted(glob.glob(os.path.join(pool_dir, "index*.jsonl"))):
        recs += [json.loads(line) for line in open(f) if line.strip()]
    return recs


def _blob(pool_dir: str, rec: dict, _cache: Dict[str, bytes]) -> List[int]:
    shard = rec["shard"]
    if shard not in _cache:
        _cache[shard] = open(os.path.join(pool_dir, shard), "rb").read()
    b = _cache[shard][rec["off"] : rec["off"] + rec["len"]]
    return list(b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, help="packed-pool dir (shard_*.bin + index*.jsonl)")
    ap.add_argument("--out", required=True, help="output dir for PNGs + contact sheets")
    ap.add_argument("--per-band", type=int, default=12, help="levels rendered per difficulty band")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ground-truth", action="store_true",
                    help="use the pixel-faithful renderer instead of the labelled schematic")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from nclone.map_generation.compose.render_image import render_schematic, render_ground_truth

    recs = _read_index(args.pool)
    if not recs:
        raise SystemExit(f"no index records under {args.pool}")
    by_band: Dict[str, List[dict]] = defaultdict(list)
    for r in recs:
        by_band[r.get("band", "unknown")].append(r)
    logger.info("pool: %d levels; bands: %s", len(recs),
                {b: len(by_band[b]) for b in _BAND_ORDER if b in by_band})

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    cache: Dict[str, bytes] = {}
    render = render_ground_truth if args.ground_truth else render_schematic
    bands_present = [b for b in _BAND_ORDER if by_band.get(b)]
    rendered: Dict[str, List[str]] = {}

    for band in bands_present:
        sample = by_band[band]
        rng.shuffle(sample)
        sample = sample[: args.per_band]
        band_dir = os.path.join(args.out, band)
        os.makedirs(band_dir, exist_ok=True)
        paths: List[str] = []
        for r in sample:
            md = _blob(args.pool, r, cache)
            title = (f"{band} | demo_len={r.get('demo_len','?')} "
                     f"fs={('%.0f' % r['fs_cost']) if r.get('fs_cost') is not None else '-'} "
                     f"mines={r.get('n_mines','?')} gold={r.get('n_gold','?')} {r.get('op','')}")
            out_png = os.path.join(band_dir, f"{r['id']}.png")
            try:
                if args.ground_truth:
                    render(md, out_png)
                else:
                    render(md, out_png, title=title)
                paths.append(out_png)
            except Exception as e:  # pragma: no cover
                logger.warning("render failed for %s: %s", r["id"], e)
        rendered[band] = paths
        logger.info("band %s: rendered %d/%d", band, len(paths), len(sample))

    _contact_sheets(args.out, rendered)
    logger.info("DONE -> %s", args.out)


def _contact_sheets(out_dir: str, rendered: Dict[str, List[str]]) -> None:
    """One montage per band (grid of the rendered PNGs) for quick eyeballing."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for band, paths in rendered.items():
        if not paths:
            continue
        n = len(paths)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 2.2))
        axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
        for ax, p in zip(axes, paths):
            ax.imshow(plt.imread(p))
            ax.axis("off")
        for ax in axes[len(paths):]:
            ax.axis("off")
        fig.suptitle(f"band {band} — {n} levels", fontsize=12)
        fig.tight_layout()
        sheet = os.path.join(out_dir, f"contact_{band}.png")
        fig.savefig(sheet, dpi=110, bbox_inches="tight")
        plt.close(fig)
        logger.info("contact sheet -> %s", sheet)


if __name__ == "__main__":
    main()
