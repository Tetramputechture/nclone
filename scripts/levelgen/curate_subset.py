"""curate_subset.py — pick a small, diverse, *labeled* subset from a packed level pool.

The confirmation artifact for the level-generation pipeline: given a materialized CME
pool (``cme_run.py`` output), select ~N levels that span the three generation paths
(``seed`` / ``mutant_faithful`` / ``grammar``), the difficulty bands (d0..d4), and as many
distinct mechanic compositions as possible, render each to a labelled schematic, and write
a ``labels.json`` manifest + a Markdown summary so a human can eyeball that the pipeline
produces valid, varied, correctly-labeled levels.

Selection is greedy max-coverage over the (source, band, mechanic-combo) key: it first
takes one level from every distinct key, then fills the remaining budget by least-covered
key, so a 25-level subset shows the widest spread the pool supports.

Usage
-----
    python -m scripts.levelgen.curate_subset --pool datasets/levelgen/pipeline_check \\
        --out viz/pipeline_check --n 25
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from collections import defaultdict
from typing import Dict, List

logger = logging.getLogger("levelgen.curate_subset")


def _read_index(pool_dir: str) -> List[dict]:
    recs: List[dict] = []
    for f in sorted(glob.glob(os.path.join(pool_dir, "index*.jsonl"))):
        recs += [json.loads(line) for line in open(f) if line.strip()]
    return recs


def _blob(pool_dir: str, rec: dict, cache: Dict[str, bytes]) -> List[int]:
    shard = rec["shard"]
    if shard not in cache:
        cache[shard] = open(os.path.join(pool_dir, shard), "rb").read()
    return list(cache[shard][rec["off"] : rec["off"] + rec["len"]])


def _key(r: dict):
    return (r.get("source", "?"), r.get("band", "?"), r.get("mech", "?"))


def _greedy_facets(remaining: List[dict], n: int, cov_band: set, cov_mech: set,
                   cov_cell: set) -> List[dict]:
    """Greedy: each pick maximizes newly-covered band + mechanic-combo facets, tie-break fitness."""
    out: List[dict] = []
    while remaining and len(out) < n:
        def gain(r):
            new = (r.get("band") not in cov_band) + (r.get("mech") not in cov_mech)
            return (new, _key(r) not in cov_cell, r.get("fitness", 0.0))
        best = max(remaining, key=gain)
        remaining.remove(best)
        out.append(best)
        cov_band.add(best.get("band")); cov_mech.add(best.get("mech")); cov_cell.add(_key(best))
    return out


def _select(recs: List[dict], n: int) -> List[dict]:
    """Source-stratified, then greedy facet coverage within and across sources.

    Each generation path (``seed`` / ``mutant_faithful`` / ``grammar``) gets a fair quota
    (an equal split, capped by how many that path actually produced and topped up by the
    largest-remainder of the unused budget), so the confirmation subset *shows all three
    paths working*, not just the most prolific one. Within each path the picks greedily
    cover difficulty bands then distinct mechanic compositions; a final global pass spends
    any leftover budget on the widest remaining variety.
    """
    by_src: Dict[str, List[dict]] = defaultdict(list)
    for r in recs:
        by_src[r.get("source", "?")].append(r)
    srcs = sorted(by_src, key=lambda s: -len(by_src[s]))

    # quota: equal split, capped by availability, largest-remainder top-up of the surplus
    quota = {s: min(len(by_src[s]), n // len(srcs)) for s in srcs}
    while sum(quota.values()) < n and any(quota[s] < len(by_src[s]) for s in srcs):
        for s in sorted(srcs, key=lambda s: (quota[s], -len(by_src[s]))):
            if quota[s] < len(by_src[s]):
                quota[s] += 1
                if sum(quota.values()) >= n:
                    break

    cov_band: set = set()
    cov_mech: set = set()
    cov_cell: set = set()
    picked: List[dict] = []
    leftovers: List[dict] = []
    for s in srcs:
        ranked = sorted(by_src[s], key=lambda r: -r.get("fitness", 0.0))
        got = _greedy_facets(ranked, quota[s], cov_band, cov_mech, cov_cell)
        picked += got
        leftovers += ranked  # _greedy_facets removed the chosen ones in place
    # spend any shortfall (a path had fewer than its quota) on global variety
    if len(picked) < n:
        picked += _greedy_facets(leftovers, n - len(picked), cov_band, cov_mech, cov_cell)
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, help="packed-pool dir (shard_*.bin + index*.jsonl)")
    ap.add_argument("--out", required=True, help="output dir for PNGs + labels.json + summary.md")
    ap.add_argument("--n", type=int, default=25, help="target subset size")
    ap.add_argument("--ground-truth", action="store_true",
                    help="pixel-faithful game renderer instead of the labelled schematic")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from nclone.map_generation.compose.render_image import render_schematic, render_ground_truth

    recs = _read_index(args.pool)
    if not recs:
        raise SystemExit(f"no index records under {args.pool}")
    picked = _select(recs, args.n)
    logger.info("pool=%d levels; picked %d covering %d distinct (source,band,mech) keys",
                len(recs), len(picked), len({_key(r) for r in picked}))

    os.makedirs(args.out, exist_ok=True)
    cache: Dict[str, bytes] = {}
    render = render_ground_truth if args.ground_truth else render_schematic
    labels: List[dict] = []
    paths: List[str] = []
    for i, r in enumerate(picked):
        md = _blob(args.pool, r, cache)
        png = os.path.join(args.out, f"{i:02d}_{r.get('source','?')}_{r.get('band','?')}_{r['id']}.png")
        title = (f"#{i:02d} {r.get('source','?')} | {r.get('band','?')} "
                 f"fs={r.get('fs_cost', 0):.0f} | {r.get('mech','?')} | "
                 f"mines={r.get('n_mines','?')} gold={r.get('n_gold','?')} struct={r.get('struct','?')}")
        try:
            if args.ground_truth:
                render(md, png)
            else:
                render(md, png, title=title)
            paths.append(png)
        except Exception as e:  # pragma: no cover
            logger.warning("render failed for %s: %s", r["id"], e)
        labels.append({
            "idx": i, "id": r["id"], "png": os.path.basename(png),
            "source": r.get("source"), "band": r.get("band"), "fs_cost": r.get("fs_cost"),
            "mechanic": r.get("mech"), "cell": r.get("cell"),
            "n_mines": r.get("n_mines"), "n_gold": r.get("n_gold"),
            "struct": r.get("struct"), "fitness": r.get("fitness"),
            "has_demo": "demo" in r or bool(r.get("demo_len")), "parent": r.get("parent"),
        })

    json.dump(labels, open(os.path.join(args.out, "labels.json"), "w"), indent=2)
    _write_summary(args.out, labels, len(recs))
    _contact_sheet(args.out, paths)
    logger.info("DONE -> %s (labels.json, summary.md, contact.png, %d PNGs)", args.out, len(paths))


def _write_summary(out: str, labels: List[dict], pool_size: int) -> None:
    by_src = defaultdict(int)
    by_band = defaultdict(int)
    mechs = set()
    for r in labels:
        by_src[r["source"]] += 1
        by_band[r["band"]] += 1
        mechs.add(r["mechanic"])
    lines = [
        "# Curated level-generation confirmation subset",
        "",
        f"- pool size: **{pool_size}** materialized levels",
        f"- curated: **{len(labels)}** levels",
        f"- sources: {dict(by_src)}",
        f"- bands: {dict(sorted(by_band.items()))}",
        f"- distinct mechanic compositions: **{len(mechs)}**",
        "",
        "| # | source | band | fs_cost | mechanic | mines | gold | struct | demo |",
        "|---|--------|------|---------|----------|-------|------|--------|------|",
    ]
    for r in labels:
        fs = f"{r['fs_cost']:.0f}" if r.get("fs_cost") is not None else "-"
        lines.append(
            f"| {r['idx']:02d} | {r['source']} | {r['band']} | {fs} | {r['mechanic']} | "
            f"{r['n_mines']} | {r['n_gold']} | {r['struct']} | {'Y' if r['has_demo'] else '-'} |")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")


def _contact_sheet(out: str, paths: List[str]) -> None:
    if not paths:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(paths)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.1))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax, p in zip(axes, paths):
        ax.imshow(plt.imread(p))
        ax.axis("off")
    for ax in axes[len(paths):]:
        ax.axis("off")
    fig.suptitle(f"curated subset — {n} levels", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "contact.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
