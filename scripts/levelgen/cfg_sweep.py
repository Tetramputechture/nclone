"""CFG calibration sweep for the tile stage (A1 post-train step).

The W0 monitoring miss was trusting loss while the *samples* were broken. This tool
sweeps the classifier-free-guidance scale and reports the **sample scalars** that
actually matter, per difficulty band:

* ``fill``          — mean tile density (the additive path collapses to ~0.94 here);
* ``tile_kl_unigram`` / ``tile_kl_patch`` — KL of the generated tile distribution vs a
  reference dataset (lower = closer to the data; success target unigram < 0.21);
* ``diversity``     — mean pairwise Hamming across samples (mode-collapse guard);
* ``structure``     — mean structure score (rich vs sparse).

Use it to pick the CFG operating point for a ``cond_mode=prepend`` model and to confirm
it beats the procgen / additive baseline. Run after training a prepend tile model.

Example::

    python -m scripts.levelgen.cfg_sweep --tile-ckpt ckpt/tile_prepend.pt \\
        --data datasets/levelgen/tiles_subset_fs_mir.h5 \\
        --cfg 0 1 1.5 2 3 --per-cfg 64 --difficulty d0 d2 d4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from nclone.levelgen.models.tile_transformer import TileConfig, TileTransformer
from nclone.levelgen.tokenizer import DIFFICULTY_BANDS, MECHANIC_TAGS, SOURCES
from nclone.levelgen.structure import describe_tiles
from scripts.levelgen.eval_generator import unigram_dist, patch2x2_dist, _kl

logger = logging.getLogger("levelgen.cfg_sweep")


def _load_tile(path, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    m = TileTransformer(TileConfig(**blob["config"])).to(device)
    m.load_state_dict(blob["state_dict"])
    return m.eval(), blob["config"]


def _ref_tiles(data_path, n):
    from nclone.levelgen.dataset import TileDataset

    ds = TileDataset(data_path)
    stride = max(1, len(ds) // n)
    return [ds[i]["tiles"].numpy() for i in range(0, stride * n, stride)]


def _diversity(tiles_np):
    """Mean pairwise Hamming distance (fraction differing) across sampled grids."""
    if len(tiles_np) < 2:
        return 0.0
    arr = np.stack(tiles_np)
    m = min(len(arr), 64)
    arr = arr[:m]
    tot = cnt = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            tot += np.mean(arr[i] != arr[j])
            cnt += 1
    return float(tot / max(cnt, 1))


def _cond(diff_band, mechanic, n, device):
    c = {
        "source": torch.full((n,), SOURCES.index("human"), dtype=torch.long, device=device),
        "difficulty": torch.full((n,), DIFFICULTY_BANDS.index(diff_band), dtype=torch.long, device=device),
        "subset": torch.ones(n, dtype=torch.long, device=device),
        "mech": torch.zeros(n, len(MECHANIC_TAGS), device=device),
    }
    if mechanic in MECHANIC_TAGS:
        c["mech"][:, MECHANIC_TAGS.index(mechanic)] = 1.0
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description="CFG calibration sweep (sample scalars)")
    ap.add_argument("--tile-ckpt", required=True)
    ap.add_argument("--data", required=True, help="HDF5 tiles dataset for the KL reference")
    ap.add_argument("--cfg", type=float, nargs="+", default=[0.0, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--difficulty", nargs="+", default=["d0", "d2", "d4"])
    ap.add_argument("--mechanic", default="mine", choices=list(MECHANIC_TAGS))
    ap.add_argument("--per-cfg", type=int, default=64, help="samples per (cfg, difficulty)")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gumbel-temp", type=float, default=0.0)
    ap.add_argument("--kl-ref-n", type=int, default=2000)
    ap.add_argument("--out", default=None, help="write the sweep table JSON here")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model, config = _load_tile(args.tile_ckpt, args.device)
    logger.info("loaded tile model cond_mode=%s mode=%s", config.get("cond_mode", "add"), config.get("mode"))
    ref = _ref_tiles(args.data, args.kl_ref_n)
    ref_uni, ref_patch = unigram_dist(ref), patch2x2_dist(ref)
    logger.info("KL reference built from %d levels", len(ref))

    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    rows = []
    for band in args.difficulty:
        for cfg_scale in args.cfg:
            cond = _cond(band, args.mechanic, args.per_cfg, args.device)
            tiles = model.sample(
                args.per_cfg, cond=cond, steps=args.steps, temperature=args.temperature,
                cfg_scale=cfg_scale, device=args.device, generator=gen, gumbel_temp=args.gumbel_temp,
            )
            tnp = [t.cpu().numpy() for t in tiles]
            fill = float(np.mean([(t != 0).mean() for t in tnp]))
            kl_u = _kl(unigram_dist(tnp), ref_uni)
            kl_p = _kl(patch2x2_dist(tnp), ref_patch)
            div = _diversity(tnp)
            struct = float(np.mean([describe_tiles(t.tolist()).structure_score for t in tnp]))
            row = {
                "difficulty": band, "cfg": cfg_scale, "fill": round(fill, 4),
                "tile_kl_unigram": round(kl_u, 4), "tile_kl_patch": round(kl_p, 4),
                "diversity": round(div, 4), "structure": round(struct, 4),
            }
            rows.append(row)
            logger.info("band=%s cfg=%.1f | fill=%.3f kl_uni=%.3f kl_patch=%.3f div=%.3f struct=%.3f",
                        band, cfg_scale, fill, kl_u, kl_p, div, struct)

    # flag the recommended operating point: lowest unigram-KL with fill in a sane band
    sane = [r for r in rows if 0.2 <= r["fill"] <= 0.6]
    best = min(sane or rows, key=lambda r: r["tile_kl_unigram"])
    logger.info("RECOMMENDED cfg=%.1f (band=%s): kl_uni=%.3f fill=%.3f (target kl_uni<0.21)",
                best["cfg"], best["difficulty"], best["tile_kl_unigram"], best["fill"])
    if args.out:
        Path(args.out).write_text(json.dumps({"config": config, "rows": rows, "recommended": best}, indent=2))
        logger.info("wrote sweep -> %s", args.out)


if __name__ == "__main__":
    main()
