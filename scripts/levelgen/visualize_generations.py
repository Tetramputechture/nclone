"""Validate the trained level generator: sample a few levels, gate them for
solvability, report validation stats, and render each to a PNG.

Reuses the production sampling/gate/assemble path from ``generate_levels`` so what
you see here is exactly what the benchmark writer would keep.

Example::

    python -m scripts.levelgen.visualize_generations --tile-ckpt ckpt/tile.pt \\
        --entity-ckpt ckpt/entity.pt --n 8 --difficulty d2 --mechanic mine \\
        --n-mines 4 --out generated_preview --ground-truth
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import torch

from nclone.levelgen.tokenizer import (
    DIFFICULTY_BANDS,
    MECHANIC_TAGS,
    SOURCES,
    assemble_map_data,
)
from scripts.levelgen.generate_levels import (
    _maybe_load_entity,
    _maybe_load_tile,
    _template_header,
    gate_with_timeout,
    place_and_gate,
)

logger = logging.getLogger("levelgen.visualize")

NINJA = (86, 42, 0)
HUMAN_FILL = 0.58  # training-distribution tile density (subset+official, measured via eval_generator)


def _pairwise_diversity(tiles) -> float:
    """Mean normalized Hamming distance between all sample pairs (0=identical, 1=all differ)."""
    import torch as _t
    n = tiles.shape[0]
    if n < 2:
        return float("nan")
    tot, npairs = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += (tiles[i] != tiles[j]).float().mean().item()
            npairs += 1
    return tot / max(npairs, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate + visualize generated levels")
    ap.add_argument("--tile-ckpt", required=True)
    ap.add_argument("--entity-ckpt", required=True)
    ap.add_argument("--out", default="generated_preview")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--difficulty", default="d2", choices=list(DIFFICULTY_BANDS))
    ap.add_argument("--mechanic", default="mine", choices=list(MECHANIC_TAGS))
    ap.add_argument("--n-mines", type=int, default=4)
    ap.add_argument("--n-gold", type=int, default=0)
    ap.add_argument("--cfg-scale", type=float, default=2.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gumbel-temp", type=float, default=0.0, help="MaskGIT commit-ranking Gumbel noise (diversity)")
    ap.add_argument("--steps", type=int, default=12, help="maskgit decode steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-timeout", type=float, default=30.0, help="per-level gate cap (s); 0=off (legacy fixed-spawn gate only)")
    ap.add_argument("--no-gate", action="store_true", help="skip solvability gate (render all)")
    ap.add_argument("--place-and-gate", action="store_true",
                    help="place ninja on a reachable floor surface before gating (Stage A fix)")
    ap.add_argument("--spawn-prefer", default="hardest", choices=["hardest", "central"],
                    help="which reachable floor cell to spawn on")
    ap.add_argument("--ground-truth", action="store_true", help="also render the pixel-faithful game frame")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from nclone.map_generation.compose.render_image import render_schematic

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    tile_model = _maybe_load_tile(args.tile_ckpt, args.device).eval()
    ent_model = _maybe_load_entity(args.entity_ckpt, args.device).eval()
    header = _template_header()

    cond = {
        "source": torch.full((args.n,), SOURCES.index("generated"), dtype=torch.long),
        "difficulty": torch.full((args.n,), DIFFICULTY_BANDS.index(args.difficulty), dtype=torch.long),
        "subset": torch.ones(args.n, dtype=torch.long),
        "mech": torch.zeros(args.n, len(MECHANIC_TAGS)),
    }
    cond["mech"][:, MECHANIC_TAGS.index(args.mechanic)] = 1.0

    logger.info("Sampling %d levels (%s/%s, steps=%d, cfg=%.1f, temp=%.1f) ...",
                args.n, args.mechanic, args.difficulty, args.steps, args.cfg_scale, args.temperature)
    tiles = tile_model.sample(args.n, cond=cond, steps=args.steps, temperature=args.temperature,
                              cfg_scale=args.cfg_scale, device=args.device, generator=gen,
                              gumbel_temp=args.gumbel_temp)
    ent_sets = ent_model.constrained_generate(
        tiles, n_mines=args.n_mines, n_gold=args.n_gold, temperature=args.temperature, generator=gen
    )

    n_solvable = 0
    d_stars = []
    filled_fracs = []
    for i in range(args.n):
        tile_list = tiles[i].tolist()

        # tile-grid validation stats
        counts = Counter(tile_list)
        n_empty = counts.get(0, 0)
        filled = 1.0 - n_empty / len(tile_list)
        distinct = len([t for t in counts if t != 0])
        filled_fracs.append(filled)

        if args.no_gate:
            md = assemble_map_data(tile_list, NINJA, ent_sets[i], header)
            d_star = None
        elif args.place_and_gate:
            md, d_star = place_and_gate(tile_list, ent_sets[i], header, prefer=args.spawn_prefer)
            if md is None:  # repair failed -> still render the raw (fixed-spawn) layout
                md = assemble_map_data(tile_list, NINJA, ent_sets[i], header)
        else:
            md = assemble_map_data(tile_list, NINJA, ent_sets[i], header)
            d_star = gate_with_timeout(md, args.gate_timeout)
        solvable = d_star is not None
        n_solvable += int(solvable)
        if solvable:
            d_stars.append(d_star)

        tag = "UNGATED" if args.no_gate else (f"d*={d_star:.0f}f" if solvable else "UNSOLVABLE")
        title = f"#{i} {args.mechanic}/{args.difficulty} | {tag} | fill={filled:.0%} types={distinct}"
        png = out / f"level_{i:02d}_{'solv' if solvable else ('ungated' if args.no_gate else 'unsolv')}.png"
        render_schematic(md, str(png), title=title)
        if args.ground_truth:
            try:
                from nclone.map_generation.compose.render_image import render_ground_truth
                render_ground_truth(md, str(out / f"level_{i:02d}_gameframe.png"))
            except Exception as e:
                logger.warning("ground-truth render failed for #%d: %s", i, e)
        logger.info("%s -> %s", title, png.name)

    # ---- validation summary ----
    diversity = _pairwise_diversity(tiles)
    gate_label = "place-and-gate" if args.place_and_gate else "fixed-spawn gate"
    print("\n==================== VALIDATION SUMMARY ====================")
    print(f"config                 : {args.mechanic}/{args.difficulty} "
          f"cfg={args.cfg_scale} temp={args.temperature} steps={args.steps} seed={args.seed}")
    print(f"levels sampled         : {args.n}")
    if not args.no_gate:
        rate = 100 * n_solvable / args.n
        print(f"solvable ({gate_label}): {n_solvable}/{args.n}  ({rate:.0f}%)")
        if d_stars:
            print(f"d* (frames) solvable   : min={min(d_stars):.0f} "
                  f"mean={sum(d_stars)/len(d_stars):.0f} max={max(d_stars):.0f}")
    print(f"tile fill fraction     : min={min(filled_fracs):.0%} "
          f"mean={sum(filled_fracs)/len(filled_fracs):.0%} max={max(filled_fracs):.0%} "
          f"(human ~{HUMAN_FILL:.0%})")
    print(f"pairwise diversity     : {diversity:.3f}  (0=identical, 1=all tiles differ)")
    print(f"images written to      : {out}/")
    print("===========================================================")


if __name__ == "__main__":
    main()
