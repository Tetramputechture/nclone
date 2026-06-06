"""Phase 2: sample -> assemble -> solvability gate -> benchmark .pkl.

Pipeline per requested level:
  1. tile transformer samples a 42x23 grid (conditioned on difficulty/mechanic/subset);
  2. entity transformer constrained-decodes a subset-valid, exactly-counted entity set
     (1 exit door + 1 exit switch + k mines + m gold) placed via the model;
  3. ``assemble_map_data`` builds a loader-valid map_data;
  4. the lattice oracle gates it: keep iff ``available and spawn_finite`` (spawn read
     from the native sim); the realized ``d*`` becomes the verified difficulty label.

Survivors are written in the existing benchmark layout (``MechanicBenchmarkLoader``
consumes them unchanged). Without ``--tile-ckpt``/``--entity-ckpt`` the models are
untrained -- useful only to exercise the gate end-to-end (few/zero will pass).

Example::

    python -m scripts.levelgen.generate_levels --tile-ckpt ckpt/tile.pt \\
        --entity-ckpt ckpt/entity.pt --n 200 --difficulty d2 --mechanic mine \\
        --n-mines 6 --out datasets/levelgen/generated
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import List, Optional

import torch

from nclone.levelgen.tokenizer import (
    DIFFICULTY_BANDS,
    MECHANIC_TAGS,
    SOURCES,
    Entity,
    assemble_map_data,
)
from nclone.levelgen.models.tile_transformer import TileConfig, TileTransformer
from nclone.levelgen.models.entity_transformer import EntityConfig, EntityTransformer

logger = logging.getLogger("levelgen.generate")


# ------------------------------------------------------------------- template
def _template_header() -> bytes:
    """A loader-valid 184-byte header, taken from a fresh procgen level."""
    from nclone.map_generation.generator_factory import GeneratorFactory

    g = GeneratorFactory.create_from_preset("single_chamber", "flat_run_t1", 0)
    g.generate(seed=0)
    return bytes(list(g.map_data())[:184])


# ----------------------------------------------------------------------- gate
def gate(map_data: List[int]) -> Optional[float]:
    """Return realized ``d*`` (frames) if solvable, else None."""
    try:
        import nclone_native
        from nclone.value_function.lattice_distance import LatticeDistanceSource

        sim = nclone_native.Simulator.load(list(map_data))
        spawn = sim.get_ninja_pos()
        src = LatticeDistanceSource(list(map_data), use_cache=True, name="levelgen")
        if not src.available or not src.spawn_finite(*spawn):
            return None
        d = src.cost_to_go(spawn[0], spawn[1], switch_activated=False)
        return float(d) if d != float("inf") else None
    except Exception as e:  # pragma: no cover
        logger.debug("gate error: %s", e)
        return None


def _gate_worker(map_data, q):  # pragma: no cover - runs in subprocess
    q.put(gate(map_data))


def gate_with_timeout(map_data: List[int], timeout: float) -> Optional[float]:
    """Gate in a worker process with a wall-clock cap.

    The lattice precompute can be pathologically slow on chaotic/troll geometry
    (random tile grids in particular), so a timeout keeps generation throughput
    bounded -- a level we cannot gate cheaply is simply skipped.
    """
    if timeout <= 0:
        return gate(map_data)
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_gate_worker, args=(list(map_data), q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None
    try:
        return q.get_nowait()
    except Exception:
        return None


# ----------------------------------------------------------------- benchmark io
def write_benchmark(levels, out_dir: Path, split: str, bucket: str):
    """Write levels in the MechanicBenchmarkLoader layout + manifest/metadata."""
    out_dir = Path(out_dir)
    tier_ids = {}
    for lv in levels:
        tier = lv["tier"]
        d = out_dir / split / bucket / f"T{tier}"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{lv['level_id']}.pkl", "wb") as f:
            pickle.dump(lv, f, protocol=4)
        tier_ids.setdefault(tier, []).append(lv["level_id"])

    split_meta = {
        "split": split,
        "total_levels": len(levels),
        # flat level_ids list is what MechanicBenchmarkLoader indexes on
        "level_ids": [lv["level_id"] for lv in levels],
        "buckets": {
            bucket: {
                "tiers": {
                    str(t): {
                        "level_ids": ids,
                        "path": f"{split}/{bucket}/T{t}",
                        "count": len(ids),
                    }
                    for t, ids in sorted(tier_ids.items())
                }
            }
        },
    }
    (out_dir / split).mkdir(parents=True, exist_ok=True)
    with open(out_dir / split / f"{split}_metadata.json", "w") as f:
        json.dump(split_meta, f, indent=2)
    manifest = {
        "version": 1,
        "source": "scripts.levelgen.generate_levels",
        "schema": "benchmark-v1",
        "buckets": [bucket],
        "tiers": sorted(tier_ids.keys()),
        "splits": [split],
        "generation_info": {"generator": "transformer"},
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _maybe_load_tile(path, device):
    if not path:
        return TileTransformer(TileConfig(d_model=128, n_layers=2, n_heads=4, d_ff=256)).to(device)
    blob = torch.load(path, map_location=device, weights_only=False)
    m = TileTransformer(TileConfig(**blob["config"])).to(device)
    m.load_state_dict(blob["state_dict"])
    return m


def _maybe_load_entity(path, device):
    if not path:
        return EntityTransformer(EntityConfig(d_model=128, n_layers=2, n_enc_layers=2, n_heads=4, d_ff=256)).to(device)
    blob = torch.load(path, map_location=device, weights_only=False)
    m = EntityTransformer(EntityConfig(**blob["config"])).to(device)
    m.load_state_dict(blob["state_dict"])
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate, gate, and write benchmark levels")
    ap.add_argument("--tile-ckpt", default=None)
    ap.add_argument("--entity-ckpt", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64, help="number of solvable levels wanted")
    ap.add_argument("--max-attempts", type=int, default=None, help="cap raw samples")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--split", default="train")
    ap.add_argument("--bucket", default="GENERATED")
    ap.add_argument("--difficulty", default="d2", choices=list(DIFFICULTY_BANDS))
    ap.add_argument("--mechanic", default="mine", choices=list(MECHANIC_TAGS))
    ap.add_argument("--n-mines", type=int, default=4)
    ap.add_argument("--n-gold", type=int, default=0)
    ap.add_argument("--cfg-scale", type=float, default=1.5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=12, help="maskgit decode steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-timeout", type=float, default=30.0, help="per-level gate wall-clock cap (s); 0=off")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    torch.manual_seed(args.seed)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    tile_model = _maybe_load_tile(args.tile_ckpt, args.device).eval()
    ent_model = _maybe_load_entity(args.entity_ckpt, args.device).eval()
    if not args.tile_ckpt or not args.entity_ckpt:
        logger.warning("Running with UNTRAINED models -- gate will reject most/all samples")

    header = _template_header()

    tier = max(1, DIFFICULTY_BANDS.index(args.difficulty))  # d0->1 .. d4->5(clamped to 4 below)
    tier = min(tier, 4)

    cond = {
        "source": torch.full((args.batch_size,), SOURCES.index("generated"), dtype=torch.long),
        "difficulty": torch.full((args.batch_size,), DIFFICULTY_BANDS.index(args.difficulty), dtype=torch.long),
        "subset": torch.ones(args.batch_size, dtype=torch.long),
        "mech": torch.zeros(args.batch_size, len(MECHANIC_TAGS)),
    }
    cond["mech"][:, MECHANIC_TAGS.index(args.mechanic)] = 1.0

    kept = []
    attempts = 0
    max_attempts = args.max_attempts or args.n * 20
    while len(kept) < args.n and attempts < max_attempts:
        bs = args.batch_size
        tiles = tile_model.sample(bs, cond=cond, steps=args.steps, temperature=args.temperature,
                                  cfg_scale=args.cfg_scale, device=args.device, generator=gen)
        ent_sets = ent_model.constrained_generate(
            tiles, n_mines=args.n_mines, n_gold=args.n_gold, temperature=args.temperature, generator=gen
        )
        for i in range(bs):
            attempts += 1
            tile_list = tiles[i].tolist()
            # ninja spawn from the sim default; place at a fixed interior unit, refined by gate
            ents = ent_sets[i]
            md = assemble_map_data(tile_list, (86, 42, 0), ents, header)
            d_star = gate_with_timeout(md, args.gate_timeout)
            if d_star is None:
                continue
            from scripts.levelgen.common import map_hash_int

            lid = map_hash_int(md)
            kept.append(
                {
                    "level_id": lid,
                    "seed": args.seed,
                    "bucket": args.bucket,
                    "tier": tier,
                    "generator": "transformer",
                    "preset": f"{args.mechanic}_{args.difficulty}",
                    "map_data": md,
                    "secondary": [],
                    "metadata": {
                        "description": f"generated {args.mechanic} {args.difficulty}",
                        "split": args.split,
                        "source": "generated",
                        "map_hash": lid,
                        "d_star": d_star,
                        "index": len(kept),
                    },
                }
            )
            if len(kept) >= args.n:
                break
        logger.info("kept %d / wanted %d (attempts=%d)", len(kept), args.n, attempts)

    if not kept:
        logger.warning("No solvable levels produced in %d attempts.", attempts)
        return
    write_benchmark(kept, Path(args.out), args.split, args.bucket)
    logger.info("Wrote %d solvable levels to %s (acceptance=%.1f%%)", len(kept), args.out,
                100 * len(kept) / max(attempts, 1))


if __name__ == "__main__":
    main()
