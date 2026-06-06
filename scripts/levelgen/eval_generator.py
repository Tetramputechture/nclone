"""Phase 0, step 5: generator-quality metrics (reused every phase).

Metrics over tile grids:
  * tile-pattern KL divergence vs the human corpus -- unigram tile-type and 2x2
    patch distributions (the headline "procgen vs human gap" number).
  * expressive-range summaries (tile density / mine count / entity count histograms).

Each side is a *tile source*; supported source specs:
  ``corpus:<dir>``     a levelgen shard corpus (uses map_data tile region)
  ``h5:<path>``        a build_dataset HDF5 (uses the ``tiles`` array)
  ``procgen:<N>[:seed]`` sample N levels across BUCKET_CATALOG via GeneratorFactory

Example -- the headline number::

    python -m scripts.levelgen.eval_generator \\
        --human corpus:datasets/levelgen/corpus --candidate procgen:2000
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Iterable, List, Tuple

import numpy as np

from nclone.levelgen.tokenizer import TILE_START, TILE_END
from nclone.levelgen.labeling import entity_counts

logger = logging.getLogger("levelgen.eval")

N_TILE_TYPES = 38  # 0..37
GRID_W, GRID_H = 42, 23


# --------------------------------------------------------------------- sources
def _tiles_from_corpus(path: str) -> List[np.ndarray]:
    from scripts.levelgen.common import iter_records

    out = []
    for r in iter_records(__import__("pathlib").Path(path)):
        out.append(np.array(r.map_data[TILE_START:TILE_END], np.uint8))
    return out


def _tiles_from_h5(path: str) -> List[np.ndarray]:
    import h5py

    with h5py.File(path, "r") as h:
        return [np.array(row, np.uint8) for row in h["tiles"][:]]


def _maps_from_procgen(n: int, seed: int) -> List[List[int]]:
    from nclone.map_generation.generator_factory import GeneratorFactory
    from nclone.map_generation.generator_configs import BUCKET_CATALOG

    entries: List[Tuple[str, str]] = []
    for bucket, tiers in BUCKET_CATALOG.items():
        for tier, lst in tiers.items():
            for gen_type, preset, _w in lst:
                entries.append((gen_type, preset))
    if not entries:
        raise SystemExit("BUCKET_CATALOG is empty")
    rng = np.random.default_rng(seed)
    maps: List[List[int]] = []
    attempts = 0
    while len(maps) < n and attempts < n * 4:
        gen_type, preset = entries[int(rng.integers(len(entries)))]
        s = int(rng.integers(1 << 30))
        attempts += 1
        try:
            g = GeneratorFactory.create_from_preset(gen_type, preset, s)
            g.generate(seed=s)
            maps.append(list(g.map_data()))
        except Exception:
            continue
    return maps


def load_source(spec: str):
    """Return (tile_arrays, map_datas_or_None). map_datas present only for procgen."""
    kind, _, rest = spec.partition(":")
    if kind == "corpus":
        return _tiles_from_corpus(rest), None
    if kind == "h5":
        return _tiles_from_h5(rest), None
    if kind == "procgen":
        parts = rest.split(":")
        n = int(parts[0])
        seed = int(parts[1]) if len(parts) > 1 else 0
        maps = _maps_from_procgen(n, seed)
        tiles = [np.array(m[TILE_START:TILE_END], np.uint8) for m in maps]
        return tiles, maps
    raise SystemExit(f"unknown source spec: {spec}")


# --------------------------------------------------------------------- metrics
def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def unigram_dist(tiles: List[np.ndarray]) -> np.ndarray:
    hist = np.zeros(N_TILE_TYPES, np.float64)
    for t in tiles:
        b = np.bincount(np.clip(t, 0, N_TILE_TYPES - 1), minlength=N_TILE_TYPES)
        hist += b
    return hist / max(hist.sum(), 1)


def patch2x2_dist(tiles: List[np.ndarray], bins: int = 4096) -> np.ndarray:
    """Hashed distribution of 2x2 binary (solid/empty) tile patches.

    Reduces each tile to solid(!=0)/empty then hashes the 2x2 occupancy (16 codes);
    here a 2x2 over the 4-bit occupancy is exact (<=16 codes), so ``bins`` caps it.
    """
    hist = np.zeros(min(bins, 16), np.float64)
    for t in tiles:
        g = (t.reshape(GRID_H, GRID_W) != 0).astype(np.uint8)
        a = g[:-1, :-1]
        b = g[:-1, 1:]
        c = g[1:, :-1]
        d = g[1:, 1:]
        code = (a << 3) | (b << 2) | (c << 1) | d
        hist += np.bincount(code.ravel(), minlength=16)
    return hist / max(hist.sum(), 1)


def expressive_range(tiles: List[np.ndarray], maps) -> dict:
    dens = [float((t != 0).mean()) for t in tiles]
    out = {
        "n": len(tiles),
        "tile_density_mean": float(np.mean(dens)) if dens else 0.0,
        "tile_density_std": float(np.std(dens)) if dens else 0.0,
    }
    if maps is not None:
        mines = [sum(entity_counts(m).get(t, 0) for t in (1, 21)) for m in maps]
        nent = [sum(entity_counts(m).values()) for m in maps]
        out["mine_count_mean"] = float(np.mean(mines)) if mines else 0.0
        out["entity_count_mean"] = float(np.mean(nent)) if nent else 0.0
    return out


def evaluate(human_spec: str, cand_spec: str) -> dict:
    h_tiles, h_maps = load_source(human_spec)
    c_tiles, c_maps = load_source(cand_spec)
    logger.info("human=%d levels, candidate=%d levels", len(h_tiles), len(c_tiles))

    hu, cu = unigram_dist(h_tiles), unigram_dist(c_tiles)
    hp, cp = patch2x2_dist(h_tiles), patch2x2_dist(c_tiles)
    report = {
        "human_spec": human_spec,
        "candidate_spec": cand_spec,
        "tile_unigram_KL": _kl(cu, hu),  # KL(candidate || human)
        "tile_unigram_KL_sym": 0.5 * (_kl(cu, hu) + _kl(hu, cu)),
        "patch2x2_KL": _kl(cp, hp),
        "patch2x2_KL_sym": 0.5 * (_kl(cp, hp) + _kl(hp, cp)),
        "human_expressive": expressive_range(h_tiles, h_maps),
        "candidate_expressive": expressive_range(c_tiles, c_maps),
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Generator quality metrics (tile KL etc.)")
    ap.add_argument("--human", required=True, help="reference source spec (corpus:/h5:/procgen:)")
    ap.add_argument("--candidate", required=True, help="candidate source spec")
    ap.add_argument("--out", default=None, help="optional json output path")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    report = evaluate(args.human, args.candidate)
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
