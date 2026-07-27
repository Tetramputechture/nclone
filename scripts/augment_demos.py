#!/usr/bin/env python
"""Demo augmentation: multiply harvested ``.replay`` demos via demo-PRESERVING
transforms, each RE-VALIDATED on the native sim so every output is a
guaranteed-winning demo.

Transforms (all exact / solvability-preserving):
  * identity         -- the original (copied through, re-validated as a sanity gate)
  * hmirror          -- horizontal reflection of map + L/R-swapped inputs
                        (N++ physics are symmetric about the vertical centre line,
                        so the mirrored tape wins iff the original does). Genuinely
                        new geometry -> the highest-value augmentation.
  * hmirror+unreach  -- (optional) mirror then randomize tiles UNREACHABLE from
                        spawn, for observation diversity without changing the route.

Usage:
  python -m scripts.augment_demos --in /demos/all_v4 --out /demos/all_v4_aug \
      --num-workers 60 --randomize 0          # mirror only (2x)
  ... --randomize 2                             # + 2 unreachable-randomized variants each
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from datetime import datetime
from typing import List, Optional, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")


def _decode(byte: int) -> Tuple[int, int]:
    """N++ input byte -> (horizontal, jump). bit0=jump, bit1=right, bit2=left."""
    j = byte & 0x01
    right = 1 if (byte & 0x02) else 0
    left = 1 if (byte & 0x04) else 0
    return right - left, j


def _validate(map_data: List[int], inputs: List[int]) -> Optional[int]:
    """Replay the input tape on the native sim; return winning frame count or None."""
    import nclone_native as Nnative
    sim = Nnative.Simulator.load([int(b) & 0xFF for b in map_data])
    sim.fast_reset()
    for i, b in enumerate(inputs):
        h, j = _decode(int(b))
        sim.tick(h, j)
        if sim.has_won():
            return i + 1
        if sim.has_died():
            return None
    return None


def _write(out_dir: str, name: str, map_data: List[int], inputs: List[int]) -> str:
    from nclone.replay.gameplay_recorder import CompactReplay
    now = datetime.now()
    replay = CompactReplay(
        episode_id=name,
        map_data=bytes(int(b) & 0xFF for b in map_data),
        input_sequence=[int(b) & 0xFF for b in inputs],
        level_id=name,
        start_time=now, end_time=now, success=True,
    )
    path = os.path.join(out_dir, f"{name}.replay")
    with open(path, "wb") as f:
        f.write(replay.to_binary())
    return path


def _augment_one(args):
    path, out_dir, n_randomize = args
    import logging
    logging.disable(logging.CRITICAL)
    from nclone.replay.gameplay_recorder import CompactReplay
    from nclone.map_augmentation.transforms import mirror_map_data
    from nclone.map_augmentation.attract.horizontal_mirror import INPUT_MIRROR_MAP

    r = CompactReplay.from_binary(open(path, "rb").read())
    md = list(r.map_data)
    inputs = list(r.input_sequence)
    base = os.path.splitext(os.path.basename(path))[0]

    out = []
    # identity: copy through (re-validate to gate any corrupt base demo)
    f0 = _validate(md, inputs)
    if f0:
        out.append(_write(out_dir, base, md, inputs[:f0]))
    # hmirror
    md_m = mirror_map_data(md)
    inp_m = [INPUT_MIRROR_MAP.get(int(b), int(b)) for b in inputs]
    fm = _validate(md_m, inp_m)
    if fm:
        out.append(_write(out_dir, f"{base}_hm", md_m, inp_m[:fm]))

    # optional: unreachable-tile-randomized variants (observation diversity)
    if n_randomize > 0:
        try:
            from nclone.map_augmentation.attract.unreachable_tile_randomizer import (
                UnreachableTileRandomizer,
            )
            from nclone.map_augmentation.map_data_attract_bridge import (
                map_data_to_attract, attract_to_map_data,
            )
            for variant, (mdv, inpv) in enumerate(
                [(md, inputs), (md_m, inp_m)]
            ):
                fv = f0 if variant == 0 else fm
                if not fv:
                    continue
                for k in range(n_randomize):
                    aug = UnreachableTileRandomizer(seed=1000 * variant + k)
                    ad = map_data_to_attract(mdv)
                    ad2 = aug.augment(ad)
                    mdr = attract_to_map_data(ad2, mdv)
                    fr = _validate(mdr, inpv[:fv])
                    if fr:
                        tag = "hm" if variant else "id"
                        out.append(_write(out_dir, f"{base}_{tag}_rnd{k}", mdr, inpv[:fr]))
        except Exception:
            pass  # bridge/randomizer unavailable -> mirror-only volume still stands
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True, help="dir of base .replay demos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--randomize", type=int, default=0,
                    help="N unreachable-randomized variants per (orig+mirror)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.inp, "*.replay")))
    if args.limit:
        files = files[: args.limit]
    print(f"Augmenting {len(files)} base demos -> {args.out} "
          f"(mirror + {args.randomize} randomize, {args.num_workers} workers)", flush=True)

    work = [(f, args.out, args.randomize) for f in files]
    t0 = time.time()
    written = 0
    import multiprocessing
    if args.num_workers <= 1:
        it = map(_augment_one, work)
        pool = None
    else:
        pool = multiprocessing.Pool(args.num_workers, maxtasksperchild=200)
        it = pool.imap_unordered(_augment_one, work)
    done = 0
    for paths in it:
        written += len(paths)
        done += 1
        if done % 500 == 0 or done == len(work):
            print(f"  {done}/{len(work)} base done, {written} demos written "
                  f"({time.time()-t0:.0f}s)", flush=True)
    if pool:
        pool.close(); pool.join()
    print(f"\nDONE: {written} augmented demos from {len(files)} base "
          f"({written/max(1,len(files)):.1f}x) in {time.time()-t0:.0f}s -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
