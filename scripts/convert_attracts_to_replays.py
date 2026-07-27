#!/usr/bin/env python3
"""Convert raw N++ attract binaries into validated BC/DAPG ``.replay`` demos.

Each attract binary is decoded (map + input tape), re-simulated on the native
engine, and written out as a ``CompactReplay`` ONLY if the demo actually wins.
Attract demos are expert playthroughs that collect gold to maximize score, so the
winners are ideal imitation data for the gold-aware generalist.

Two things that are easy to get wrong (both handled here):
  * the attract ``demo_data`` bytes are BIT-PACKED (bit0=jump, bit1=right,
    bit2=left) — decoded via ``decode_input_to_controls``. (nclone's own
    demo_ingestion._INPUT_TABLE uses a DIFFERENT semantic table and only ever
    runs tiles-only, so it never validated wins — do NOT copy it here.)
  * the map is rebuilt with ``NppAttractDecoder.create_nclone_map`` and the demo
    is one input per SIM FRAME (native ``tick`` per byte).

Usage:
  python scripts/convert_attracts_to_replays.py \
      --in-dir /path/to/attracts_subset --out-dir /path/to/attract_replays \
      --workers 16 [--limit N] [--min-gold-fraction 0.0]
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")


def _convert_one(args):
    path, out_dir, min_gold_fraction = args
    try:
        from nclone.replay.npp_attract_decoder import NppAttractDecoder
        from nclone.replay.replay_executor import decode_input_to_controls
        from nclone.native_headless import NativePlayHeadless
        from nclone.constants.entity_types import EntityType
        from nclone.replay.gameplay_recorder import CompactReplay

        dec = NppAttractDecoder()
        res = dec.decode_npp_attract_file(path)
        demo = res.get("demo_data", []) or []
        if not demo:
            return ("no_demo", path, 0, 0)
        mapb = list(dec.create_nclone_map(path))

        h = NativePlayHeadless()
        h.load_map_from_map_data(mapb)
        total_gold = len(h.sim.entity_dic.get(EntityType.GOLD, []))
        for b in demo:
            hor, jmp = decode_input_to_controls(int(b))
            h.tick(hor, jmp)
            if h.ninja_has_won() or h.ninja_has_died():
                break
        if not h.ninja_has_won():
            return ("no_win", path, 0, total_gold)
        gold = int(getattr(h.sim.ninja, "gold_collected", 0))
        if total_gold > 0 and (gold / total_gold) < min_gold_fraction:
            return ("low_gold", path, gold, total_gold)

        name = Path(path).name
        now = datetime.now(timezone.utc)
        cr = CompactReplay(
            episode_id=f"attract_{name}",
            map_data=bytes(mapb),
            input_sequence=[int(b) & 0xFF for b in demo],
            level_id=name,
            start_time=now,
            end_time=now,
            success=True,
        )
        Path(out_dir, f"attract_{name}.replay").write_bytes(cr.to_binary())
        return ("ok", path, gold, total_gold)
    except Exception as e:  # noqa: BLE001
        return (f"err:{type(e).__name__}", path, 0, 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-gold-fraction", type=float, default=0.0,
                    help="Reject winning demos collecting < this fraction of gold (0=keep all wins).")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, "*")))
    files = [f for f in files if os.path.isfile(f) or os.path.islink(f)]
    if args.limit:
        files = files[: args.limit]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    tasks = [(f, args.out_dir, args.min_gold_fraction) for f in files]
    print(f"Converting {len(files)} attract files with {args.workers} workers -> {args.out_dir}")

    counts: dict[str, int] = {}
    gold_sum = tot_sum = 0
    with mp.Pool(args.workers) as pool:
        for i, (status, _p, gold, tg) in enumerate(pool.imap_unordered(_convert_one, tasks, chunksize=8), 1):
            key = status if status in ("ok", "no_win", "no_demo", "low_gold") else "error"
            counts[key] = counts.get(key, 0) + 1
            if status == "ok":
                gold_sum += gold
                tot_sum += tg
            if i % 1000 == 0:
                print(f"  {i}/{len(files)}  ok={counts.get('ok',0)} no_win={counts.get('no_win',0)} err={counts.get('error',0)}")

    ok = counts.get("ok", 0)
    print(f"\nDONE: {ok} valid winning demos written "
          f"(gold {gold_sum}/{tot_sum} = {100*gold_sum/max(tot_sum,1):.0f}% collected)")
    print(f"  breakdown: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
