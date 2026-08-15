#!/usr/bin/env python3
"""Diff every dual-laser field against the game, frame by frame.

Field offsets from round-9 findings §1.5 (`field_hunt.py` located them):

    0x10/0x18 xpos/ypos   0x58/0x60 anchor   0x68/0x70 normal
    0x78/0x80 xend/yend   0xB0 beam_angle

Run with `NCLONE_AB_LASER_SEAT=1`, otherwise the seating is knowingly wrong and
everything downstream is meaningless.

    NCLONE_AB_LASER_SEAT=1 python tools/npptrace/compare_laser.py 10134679
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from read_entities import Capture  # noqa: E402

ATTRACTS = os.environ.get("NCLONE_ATTRACTS", "/home/tetra/projects/attracts")
TRACES = os.environ.get(
    "NCLONE_TRACES", os.path.join(HERE, "..", "..", "docs", "game_traces")
)
POOL = 21

FIELDS = [
    ("xpos", 0x10), ("ypos", 0x18),
    ("xanchor", 0x58), ("yanchor", 0x60),
    ("xnormal", 0x68), ("ynormal", 0x70),
    ("xend", 0x78), ("yend", 0x80),
    ("beam_angle", 0xB0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    if os.environ.get("NCLONE_AB_LASER_SEAT") != "1":
        print("WARNING: NCLONE_AB_LASER_SEAT is not 1; seating is knowingly wrong.\n")

    cap = Capture(os.path.join(TRACES, args.demo))
    p = next((q for q in cap.pools if q["i"] == POOL), None)
    if p is None:
        raise SystemExit("%s has no dual lasers" % args.demo)

    from nclone.replay.npp_attract_decoder import NppAttractDecoder
    from nclone.replay.replay_executor import decode_input_to_controls
    from nclone.nplay_headless import NPlayHeadless

    dec = NppAttractDecoder()
    path = os.path.join(ATTRACTS, args.demo)
    demo = dec.decode_npp_attract_file(path).get("demo_data", []) or []
    mapb = list(dec.create_nclone_map(path))
    h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    h.load_map_from_map_data(mapb)
    lasers = [e for lst in h.sim.entity_dic.values() for e in lst
              if getattr(e, "type", None) == 23]

    n = min(args.frames, cap.n, len(demo))
    first = {}
    for f in range(1, n + 1):
        hor, jmp = decode_input_to_controls(int(demo[f - 1]))
        h.tick(hor, jmp)
        rec = cap.frame(f)
        base = 4 + cap.layout["ninjaSlab"] + p["offset"]
        for i, e in enumerate(lasers):
            if i >= p["count"]:
                break
            blob = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            for name, off in FIELDS:
                mine = getattr(e, name, None)
                if mine is None:
                    continue
                theirs = struct.unpack_from("<d", blob, off)[0]
                if abs(mine - theirs) > args.tol and (i, name) not in first:
                    first[(i, name)] = (f, mine, theirs)

    print("%s: %d lasers, %d frames compared" % (args.demo, p["count"], n))
    if not first:
        print("  every field matches on every laser -- geometry is exact")
        return
    print("  %-4s %-11s %-6s %-22s %-22s %s"
          % ("idx", "field", "frame", "nclone", "game", "delta"))
    for (i, name), (f, mine, theirs) in sorted(first.items(), key=lambda kv: kv[1][0]):
        print("  %-4d %-11s %-6d %-22.12g %-22.12g %+.6g"
              % (i, name, f, mine, theirs, mine - theirs))


if __name__ == "__main__":
    main()
