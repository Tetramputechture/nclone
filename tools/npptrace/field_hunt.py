#!/usr/bin/env python3
"""Locate an entity's fields inside its captured slab by matching nclone's values.

Replays the demo, and for each frame asks: which byte offset in the game's slab
holds a double equal to nclone's `<attr>`? Offsets that agree on every frame are
the field. Works even when nclone's value is only *nearly* right -- pass
`--tol` to match approximately.

    python tools/npptrace/field_hunt.py 1000622 --type 23 \
        --attrs xpos,ypos,xend,yend,beam_angle,xnormal,ynormal,xanchor,yanchor
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo")
    ap.add_argument("--type", type=int, required=True)
    ap.add_argument("--attrs", required=True)
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--tol", type=float, default=0.0)
    ap.add_argument("--index", type=int, default=0, help="which entity of that type")
    args = ap.parse_args()

    attrs = args.attrs.split(",")
    pool = args.type - 2
    cap = Capture(os.path.join(TRACES, args.demo))
    p = next((q for q in cap.pools if q["i"] == pool), None)
    if p is None:
        raise SystemExit("no pool %d in %s" % (pool, args.demo))
    stride = p["stride"]

    from nclone.replay.npp_attract_decoder import NppAttractDecoder
    from nclone.replay.replay_executor import decode_input_to_controls
    from nclone.nplay_headless import NPlayHeadless

    dec = NppAttractDecoder()
    path = os.path.join(ATTRACTS, args.demo)
    demo = dec.decode_npp_attract_file(path).get("demo_data", []) or []
    mapb = list(dec.create_nclone_map(path))
    h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    h.load_map_from_map_data(mapb)
    sim = h.sim
    ents = [e for lst in sim.entity_dic.values() for e in lst
            if getattr(e, "type", None) == args.type]
    ent = ents[args.index]

    n = min(args.frames, cap.n, len(demo))
    cands = {a: set(range(0, stride - 7, 4)) for a in attrs}
    for f in range(1, n + 1):
        hor, jmp = decode_input_to_controls(int(demo[f - 1]))
        h.tick(hor, jmp)
        rec = cap.frame(f)
        base = 4 + cap.layout["ninjaSlab"] + p["offset"] + args.index * stride
        blob = rec[base:base + stride]
        for a in attrs:
            want = getattr(ent, a, None)
            if want is None:
                cands[a] = set()
                continue
            keep = set()
            for off in cands[a]:
                got = struct.unpack_from("<d", blob, off)[0]
                if got == want or (args.tol and abs(got - want) <= args.tol):
                    keep.add(off)
            cands[a] = keep

    print("%s type %d entity %d, stride %d, %d frames, tol=%g"
          % (args.demo, args.type, args.index, stride, n, args.tol))
    for a in attrs:
        offs = sorted(cands[a])
        print("   %-12s -> %s" % (a, ["0x%02X" % o for o in offs] or "no match"))


if __name__ == "__main__":
    main()
