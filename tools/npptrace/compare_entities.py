#!/usr/bin/env python3
"""Compare nclone's entities of one type against the game's, frame by frame.

Generalises `compare_drones.py` to any pool. Pool index is `type - 2`
(round 9 findings §1.1b), so `--type 23` reads pool 21.

    python tools/npptrace/compare_entities.py 1000622 --type 23 --frames 40
"""
from __future__ import annotations

import argparse
import math
import os
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
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--show", type=int, default=6, help="frames to print in full")
    args = ap.parse_args()

    pool = args.type - 2
    cap = Capture(os.path.join(TRACES, args.demo))
    if not any(p["i"] == pool for p in cap.pools):
        raise SystemExit("demo %s has no pool %d (type %d); pools present: %s"
                         % (args.demo, pool, args.type,
                            [p["i"] for p in cap.pools]))

    from nclone.replay.npp_attract_decoder import NppAttractDecoder
    from nclone.replay.replay_executor import decode_input_to_controls
    from nclone.nplay_headless import NPlayHeadless

    dec = NppAttractDecoder()
    p = os.path.join(ATTRACTS, args.demo)
    demo = dec.decode_npp_attract_file(p).get("demo_data", []) or []
    mapb = list(dec.create_nclone_map(p))
    h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    h.load_map_from_map_data(mapb)
    sim = h.sim

    def mine():
        out = []
        for lst in sim.entity_dic.values():
            for e in lst:
                if getattr(e, "type", None) == args.type:
                    out.append(e)
        return out

    def theirs(f):
        return [(x, y) for (pi, _e, x, y, _b) in cap.entities(f) if pi == pool]

    n = min(args.frames, cap.n, len(demo))
    ents = mine()
    g0 = theirs(1)
    print("%s type %d: nclone has %d, game pool %d has %d"
          % (args.demo, args.type, len(ents), pool, len(g0)))

    first_bad = [None] * len(ents)
    for f in range(1, n + 1):
        hor, jmp = decode_input_to_controls(int(demo[f - 1]))
        h.tick(hor, jmp)
        g = theirs(f)
        if f <= args.show:
            print("  frame %d" % f)
            for i, e in enumerate(ents):
                if i >= len(g):
                    break
                d = math.hypot(e.xpos - g[i][0], e.ypos - g[i][1])
                print("     [%d] nclone=(%9.3f, %9.3f)  game=(%9.3f, %9.3f)  d=%.4f%s"
                      % (i, e.xpos, e.ypos, g[i][0], g[i][1], d,
                         "" if d < 1e-9 else "   <-- DIFFERS"))
        for i, e in enumerate(ents):
            if first_bad[i] is not None or i >= len(g):
                continue
            if e.xpos != g[i][0] or e.ypos != g[i][1]:
                first_bad[i] = (f, (e.xpos, e.ypos), g[i])

    print("\n  %5s  %-9s %-24s %-24s %s" % ("idx", "diverges", "nclone", "game", "dist"))
    for i, fb in enumerate(first_bad):
        if fb is None:
            print("  %5d  %-9s matches for all %d frames" % (i, "never", n))
        else:
            f, s, g = fb
            print("  %5d  %-9d (%9.3f, %9.3f) (%9.3f, %9.3f) %.4f"
                  % (i, f, s[0], s[1], g[0], g[1],
                     math.hypot(s[0] - g[0], s[1] - g[1])))


if __name__ == "__main__":
    main()
