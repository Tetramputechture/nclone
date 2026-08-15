#!/usr/bin/env python3
"""Match nclone's drones to the game's by spawn, then report the first frame each
one's position diverges.

The killing drone is usually not the first to go wrong -- the interesting number
is which drone breaks earliest and on which frame, because that is the turn
decision the direction table gets wrong.

    python tools/npptrace/compare_drones.py 10131134 [--pool 12] [--frames 120]
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
DRONE_TYPES = (12, 13, 14, 15, 26)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo")
    ap.add_argument("--pool", type=int, default=12)
    ap.add_argument("--frames", type=int, default=120)
    args = ap.parse_args()

    traces = os.environ.get(
        "NCLONE_TRACES", os.path.join(HERE, "..", "..", "docs", "game_traces")
    )
    stem = os.path.join(traces, args.demo)
    cap = Capture(stem)
    if not any(p["i"] == args.pool for p in cap.pools):
        raise SystemExit("no pool %d in %s" % (args.pool, args.demo))

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

    def sim_drones():
        out = []
        for lst in sim.entity_dic.values():
            for e in lst:
                if getattr(e, "type", None) in DRONE_TYPES:
                    out.append(e)
        return out

    def game_drones(f):
        return [(x, y) for (pi, _ei, x, y, _b) in cap.entities(f) if pi == args.pool]

    n = min(args.frames, cap.n, len(demo))

    hor, jmp = decode_input_to_controls(int(demo[0]))
    h.tick(hor, jmp)
    sd = sim_drones()
    gd = game_drones(1)
    if len(sd) != len(gd):
        print("count mismatch: nclone %d, game %d" % (len(sd), len(gd)))

    pairing, used = [], set()
    for e in sd:
        best, bj = None, None
        for j, (gx, gy) in enumerate(gd):
            if j in used:
                continue
            d = math.hypot(e.xpos - gx, e.ypos - gy)
            if best is None or d < best:
                best, bj = d, j
        pairing.append(bj)
        used.add(bj)
        if best and best > 1e-9:
            print("  paired at distance %.6f on frame 1 (not an exact spawn match)" % best)

    first_bad = [None] * len(sd)
    for f in range(2, n + 1):
        hor, jmp = decode_input_to_controls(int(demo[f - 1]))
        h.tick(hor, jmp)
        gd = game_drones(f)
        for i, e in enumerate(sd):
            if first_bad[i] is not None:
                continue
            gx, gy = gd[pairing[i]]
            if e.xpos != gx or e.ypos != gy:
                first_bad[i] = (f, (e.xpos, e.ypos), (gx, gy))

    print("\n%s: %d drones in pool %d, %d frames compared"
          % (args.demo, len(sd), args.pool, n))
    print("  %5s  %-9s %-24s %-24s %s" % ("drone", "diverges", "nclone", "game", "dist"))
    for i, fb in enumerate(first_bad):
        if fb is None:
            print("  %5d  %-9s matches for all %d frames" % (i, "never", n))
        else:
            f, s, g = fb
            print("  %5d  %-9d (%9.3f, %9.3f) (%9.3f, %9.3f) %.3f"
                  % (i, f, s[0], s[1], g[0], g[1],
                     math.hypot(s[0] - g[0], s[1] - g[1])))


if __name__ == "__main__":
    main()
