#!/usr/bin/env python3
"""Diff an nclone trace against captured game ground truth, bit-for-bit.

    python tools/npptrace/diff_trace.py docs/game_traces/10131738.jsonl /tmp/p.jsonl

Reports the first frame where any field's IEEE-754 bit pattern differs, the ULP
delta per field, and how the error grows afterwards.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys

FIELDS = ("xpos", "ypos", "xspeed", "yspeed", "floor_x", "floor_y")


def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def ulp_delta(a: float, b: float) -> int:
    """Signed distance in representable doubles (monotonic across zero)."""
    def key(v):
        u = bits(v)
        return -(u & 0x7FFFFFFFFFFFFFFF) if u >> 63 else u
    return key(a) - key(b)


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("sim")
    ap.add_argument("--context", type=int, default=2)
    ap.add_argument("--max-report", type=int, default=1)
    args = ap.parse_args()

    g = {r["frame"]: r for r in load(args.game)}
    s = {r["frame"]: r for r in load(args.sim)}
    common = sorted(set(g) & set(s))
    if not common:
        print("no overlapping frames")
        return 2

    # STATE first. A wrongly-dead ninja (state 6) makes every float diverge as a
    # side effect, and reporting the float is reporting the symptom.
    death = None
    for f in common:
        if g[f]["state"] != s[f]["state"]:
            death = (f, g[f]["state"], s[f]["state"])
            break

    first = None
    for f in common:
        bad = [k for k in FIELDS if bits(g[f][k]) != bits(s[f][k])]
        if bad:
            first = (f, bad)
            break

    if death and (first is None or death[0] <= first[0]):
        f, gs, ss = death
        tag = "  <-- SIM KILLS THE NINJA HERE" if ss in (6, 7) and gs not in (6, 7) else ""
        print("\nSTATE DIVERGES FIRST  frame %d: game state=%d, sim state=%d%s"
              % (f, gs, ss, tag))

    print("game frames=%d  sim frames=%d  overlap=%d"
          % (len(g), len(s), len(common)))
    if first is None:
        print("\nBIT-EXACT across all %d overlapping frames." % len(common))
        extra = sorted(set(g) - set(s))
        if extra:
            print("(game continues to frame %d; sim stopped at %d)"
                  % (max(g), max(s)))
        return 0

    f, bad = first
    print("\nFIRST DIVERGENCE  frame %d   fields: %s" % (f, ", ".join(bad)))
    for k in FIELDS:
        gv, sv = g[f][k], s[f][k]
        d = ulp_delta(sv, gv)
        mark = "" if d == 0 else "   <-- %+d ULP" % d
        print("   %-8s game=%-24.17g sim=%-24.17g%s" % (k, gv, sv, mark))
    print("   %-8s game=%-24d sim=%-24d" % ("state", g[f]["state"], s[f]["state"]))

    lo = max(min(common), f - args.context)
    print("\n  preceding frames (should be identical):")
    for pf in range(lo, f):
        same = all(bits(g[pf][k]) == bits(s[pf][k]) for k in FIELDS)
        print("    frame %-5d %s" % (pf, "identical" if same else "DIFFERS"))

    print("\n  error growth:")
    for pf in common:
        if pf < f:
            continue
        dx = abs(g[pf]["xpos"] - s[pf]["xpos"])
        dy = abs(g[pf]["ypos"] - s[pf]["ypos"])
        if pf in (f, f + 10, f + 25, f + 50) or pf % 100 == 0 or pf == common[-1]:
            print("    frame %-5d dx=%.6e  dy=%.6e" % (pf, dx, dy))
    return 1


if __name__ == "__main__":
    sys.exit(main())
