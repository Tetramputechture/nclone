#!/usr/bin/env python3
"""Read a round-9 entity capture.

    python tools/npptrace/read_entities.py docs/game_traces/10131134 --frame 56
    python tools/npptrace/read_entities.py docs/game_traces/10131134 --pool 12 --field 0x10
    python tools/npptrace/read_entities.py docs/game_traces/10131134 --near-ninja 56

Capture layout (see docs/ENTITY_RE_ROUND9_FINDINGS.md §1.3):

    <id>.layout.json   pool table: index, count, stride, vtableRva, byte offset
    <id>.ent.bin.gz    per frame: <u32 frame><512 B ninja><entity slabs, pool order>
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import struct

NINJA_SLAB = 0x200
F_XPOS, F_YPOS = 0x20, 0x28
# Confirmed across two classes; see round-9 findings §1.2.
E_XPOS, E_YPOS = 0x10, 0x18


class Capture:
    def __init__(self, stem):
        self.stem = stem
        self.layout = json.load(open(stem + ".layout.json"))
        self.rec = self.layout["recBytes"]
        self.pools = self.layout["pools"]
        with gzip.open(stem + ".ent.bin.gz", "rb") as fh:
            self.data = fh.read()
        self.n = len(self.data) // self.rec

    def frame(self, f):
        """1-indexed, matching the ninja trace."""
        o = (f - 1) * self.rec
        if not (0 <= o < len(self.data)):
            raise IndexError("frame %d outside 1..%d" % (f, self.n))
        return self.data[o:o + self.rec]

    def ninja(self, f):
        r = self.frame(f)
        s = r[4:4 + NINJA_SLAB]
        return (struct.unpack_from("<d", s, F_XPOS)[0],
                struct.unpack_from("<d", s, F_YPOS)[0])

    def entities(self, f):
        """[(pool_index, entity_index, x, y, raw_bytes)]"""
        r = self.frame(f)
        base = 4 + NINJA_SLAB
        out = []
        for p in self.pools:
            for e in range(p["count"]):
                o = base + p["offset"] + e * p["stride"]
                b = r[o:o + p["stride"]]
                x = struct.unpack_from("<d", b, E_XPOS)[0]
                y = struct.unpack_from("<d", b, E_YPOS)[0]
                out.append((p["i"], e, x, y, b))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="path without the .layout.json / .ent.bin.gz suffix")
    ap.add_argument("--frame", type=int)
    ap.add_argument("--pool", type=int)
    ap.add_argument("--field", type=lambda s: int(s, 0),
                    help="dump this byte offset as a double, per entity per frame")
    ap.add_argument("--near-ninja", type=int, metavar="FRAME",
                    help="rank entities by distance to the ninja on FRAME")
    args = ap.parse_args()

    cap = Capture(args.stem)
    print("%s: %d frames, rec=%d, %d pools"
          % (os.path.basename(args.stem), cap.n, cap.rec, len(cap.pools)))
    for p in cap.pools:
        print("   pool %-3d count=%-4d stride=%-5d vtableRva=%s"
              % (p["i"], p["count"], p["stride"],
                 ("0x%X" % p["vtableRva"]) if p.get("vtableRva") is not None else "-"))

    if args.near_ninja:
        f = args.near_ninja
        nx, ny = cap.ninja(f)
        print("\nframe %d  ninja=(%.6f, %.6f)" % (f, nx, ny))
        rows = []
        for pi, ei, x, y, _ in cap.entities(f):
            rows.append((math.hypot(x - nx, y - ny), pi, ei, x, y))
        rows.sort()
        print("  %8s  %5s %5s  %s" % ("dist", "pool", "idx", "position"))
        for d, pi, ei, x, y in rows[:12]:
            print("  %8.3f  %5d %5d  (%9.3f, %9.3f)" % (d, pi, ei, x, y))
        return

    if args.frame:
        nx, ny = cap.ninja(args.frame)
        print("\nframe %d  ninja=(%.6f, %.6f)" % (args.frame, nx, ny))
        for pi, ei, x, y, _ in cap.entities(args.frame):
            if args.pool is not None and pi != args.pool:
                continue
            print("   pool %-3d [%2d]  (%9.3f, %9.3f)" % (pi, ei, x, y))
        return

    if args.field is not None and args.pool is not None:
        print("\npool %d, double at +0x%02X, per frame:" % (args.pool, args.field))
        for f in range(1, min(cap.n, 40) + 1):
            r = cap.frame(f)
            base = 4 + NINJA_SLAB
            p = next(q for q in cap.pools if q["i"] == args.pool)
            vals = []
            for e in range(p["count"]):
                o = base + p["offset"] + e * p["stride"]
                vals.append(struct.unpack_from("<d", r, o + args.field)[0])
            print("  f%-4d %s" % (f, "  ".join("%10.4f" % v for v in vals)))


if __name__ == "__main__":
    main()
