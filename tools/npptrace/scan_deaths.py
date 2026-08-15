#!/usr/bin/env python3
"""Scan every entity capture for frames where the ninja is dead.

    python tools/npptrace/scan_deaths.py

Ninja state lives at +0xD0 in the ninja slab. 6 == dead/ragdoll, 7 == dead
(no ragdoll), 8 == celebrating. Captures taken with the wide slab
(ninjaSlab >= 0xD40) also carry the 13 bone pairs at +0xC60.
"""
from __future__ import annotations

import glob
import json
import gzip
import os
import struct
import sys

F_STATE = 0xD0
F_BONES = 0xC60


def stems():
    seen = []
    for pat in ("docs/game_traces/*.layout.json",
                "tools/npptrace/out*/*.layout.json"):
        for p in sorted(glob.glob(pat)):
            seen.append(p[: -len(".layout.json")])
    return seen


def main():
    rows = []
    for stem in stems():
        try:
            layout = json.load(open(stem + ".layout.json"))
        except Exception as exc:  # noqa: BLE001
            print("!! %s: %s" % (stem, exc))
            continue
        slab = layout.get("ninjaSlab", 0x200)
        rec = layout["recBytes"]
        if not os.path.exists(stem + ".ent.bin.gz"):
            continue
        with gzip.open(stem + ".ent.bin.gz", "rb") as fh:
            data = fh.read()
        n = len(data) // rec
        if slab <= F_STATE:
            continue
        states = {}
        first_dead = None
        for f in range(n):
            o = f * rec + 4 + F_STATE
            st = struct.unpack_from("<i", data, o)[0]
            states[st] = states.get(st, 0) + 1
            if first_dead is None and st in (6, 7):
                first_dead = f + 1
        rows.append((stem, n, slab, first_dead, states))

    wide = [r for r in rows if r[2] >= 0xD40]
    dead = [r for r in rows if r[3] is not None]
    wide_dead = [r for r in wide if r[3] is not None]
    print("captures scanned : %d" % len(rows))
    print("wide slab        : %d" % len(wide))
    print("with a dead frame: %d" % len(dead))
    print("wide AND dead    : %d" % len(wide_dead))
    print()
    for stem, n, slab, fd, states in rows:
        if fd is None:
            continue
        print("%-46s frames=%-6d slab=0x%X first_dead=%-6d states=%s"
              % (os.path.basename(stem), n, slab, fd,
                 sorted(states.items())))
    if not dead:
        print("(no capture contains a dead ninja)")
        print()
        print("state histogram across all captures:")
        agg = {}
        for _, _, _, _, states in rows:
            for k, v in states.items():
                agg[k] = agg.get(k, 0) + v
        for k in sorted(agg):
            print("   state %-4d %d frames" % (k, agg[k]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
