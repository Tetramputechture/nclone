#!/usr/bin/env python3
"""Tabulate the direction each drone actually moves first, against the map's
orientation and mode bytes.

`drone_dirlist` in `nclone-native/src/ab.rs` is tagged **[I]** -- inherited from
the original project, never verified. Round 9's entity capture shows nclone
picking a different first move from the game for half the drones on 10131134.
This reads the truth straight out of the capture.

    python tools/npptrace/drone_dirs.py            # every captured demo with drones
    python tools/npptrace/drone_dirs.py 10131134
"""
from __future__ import annotations

import glob
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
# pool index = type - 2 (round 9 findings §1.1b)
DRONE_TYPES = {12: 10, 13: 11, 14: 12, 15: 13, 26: 24}
DIRNAME = {(1, 0): "right", (0, 1): "down", (-1, 0): "left", (0, -1): "up"}


def map_entities(demo):
    from nclone.replay.npp_attract_decoder import NppAttractDecoder

    dec = NppAttractDecoder()
    mapb = list(dec.create_nclone_map(os.path.join(ATTRACTS, demo)))
    out = []
    i = 1230
    while i + 4 < len(mapb):
        t = int(mapb[i])
        if t:
            out.append({"type": t, "x": int(mapb[i + 1]), "y": int(mapb[i + 2]),
                        "orientation": int(mapb[i + 3]), "mode": int(mapb[i + 4])})
        i += 5
    return out


def direction(p0, p1):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    if abs(dx) >= abs(dy):
        return (1 if dx > 0 else -1, 0)
    return (0, 1 if dy > 0 else -1)


def run(demo):
    stem = os.path.join(TRACES, demo)
    if not os.path.exists(stem + ".layout.json"):
        return []
    cap = Capture(stem)
    ents = map_entities(demo)
    rows = []
    for t, pool in DRONE_TYPES.items():
        if not any(p["i"] == pool for p in cap.pools):
            continue
        # map entities of this type, in file order -- the pool is filled in the
        # same order the loader creates them
        me = [e for e in ents if e["type"] == t]
        g1 = [(x, y) for (pi, _e, x, y, _b) in cap.entities(1) if pi == pool]
        g3 = [(x, y) for (pi, _e, x, y, _b) in cap.entities(3) if pi == pool]
        if len(me) != len(g1):
            print("  %s type %d: map has %d, pool has %d -- skipping"
                  % (demo, t, len(me), len(g1)))
            continue
        for k, e in enumerate(me):
            d = direction(g1[k], g3[k])
            rows.append((demo, t, e["orientation"], e["mode"],
                         DIRNAME.get(d, "still"), g1[k]))
    return rows


def main():
    demos = sys.argv[1:]
    if not demos:
        demos = sorted(os.path.basename(p).replace(".layout.json", "")
                       for p in glob.glob(os.path.join(TRACES, "*.layout.json")))
    allrows = []
    for d in demos:
        if not os.path.exists(os.path.join(ATTRACTS, d)):
            continue
        allrows += run(d)

    if not allrows:
        print("no drone pools in the captured set")
        return
    print("%-12s %5s %5s %5s  %-6s %s" %
          ("demo", "type", "orient", "mode", "moves", "spawn"))
    for demo, t, o, m, dr, pos in allrows:
        print("%-12s %5d %6d %5d  %-6s (%8.3f, %8.3f)" % (demo, t, o, m, dr, pos[0], pos[1]))

    print("\n=== (type, orientation, mode) -> first direction observed ===")
    tab = {}
    for demo, t, o, m, dr, _ in allrows:
        tab.setdefault((t, o, m), set()).add(dr)
    for k in sorted(tab):
        v = sorted(tab[k])
        flag = "" if len(v) == 1 else "   <-- AMBIGUOUS, depends on surroundings"
        print("  type %-3d orient %-2d mode %-2d -> %s%s" % (k[0], k[1], k[2], v, flag))


if __name__ == "__main__":
    main()
