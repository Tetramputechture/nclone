#!/usr/bin/env python3
"""Name the Simulator's entity pools by correlating pool counts against each
map's known entity-type histogram.

The pool index is not obviously the entity type, and the vtable RVAs are not in
any doc. But every captured demo gives one equation -- "pool P held N entities on
a map with this type histogram" -- and a handful of demos over different maps
pins the mapping.

    python tools/npptrace/identify_pools.py CAPTURE_DIR
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

ATTRACTS = os.environ.get("NCLONE_ATTRACTS", "/home/tetra/projects/attracts")
# Switch halves of locked (7) / trap (9) doors are not standalone entities.
PHANTOM = frozenset({7, 9})


def type_hist(demo_id):
    from nclone.replay.npp_attract_decoder import NppAttractDecoder

    dec = NppAttractDecoder()
    mapb = list(dec.create_nclone_map(os.path.join(ATTRACTS, demo_id)))
    h = collections.Counter()
    i = 1230
    while i + 4 < len(mapb):
        t = int(mapb[i])
        if t:
            h[t] += 1
        i += 5
    return h


def main():
    cap_dir = sys.argv[1]
    obs = []  # (demo, {pool: count}, {type: count})
    for lay in sorted(glob.glob(os.path.join(cap_dir, "*.layout.json"))):
        demo = os.path.basename(lay).replace(".layout.json", "")
        if not os.path.exists(os.path.join(ATTRACTS, demo)):
            continue
        L = json.load(open(lay))
        pools = {p["i"]: p["count"] for p in L["pools"]}
        obs.append((demo, pools, type_hist(demo), L))

    if not obs:
        print("no identified captures in %s" % cap_dir)
        return

    print("%-12s %s" % ("demo", "pools (index:count)"))
    for demo, pools, hist, _ in obs:
        print("  %-12s %-42s types %s"
              % (demo,
                 " ".join("%d:%d" % kv for kv in sorted(pools.items())),
                 " ".join("%d:%d" % kv for kv in sorted(hist.items()))))

    # A pool index maps to a type if their counts agree on EVERY demo where
    # either appears.
    all_pools = sorted({p for _, pl, _, _ in obs for p in pl})
    all_types = sorted({t for _, _, h, _ in obs for t in h})
    print("\nconsistent pool -> type assignments:")
    for p in all_pools:
        cands = []
        for t in all_types:
            if all(pl.get(p, 0) == h.get(t, 0) for _, pl, h, _ in obs):
                cands.append(t)
        vt = None
        for _, _, _, L in obs:
            for q in L["pools"]:
                if q["i"] == p and q.get("vtableRva") is not None:
                    vt = q["vtableRva"]
                    break
            if vt:
                break
        counts = sorted({pl.get(p, 0) for _, pl, _, _ in obs})
        print("   pool %-3d vtableRva=%-10s counts=%-18s -> %s"
              % (p, ("0x%X" % vt) if vt else "-", counts,
                 cands if cands else "NO consistent type"))


if __name__ == "__main__":
    main()
