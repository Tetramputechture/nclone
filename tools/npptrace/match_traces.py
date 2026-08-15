#!/usr/bin/env python3
"""Resolve captured traces to attract ids by replaying candidates in nclone.

Several demos in a capture batch can share a map, so the ninja spawn is not a
unique key. This replays each candidate and scores it by how many LEADING frames
agree with the capture: the true pairing agrees for hundreds of frames, a wrong
one diverges within a handful.

    python tools/npptrace/match_traces.py CAPTURE_DIR id1 id2 id3 ...
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PY = os.path.join(ROOT, ".venv", "bin", "python")
FIELDS = ("xpos", "ypos", "xspeed", "yspeed")


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def leading_match(a, b):
    n = 0
    for ra, rb in zip(a, b):
        if all(bits(ra[k]) == bits(rb[k]) for k in FIELDS):
            n += 1
        else:
            break
    return n


def approx_lead(a, b, tol=1e-6):
    n = 0
    for ra, rb in zip(a, b):
        if all(abs(ra[k] - rb[k]) <= tol for k in FIELDS):
            n += 1
        else:
            break
    return n


def sim(demo_id, engine):
    out = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
    subprocess.run([PY, os.path.join(HERE, "sim_trace.py"), demo_id,
                    "--engine", engine, "-o", out],
                   capture_output=True, cwd=ROOT)
    try:
        return load(out)
    finally:
        os.unlink(out)


def main():
    cap_dir = sys.argv[1]
    ids = sys.argv[2:]
    caps = sorted(f for f in os.listdir(cap_dir) if f.endswith(".jsonl"))
    sims = {}
    for i in ids:
        sims[i] = sim(i, "python")
        print("replayed %s (%d frames)" % (i, len(sims[i])), flush=True)

    print("\n%-22s %-12s %8s %8s" % ("capture", "best match", "exact", "approx"))
    used = set()
    for c in caps:
        g = load(os.path.join(cap_dir, c))
        scored = []
        for i, s in sims.items():
            scored.append((leading_match(g, s), approx_lead(g, s), i))
        scored.sort(reverse=True)
        best = scored[0]
        runner = scored[1] if len(scored) > 1 else (0, 0, "-")
        flag = ""
        if best[2] in used:
            flag = "  (DUPLICATE)"
        used.add(best[2])
        if max(best[0], best[1]) < 5:
            flag += "  (WEAK)"
        print("%-22s %-12s %8d %8d   runner-up %s(%d/%d)%s"
              % (c, best[2], best[0], best[1], runner[2], runner[0], runner[1], flag))


if __name__ == "__main__":
    main()
