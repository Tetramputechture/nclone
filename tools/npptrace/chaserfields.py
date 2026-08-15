#!/usr/bin/env python3
"""Score our chaser against every capture, field by field.

Same shape as the run that localised both shooter-drone bugs: only frames on
which our ninja is still exactly on the game's are counted, and the count of
frames dropped after divergence is printed so the filter cannot hide a collapse.
"""
import glob
import json
import math
import os
import struct
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone")
sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402

POOL, ETYPE = 13, 15
dec = NppAttractDecoder()
tot, ok = {}, {}
after = 0
firstbad = []


def bump(k, good):
    tot[k] = tot.get(k, 0) + 1
    if good:
        ok[k] = ok.get(k, 0) + 1


for s in sorted(glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json")):
    stem = s.replace(".layout.json", "")
    demo = os.path.basename(stem)
    if not demo.isdigit():
        continue
    nj_path = stem + ".ninja.jsonl"
    path = "/home/tetra/projects/attracts/%s" % demo
    if not (os.path.exists(nj_path) and os.path.exists(path)):
        continue
    try:
        cap = Capture(stem)
    except Exception:
        continue
    p = next((q for q in cap.pools if q["i"] == POOL and q["count"]), None)
    if p is None:
        continue
    gninja = [json.loads(l) for l in open(nj_path) if l.strip()]
    tape = dec.decode_npp_attract_file(path).get("demo_data") or []
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    h.load_map_from_map_data(list(dec.create_nclone_map(path)))
    nj = h.sim.ninja
    ours = h.sim.entity_dic.get(ETYPE, [])
    if p["count"] != len(ours):
        print("SKIP %s: pool %d vs ours %d" % (demo, p["count"], len(ours)))
        continue
    tracking = True
    bad = None
    for f in range(1, min(cap.n, len(tape), len(gninja)) + 1):
        hor, jmp = decode_input_to_controls(int(tape[f - 1]))
        h.tick(hor, jmp)
        g = gninja[f - 1]
        if tracking and (abs(nj.xpos - g["xpos"]) > 1e-9
                         or abs(nj.ypos - g["ypos"]) > 1e-9):
            tracking = False
        r = cap.frame(f)
        for e, ent in enumerate(ours):
            if not tracking:
                after += 1
                continue
            b = r[base + e * p["stride"]: base + (e + 1) * p["stride"]]
            glatch = b[0x70]
            gx, gy = struct.unpack_from("<dd", b, 0x10)
            gtx, gty = struct.unpack_from("<dd", b, 0x30)
            gdir = struct.unpack_from("<i", b, 0x40)[0]
            gcommit = struct.unpack_from("<i", b, 0x88)[0]
            good_latch = (glatch == 1) == bool(ent.chasing)
            bump("latch +0x70", good_latch)
            bump("position", abs(gx - ent.xpos) < 1e-9 and abs(gy - ent.ypos) < 1e-9)
            bump("target +0x30", abs(gtx - ent.xtarget) < 1e-9
                 and abs(gty - ent.ytarget) < 1e-9)
            bump("dir +0x40", gdir == ent.dir)
            if glatch == 1:
                bump("commit +0x88 (latched)", gcommit == ent.chase_dir)
            if bad is None and not good_latch:
                bad = (f, e, glatch, ent.chasing)
        if h.ninja_has_died() or h.ninja_has_won():
            break
    if bad:
        firstbad.append((demo, bad))

print("\nchaser vs capture, frames where our ninja is still exact:")
for k in ("latch +0x70", "position", "target +0x30", "dir +0x40",
          "commit +0x88 (latched)"):
    if k in tot:
        print("   %-24s %7d / %-7d = %6.2f%%"
              % (k, ok.get(k, 0), tot[k], 100.0 * ok.get(k, 0) / tot[k]))
print("   (%d drone-frames after our ninja left the game's track, not counted)"
      % after)
print("\nfirst latch disagreement per demo:")
for demo, (f, e, gl, ol) in firstbad:
    print("   %-10s f%-6d drone %d  game=%d ours=%s" % (demo, f, e, gl, ol))
if not firstbad:
    print("   none")
