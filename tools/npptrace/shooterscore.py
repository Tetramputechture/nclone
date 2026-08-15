#!/usr/bin/env python3
"""Score our chaingun against every capture, field by field.

Only frames on which our ninja is still exactly on the game's are counted --
past that point any drone disagreement is downstream of a ninja bug and says
nothing about the drone.  The 'after divergence' column is printed too so the
filter cannot quietly hide a collapse.
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

POOL = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ETYPE = POOL + 2
dec = NppAttractDecoder()
stems = sorted(glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json"))

tot = {}
ok = {}
after = 0
firstbad = []


def bump(k, good):
    tot[k] = tot.get(k, 0) + 1
    if good:
        ok[k] = ok.get(k, 0) + 1


for s in stems:
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
    nframes = min(cap.n, len(tape), len(gninja))
    for f in range(1, nframes + 1):
        hor, jmp = decode_input_to_controls(int(tape[f - 1]))
        h.tick(hor, jmp)
        g = gninja[f - 1]
        if tracking and (abs(nj.xpos - g["xpos"]) > 1e-9
                         or abs(nj.ypos - g["ypos"]) > 1e-9):
            tracking = False
        r = cap.frame(f)
        for e, ent in enumerate(ours):
            b = r[base + e * p["stride"]: base + (e + 1) * p["stride"]]
            gst = struct.unpack_from("<i", b, 0x7C)[0]
            gx, gy = struct.unpack_from("<dd", b, 0x10)
            gshots = struct.unpack_from("<i", b, 0x88)[0]
            gtick = struct.unpack_from("<i", b, 0x8C)[0]
            gtimer = struct.unpack_from("<i", b, 0x78)[0]
            if not tracking:
                after += 1
                continue
            good_state = (gst == ent.state)
            bump("state", good_state)
            bump("position", abs(gx - ent.xpos) < 1e-9 and abs(gy - ent.ypos) < 1e-9)
            if POOL == 10:
                bump("shots", gshots == ent.shots)
                bump("shot_tick", gtick == ent.shot_tick)
            else:
                # type 13: `+0x8C` is the discharge frame counter and `+0x90` the
                # frozen beam direction; `+0xA0` is the endpoint (not `+0xC0`).
                bump("fire_count", gtick == ent.shot_tick)
                gdx, gdy = struct.unpack_from("<dd", b, 0x90)
                if gst in (1, 2):
                    bump("beam_dir<1e-9",
                         math.hypot(gdx - ent.fire_dx, gdy - ent.fire_dy) < 1e-9)
            bump("timer", gtimer == ent.timer)
            if gst == 2:
                gex, gey = struct.unpack_from("<dd", b, 0xC0 if POOL == 10 else 0xA0)
                bump("beam_end<0.001",
                     math.hypot(gex - ent.xend, gey - ent.yend) < 1e-3)
            if bad is None and not good_state:
                bad = (f, e, gst, ent.state)
        if h.ninja_has_died() or h.ninja_has_won():
            break
    if bad:
        firstbad.append((demo, bad))

print("\nshooter drone vs capture, frames where our ninja is still exact:")
for k in ("state", "timer", "position", "shots", "shot_tick", "fire_count",
          "beam_dir<1e-9", "beam_end<0.001"):
    if k in tot:
        print("   %-16s %7d / %-7d = %6.2f%%"
              % (k, ok.get(k, 0), tot[k], 100.0 * ok.get(k, 0) / tot[k]))
print("   (%d drone-frames after our ninja left the game's track, not counted)"
      % after)
print("\ndemos whose drone state ever disagrees (first disagreement):")
for demo, (f, e, gst, ost) in firstbad:
    print("   %-10s f%-6d drone %d  game=%d ours=%d" % (demo, f, e, gst, ost))
if not firstbad:
    print("   none")
