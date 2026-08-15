"""10136122: drones are in the right place and 127px away, yet we die.

That is a kill-test failure with no position error to hide behind, and it is
visible in the SHIPPING config (no experimental flags). Print each drone's state
and beam alongside the game's, at and just before the death.
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

DEMO, POOL, ETYPE = "10136122", 10, 12
stem = next(os.path.join(c, DEMO)
            for c in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*")
            if os.path.exists(os.path.join(c, DEMO) + ".layout.json"))
cap = Capture(stem)
p = next(q for q in cap.pools if q["i"] == POOL and q["count"])
base = 4 + cap.layout["ninjaSlab"] + p["offset"]
gninja = [json.loads(l) for l in open(stem + ".ninja.jsonl") if l.strip()]
dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/%s" % DEMO
tape = dec.decode_npp_attract_file(path).get("demo_data") or []
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(list(dec.create_nclone_map(path)))
ours = h.sim.entity_dic[ETYPE]
nj = h.sim.ninja

for f in range(1, min(cap.n, len(tape), len(gninja)) + 1):
    hor, jmp = decode_input_to_controls(int(tape[f - 1]))
    h.tick(hor, jmp)
    dead = h.ninja_has_died()
    if f >= 136 or dead:
        rec = cap.frame(f)
        n = gninja[f - 1]
        print("\nframe %d%s  ninja ours (%.2f,%.2f) game (%.2f,%.2f)"
              % (f, "  *** WE DIE ***" if dead else "", nj.xpos, nj.ypos,
                 n["xpos"], n["ypos"]))
        for i, e in enumerate(ours):
            b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            gst = struct.unpack_from("<i", b, 0x7C)[0]
            gxe, gye = struct.unpack_from("<dd", b, 0xC0)
            print("   drone %d state ours=%s game=%d | beam ours (%.2f,%.2f)"
                  " game (%.2f,%.2f) | drone (%.2f,%.2f)"
                  % (i, getattr(e, "state", "?"), gst,
                     getattr(e, "xend", 0.0), getattr(e, "yend", 0.0),
                     gxe, gye, e.xpos, e.ypos))
    if dead or f > 144:
        break
