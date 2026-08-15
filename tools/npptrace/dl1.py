"""Exact frame-1 positions for every dual laser on one demo.

Ordering is identity, so entities 2 and 3 diverging on frame 1 is a real
numerical difference at spawn or on the first step -- print it.
"""
import glob
import json
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

POOL, ETYPE = 21, 23
DEMO = sys.argv[1] if len(sys.argv) > 1 else "10131334"
stem = next(os.path.join(c, DEMO)
            for c in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*")
            if os.path.exists(os.path.join(c, DEMO) + ".layout.json"))
cap = Capture(stem)
p = next(q for q in cap.pools if q["i"] == POOL and q["count"])
base = 4 + cap.layout["ninjaSlab"] + p["offset"]
dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/%s" % DEMO
tape = dec.decode_npp_attract_file(path).get("demo_data") or []
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(list(dec.create_nclone_map(path)))
ours = h.sim.entity_dic[ETYPE]

print("%s: %d dual lasers, stride 0x%X" % (DEMO, p["count"], p["stride"]))
print("\nBEFORE any tick (spawn):")
for i, e in enumerate(ours):
    print("   ours %d  (%.6f, %.6f)  mode=%s" % (i, e.xpos, e.ypos,
                                                 getattr(e, "mode", "?")))
for f in (1, 2, 3):
    hor, jmp = decode_input_to_controls(int(tape[f - 1]))
    h.tick(hor, jmp)
    rec = cap.frame(f)
    print("\nframe %d:" % f)
    for i in range(min(p["count"], len(ours))):
        b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
        gx, gy = struct.unpack_from("<dd", b, 0x10)
        e = ours[i]
        d = ((e.xpos - gx) ** 2 + (e.ypos - gy) ** 2) ** 0.5
        flag = "  <<<" if d > 1e-9 else ""
        print("   %d  game (%12.6f,%12.6f)  ours (%12.6f,%12.6f)  d=%.3e%s"
              % (i, gx, gy, e.xpos, e.ypos, d, flag))
