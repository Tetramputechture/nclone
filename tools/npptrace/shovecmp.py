"""Shove thwump (type 28, pool 26): the game's fields beside ours, every frame it moves.

Usage: shovecmp.py <demo> [maxframe]
"""
import glob
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

demo = sys.argv[1]
MAXF = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9

caps = {}
for s in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json"):
    caps[os.path.basename(s).replace(".layout.json", "")] = s.replace(".layout.json", "")
cap = Capture(caps[demo])
p = next(q for q in cap.pools if q["i"] == 26 and q["count"])
base = 4 + cap.layout["ninjaSlab"] + p["offset"]
print("%s: pool 26 count=%d stride=0x%x" % (demo, p["count"], p["stride"]))

dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/%s" % demo
tape = dec.decode_npp_attract_file(path).get("demo_data") or []
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(list(dec.create_nclone_map(path)))
ours = h.sim.entity_dic[28]
nj = h.sim.ninja
print("ours: %d shove thwumps\n" % len(ours))

prev = None
for f in range(1, min(cap.n, len(tape), MAXF) + 1):
    hor, jmp = decode_input_to_controls(int(tape[f - 1]))
    h.tick(hor, jmp)
    rec = cap.frame(f)
    row = []
    for i in range(min(p["count"], len(ours))):
        b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
        gx, gy = struct.unpack_from("<dd", b, 0x10)
        gst = struct.unpack_from("<i", b, 0x30)[0] if p["stride"] >= 0x34 else None
        o = ours[i]
        row.append((gx, gy, gst, o.xpos, o.ypos, o.state))
    key = tuple(round(v, 4) if isinstance(v, float) else v for r in row for v in r)
    if key != prev:
        prev = key
        for i, (gx, gy, gst, ox, oy, ost) in enumerate(row):
            flag = "" if (abs(gx - ox) < 1e-9 and abs(gy - oy) < 1e-9) else "   <-- DIVERGED"
            print("f=%-5d T%d game (%8.3f,%8.3f) st=%-3s | ours (%8.3f,%8.3f) st=%-3s | ninja (%8.3f,%8.3f)%s"
                  % (f, i, gx, gy, gst, ox, oy, ost, nj.xpos, nj.ypos, flag))
    if h.ninja_has_died() or h.ninja_has_won():
        print("END f=%d died=%s" % (f, h.ninja_has_died()))
        break
