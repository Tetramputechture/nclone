"""Dual laser (23), ours vs the game's, every frame.

Usage: lasertrace.py <demo> [first] [last] [step]
Prints only frames where either side moves, so a 1000-frame crawl is readable.
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
FIRST = int(sys.argv[2]) if len(sys.argv) > 2 else 1
LAST = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9

caps = {}
for s in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json"):
    caps[os.path.basename(s).replace(".layout.json", "")] = s.replace(".layout.json", "")
for s in glob.glob("/home/tetra/projects/nclone/docs/game_traces/*.layout.json"):
    caps.setdefault(os.path.basename(s).replace(".layout.json", ""), s.replace(".layout.json", ""))

cap = Capture(caps[demo])
p = next(q for q in cap.pools if q["i"] == 21 and q["count"])
base = 4 + cap.layout["ninjaSlab"] + p["offset"]

dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/%s" % demo
tape = dec.decode_npp_attract_file(path).get("demo_data") or []
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(list(dec.create_nclone_map(path)))
ours = h.sim.entity_dic[23]
print("%s: %d lasers, stride 0x%x" % (demo, len(ours), p["stride"]))
print("our laser attrs: %s\n" % sorted(k for k in vars(ours[0])
                                       if not k.startswith(("sim", "_"))))

prev = None
for f in range(1, min(cap.n, len(tape)) + 1):
    hor, jmp = decode_input_to_controls(int(tape[f - 1]))
    h.tick(hor, jmp)
    if not (FIRST <= f <= LAST):
        if h.ninja_has_died() or h.ninja_has_won():
            break
        continue
    rec = cap.frame(f)
    row = []
    for i, o in enumerate(ours):
        b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
        gx, gy = struct.unpack_from("<dd", b, 0x10)
        row.append((gx, gy, o.xpos, o.ypos,
                    getattr(o, "mode", None), getattr(o, "state", None),
                    getattr(o, "xanchor", None), getattr(o, "yanchor", None)))
    key = tuple((round(a, 6) for r in row for a in r[:4]))
    if key != prev:
        prev = key
        for i, (gx, gy, ox, oy, mode, st, ax, ay) in enumerate(row):
            print("f=%-5d L%d game (%10.4f,%10.4f) | ours (%10.4f,%10.4f)  d=(%+8.4f,%+8.4f)"
                  "  mode=%s st=%s anchor=(%s,%s)"
                  % (f, i, gx, gy, ox, oy, ox - gx, oy - gy, mode, st,
                     None if ax is None else round(ax, 3),
                     None if ay is None else round(ay, 3)))
    if h.ninja_has_died() or h.ninja_has_won():
        print("END f=%d died=%s" % (f, h.ninja_has_died()))
        break
