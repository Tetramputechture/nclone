"""Score BOTH chaser models against every captured chaser, frame by frame.

The two smoke regressions have no capture, so judge the model where ground truth
exists: how long does each chaser stay bit-exact with the game? Run this with and
without the flags and compare. A model that is right should push every drone's
first divergence later, not just the ones that happen to flip a demo.
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

POOL, ETYPE = 13, 15
dec = NppAttractDecoder()
rows = []
total_exact = 0
total_frames = 0

for s in sorted(glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json")):
    stem = s.replace(".layout.json", "")
    demo = os.path.basename(stem)
    if not demo.isdigit():
        continue  # capNN captures have no resolved demo id
    path = "/home/tetra/projects/attracts/%s" % demo
    if not os.path.exists(path):
        continue
    try:
        cap = Capture(stem)
    except Exception:
        continue
    p = next((q for q in cap.pools if q["i"] == POOL and q["count"]), None)
    if p is None:
        continue
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    tape = dec.decode_npp_attract_file(path).get("demo_data") or []
    h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    h.load_map_from_map_data(list(dec.create_nclone_map(path)))
    ours = h.sim.entity_dic.get(ETYPE, [])
    n = min(cap.n, len(tape))
    first = [None] * min(p["count"], len(ours))
    for f in range(1, n + 1):
        hor, jmp = decode_input_to_controls(int(tape[f - 1]))
        h.tick(hor, jmp)
        rec = cap.frame(f)
        for i in range(len(first)):
            if first[i] is not None:
                continue
            b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            gx, gy = struct.unpack_from("<dd", b, 0x10)
            e = ours[i]
            if abs(e.xpos - gx) > 1e-9 or abs(e.ypos - gy) > 1e-9:
                first[i] = f
        if h.ninja_has_died() or h.ninja_has_won():
            break
    for i, fv in enumerate(first):
        exact = (fv - 1) if fv is not None else n
        total_exact += exact
        total_frames += n
        rows.append((demo, i, exact, n, fv))

print("%-10s %-4s %8s %8s  %s" % ("demo", "ent", "exact", "of", "first div"))
for r in sorted(rows):
    print("%-10s %-4d %8d %8d  %s" % (r[0], r[1], r[2], r[3], r[4] or "NONE"))
print("\n%d chasers over %d captures" % (len(rows), len({r[0] for r in rows})))
print("TOTAL bit-exact chaser frames: %d / %d = %.2f%%"
      % (total_exact, total_frames, 100.0 * total_exact / max(total_frames, 1)))
print("chasers exact for the whole demo: %d / %d"
      % (sum(1 for r in rows if r[4] is None), len(rows)))
