"""Error distribution of the rocket PROJECTILE, and the first diverging flight.

§45 established the projectile is the kill agent and drifts sub-0.01px at first.
Quantify it, and dump one flight frame-by-frame from its launch so the source is
readable: position, direction, speed, step and odometer against the game's own
+0x50/+0x58/+0x60/+0x68/+0x70/+0x78/+0x80.
"""
import collections
import glob
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

POOL, ETYPE = 16, 18
dec = NppAttractDecoder()
buckets = collections.Counter()
tot = 0
dumped = False

for s in sorted(glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json")):
    stem = s.replace(".layout.json", "")
    demo = os.path.basename(stem)
    if not demo.isdigit():
        continue
    path = "/home/tetra/projects/attracts/%s" % demo
    if not os.path.exists(path):
        continue
    try:
        cap = Capture(stem)
    except Exception:
        continue
    p = next((q for q in cap.pools if q["i"] == POOL and q["count"]), None)
    if p is None or p["stride"] < 0x88:
        continue
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    tape = dec.decode_npp_attract_file(path).get("demo_data") or []
    h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    h.load_map_from_map_data(list(dec.create_nclone_map(path)))
    ours = h.sim.entity_dic.get(ETYPE, [])
    if p["count"] != len(ours):
        continue
    prev_flying = [False] * len(ours)
    rows = []
    for f in range(1, min(cap.n, len(tape)) + 1):
        hor, jmp = decode_input_to_controls(int(tape[f - 1]))
        h.tick(hor, jmp)
        rec = cap.frame(f)
        for i, e in enumerate(ours):
            flying = getattr(e, "state", None) == getattr(e, "FLYING", 99)
            if not flying:
                prev_flying[i] = False
                continue
            b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            gx, gy = struct.unpack_from("<dd", b, 0x50)
            gdx, gdy = struct.unpack_from("<dd", b, 0x60)
            gsp = struct.unpack_from("<d", b, 0x70)[0]
            gstep = struct.unpack_from("<d", b, 0x78)[0]
            gtr = struct.unpack_from("<d", b, 0x80)[0]
            pr = e.projectile
            d = math.hypot(pr.xpos - gx, pr.ypos - gy)
            tot += 1
            for lim in (1e-12, 1e-6, 1e-3, 0.01, 0.1, 1, 5, 1e9):
                if d <= lim:
                    buckets[lim] += 1
                    break
            if not dumped and not prev_flying[i]:
                rows = [(demo, i)]
            if not dumped and rows and rows[0] == (demo, i) and len(rows) < 13:
                rows.append((f, gx, gy, pr.xpos, pr.ypos, gsp, pr.speed,
                             gstep, pr.step, gtr, pr.travel, gdx, pr.xdir))
            prev_flying[i] = True
        if h.ninja_has_died() or h.ninja_has_won():
            break
    if rows and len(rows) > 6 and not dumped:
        dumped = True
        print("=== first captured flight: %s rocket %d ===" % rows[0])
        print("%5s | %-19s %-19s | %-15s | %-15s | %-15s"
              % ("frame", "game pos", "ours pos", "speed g/o", "step g/o", "travel g/o"))
        for r in rows[1:]:
            print("%5d | %8.3f,%8.3f %8.3f,%8.3f | %6.4f %6.4f | %6.4f %6.4f | %6.2f %6.2f"
                  % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10]))

print("\nprojectile frames (flying): %d" % tot)
names = {1e-12: "bit-exact", 1e-6: "<=1e-6", 1e-3: "<=1e-3", 0.01: "<=0.01px",
         0.1: "<=0.1px", 1: "<=1px", 5: "<=5px", 1e9: ">5px"}
run = 0
for lim in (1e-12, 1e-6, 1e-3, 0.01, 0.1, 1, 5, 1e9):
    run += buckets[lim]
    print("   %-10s %6d  (cumulative %.1f%%)"
          % (names[lim], buckets[lim], 100.0 * run / max(tot, 1)))
