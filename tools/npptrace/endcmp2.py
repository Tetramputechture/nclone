"""Re-measure the endpoint agreement, restricted to frames actually discharging.

Frame 170 on 10131718 is the 2->3 transition (§17 records it). The game keeps the
last shot's endpoint in +0xC0 while our code resets xend to the drone -- that is
a stale-vs-reset artefact at a state boundary, not a raycast failure, and it
inflated the 57.2%. Compare only while state == 2, and report the size
distribution rather than one pass/fail threshold.
"""
import collections
import glob
import math
import os
import struct
import sys

os.environ["NCLONE_AB_DRONE_FULL_SHOT"] = "1"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone")
sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402

POOL, ETYPE = 10, 12
dec = NppAttractDecoder()
buckets = collections.Counter()
collapsed = 0
tot = 0

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
    if p is None or p["stride"] < 0xD8:
        continue
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    tape = dec.decode_npp_attract_file(path).get("demo_data") or []
    h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    h.load_map_from_map_data(list(dec.create_nclone_map(path)))
    ours = h.sim.entity_dic.get(ETYPE, [])
    if not ours:
        continue
    for f in range(1, min(cap.n, len(tape)) + 1):
        hor, jmp = decode_input_to_controls(int(tape[f - 1]))
        h.tick(hor, jmp)
        rec = cap.frame(f)
        for i in range(min(p["count"], len(ours))):
            b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            gst = struct.unpack_from("<i", b, 0x7C)[0]
            e = ours[i]
            # BOTH must be discharging: otherwise +0xC0 is stale.
            if gst != 2 or getattr(e, "state", None) != 2:
                continue
            gxe, gye = struct.unpack_from("<dd", b, 0xC0)
            if gxe == 0.0 and gye == 0.0:
                continue
            oxe, oye = getattr(e, "xend", None), getattr(e, "yend", None)
            if oxe is None:
                continue
            tot += 1
            if abs(oxe - e.xpos) < 1e-9 and abs(oye - e.ypos) < 1e-9:
                collapsed += 1
                continue
            d = math.hypot(oxe - gxe, oye - gye)
            for lim in (0.001, 0.1, 0.5, 1, 2, 5, 20, 100, 1e9):
                if d <= lim:
                    buckets[lim] += 1
                    break
        if h.ninja_has_died() or h.ninja_has_won():
            break

print("frames where BOTH engines are discharging: %d\n" % tot)
print("beam collapsed to the drone (raycast returned None): %d = %.1f%%"
      % (collapsed, 100.0 * collapsed / max(tot, 1)))
print("\nendpoint error distribution:")
names = {0.001: "exact", 0.1: "<=0.1px", 0.5: "<=0.5px", 1: "<=1px",
         2: "<=2px", 5: "<=5px", 20: "<=20px", 100: "<=100px", 1e9: ">100px"}
run = 0
for lim in (0.001, 0.1, 0.5, 1, 2, 5, 20, 100, 1e9):
    run += buckets[lim]
    print("   %-8s %5d   (cumulative %.1f%%)"
          % (names[lim], buckets[lim], 100.0 * run / max(tot, 1)))
