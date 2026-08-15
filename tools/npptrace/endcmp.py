"""Compare our shot endpoint against the game's own, shot by shot.

+0xC0/+0xC8 is the game's traced beam endpoint -- the exact quantity
`_trace_beam` computes into self.xend/self.yend. We raycast against tile
geometry too, so if the 88 false kills are an occlusion problem it will show up
here as endpoints that disagree.
"""
import glob
import json
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
match = miss = 0
rows = []

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
    n = min(cap.n, len(tape))
    for f in range(1, n + 1):
        hor, jmp = decode_input_to_controls(int(tape[f - 1]))
        h.tick(hor, jmp)
        rec = cap.frame(f)
        for i in range(min(p["count"], len(ours))):
            b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            gxe, gye = struct.unpack_from("<dd", b, 0xC0)
            if gxe == 0.0 and gye == 0.0:
                continue  # not discharging
            e = ours[i]
            oxe = getattr(e, "xend", None)
            oye = getattr(e, "yend", None)
            if oxe is None:
                continue
            d = math.hypot(oxe - gxe, oye - gye)
            if d < 0.5:
                match += 1
            else:
                miss += 1
                if len(rows) < 10:
                    rows.append((demo, i, f, round(gxe, 2), round(gye, 2),
                                 round(oxe, 2), round(oye, 2), round(d, 1)))
        if h.ninja_has_died() or h.ninja_has_won():
            break

tot = match + miss
print("shot-endpoint frames compared: %d" % tot)
print("   endpoint matches the game (<0.5px): %d = %.1f%%"
      % (match, 100.0 * match / max(tot, 1)))
print("   disagrees:                          %d = %.1f%%"
      % (miss, 100.0 * miss / max(tot, 1)))
print("\nsample disagreements (demo, ent, frame, game xend/yend, ours, dist):")
for r in rows:
    print("   %s" % (r,))
