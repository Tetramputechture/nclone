"""Classify each rocket flight: wrong from launch, or drifts mid-flight?

75.3% of flying frames are now bit-exact but 1173 are >5px out. If the bad frames
belong to flights that are wrong from their FIRST frame, the residual is still
the launch aim. If they start exact and diverge later, it is the steering.
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
kinds = collections.Counter()
samples = []

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
    was_flying = [False] * len(ours)
    flight = [None] * len(ours)
    for f in range(1, min(cap.n, len(tape)) + 1):
        hor, jmp = decode_input_to_controls(int(tape[f - 1]))
        h.tick(hor, jmp)
        rec = cap.frame(f)
        for i, e in enumerate(ours):
            flying = getattr(e, "state", None) == getattr(e, "FLYING", 99)
            if flying and not was_flying[i]:
                flight[i] = {"start": f, "first_bad": None, "n": 0}
            if not flying:
                if flight[i] and flight[i]["n"] > 2:
                    fl = flight[i]
                    if fl["first_bad"] is None:
                        kinds["exact for the whole flight"] += 1
                    elif fl["first_bad"] == fl["start"]:
                        kinds["wrong from the FIRST frame"] += 1
                        if len(samples) < 6:
                            samples.append((demo, i, fl["start"], fl["n"]))
                    else:
                        kinds["starts exact, drifts later"] += 1
                    flight[i] = None
                was_flying[i] = False
                continue
            was_flying[i] = True
            b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            gx, gy = struct.unpack_from("<dd", b, 0x50)
            pr = e.projectile
            flight[i]["n"] += 1
            if (flight[i]["first_bad"] is None
                    and math.hypot(pr.xpos - gx, pr.ypos - gy) > 1e-9):
                flight[i]["first_bad"] = f
        if h.ninja_has_died() or h.ninja_has_won():
            break

tot = sum(kinds.values())
print("completed flights: %d\n" % tot)
for k, v in kinds.most_common():
    print("   %-30s %4d  %5.1f%%" % (k, v, 100.0 * v / max(tot, 1)))
print("\nsample flights wrong from the first frame (demo, ent, start, len):")
for r in samples:
    print("   %s" % (r,))
