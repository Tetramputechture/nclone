"""Solve for the gun's lerp TARGET instead of assuming it.

The one-frame steps decay by 0.8 exactly, so the 20% lerp is right and only the
target is wrong. Invert it: target = prev + (actual - prev) / 0.2, then ask what
that angle is -- the ninja's bearing, or one of the four axis directions.
"""
import collections
import glob
import math
import os
import struct
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402

LERP = 0.2
POOLS = (10, 11)
AXES = {0.0: "right", math.pi / 2: "down", math.pi: "left", -math.pi / 2: "up"}
hits = collections.defaultdict(collections.Counter)


def wrap(d):
    while d >= math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


for d in ["/home/tetra/projects/nclone/tools/npptrace/out9",
          "/home/tetra/projects/nclone/tools/npptrace/out9_batch1",
          "/home/tetra/projects/nclone/tools/npptrace/out10",
          "/home/tetra/projects/nclone/tools/npptrace/out11"]:
    for s in sorted(glob.glob(d + "/*.layout.json")):
        demo = os.path.basename(s).replace(".layout.json", "")
        if demo.startswith("cap"):
            continue
        cap = Capture(os.path.join(d, demo))
        pools = [q for q in cap.pools if q["i"] in POOLS and q["count"]]
        if not pools:
            continue
        dec = NppAttractDecoder()
        path = "/home/tetra/projects/attracts/" + demo
        inputs = dec.decode_npp_attract_file(path).get("demo_data", []) or []
        h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
        h.load_map_from_map_data(list(dec.create_nclone_map(path)))
        n = min(cap.n, len(inputs))
        prev = {}
        for f in range(1, n + 1):
            hor, jmp = decode_input_to_controls(int(inputs[f - 1]))
            h.tick(hor, jmp)
            ninja = h.sim.ninja
            if ninja is None or h.ninja_has_died():
                break   # our ninja's position is frozen from here; bearings lie
            rec = cap.frame(f)
            for p in pools:
                base = 4 + cap.layout["ninjaSlab"] + p["offset"]
                for i in range(p["count"]):
                    b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
                    gun = struct.unpack_from("<f", b, 0x50)[0]
                    st = struct.unpack_from("<i", b, 0x7C)[0]
                    gx = struct.unpack_from("<d", b, 0x10)[0]
                    gy = struct.unpack_from("<d", b, 0x18)[0]
                    key = (demo, p["i"], i)
                    if key in prev:
                        pgun, _ = prev[key]
                        step = wrap(gun - pgun)
                        if abs(step) < 1e-6:
                            hits[st]["already there"] += 1
                        else:
                            target = pgun + step / LERP
                            bearing = math.atan2(ninja.ypos - gy, ninja.xpos - gx)
                            e0 = struct.unpack_from("<f", b, 0xE0)[0]
                            if abs(wrap(target - e0)) < 3e-3:
                                hits[st]["+0xE0 latched"] += 1
                            elif abs(wrap(target - bearing)) < 3e-3:
                                hits[st]["ninja bearing"] += 1
                            else:
                                lab = None
                                for ax, name in AXES.items():
                                    if abs(wrap(target - ax)) < 3e-3:
                                        lab = "axis:" + name
                                        break
                                hits[st][lab or "other"] += 1
                    prev[key] = (gun, st)

print("what the gun is lerping toward, by state:")
for st in sorted(hits):
    tot = sum(hits[st].values())
    print("  state %d  (%d frames)" % (st, tot))
    for k, v in hits[st].most_common(6):
        print("      %-16s %6d  %5.1f%%" % (k, v, 100.0 * v / max(tot, 1)))
