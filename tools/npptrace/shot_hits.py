"""Characterise the shots our lethality test would kill on but the game does not.

Restricted to the six demos that produced a false kill, and doing no work beyond
what is needed: the drone-to-ninja range, the distance the shot's raycast
travelled, and where along the beam the ninja sits.
"""
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
from nclone.physics import beam_hits_ninja, get_raycast_distance  # noqa: E402

DEMOS = ["10132175", "10134101", "10134102", "10136116", "10138394", "10138797"]
DIRS = ["/home/tetra/projects/nclone/tools/npptrace/out9",
        "/home/tetra/projects/nclone/tools/npptrace/out9_batch1"]
POOLS = (10, 11)

print("%-10s %-6s %-4s %-9s %-9s %-9s %s"
      % ("demo", "frame", "shot", "range", "raycast", "along", "perp"))
hits = []
extra = []
alls = []
for demo in DEMOS:
    stem = next((os.path.join(d, demo) for d in DIRS
                 if os.path.exists(os.path.join(d, demo + ".layout.json"))), None)
    cap = Capture(stem)
    pools = [q for q in cap.pools if q["i"] in POOLS and q["count"]]
    dec = NppAttractDecoder()
    path = "/home/tetra/projects/attracts/" + demo
    inputs = dec.decode_npp_attract_file(path).get("demo_data", []) or []
    h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    h.load_map_from_map_data(list(dec.create_nclone_map(path)))
    n = min(cap.n, len(inputs))
    for f in range(1, n + 1):
        hor, jmp = decode_input_to_controls(int(inputs[f - 1]))
        h.tick(hor, jmp)
        ninja = h.sim.ninja
        if ninja is None or not ninja.is_valid_target():
            continue
        rec = cap.frame(f)
        for p in pools:
            base = 4 + cap.layout["ninjaSlab"] + p["offset"]
            for i in range(p["count"]):
                b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
                if struct.unpack_from("<i", b, 0x7C)[0] != 2:
                    continue
                if struct.unpack_from("<i", b, 0x8C)[0] != 0:
                    continue
                gx = struct.unpack_from("<d", b, 0x10)[0]
                gy = struct.unpack_from("<d", b, 0x18)[0]
                dx = struct.unpack_from("<d", b, 0x90)[0]
                dy = struct.unpack_from("<d", b, 0x98)[0]
                length = get_raycast_distance(h.sim, gx, gy, dx, dy)
                if length:
                    xe, ye = gx + dx * length, gy + dy * length
                else:
                    xe, ye, length = gx, gy, None
                rng = math.hypot(ninja.xpos - gx, ninja.ypos - gy)
                shot_i = struct.unpack_from("<i", b, 0x88)[0]
                drone = next((e for lst in h.sim.entity_dic.values()
                              for e in lst
                              if getattr(e, "type", None) == p["i"] + 2
                              and abs(e.xpos - gx) < 1e-6
                              and abs(e.ypos - gy) < 1e-6), None)
                lane = (drone.ninja_in_corridor(drone.dir, ninja)
                        if drone is not None else None)
                along = (ninja.xpos - gx) * dx + (ninja.ypos - gy) * dy
                perp = abs((ninja.xpos - gx) * dy - (ninja.ypos - gy) * dx)
                alls.append(rng)
                if beam_hits_ninja(gx, gy, xe, ye, ninja, length):
                    hits.append((rng, along, perp, length or 0.0))
                    extra.append((demo, f, shot_i, lane, ninja.state,
                                  round(perp, 2), round(rng, 1)))
                    print("%-10s %-6d %-4d %-9.1f %-9.1f %-9.1f %.2f"
                          % (demo, f, i, rng, length or 0.0, along, perp))


def stat(name, xs):
    xs = sorted(xs)
    if not xs:
        print("  %-18s none" % name)
        return
    print("  %-18s n=%-4d min %.1f  median %.1f  max %.1f"
          % (name, len(xs), xs[0], xs[len(xs) // 2], xs[-1]))


print()
stat("range, false kills", [r[0] for r in hits])
stat("range, all shots", alls)
stat("raycast len, kills", [r[3] for r in hits])
import collections as _c
print()
print("%-10s %-6s %-5s %-7s %-6s %-6s %s"
      % ("demo", "frame", "shot", "inlane", "nstate", "perp", "range"))
for e in extra:
    print("%-10s %-6d %-5s %-7s %-6s %-6s %s" % e)
print()
print("in-lane at the moment of the false kill: %s"
      % _c.Counter(e[3] for e in extra).most_common())
print("shot index within the burst: %s"
      % _c.Counter(e[2] for e in extra).most_common())
