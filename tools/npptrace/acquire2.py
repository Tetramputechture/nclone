"""Is acquisition simply the first frame the drone has line of sight?

For each drone, walk the demo and note the first frame `raycast_vs_player`
succeeds while the game still has it in state 0, then compare against the frame
the game actually moved it to state 1. Only drones whose position still matches
the game's are counted, so the test is not confounded by drift.
"""
import collections
import glob
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
from nclone.physics import raycast_vs_player  # noqa: E402

POOLS = (10, 11)
rows = []
for d in ["/home/tetra/projects/nclone/tools/npptrace/out9",
          "/home/tetra/projects/nclone/tools/npptrace/out9_batch1"]:
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
        first_ray = {}
        done = set()
        for f in range(1, n + 1):
            hor, jmp = decode_input_to_controls(int(inputs[f - 1]))
            h.tick(hor, jmp)
            rec = cap.frame(f)
            ninja = h.sim.ninja
            if ninja is None:
                continue
            for p in pools:
                base = 4 + cap.layout["ninjaSlab"] + p["offset"]
                for i in range(p["count"]):
                    key = (demo, p["i"], i)
                    if key in done:
                        continue
                    b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
                    gx = struct.unpack_from("<d", b, 0x10)[0]
                    gy = struct.unpack_from("<d", b, 0x18)[0]
                    gs = struct.unpack_from("<i", b, 0x7C)[0]
                    # Cast from the GAME's drone position, so our drone's own
                    # patrol cannot confound it.
                    if key not in first_ray and raycast_vs_player(
                            h.sim, gx, gy, ninja.xpos, ninja.ypos):
                        first_ray[key] = f
                    if gs == 1:
                        done.add(key)
                        rows.append((demo, p["i"], i, f, first_ray.get(key)))

print("%-10s %-5s %-4s %-8s %-10s %s"
      % ("demo", "pool", "idx", "acquire", "first ray", "delta"))
deltas = collections.Counter()
for demo, pool, i, gf, fr in rows:
    d = (gf - fr) if fr is not None else None
    deltas[d] += 1
    print("%-10s %-5d %-4d %-8d %-10s %s" % (demo, pool, i, gf, fr, d))
print("\nacquire_frame - first_line_of_sight:")
for d, n in sorted(deltas.items(), key=lambda kv: (kv[0] is None, kv[0])):
    print("   %-8s %d" % (d, n))
