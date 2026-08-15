"""Bracket the toggle mine's thresholds, pairing mines by POSITION.

Pool order is not our entity order, so index pairing put the ninja 80 px from a
mine it was supposedly touching. `+0x10`/`+0x18` is the mine position and
`+0x28` its radius; `+0x30` is the state.
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

POOL = 19
brackets = collections.defaultdict(list)
radii = collections.Counter()

for d in ["/home/tetra/projects/nclone/tools/npptrace/out9",
          "/home/tetra/projects/nclone/tools/npptrace/out9_batch1"]:
    for s in sorted(glob.glob(d + "/*.layout.json")):
        demo = os.path.basename(s).replace(".layout.json", "")
        if demo.startswith("cap"):
            continue
        cap = Capture(os.path.join(d, demo))
        p = next((q for q in cap.pools if q["i"] == POOL), None)
        if p is None or not p["count"]:
            continue
        dec = NppAttractDecoder()
        path = "/home/tetra/projects/attracts/" + demo
        inputs = dec.decode_npp_attract_file(path).get("demo_data", []) or []
        h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
        h.load_map_from_map_data(list(dec.create_nclone_map(path)))
        base = 4 + cap.layout["ninjaSlab"] + p["offset"]
        n = min(cap.n, len(inputs))
        prev = {}
        for f in range(1, n + 1):
            hor, jmp = decode_input_to_controls(int(inputs[f - 1]))
            h.tick(hor, jmp)
            rec = cap.frame(f)
            ninja = h.sim.ninja
            if ninja is None:
                continue
            for i in range(p["count"]):
                b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
                mx = struct.unpack_from("<d", b, 0x10)[0]
                my = struct.unpack_from("<d", b, 0x18)[0]
                gs = struct.unpack_from("<i", b, 0x30)[0]
                radii[(gs, round(struct.unpack_from("<d", b, 0x28)[0], 4))] += 1
                dist = math.hypot(ninja.xpos - mx, ninja.ypos - my)
                key = (demo, i)
                if key in prev:
                    ps, pd = prev[key]
                    if gs != ps:
                        brackets[(ps, gs)].append((pd, dist, demo, i, f))
                prev[key] = (gs, dist)

print("captured radius (+0x28) by state: %s" % radii.most_common())
print()
for (a, b), rows in sorted(brackets.items()):
    on = sorted(r[1] for r in rows)
    before = sorted(r[0] for r in rows)
    print("state %d -> %d : %d transitions" % (a, b, len(rows)))
    print("   distance ON:     min %.3f  max %.3f" % (on[0], on[-1]))
    print("   distance BEFORE: min %.3f  max %.3f" % (before[0], before[-1]))
    for r in sorted(rows, key=lambda r: r[1])[:10]:
        print("      %-9s mine %d f%-5d before %8.3f  on %8.3f"
              % (r[2], r[3], r[4], r[0], r[1]))
