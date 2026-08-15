"""How many discharges does our drone run, against how many the game runs?

Counting outcomes cannot say whether the model is too eager; counting cycles
can. Stops at the port's own death frame, because past that our ninja is frozen
and every comparison is meaningless.
"""
import collections
import glob
import os
import struct
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["NCLONE_AB_DRONE_FULL_SHOT"] = "1"
sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402

DIRS = ["/home/tetra/projects/nclone/tools/npptrace/out12",
        "/home/tetra/projects/nclone/tools/npptrace/out9"]
total = collections.Counter()
print("%-10s %-7s %-8s %-8s %s" % ("demo", "frames", "game", "port", "note"))

for d in DIRS:
    for s in sorted(glob.glob(d + "/*.layout.json")):
        demo = os.path.basename(s).replace(".layout.json", "")
        if demo.startswith("cap"):
            continue
        cap = Capture(os.path.join(d, demo))
        p = next((q for q in cap.pools if q["i"] == 10 and q["count"]), None)
        if p is None:
            continue
        dec = NppAttractDecoder()
        path = "/home/tetra/projects/attracts/" + demo
        inputs = dec.decode_npp_attract_file(path).get("demo_data", []) or []
        h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
        h.load_map_from_map_data(list(dec.create_nclone_map(path)))
        ds = [e for lst in h.sim.entity_dic.values() for e in lst
              if getattr(e, "type", None) == 12]
        base = 4 + cap.layout["ninjaSlab"] + p["offset"]
        gprev, pprev = {}, {}
        gfire = pfire = 0
        last = 0
        for f in range(1, min(cap.n, len(inputs)) + 1):
            hor, jmp = decode_input_to_controls(int(inputs[f - 1]))
            h.tick(hor, jmp)
            if h.ninja_has_died():
                break
            last = f
            rec = cap.frame(f)
            for i in range(min(p["count"], len(ds))):
                b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
                gs = struct.unpack_from("<i", b, 0x7C)[0]
                if gprev.get(i) != 2 and gs == 2:
                    gfire += 1
                gprev[i] = gs
                ps = ds[i].state
                if pprev.get(i) != 2 and ps == 2:
                    pfire += 1
                pprev[i] = ps
        note = "" if pfire == gfire else ("port fires %+d" % (pfire - gfire))
        print("%-10s %-7d %-8d %-8d %s" % (demo, last, gfire, pfire, note))
        total["game"] += gfire
        total["port"] += pfire
        total["demos"] += 1

print("\nover %d demos with a chaingun: game %d discharges, port %d"
      % (total["demos"], total["game"], total["port"]))
