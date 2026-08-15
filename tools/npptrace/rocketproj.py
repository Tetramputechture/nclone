"""Score the rocket PROJECTILE, not the launcher.

`entscore.py 16` reported 100% -- but it compared `+0x10`, the launcher body,
which never moves much. The thing that kills is the projectile, at
`+0x50/+0x58` (with direction at +0x60/+0x68, speed at +0x70, distance at +0x80).
That is why we die to rockets whose bodies are 29-47 px away while our ninja
tracks the game exactly.
"""
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

# Which attribute holds our projectile position?
probe = None
tot = match = 0
firsts = []

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
    if probe is None and ours:
        e = ours[0]
        cands = [a for a in dir(e)
                 if not a.startswith("__") and "x" in a.lower()
                 and isinstance(getattr(e, a, None), float)]
        probe = cands
        print("our rocket float attrs containing 'x': %s\n" % cands)
    first = [None] * len(ours)
    n = min(cap.n, len(tape))
    for f in range(1, n + 1):
        hor, jmp = decode_input_to_controls(int(tape[f - 1]))
        h.tick(hor, jmp)
        rec = cap.frame(f)
        for i in range(len(ours)):
            b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            gx, gy = struct.unpack_from("<dd", b, 0x50)
            if gx == 0.0 and gy == 0.0:
                continue
            e = ours[i]
            if getattr(e, "state", None) != getattr(e, "FLYING", 99):
                continue
            pr = getattr(e, "projectile", None); ox = getattr(pr, "xpos", None) if pr else None
            oy = getattr(pr, "ypos", None) if pr else None
            if ox is None:
                continue
            tot += 1
            if math.hypot(ox - gx, oy - gy) < 1e-9:
                match += 1
            elif first[i] is None:
                first[i] = (demo, i, f, round(gx, 2), round(gy, 2),
                            round(ox, 2), round(oy, 2))
                if len(firsts) < 6:
                    firsts.append(first[i])
        if h.ninja_has_died() or h.ninja_has_won():
            break

if tot:
    print("projectile frames compared: %d" % tot)
    print("   bit-exact: %d = %.2f%%" % (match, 100.0 * match / tot))
    print("\nfirst divergences (demo, ent, frame, game x/y, ours x/y):")
    for r in firsts:
        print("   %s" % (r,))
else:
    print("Could not find our projectile position attribute -- see the attr list")
    print("above and adjust the getattr() probe.")
