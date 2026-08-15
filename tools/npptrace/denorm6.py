"""As denorm5, but with the x the collision actually sees.

The contact is evaluated after the frame's integration, so the x is the one at
the target frame, while the distance comes from the y ENTERING it plus gravity.
"""
import json
import math
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402

ROOT = "/home/tetra/projects/nclone/tools/npptrace"
GRAV = 0.06666666666666665
CASES = [("10137734", 9), ("10132599", 8), ("10137755", 20), ("10131474", 90),
         ("10133925", 213)]


def trace_for(demo):
    for d in ("out9", "out9_batch1"):
        p = "%s/%s/%s.ninja.jsonl" % (ROOT, d, demo)
        if os.path.exists(p):
            return p


for demo, target in CASES:
    tp = trace_for(demo)
    if tp is None:
        continue
    game = [json.loads(l) for l in open(tp) if l.strip()]
    dec = NppAttractDecoder()
    path = "/home/tetra/projects/attracts/" + demo
    inputs = dec.decode_npp_attract_file(path).get("demo_data", []) or []
    h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    h.load_map_from_map_data(list(dec.create_nclone_map(path)))
    n = h.sim.ninja
    y_before = None
    for f in range(1, target + 1):
        if f == target:
            y_before = n.ypos
        hor, jmp = decode_input_to_controls(int(inputs[f - 1]))
        h.tick(hor, jmp)
    x = n.xpos
    want = game[target - 1].get("floor_x", 0.0)
    floor_y = round((n.ypos + 10.0) / 12.0) * 12.0
    dist = floor_y - (y_before + GRAV)
    print("%s f%d: x=%.17g  y_before=%g  floor_y=%g  dist=%.17g"
          % (demo, target, x, y_before, floor_y, dist))
    print("   game floor_x = %.17g" % want)
    tilex = math.floor(x / 24.0) * 24.0
    halfx = math.floor(x / 12.0) * 12.0
    best = None
    for name, x1, px in (("12px half R->L", halfx + 12.0, -12.0),
                         ("12px half L->R", halfx, 12.0),
                         ("24px tile R->L", tilex + 24.0, -24.0),
                         ("24px tile L->R", tilex, 24.0)):
        t = (x - x1) * px
        lensq = px * px
        if not (0 < t < lensq):
            print("      %-16s outside" % name)
            continue
        a = x1 + (t / lensq) * px
        got = (x - a) / dist if dist else 0.0
        hit = "   <== MATCH" if got == want else ""
        if hit:
            best = name
        print("      %-16s dx=%-24.17g floor_x=%-24.17g%s" % (name, x - a, got, hit))
    print("   -> %s\n" % (best or "no candidate reproduces it"))
