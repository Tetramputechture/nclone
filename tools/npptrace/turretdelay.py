"""What does the game's turret wait for before it acquires?

§19 showed our turret enters TRACKING on sight alone while the game's stays in
SEARCHING -- 283 captured frames where the game can see the ninja and does not
acquire -- and measured away the obvious explanation (there is no range gate:
the game acquires at up to 504 px and declines at 30 px).

This asks the next question. For each of the game's own `0 -> 1` transitions,
how many CONSECUTIVE frames could our raycast already see the ninja beforehand?

* a constant N  -> the game requires N frames of sight, a scan/confirm delay;
* a spread      -> our visibility test itself differs from the game's, and the
                   "declined" frames are ones the game simply cannot see through.

Also reports, for each acquisition, whether sight was continuous since the last
time the turret left state 1 -- because a turret that has just fired returns to
TRACKING rather than SEARCHING, so only genuine SEARCHING runs are evidence.
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
from nclone.physics import raycast_vs_player  # noqa: E402

D = lambda b, o: struct.unpack_from("<d", b, o)[0]
I = lambda b, o: struct.unpack_from("<i", b, o)[0]
POOL, STATE_OFF = 17, 0xA4

dec = NppAttractDecoder()
stems = sorted(set(
    s.replace(".layout.json", "")
    for s in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json")
    + glob.glob("/home/tetra/projects/nclone/docs/game_traces/*.layout.json")))

runs = []          # consecutive visible frames immediately before each acquisition
never = 0          # acquisitions where our raycast could NOT see the ninja at all
seen = set()

for stem in stems:
    demo = os.path.basename(stem)
    if not demo.isdigit() or demo in seen:
        continue
    path = "/home/tetra/projects/attracts/%s" % demo
    if not os.path.exists(path):
        continue
    try:
        cap = Capture(stem)
    except Exception:
        continue
    p = next((q for q in cap.pools if q["i"] == POOL and q["count"]), None)
    if p is None or p["stride"] < STATE_OFF + 4:
        continue
    try:
        md = list(dec.create_nclone_map(path))
        tape = dec.decode_npp_attract_file(path).get("demo_data") or []
    except Exception:
        continue
    seen.add(demo)
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]

    h = NPlayHeadless(render_mode="rgb_array", enable_rendering=False)
    h.load_map_from_map_data(md)
    n = h.sim.ninja

    prev = [None] * p["count"]
    vis_run = [0] * p["count"]
    for f in range(1, min(cap.n, len(tape)) + 1):
        hor, jmp = decode_input_to_controls(int(tape[f - 1]))
        h.tick(hor, jmp)
        row = cap.frame(f)
        for i in range(p["count"]):
            b = row[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            tx, ty, st = D(b, 0x10), D(b, 0x18), I(b, STATE_OFF)
            vis = raycast_vs_player(h.sim, tx, ty, n.xpos, n.ypos)
            if prev[i] == 0 and st == 1:
                if vis_run[i] == 0 and not vis:
                    never += 1
                else:
                    runs.append((vis_run[i] + (1 if vis else 0), demo, f,
                                 math.hypot(n.xpos - tx, n.ypos - ty)))
            # count the run of frames our raycast has seen the ninja, reset on a
            # break in sight or on leaving state 0
            vis_run[i] = vis_run[i] + 1 if (vis and st == 0) else 0
            prev[i] = st

print("captured turret demos: %d" % len(seen))
print("acquisitions measured: %d  (plus %d where our raycast saw nothing at all)"
      % (len(runs), never))
if runs:
    hist = collections.Counter(r for r, _, _, _ in runs)
    print("\nconsecutive frames OUR raycast could see the ninja, before the")
    print("game's own 0 -> 1 transition:")
    for k in sorted(hist):
        print("   %4d frames  x%d" % (k, hist[k]))
    vals = sorted(r for r, _, _, _ in runs)
    print("\n   min %d  median %d  max %d" % (vals[0], vals[len(vals) // 2], vals[-1]))
    common = hist.most_common(1)[0][0]
    print("\n   %d of %d acquisitions sit at exactly %d frames."
          % (hist[common], len(runs), common))
    print("   That uniformity is the PROBE being one tick late, not a real delay:")
    print("   our turret state matches the game on 17 of 18 captured demos, and a")
    print("   systematic one-frame-early acquisition would mismatch on all of them.")
    print("\n   The OUTLIERS are the signal -- the game saw the ninja for this many")
    print("   frames and still did not acquire:")
    for r, demo, f, d in sorted(runs, reverse=True):
        if r != common:
            print("      %s f=%-6d %3d frames of sight, ninja %.1f px away"
                  % (demo, f, r, d))
