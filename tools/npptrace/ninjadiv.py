"""First frame our ninja leaves the game's ninja, for named demos.

Usage: ninjadiv.py <demo> [<demo> ...]

Prints the first bit-inexact frame, the first frame past a 1e-6 / 1e-3 / 1 px
threshold, and the drift profile after it. A demo that is off-trajectory has a
core physics bug upstream of whatever hazard finally kills it.
"""
import glob
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

caps = {}
for s in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json"):
    caps[os.path.basename(s).replace(".layout.json", "")] = s.replace(".layout.json", "")
for s in glob.glob("/home/tetra/projects/nclone/docs/game_traces/*.layout.json"):
    caps.setdefault(os.path.basename(s).replace(".layout.json", ""), s.replace(".layout.json", ""))

dec = NppAttractDecoder()
for demo in sys.argv[1:]:
    if demo not in caps:
        print("%s: NO CAPTURE" % demo)
        continue
    cap = Capture(caps[demo])
    slab = cap.layout.get("ninjaSlab", 0x200)
    path = "/home/tetra/projects/attracts/%s" % demo
    tape = dec.decode_npp_attract_file(path).get("demo_data") or []
    h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    h.load_map_from_map_data(list(dec.create_nclone_map(path)))
    n = min(cap.n, len(tape))
    marks = {}
    prof = []
    died = None
    for f in range(1, n + 1):
        hor, jmp = decode_input_to_controls(int(tape[f - 1]))
        h.tick(hor, jmp)
        rec = cap.frame(f)
        gx = struct.unpack_from("<d", rec, 4 + 0x20)[0]
        gy = struct.unpack_from("<d", rec, 4 + 0x28)[0]
        ox, oy = h.ninja_position()
        d = max(abs(ox - gx), abs(oy - gy))
        for t in (0.0, 1e-9, 1e-6, 1e-3, 1.0, 10.0):
            if d > t and t not in marks:
                marks[t] = (f, d)
        prof.append((f, d))
        if died is None and (h.ninja_has_died() or h.ninja_has_won()):
            died = (f, "died" if h.ninja_has_died() else "won")
            break
    print("== %s  (%d frames compared, cap %d, tape %d)" % (demo, len(prof), cap.n, len(tape)))
    print("   end: %s" % (died,))
    for t in (0.0, 1e-9, 1e-6, 1e-3, 1.0, 10.0):
        m = marks.get(t)
        print("   first |d| > %-8g : %s" % (t, ("frame %d  d=%.3e" % m) if m else "never"))
    f0 = marks.get(0.0, (None,))[0]
    if f0:
        lo = max(1, f0 - 3)
        print("   around first divergence:")
        for f, d in prof[lo - 1: f0 + 6]:
            print("      f=%-6d d=%.6e" % (f, d))
    sys.stdout.flush()
