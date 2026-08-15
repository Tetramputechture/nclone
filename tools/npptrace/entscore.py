"""Score ANY entity type's positions against the game, across every capture.

Generalisation of chaserscore.py. Usage: entscore.py <pool> [--list]
Pool index = type - 2. Reports bit-exact frames before first divergence, which
is a ground-truth objective rather than a count of demo flips.
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

POOL = int(sys.argv[1])
ETYPE = POOL + 2
SHOW = "--list" in sys.argv
dec = NppAttractDecoder()
rows = []
tot_exact = tot_frames = 0

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
    if p is None:
        continue
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    tape = dec.decode_npp_attract_file(path).get("demo_data") or []
    h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    h.load_map_from_map_data(list(dec.create_nclone_map(path)))
    ours = h.sim.entity_dic.get(ETYPE, [])
    if not ours:
        continue
    n = min(cap.n, len(tape))
    first = [None] * min(p["count"], len(ours))
    for f in range(1, n + 1):
        hor, jmp = decode_input_to_controls(int(tape[f - 1]))
        h.tick(hor, jmp)
        rec = cap.frame(f)
        for i in range(len(first)):
            if first[i] is not None:
                continue
            b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            gx, gy = struct.unpack_from("<dd", b, 0x10)
            e = ours[i]
            if abs(e.xpos - gx) > 1e-9 or abs(e.ypos - gy) > 1e-9:
                first[i] = f
        if h.ninja_has_died() or h.ninja_has_won():
            break
    for i, fv in enumerate(first):
        exact = (fv - 1) if fv is not None else n
        tot_exact += exact
        tot_frames += n
        rows.append((demo, i, exact, n, fv))

if SHOW:
    print("%-10s %-4s %8s %8s  %s" % ("demo", "ent", "exact", "of", "first div"))
    for r in sorted(rows, key=lambda r: r[2]):
        print("%-10s %-4d %8d %8d  %s" % (r[0], r[1], r[2], r[3], r[4] or "NONE"))
print("\npool %d (type %d): %d entities over %d captures"
      % (POOL, ETYPE, len(rows), len({r[0] for r in rows})))
print("TOTAL bit-exact frames: %d / %d = %.2f%%"
      % (tot_exact, tot_frames, 100.0 * tot_exact / max(tot_frames, 1)))
print("exact for the whole demo: %d / %d"
      % (sum(1 for r in rows if r[4] is None), len(rows)))
