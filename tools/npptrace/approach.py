"""Closest approach between the ninja and a hazard, with `kill` neutered.

    python tools/npptrace/approach.py <type> <bound> <demolist> [limit]

The game wins every one of these demos, so its ninja never comes within the
hazard's true bound. Replaying with the kill disabled and histogramming our
closest approach says which kind of bug it is:

  * a tight cluster just INSIDE the bound  -> a boundary case: the test, its
    radius, or the frame it runs on;
  * a broad spread well inside            -> we are simply somewhere the game's
    ninja is not, and the hazard is the messenger.
"""
import collections
import math
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone")

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402

ETYPE = int(sys.argv[1])
BOUND = float(sys.argv[2])
demos = [l.strip().split("/")[-1] for l in open(sys.argv[3]) if l.strip()]
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else 200
demos = demos[:LIMIT]

dec = NppAttractDecoder()
hist = collections.Counter()
rows = []
for demo in demos:
    path = "/home/tetra/projects/attracts/%s" % demo
    try:
        tape = dec.decode_npp_attract_file(path).get("demo_data") or []
        h = NPlayHeadless(render_mode="rgb_array", enable_rendering=False)
        h.load_map_from_map_data(list(dec.create_nclone_map(path)))
    except Exception:
        continue
    ents = h.sim.entity_dic.get(ETYPE, [])
    if not ents:
        continue
    nj = h.sim.ninja
    nj.kill = lambda *a, **k: None
    best = 1e9
    at = None
    for f in range(1, len(tape) + 1):
        hor, jmp = decode_input_to_controls(int(tape[f - 1]))
        h.tick(hor, jmp)
        for e in ents:
            d = math.hypot(nj.xpos - e.xpos, nj.ypos - e.ypos)
            if d < best:
                best, at = d, f
        if h.ninja_has_won():
            break
    rows.append((demo, best, at, h.ninja_has_won()))
    hist[round(best)] += 1

print("type %d, our kill bound %.2f -- closest approach with kill disabled, %d demos\n"
      % (ETYPE, BOUND, len(rows)))
print("closest-approach histogram (px, rounded):")
for d in sorted(hist):
    bar = "#" * min(60, hist[d])
    flag = "" if d >= BOUND else "  <- inside our bound"
    print("   %5d  %4d  %s%s" % (d, hist[d], bar, flag))
inside = [r for r in rows if r[1] < BOUND]
print("\n%d/%d come inside %.2f" % (len(inside), len(rows), BOUND))
if inside:
    inside.sort(key=lambda r: -r[1])
    print("nearest-miss end of the distribution (largest closest-approach that still kills):")
    for demo, d, at, won in inside[:10]:
        print("   %-10s closest %8.4f at f=%-6s  (won with kill off: %s)" % (demo, d, at, won))
    print("deepest end:")
    for demo, d, at, won in inside[-5:]:
        print("   %-10s closest %8.4f at f=%-6s  (won with kill off: %s)" % (demo, d, at, won))
