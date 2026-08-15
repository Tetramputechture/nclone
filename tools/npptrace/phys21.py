"""The 21 physics deaths -- the joint-largest cause, never attacked as a category.

Look for structure: do they share a level, a ninja state, a place on the map, a
tile type under the ninja? A physics death means our ninja is somewhere the real
one was not, so the shared feature is the bug.
"""
import collections
import json
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone")

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402

rows = json.load(open("/tmp/cause69.json"))
phys = [r for r in rows if r[1] == "physics"]
print("%d physics deaths\n" % len(phys))

dec = NppAttractDecoder()
out = []
maps = collections.Counter()
states = collections.Counter()
tiles = collections.Counter()
for bucket, cause, demo_id, frames, total, frac in sorted(phys, key=lambda r: r[2]):
    path = "/home/tetra/projects/attracts/%s" % demo_id
    try:
        res = dec.decode_npp_attract_file(path)
        tape = res.get("demo_data") or []
        mapb = list(dec.create_nclone_map(path))
        h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
        h.load_map_from_map_data(mapb)
        sim = h.sim
        prev = []
        for i, b in enumerate(tape, 1):
            hor, jmp = decode_input_to_controls(int(b))
            h.tick(hor, jmp)
            n = sim.ninja
            prev.append((i, hor, jmp, n.state, n.xpos, n.ypos, n.xspeed, n.yspeed))
            if len(prev) > 6:
                prev.pop(0)
            if h.ninja_has_died() or h.ninja_has_won():
                break
        n = sim.ninja
        fp = sum(mapb)
        maps[fp] += 1
        # State on the frame BEFORE death is the informative one.
        pstate = prev[-2][3] if len(prev) >= 2 else -1
        states[pstate] += 1
        tx, ty = int(n.xpos // 24), int(n.ypos // 24)
        tid = None
        try:
            tid = sim.tile_dic.get((tx, ty))
        except Exception:
            pass
        tiles[tid] += 1
        out.append((demo_id, frames, total, round(frac, 3), pstate,
                    round(n.xpos, 1), round(n.ypos, 1),
                    round(n.xspeed, 2), round(n.yspeed, 2), tid, fp))
    except Exception as e:
        out.append((demo_id, -1, -1, -1, -1, 0, 0, 0, 0, None, "ERR %s" % e))

print("%-9s %6s %6s %5s %5s %8s %8s %7s %7s %5s" %
      ("id", "frame", "of", "frac", "state", "x", "y", "vx", "vy", "tile"))
for r in out:
    print("%-9s %6s %6s %5s %5s %8s %8s %7s %7s %5s"
          % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]))

print("\nninja state on the frame before death:")
for k, v in states.most_common():
    print("   state %-3s %d" % (k, v))
print("\ntile under the death point:")
for k, v in tiles.most_common():
    print("   tile %-5s %d" % (k, v))
print("\nlevels (map checksum -> how many of the 21):")
for k, v in maps.most_common(6):
    print("   %-12s %d" % (k, v))
print("\nvertical speed at death: %d of %d are at terminal (|vy|>11.9)"
      % (sum(1 for r in out if isinstance(r[8], float) and abs(r[8]) > 11.9), len(out)))
