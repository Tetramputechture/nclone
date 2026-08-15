"""Three demos die to a mine at ~(255,252). Which mine, in what state, how close?

These are winning replays, so the real ninja passed that spot alive. Either our
ninja arrives in the wrong place (upstream drift) or the mine is wrong. Three
near-identical death positions from three different demos argues for the mine.
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone")

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402

IDS = ["10138350", "10138750", "10138974"]
ATTRACT = "/home/tetra/projects/attracts"
dec = NppAttractDecoder()

for demo_id in IDS:
    path = os.path.join(ATTRACT, demo_id)
    res = dec.decode_npp_attract_file(path)
    demo = res.get("demo_data", []) or []
    mapb = list(dec.create_nclone_map(path))
    h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    h.load_map_from_map_data(mapb)
    sim = h.sim
    frames = 0
    for b in demo:
        hor, jmp = decode_input_to_controls(int(b))
        h.tick(hor, jmp)
        frames += 1
        if h.ninja_has_died() or h.ninja_has_won():
            break
    nj = sim.ninja
    print("\n=== %s  died frame %d at (%.2f, %.2f) vel (%.3f, %.3f)"
          % (demo_id, frames, nj.xpos, nj.ypos, nj.xspeed, nj.yspeed))
    # Map fingerprint: same level replayed by different players?
    print("    map fingerprint: %d bytes, sum %d" % (len(mapb), sum(mapb)))
    mines = []
    for lst in sim.entity_dic.values():
        for e in lst:
            if type(e).__name__.lower().find("mine") >= 0:
                d = ((e.xpos - nj.xpos) ** 2 + (e.ypos - nj.ypos) ** 2) ** 0.5
                mines.append((d, e))
    mines.sort(key=lambda t: t[0])
    print("    %d mines on the map; nearest 4 to the death point:" % len(mines))
    for d, e in mines[:4]:
        print("      d=%7.3f  at (%7.2f, %7.2f)  state=%s  radius=%s  active=%s"
              % (d, e.xpos, e.ypos, getattr(e, "state", "?"),
                 getattr(e, "RADII", {}).get(getattr(e, "state", None), "?")
                 if hasattr(e, "RADII") else getattr(e, "radius", "?"),
                 getattr(e, "active", "?")))
