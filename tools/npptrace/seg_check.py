"""Print the raw half-edge states and the emitted segments around one point.

A merged face must span exactly the two half-cells that carry the same non-zero
state; anything longer is the join reaching past its evidence.
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402
from nclone.physics import gather_segments_from_region  # noqa: E402

demo = sys.argv[1]
px, py = float(sys.argv[2]), float(sys.argv[3])
dec = NppAttractDecoder()
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(
    list(dec.create_nclone_map("/home/tetra/projects/attracts/" + demo)))
sim = h.sim

cx, cy = int(px // 24), int(py // 24)
print("tiles rows %d..%d, cols %d..%d:" % (cy - 1, cy + 1, cx - 1, cx + 1))
for yy in range(cy - 1, cy + 2):
    print("   " + " ".join("%3d" % sim.tile_dic.get((xx, yy), 0)
                           for xx in range(cx - 1, cx + 2)))

print("\nhorizontal segments within 30px:")
for s in gather_segments_from_region(sim, px - 30, py - 30, px + 30, py + 30):
    if getattr(s, "px", None) is None:
        continue
    if s.py != 0:
        continue
    print("   (%7.1f,%7.1f) -> (%7.1f,%7.1f)   px=%+g  length=%g"
          % (s.x1, s.y1, s.x2, s.y2, s.px, abs(s.px)))
print("\nvertical segments within 30px:")
for s in gather_segments_from_region(sim, px - 30, py - 30, px + 30, py + 30):
    if getattr(s, "px", None) is None or s.px != 0:
        continue
    print("   (%7.1f,%7.1f) -> (%7.1f,%7.1f)   py=%+g  length=%g"
          % (s.x1, s.y1, s.x2, s.y2, s.py, abs(s.py)))
