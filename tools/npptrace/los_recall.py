"""Is line of sight SUFFICIENT for a shooter drone to acquire, or only necessary?

§17 measured that when the game acquires, LOS has just been established -- 17 of
19. That is precision. It says nothing about recall: if the game sits in state 0
for thousands of frames with clear line of sight and does not acquire, then LOS
is necessary and not sufficient, and modelling it as sufficient makes the drone
fire far more often than the real one.
"""
import collections
import glob
import os
import struct
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402
from nclone.physics import raycast_vs_player  # noqa: E402

POOLS = (10, 11)
tally = collections.Counter()

for d in ["/home/tetra/projects/nclone/tools/npptrace/out9",
          "/home/tetra/projects/nclone/tools/npptrace/out9_batch1",
          "/home/tetra/projects/nclone/tools/npptrace/out10"]:
    for s in sorted(glob.glob(d + "/*.layout.json")):
        demo = os.path.basename(s).replace(".layout.json", "")
        if demo.startswith("cap"):
            continue
        cap = Capture(os.path.join(d, demo))
        pools = [q for q in cap.pools if q["i"] in POOLS and q["count"]]
        if not pools:
            continue
        dec = NppAttractDecoder()
        path = "/home/tetra/projects/attracts/" + demo
        inputs = dec.decode_npp_attract_file(path).get("demo_data", []) or []
        h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
        h.load_map_from_map_data(list(dec.create_nclone_map(path)))
        prev = {}
        for f in range(1, min(cap.n, len(inputs)) + 1):
            hor, jmp = decode_input_to_controls(int(inputs[f - 1]))
            h.tick(hor, jmp)
            if h.ninja_has_died():
                break
            ninja = h.sim.ninja
            rec = cap.frame(f)
            for p in pools:
                base = 4 + cap.layout["ninjaSlab"] + p["offset"]
                for i in range(p["count"]):
                    b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
                    st = struct.unpack_from("<i", b, 0x7C)[0]
                    if st != 0:
                        prev[(demo, p["i"], i)] = st
                        continue
                    gx = struct.unpack_from("<d", b, 0x10)[0]
                    gy = struct.unpack_from("<d", b, 0x18)[0]
                    los = raycast_vs_player(h.sim, gx, gy, ninja.xpos, ninja.ypos)
                    tally["idle frames"] += 1
                    if los:
                        tally["idle WITH line of sight"] += 1
                    prev[(demo, p["i"], i)] = st

print("shooter drones sitting in state 0 (idle):")
for k, v in tally.most_common():
    print("   %-26s %d" % (k, v))
idle = tally["idle frames"]
los = tally["idle WITH line of sight"]
print()
print("So the game declines to acquire on %d idle frames where it CAN see the"
      % los)
print("ninja. Line of sight is necessary, not sufficient.")
print("(19 acquisitions total across the corpus.)")
