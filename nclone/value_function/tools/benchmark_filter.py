"""Scan the official map corpus and select levels the lattice planner supports:
entities are a subset of {toggle mine, gold, exit door, exit switch} with exactly
one exit switch and one exit door.  Writes the valid map paths to a file and
reports the breakdown of why levels were excluded.

Entity_dic keys (empirically): 1=ToggleMine, 2=Gold, 3={EntityExit, EntityExitSwitch},
21=ToggleMineToggled (mine variant).  Anything else (drones 14/15/26, thwump 20,
one-way 11, locked door 6/7, death ball 25, bounce 17, launch/boost pad 10/24,
trap door 8, regular door 5, ...) makes a level unsupported.

Usage:
    .venv/bin/python -m nclone.value_function.tools.benchmark_filter \
        [official_dir] [out_file]
"""
import glob
import os
import sys
from collections import Counter

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from nclone.nplay_headless import NPlayHeadless

ALLOWED = {1, 2, 3, 21}  # mine, gold, exit(door+switch), toggled-mine


def scan(official_dir, out_file):
    maps = sorted(f for f in glob.glob(os.path.join(official_dir, "*", "*"))
                  if os.path.isfile(f))
    print(f"scanning {len(maps)} maps under {official_dir}")
    nh = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)

    valid = []
    excl_entity = Counter()   # unsupported entity-type key -> count of levels
    n_bad_goal = 0
    n_err = 0
    for p in maps:
        try:
            nh.load_map(p)
            ed = nh.sim.entity_dic
            keys = {k for k, v in ed.items() if v}
            extra = keys - ALLOWED
            if extra:
                for k in extra:
                    excl_entity[k] += 1
                continue
            ex = ed.get(3, [])
            n_door = sum(1 for e in ex if type(e).__name__ == "EntityExit")
            n_switch = sum(1 for e in ex if type(e).__name__ == "EntityExitSwitch")
            if n_door == 1 and n_switch == 1:
                valid.append(p)
            else:
                n_bad_goal += 1
        except Exception:
            n_err += 1

    with open(out_file, "w") as f:
        f.write("\n".join(valid) + "\n")

    print(f"\nVALID (planner-supported): {len(valid)} / {len(maps)} "
          f"({100.0*len(valid)/max(1,len(maps)):.1f}%)")
    print(f"excluded (bad/missing single switch+door): {n_bad_goal}")
    print(f"load errors: {n_err}")
    print("excluded by unsupported entity-type key (levels containing each):")
    name = {5: "regular_door", 6: "locked_door", 7: "locked_switch", 8: "trap_door",
            10: "launch_pad", 11: "one_way", 14: "drone_zap", 15: "drone_chaser",
            17: "bounce_block", 20: "thwump", 24: "boost_pad", 25: "death_ball",
            26: "mini_drone", 28: "shwump"}
    for k, c in excl_entity.most_common():
        print(f"  key {k:3d} ({name.get(k,'?'):14s}): {c}")
    print(f"\nwrote valid map list -> {out_file}")
    return valid


if __name__ == "__main__":
    official = sys.argv[1] if len(sys.argv) > 1 else "nclone/maps/official"
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/valid_maps.txt"
    scan(official, out)
