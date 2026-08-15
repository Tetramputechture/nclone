"""Score the fields that KILL, not the ones that move.

`entscore.py` compares entity POSITION, and position is not where hazards go
wrong. A toggle mine never moves, so it scores 240,219/240,219 no matter how
wrong its lethality is. A turret's body never moves either; what kills is its
beam. A chaingun's patrol can match the game perfectly while it fires on entirely
the wrong frames -- which is exactly where two of this session's bugs lived, and
both would have scored 100% under `entscore`.

The fields that decide lethality are in the captures already on disk, at offsets
the ports' own docstrings record. Nothing read them until now.

    hazardscore.py [pool ...] [--list]

With no pools, scores every pool in SPECS.
"""
import collections
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

D = lambda b, o: struct.unpack_from("<d", b, o)[0]
I = lambda b, o: struct.unpack_from("<i", b, o)[0]

#: pool -> (label, entity types, [(field name, offset, reader, our attribute,
#:          only-when predicate on the game's slab or None)])
#:
#: Offsets come from the ports' own docstrings, which record them against the
#: decompiled functions -- see `entity_toggle_mine.py`, `entity_drone_shooter_base.py`
#: and `entity_turret.py`.
SPECS = {
    19: ("toggle mine (types 1, 21)", (1, 21), [
        # `FUN_10417F30` kills on `FUN_1050C650(x, y, +0x28, ninja, R)` gated on
        # `+0x30 == 0`, so these two ARE the hazard.
        ("state  +0x30", 0x30, I, "state", None),
        # `+0x28` is written ONCE in the ctor (`FUN_10417EB0`, `DAT_10571CF0 =
        # 4.0`) and never changed, so it is a constant KILL radius -- comparable
        # only in state 0, the only state that kills. Our `RADIUS` doubles as the
        # toggle threshold (3.5 / 4.5) in the other two states, and comparing
        # that against a constant 4.0 marks every mine wrong while nothing is.
        ("kill radius +0x28", 0x28, D, "RADIUS", lambda b: I(b, 0x30) == 0),
    ]),
    10: ("chaingun drone (type 12)", (12,), [
        ("state  +0x7C", 0x7C, I, "state", None),
        ("timer  +0x78", 0x78, I, "timer", None),
        ("shots  +0x88", 0x88, I, "shots", None),
    ]),
    11: ("laser drone (type 13)", (13,), [
        ("state  +0x7C", 0x7C, I, "state", None),
        ("timer  +0x78", 0x78, I, "timer", None),
    ]),
    17: ("turret (type 19)", (19,), [
        ("state  +0xA4", 0xA4, I, "state", None),
    ]),
}

args = [a for a in sys.argv[1:] if not a.startswith("--")]
SHOW = "--list" in sys.argv
pools = [int(a) for a in args] if args else sorted(SPECS)

dec = NppAttractDecoder()
stems = sorted(set(
    s.replace(".layout.json", "")
    for s in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json")
    + glob.glob("/home/tetra/projects/nclone/docs/game_traces/*.layout.json")))


def score(pool):
    label, types, fields = SPECS[pool]
    need = max(off for _, off, _, _, _ in fields) + 8
    exact = collections.Counter()
    total = collections.Counter()
    firsts = []
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
        p = next((q for q in cap.pools if q["i"] == pool and q["count"]), None)
        if p is None or p["stride"] < need:
            continue
        try:
            md = list(dec.create_nclone_map(path))
            tape = dec.decode_npp_attract_file(path).get("demo_data") or []
        except Exception:
            continue

        h = NPlayHeadless(render_mode="rgb_array", enable_rendering=False)
        h.load_map_from_map_data(md)
        base = 4 + cap.layout["ninjaSlab"] + p["offset"]
        r1 = cap.frame(1)
        gpos = []
        for i in range(p["count"]):
            b = r1[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            gpos.append((D(b, 0x10), D(b, 0x18)))

        # A pool may hold several types and a map may carry only some of them, so
        # several groupings can match the count. Choose by POSITION agreement,
        # not by count: on `10160699` the game's pool holds one mine (the type-21
        # at (564,228)) while the map also has a type-1 at (552,216), and picking
        # the first count-matching grouping selected the wrong mine entirely --
        # then reported its state as a frame-1 divergence.
        cands = [[e for t in types for e in h.sim.entity_dic.get(t, [])]]
        cands += [h.sim.entity_dic.get(t, []) for t in types]
        best_c, best_err = None, None
        for c in cands:
            if len(c) != p["count"]:
                continue
            pend, err = list(c), 0.0
            for gx, gy in gpos:
                e = min(pend, key=lambda e: (e.xpos - gx) ** 2 + (e.ypos - gy) ** 2)
                err += (e.xpos - gx) ** 2 + (e.ypos - gy) ** 2
                pend.remove(e)
            if best_err is None or err < best_err:
                best_c, best_err = c, err
        pool_ents = best_c
        if not pool_ents:
            continue
        # Align OUR entities to the game's slab entries by NEAREST POSITION, not
        # by index. Pool 19 holds both type 1 and type 21 and the game's order
        # within it is not ours, so index alignment silently compares one entity
        # against another: on `10160699`, which has exactly one of each, it
        # reported a frame-1 state mismatch that was entirely this artifact.
        #
        # Nearest rather than exact, because the game's frame-1 slab is one tick
        # in while our entities are still at spawn -- a mover has already left
        # its spawn point by then, and exact matching scored ZERO drones.
        pending = list(pool_ents)
        ours = []
        for gx, gy in gpos:
            best = min(pending,
                       key=lambda e: (e.xpos - gx) ** 2 + (e.ypos - gy) ** 2,
                       default=None)
            if best is None:
                ours = None
                break
            ours.append(best)
            pending.remove(best)
        if ours is None:
            continue
        seen.add(demo)
        first = {}
        n = min(cap.n, len(tape))
        for f in range(1, n + 1):
            hor, jmp = decode_input_to_controls(int(tape[f - 1]))
            h.tick(hor, jmp)
            row = cap.frame(f)
            for i, e in enumerate(ours):
                b = row[base + i * p["stride"]: base + (i + 1) * p["stride"]]
                for name, off, rd, attr, only in fields:
                    if only is not None and not only(b):
                        continue
                    got = rd(b, off)
                    mine = getattr(e, attr, None)
                    if mine is None:
                        continue
                    if isinstance(got, int):
                        mine = int(mine)
                    total[name] += 1
                    if mine == got:
                        exact[name] += 1
                    elif (name, i) not in first:
                        first[(name, i)] = (f, mine, got)
        if first:
            firsts.append((demo, first))

    print("\n=== pool %d: %s ===" % (pool, label))
    print("   demos scored: %d" % len(seen))
    for name, _, _, _, _ in fields:
        t = total[name]
        if not t:
            print("   %-20s no comparable frames" % name)
            continue
        print("   %-20s %d / %d = %.2f%%" % (name, exact[name], t, 100.0 * exact[name] / t))
    print("   demos with any mismatch: %d" % len(firsts))
    if SHOW:
        for demo, first in firsts[:8]:
            print("   %s" % demo)
            for (name, i), (f, mine, got) in sorted(first.items())[:6]:
                print("      %-20s ent %-3d f=%-6d ours=%-8s game=%s" % (name, i, f, mine, got))


for pool in pools:
    if pool in SPECS:
        score(pool)
