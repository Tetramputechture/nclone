#!/usr/bin/env python3
"""Differential check: Python sim vs native sim, frame by frame, on official maps
containing a given entity type.

Usage: python scripts/diffcheck_entity_parity.py <entity_name|type_id> [n_maps] [n_frames]
       python scripts/diffcheck_entity_parity.py all [n_maps] [n_frames]

Reports the first divergence per map (field, frame, both values).

Both sims now implement every official entity type, so there is no longer a
"supported subset" to restrict the sample to -- any official map is a valid
comparison. Maps are found by scanning the corpus for the type's byte in the
entity stream, so no external manifest is needed.

**Known pre-existing divergence.** Native and Python drift by exactly 1 ulp in
`ninja.xspeed` on roughly 6% of maps; it reproduces on maps with no entities at
all beyond gold, so it predates the entity work (see
`nclone-native/ENTITY_PARITY_PLAN.md` §3b). Compare deltas, not absolutes.

**This check is necessary but not sufficient.** It samples official maps with
SYNTHETIC input, and it reported `15/15 maps exact` while the two engines
disagreed on frame 1 of the attract demo `10132650`
(`docs/ENTITY_RE_ROUND8_FINDINGS.md` §6.1). Real inputs on real trajectories are
strictly stronger, so run `scripts/verify_game_traces.py` alongside this one --
that is the guard that catches what this misses.
"""
import random
import sys
import zlib
from pathlib import Path

import nclone_native
from nclone.constants.entity_types import EntityType
from nclone.nsim import Simulator
from nclone.sim_config import SimConfig

OFFICIAL_ROOT = Path(__file__).resolve().parents[1] / "nclone" / "maps" / "official"

#: The entity section starts at 1230 with a 5-byte stride, matching
#: `map/loader.rs:load_entities`.
ENTITY_SECTION_START = 1230
ENTITY_STRIDE = 5

NINJA_FIELDS = (
    "xpos", "ypos", "xspeed", "yspeed", "xpos_old", "ypos_old",
    "xspeed_old", "yspeed_old", "state", "airborn", "walled", "wall_normal",
    "jump_duration", "jump_buffer", "floor_buffer", "wall_buffer",
    "launch_pad_buffer", "applied_gravity", "applied_drag", "applied_friction",
    "floor_count", "wall_count", "ceiling_count", "gold_collected", "doors_opened",
)

#: Exit doors are the one entity whose `active` means different things per sim by
#: design: Python's EntityExit never assigns it, native reports `switch_hit`.
#: Comparing it would compare two different questions.
_ACTIVE_FLAG_DIFFERS = {EntityType.EXIT_DOOR}

_DRONES = {
    EntityType.DRONE_CHAINGUN, EntityType.DRONE_LASER, EntityType.DRONE_ZAP,
    EntityType.DRONE_CHASER, EntityType.MINI_DRONE,
}


def project_py(e):
    """One Python entity as `(type, x, y, active, state)` the way the native
    `EntitySnapshot` reports it.

    The snapshot carries a single `state` byte per entity and what it holds is
    per-type (see `entities/mod.rs`), so the Python side has to be projected the
    same way. Comparing `e.state` naively compares two different questions and
    reports divergences that are not there.
    """
    t = int(e.type)
    x, y, active, state = float(e.xpos), float(e.ypos), bool(e.active), 0

    if t in _DRONES:
        state = int(e.dir)                          # facing, 0..3
    elif t == EntityType.FLOOR_GUARD:
        state = int(e.dir) % 3                      # signed -1/0/+1 folded to 0..2
    elif t == EntityType.THWUMP:
        state = int(e.state) % 3                    # signed, same folding
    elif t == EntityType.EYE_BAT:
        state = int(bool(e.chasing))
    elif t == EntityType.BOOST_PAD:
        state = int(bool(e.is_touching_ninja))      # one-shot latch
    elif t in (EntityType.ROCKET, EntityType.PLAYER_CONTROLLED_ROCKET):
        # The snapshot reports the PROJECTILE, not the launcher: the launcher
        # never moves, and the projectile is what the ninja can be killed by.
        x, y = float(e.projectile.xpos), float(e.projectile.ypos)
        state = int(e.state)
        if t == EntityType.ROCKET:
            active = e.state == e.FLYING
    elif t == EntityType.DUAL_LASER:
        state = int(e.mode)                         # spinner (0) vs surface (1)
    elif t == EntityType.EVIL_NINJA:
        state = int(e.state)
        active = e.state == e.PLAYBACK
    elif t in (EntityType.BOUNCE_BLOCK, EntityType.DEATH_BALL):
        active = True
    elif t in _ACTIVE_FLAG_DIFFERS:
        active = None

    if t not in _DRONES and t not in (
        EntityType.FLOOR_GUARD, EntityType.THWUMP, EntityType.EYE_BAT,
        EntityType.BOOST_PAD, EntityType.ROCKET,
        EntityType.PLAYER_CONTROLLED_ROCKET, EntityType.DUAL_LASER,
        EntityType.EVIL_NINJA,
    ):
        state = int(getattr(e, "state", 0) or 0)

    return (t, round(x, 12), round(y, 12), active, state)


def py_entities(py):
    return sorted(
        (project_py(e) for lst in py.entity_dic.values() for e in lst), key=repr
    )


def nat_entities(nat):
    return sorted(
        (
            (
                int(e.entity_type),
                round(float(e.x), 12),
                round(float(e.y), 12),
                None if int(e.entity_type) in _ACTIVE_FLAG_DIFFERS else bool(e.active),
                int(e.state),
            )
            for e in nat.get_state_dynamic().entities
        ),
        key=repr,
    )


def action_tape(n, seed):
    """(hor, jump) pairs. Jump is held in runs so the tape produces real jumps,
    wall slides and launches rather than one-frame twitches."""
    rng = random.Random(seed)
    tape, jump, held = [], 0, 0
    for _ in range(n):
        if held == 0:
            jump = 1 if rng.random() < 0.35 else 0
            held = rng.randint(1, 12)
        held -= 1
        tape.append((rng.choice([-1, 0, 1]), jump))
    return tape


def map_entity_types(map_data):
    types = set()
    i = ENTITY_SECTION_START
    while i + 4 < len(map_data):
        t = int(map_data[i])
        if t != 0:
            types.add(t)
        i += ENTITY_STRIDE
    return types


def maps_containing(type_id):
    """Official maps whose entity stream carries `type_id`."""
    hits = []
    for p in sorted(OFFICIAL_ROOT.rglob("*")):
        if not p.is_file() or p.stat().st_size <= 1235:
            continue
        if type_id in map_entity_types(p.read_bytes()):
            hits.append(p)
    return hits


def run_one(path, n_frames):
    """Replay one action tape on both sims. Returns (kind, detail); kind is None
    when the two agreed on every frame."""
    md = [int(b) for b in path.read_bytes()]
    py = Simulator(SimConfig())
    py.load(md)
    nat = nclone_native.Simulator.load(md)

    if len(py_entities(py)) != len(nat_entities(nat)):
        return "load", (f"entity count py={len(py_entities(py))} "
                        f"native={len(nat_entities(nat))}")

    tape = action_tape(n_frames, zlib.crc32(path.name.encode()) & 0xFFFF)
    for frame, (hor, jump) in enumerate(tape, start=1):
        py.tick(hor, jump)
        nat.tick(hor, jump)

        pe, ne = py_entities(py), nat_entities(nat)
        if pe != ne:
            po = [x for x in pe if x not in ne][:2]
            no = [x for x in ne if x not in pe][:2]
            return "entity", f"frame {frame}: py-only={po} native-only={no}"

        n = nat.get_state_dynamic().ninja
        for f in NINJA_FIELDS:
            pv, nv = getattr(py.ninja, f), getattr(n, f)
            if isinstance(pv, bool) or isinstance(nv, bool):
                pv, nv = bool(pv), bool(nv)
            elif isinstance(pv, float) or isinstance(nv, float):
                pv, nv = float(pv), float(nv)
            else:
                pv, nv = int(pv), int(nv)
            if pv != nv:
                return f"ninja.{f}", f"frame {frame}: py={pv!r} native={nv!r}"

        if bool(py.ninja.has_died()) != bool(nat.has_died()):
            return "death", f"frame {frame}: sims disagree the ninja died"
        if bool(py.ninja.has_won()) != bool(nat.has_won()):
            return "win", f"frame {frame}: sims disagree the ninja won"
        if py.ninja.has_died() or py.ninja.has_won():
            break
    return None, None


def check_type(type_id, name, n_maps, n_frames):
    hits = maps_containing(type_id)
    if not hits:
        print(f"{name} ({type_id}): no official map contains it -- skipped")
        return 0
    sample = random.Random(4242).sample(hits, min(n_maps, len(hits)))

    kinds = {}
    ok = 0
    for p in sample:
        try:
            kind, detail = run_one(p, n_frames)
        except Exception as exc:                     # noqa: BLE001 - report, don't abort the sweep
            kind, detail = f"exc:{type(exc).__name__}", str(exc)[:200]
        if kind is None:
            ok += 1
        else:
            kinds.setdefault(kind, []).append((p.name, detail))

    print(f"{name} ({type_id}): {ok}/{len(sample)} maps exact "
          f"({len(hits)} official maps contain it)")
    for kind, items in sorted(kinds.items()):
        print(f"  {kind}: {len(items)}")
        for map_name, detail in items[:3]:
            print(f"      {map_name}: {detail}")
    return len(sample) - ok


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    target = sys.argv[1]
    n_maps = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    n_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 400

    by_name = {
        k.lower(): v for k, v in vars(EntityType).items()
        if not k.startswith("_") and isinstance(v, int)
    }

    if target == "all":
        targets = [(v, k) for k, v in sorted(by_name.items(), key=lambda kv: kv[1])
                   if v not in (0, EntityType.LOCKED_DOOR_SWITCH,
                                EntityType.TRAP_DOOR_SWITCH)]
    elif target.isdigit():
        tid = int(target)
        name = next((k for k, v in by_name.items() if v == tid), f"type{tid}")
        targets = [(tid, name)]
    elif target.lower() in by_name:
        targets = [(by_name[target.lower()], target.lower())]
    else:
        print(f"unknown entity: {target}\nknown: {', '.join(sorted(by_name))}")
        return 2

    bad = 0
    for tid, name in targets:
        bad += check_type(tid, name, n_maps, n_frames)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
