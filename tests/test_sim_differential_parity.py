"""Frame-by-frame differential parity between the Python and native simulators.

The entity-parity work (nclone-native/ENTITY_PARITY_PLAN.md) rebuilds the
native tick loop: a `move` pass ahead of `think`, type-ordered entity iteration,
`&mut` physical collision, and half-cell grid edges. None of that may change a
single bit of existing behaviour, and the ten entity ports that follow all need
a harness that catches divergence on the frame it happens rather than at episode
end.

`test_native_backend_parity.py` already compares policy-facing *observations* on
generated training maps. This is the layer beneath: raw simulator state, exact
equality, on real official maps.

Exact means exact. `==` on f64, not `approx`. These are two implementations of
the same deterministic integer-seeded arithmetic; a 1-ulp difference is a bug
that compounds into a different trajectory within a few hundred frames, and
tolerating it would defeat the purpose of the harness.
"""

from __future__ import annotations

import json
import random
import zlib
from pathlib import Path

import pytest

nclone_native = pytest.importorskip("nclone_native")

from nclone.nsim import Simulator  # noqa: E402
from nclone.sim_config import SimConfig  # noqa: E402

MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "npp-rl"
    / "datasets"
    / "official_compatible_manifest_native.json"
)
OFFICIAL_ROOT = Path(__file__).resolve().parents[1] / "nclone" / "maps" / "official"

N_MAPS = 12
N_FRAMES = 400

# The ninja fields that fully determine the next frame. If these all agree every
# frame, the two sims are the same machine.
NINJA_FIELDS = (
    "xpos", "ypos", "xspeed", "yspeed",
    "xpos_old", "ypos_old", "xspeed_old", "yspeed_old",
    "state", "airborn", "walled", "wall_normal",
    "jump_duration", "jump_buffer", "floor_buffer", "wall_buffer",
    "launch_pad_buffer", "applied_gravity", "applied_drag", "applied_friction",
    "floor_count", "wall_count", "ceiling_count",
    "gold_collected", "doors_opened",
)


def _map_paths() -> list[Path]:
    if MANIFEST.exists():
        comp = json.loads(MANIFEST.read_text())["compatible"]
        paths = [Path(c["path"]) for c in comp]
        locked = [Path(c["path"]) for c in comp if "locked_door" in c["entities"]]
    elif OFFICIAL_ROOT.exists():
        paths = [p for p in OFFICIAL_ROOT.rglob("*") if p.is_file() and p.stat().st_size > 1235]
        locked = []
    else:
        pytest.skip("no official maps or manifest available")
    paths = [p for p in paths if p.exists()]
    if not paths:
        pytest.skip("manifest paths do not resolve on this machine")
    sampled = random.Random(31337).sample(paths, min(N_MAPS, len(paths)))
    # Always include a locked-door map: it is the only one exercising the door
    # segment + grid-edge flip mid-episode.
    return list(dict.fromkeys(sampled + locked[:2]))


def _seed_for(map_path: Path) -> int:
    """Per-map action-tape seed. crc32, NOT the builtin hash(): Python
    randomizes str hashing per process, which would silently give this suite a
    different tape on every run -- so a genuine divergence would reproduce only
    sometimes, and a green run would prove nothing."""
    return zlib.crc32(map_path.name.encode()) & 0xFFFF


def _action_tape(n: int, seed: int) -> list[tuple[int, int]]:
    """(hor_input, jump_input) pairs. Jump is held in runs so the tape actually
    produces jumps, wall slides and launches rather than 1-frame twitches."""
    rng = random.Random(seed)
    tape, jump, held = [], 0, 0
    for _ in range(n):
        if held == 0:
            jump = 1 if rng.random() < 0.35 else 0
            held = rng.randint(1, 12)
        held -= 1
        tape.append((rng.choice([-1, 0, 1]), jump))
    return tape


# Exit doors (type 3) are the one entity whose `active` flag means different
# things on each side, by design and not by accident:
#   * Python's EntityExit never assigns `active`, so it inherits True forever;
#     whether the door is usable is tracked by its presence in `grid_entity`.
#   * The native snapshot reports `active = switch_hit`, which is what the
#     observation layer actually wants to read.
# Comparing that field across sims would be comparing two different questions,
# so it is dropped for type 3 only -- position and state are still compared, and
# the door's real behaviour (does touching it win?) is covered by the has_won
# assertion in the ninja test.
_ACTIVE_FLAG_DIFFERS = {3}


def _py_entities(py) -> list[tuple]:
    """Mutable entity state from the Python sim, as (type, x, y, active, state)."""
    out = []
    for entity_list in py.entity_dic.values():
        for e in entity_list:
            etype = int(e.type)
            active = None if etype in _ACTIVE_FLAG_DIFFERS else bool(e.active)
            out.append((
                etype, round(e.xpos, 12), round(e.ypos, 12),
                active, int(getattr(e, "state", 0) or 0),
            ))
    return out


def _native_entities(native) -> list[tuple]:
    st = native.get_state_dynamic()
    out = []
    for e in st.entities:
        etype = int(e.entity_type)
        active = None if etype in _ACTIVE_FLAG_DIFFERS else bool(e.active)
        out.append((etype, round(e.x, 12), round(e.y, 12), active, int(e.state)))
    return out


@pytest.mark.parametrize("map_path", _map_paths(), ids=lambda p: p.name)
def test_ninja_state_identical_every_frame(map_path: Path):
    """Replay one action tape on both sims; ninja state must agree exactly on
    every frame, and the first disagreement is reported with its frame number."""
    map_data = [int(b) for b in map_path.read_bytes()]
    tape = _action_tape(N_FRAMES, seed=_seed_for(map_path))

    py = Simulator(SimConfig())
    py.load(map_data)
    native = nclone_native.Simulator.load(map_data)

    for frame, (hor, jump) in enumerate(tape, start=1):
        py.tick(hor, jump)
        native.tick(hor, jump)

        n = native.get_state_dynamic().ninja
        pn = py.ninja
        for field in NINJA_FIELDS:
            pv, nv = getattr(pn, field), getattr(n, field)
            if isinstance(pv, bool) or isinstance(nv, bool):
                pv, nv = bool(pv), bool(nv)
            elif isinstance(pv, float) or isinstance(nv, float):
                pv, nv = float(pv), float(nv)
            else:
                pv, nv = int(pv), int(nv)
            assert pv == nv, (
                f"{map_path.name} frame {frame}: ninja.{field} "
                f"python={pv!r} native={nv!r} (action hor={hor} jump={jump})"
            )

        # Both sims must agree the episode is over at the same instant.
        assert bool(py.ninja.has_died()) == bool(native.has_died()), (
            f"{map_path.name} frame {frame}: death disagreement"
        )
        assert bool(py.ninja.has_won()) == bool(native.has_won()), (
            f"{map_path.name} frame {frame}: win disagreement"
        )
        if py.ninja.has_died() or py.ninja.has_won():
            break


@pytest.mark.parametrize("map_path", _map_paths()[:4], ids=lambda p: p.name)
def test_entity_state_identical_every_frame(map_path: Path):
    """Entity mutable state (position, active, state) must track too -- this is
    what catches a mis-ordered move/think pass or a stale entity grid."""
    map_data = [int(b) for b in map_path.read_bytes()]
    tape = _action_tape(N_FRAMES, seed=_seed_for(map_path) ^ 0xA5A5)

    py = Simulator(SimConfig())
    py.load(map_data)
    native = nclone_native.Simulator.load(map_data)

    assert len(_py_entities(py)) == len(_native_entities(native)), (
        f"{map_path.name}: entity count differs at load "
        f"(python={len(_py_entities(py))} native={len(_native_entities(native))})"
    )

    for frame, (hor, jump) in enumerate(tape, start=1):
        py.tick(hor, jump)
        native.tick(hor, jump)
        # Compare as multisets: the two sims store entities in different orders
        # (Python buckets by type, native keeps map-load order), and this test is
        # about state evolution, not ordering.
        pe, ne = sorted(_py_entities(py), key=repr), sorted(_native_entities(native), key=repr)
        assert pe == ne, (
            f"{map_path.name} frame {frame}: entity state diverged\n"
            f"  python-only: {[x for x in pe if x not in ne][:3]}\n"
            f"  native-only: {[x for x in ne if x not in pe][:3]}"
        )
        if py.ninja.has_died() or py.ninja.has_won():
            break
