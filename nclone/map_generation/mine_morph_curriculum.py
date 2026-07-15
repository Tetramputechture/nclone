"""Geometry-morphing curriculum generator (see ``docs/GEOMETRY_MORPHING_CURRICULUM.md``).

Turns a single mine-saturated level into a ladder of progressively-tighter rungs by
carving the threadable channel (``mine_morph.morph_mines``). Each rung is written as
a faithful raw map file the trainer loads via ``--custom-map-path``; a ``manifest.json``
records the schedule + per-rung stats so a driver can walk wide -> tight, carrying the
policy forward (``--init-from`` the previous rung's checkpoint).

The terminal rung (``delta=0``) is byte-identical to the real level.

Subcommands::

    # generate the stage maps + manifest
    python -m nclone.map_generation.mine_morph_curriculum generate \
        --map nclone/test_maps/mines-extremely-difficult \
        --deltas 48 36 24 18 12 6 0 --out-dir datasets/mine_morph_curriculum

    # bracket experiment: find the smallest delta a short RRT can solve
    # (the go/no-go for the whole plan — §6 of the design doc)
    python -m nclone.map_generation.mine_morph_curriculum bracket \
        --map nclone/test_maps/mines-extremely-difficult \
        --deltas 0 6 12 24 48 --max-iter 300000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

from .mine_morph import (
    morph_mines, route_centerline, min_clearance_along, iter_mines, spawn_world,
)


def _load_base_map(map_path: str) -> List[int]:
    """Raw map-file bytes (one int per byte), exactly as the trainer's
    ``NPlayHeadless.load_map`` reads them. Reading the file directly (rather than
    round-tripping through the sim) keeps the δ=0 rung *byte-identical* to the
    real level — the sim canonicalizes a few header bytes (e.g. ninja orientation)
    that we want to preserve verbatim."""
    with open(map_path, "rb") as f:
        return [int(b) for b in f.read()]


def _stage_stats(map_data: Sequence[int], centerline) -> dict:
    """Per-rung verification stats: channel clearance, deadly-mine count, spawn D0."""
    from ..value_function.mine_threading_distance import MineThreadingDistanceSource
    src = MineThreadingDistanceSource(list(map_data))
    sp = spawn_world(map_data)
    D0 = (src.cost_to_completion(sp[0], sp[1], False) if src.available else float("inf"))
    return dict(
        clearance_px=round(min_clearance_along(map_data, centerline), 2),
        deadly_mines=sum(1 for _ in iter_mines(map_data, deadly_only=True)),
        spawn_D0=round(float(D0), 1),
        source_available=bool(src.available),
    )


def probe_solvable(map_data: Sequence[int], *, max_iter: int, kext: int,
                   contact_px: float, seed: int, log=lambda m: None) -> dict:
    """Short two-phase RRT solvability probe on the morphed map (the on-ramp proof).

    Returns dict(solvable, frames, phaseA_reached, phaseB_reached, replay_actions).
    A win means the rung is discoverable by search and yields a free BC demo.
    """
    import random
    import nclone_native
    from ..value_function.mine_threading_distance import MineThreadingDistanceSource
    from ..value_function.tools.mine_threading_rrt import (
        rrt_to_goal, validate_win, _goal_xy,
    )

    mt = MineThreadingDistanceSource(list(map_data))
    if not mt.available:
        return dict(solvable=False, frames=0, phaseA_reached=False,
                    phaseB_reached=False, replay_actions=None, note="source unavailable")

    sim = nclone_native.Simulator.load(list(map_data))
    st0 = sim.get_state()
    switch_xy = _goal_xy(st0, 4)
    exit_xy = _goal_xy(st0, 3)
    if switch_xy is None or exit_xy is None:
        return dict(solvable=False, frames=0, phaseA_reached=False,
                    phaseB_reached=False, replay_actions=None, note="no switch/exit")

    sim.fast_reset()
    spawn = tuple(sim.get_ninja_pos())
    spawn_state = sim.save_ninja_state()
    rng = random.Random(seed)
    ra = rrt_to_goal(sim, mt, spawn_state, spawn, switch_xy, is_exit=False,
                     max_iter=max_iter, kext=kext, contact_px=contact_px, rng=rng, log=log)
    if not ra["reached"]:
        return dict(solvable=False, frames=len(ra["actions"]), phaseA_reached=False,
                    phaseB_reached=False, replay_actions=None)
    sim.restore_ninja_state(ra["final_state"])
    sw_xy = tuple(sim.get_ninja_pos())
    rb = rrt_to_goal(sim, mt, sim.save_ninja_state(), sw_xy, exit_xy, is_exit=True,
                     max_iter=max_iter, kext=kext, contact_px=contact_px,
                     rng=random.Random(seed + 500), log=log)
    stitched = ra["actions"] + rb["actions"]
    won, fwin, _ = validate_win(map_data, stitched)
    return dict(solvable=bool(won), frames=int(fwin), phaseA_reached=True,
                phaseB_reached=bool(rb["reached"]),
                replay_actions=(stitched[:fwin] if won else None))


def generate_stages(
    map_path: str,
    deltas: Sequence[float], *,
    out_dir: str,
    mode: str = "carve",
    bracket: bool = False,
    bracket_max_iter: int = 300000,
    bracket_kext: int = 3,
    bracket_contact_px: float = 10.0,
    bracket_seed: int = 0,
    write_replays: bool = True,
) -> dict:
    """Write one map file per rung + a manifest. Rungs are sorted wide -> tight so
    the driver trains easiest first and the last rung is the real level (delta=0).

    If ``bracket`` is set, runs a short RRT solvability probe per rung and records
    it (and, on a win, writes the demo .replay for BC warm-start)."""
    base_md = _load_base_map(map_path)
    name = Path(map_path).stem
    centerline = route_centerline(base_md)
    base_clear = min_clearance_along(base_md, centerline)

    os.makedirs(out_dir, exist_ok=True)
    replay_dir = os.path.join(out_dir, "demos")
    if bracket and write_replays:
        os.makedirs(replay_dir, exist_ok=True)

    # wide -> tight (descending delta); dedup and ensure 0 is the terminal rung.
    rungs = sorted({float(d) for d in deltas}, reverse=True)
    stages = []
    for idx, delta in enumerate(rungs):
        md = morph_mines(base_md, delta, mode=mode, centerline=centerline)
        if delta == 0:
            assert md == base_md, "delta=0 rung must be byte-identical to the real level"
        stem = f"stage_{idx:02d}_{name}_d{int(delta)}"
        map_file = os.path.join(out_dir, stem)
        with open(map_file, "wb") as f:
            f.write(bytes(min(max(int(b), 0), 255) for b in md))
        stats = _stage_stats(md, centerline)
        entry = dict(idx=idx, delta_px=float(delta), map=os.path.abspath(map_file),
                     map_name=stem, **stats)

        if bracket:
            t0 = time.time()
            pr = probe_solvable(md, max_iter=bracket_max_iter, kext=bracket_kext,
                                contact_px=bracket_contact_px, seed=bracket_seed)
            entry["solvable"] = pr["solvable"]
            entry["probe_frames"] = pr["frames"]
            entry["probe_phaseA"] = pr["phaseA_reached"]
            entry["probe_secs"] = round(time.time() - t0, 1)
            if pr["solvable"] and write_replays and pr["replay_actions"]:
                from ..tas_tool.movie import TASMovie
                rp = os.path.join(replay_dir, f"{stem}.replay")
                TASMovie(md, pr["replay_actions"], level_id=stem).save(rp, success=True)
                entry["replay"] = os.path.abspath(rp)
        stages.append(entry)

    manifest = dict(
        base_map=os.path.abspath(map_path),
        base_map_name=name,
        mode=mode,
        base_clearance_px=round(base_clear, 2),
        centerline_pts=len(centerline),
        bracketed=bool(bracket),
        stages=stages,
    )
    mpath = os.path.join(out_dir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    manifest["_manifest_path"] = mpath
    return manifest


def _print_manifest(m: dict) -> None:
    print(f"\n{m['base_map_name']}  mode={m['mode']}  base_clearance="
          f"{m['base_clearance_px']}px  centerline={m['centerline_pts']}pts")
    print(f"{'idx':>3} {'δpx':>5} {'clear':>7} {'mines':>7} {'D0':>8}  "
          f"{'solvable' if m['bracketed'] else 'map'}")
    for s in m["stages"]:
        tail = (f"{str(s.get('solvable')):>8} ({s.get('probe_frames','?')}f, "
                f"{s.get('probe_secs','?')}s)" if m["bracketed"] else s["map_name"])
        print(f"{s['idx']:>3} {int(s['delta_px']):>5} {s['clearance_px']:>7.1f} "
              f"{s['deadly_mines']:>7} {s['spawn_D0']:>8.0f}  {tail}")
    if m["bracketed"]:
        winners = [s for s in m["stages"] if s.get("solvable")]
        if winners:
            tight = min(winners, key=lambda s: s["delta_px"])
            print(f"\n→ smallest RRT-solvable rung: δ={int(tight['delta_px'])}px "
                  f"(the curriculum's bottom-up handoff to RL)")
        else:
            print("\n→ NO rung solved by the short RRT — widen δ_max / increase "
                  "--max-iter, or this geometry needs a demo (see kill criteria §5).")
    print(f"\nmanifest: {m.get('_manifest_path')}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write stage maps + manifest")
    g.add_argument("--map", default="nclone/test_maps/mines-extremely-difficult")
    g.add_argument("--deltas", type=float, nargs="+",
                   default=[48, 36, 24, 18, 12, 6, 0])
    g.add_argument("--mode", choices=("carve", "displace"), default="carve")
    g.add_argument("--out-dir", default="datasets/mine_morph_curriculum")
    g.add_argument("--bracket", action="store_true",
                   help="also run a short RRT solvability probe per rung")
    g.add_argument("--max-iter", type=int, default=300000)
    g.add_argument("--kext", type=int, default=3)
    g.add_argument("--contact-px", type=float, default=10.0)
    g.add_argument("--seed", type=int, default=0)

    b = sub.add_parser("bracket", help="solvability bracket experiment only (no maps written... "
                                       "writes maps to a temp out-dir)")
    b.add_argument("--map", default="nclone/test_maps/mines-extremely-difficult")
    b.add_argument("--deltas", type=float, nargs="+", default=[0, 6, 12, 24, 48])
    b.add_argument("--mode", choices=("carve", "displace"), default="carve")
    b.add_argument("--out-dir", default="datasets/mine_morph_curriculum")
    b.add_argument("--max-iter", type=int, default=300000)
    b.add_argument("--kext", type=int, default=3)
    b.add_argument("--contact-px", type=float, default=10.0)
    b.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    m = generate_stages(
        args.map, args.deltas, out_dir=args.out_dir, mode=args.mode,
        bracket=(args.cmd == "bracket" or getattr(args, "bracket", False)),
        bracket_max_iter=args.max_iter, bracket_kext=args.kext,
        bracket_contact_px=args.contact_px, bracket_seed=args.seed,
    )
    _print_manifest(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
