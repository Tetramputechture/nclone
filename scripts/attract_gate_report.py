#!/usr/bin/env python3
"""Re-simulate attract demos and report which entity types gate the failures.

This is the validation oracle for the entity RE work (`docs/ENTITY_INTEGRATION_SPEC.md`
§11). `convert_attracts_to_replays.py` answers "how many demos win"; this answers
"and which unimplemented entity is standing in the way", which is what tells you
whether a freshly-ported entity actually did anything.

Emits one JSON line per attract file:

    {"f": "1000054", "types": [1, 2, 3, 4], "outcome": "win", "frames": 812}

`outcome` is one of win / die / timeout (tape ran out with no win) / no_demo / err:*.
Summarise an existing report without re-simulating via ``--summarize FILE``.

Usage:
  python scripts/attract_gate_report.py --in-dir /path/to/attracts \
      --out /tmp/gate.jsonl --limit 8000 --workers 10
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

# Entity types the ninja can never see because they are not real entities: the
# switch halves of locked (7) and trap (9) doors, which the door entity owns.
from nclone.constants.entity_types import PHANTOM_ENTITY_TYPES as PHANTOM_TYPES

#: Frames to keep simulating with no input after an attract's tape is exhausted.
#: The game does not stop when the inputs do -- see the coast block in `simulate`.
#: 60 (one second) against an observed maximum of 25, so it is a bound rather
#: than a fitted number.
COAST_FRAMES = 60


def map_entity_types(map_data: list[int]) -> set[int]:
    """Entity types present in a decoded nclone map, straight from the byte stream.

    The entity section starts at 1230 with a 5-byte stride
    ([type, x, y, orientation, mode]) -- see `map/loader.rs:load_entities`.
    """
    types: set[int] = set()
    i = 1230
    while i + 4 < len(map_data):
        t = int(map_data[i])
        if t != 0:
            types.add(t)
        i += 5
    return types


def _run_one(args):
    path, max_frames, engine, require_type = args
    name = Path(path).name
    try:
        from nclone.replay.npp_attract_decoder import NppAttractDecoder
        from nclone.replay.replay_executor import decode_input_to_controls

        dec = NppAttractDecoder()
        res = dec.decode_npp_attract_file(path)
        demo = res.get("demo_data", []) or []
        mapb = list(dec.create_nclone_map(path))
        types = sorted(map_entity_types(mapb) - PHANTOM_TYPES)
        with open(path, "rb") as fh:
            ninjas = dec.ninja_count(fh.read())
        if require_type is not None and require_type not in types:
            return None
        if not demo:
            return {"f": name, "types": types, "ninjas": ninjas,
                    "outcome": "no_demo", "frames": 0}
        if ninjas > 1:
            # A co-op recording: the tape carries inputs for several players and
            # the level is built around all of them. A single-ninja simulator
            # cannot replay it however correct its physics are, so it is out of
            # scope rather than a loss. Counting these as failures understated the
            # real win rate by ~19 points.
            return {"f": name, "types": types, "ninjas": ninjas,
                    "outcome": "coop", "frames": 0}

        if engine == "python":
            from nclone.nplay_headless import NPlayHeadless

            h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
        else:
            from nclone.native_headless import NativePlayHeadless

            h = NativePlayHeadless()
        h.load_map_from_map_data(mapb)
        outcome = "timeout"
        frames = 0
        for b in demo[:max_frames]:
            hor, jmp = decode_input_to_controls(int(b))
            h.tick(hor, jmp)
            frames += 1
            if h.ninja_has_won():
                outcome = "win"
                break
            if h.ninja_has_died():
                outcome = "die"
                break

        # The tape running out is NOT the end of the demo. The game keeps
        # simulating with no input once an attract's inputs are exhausted, and
        # the win frequently registers during that coast: the last input launches
        # the ninja INTO the exit and it arrives a frame or two later.
        #
        # Measured against a capture of `10172495` taken for this: our ninja is
        # bit-identical to the game for all 221 frames of the tape, and the
        # GAME's ninja reaches state 8 on frame 222 -- one frame past the last
        # input, with the remaining 238 captured frames being the win animation.
        # Scoring that as a timeout was our harness stopping before the game did.
        #
        # Across the 472-demo timeout bucket this recovers 50, of which 27 need
        # a single extra frame and every one lands within 25. The budget below is
        # deliberately looser than the observed maximum rather than fitted to it.
        if outcome == "timeout":
            for _ in range(COAST_FRAMES):
                h.tick(0, 0)
                frames += 1
                if h.ninja_has_won():
                    outcome = "win"
                    break
                if h.ninja_has_died():
                    outcome = "die"
                    break

        row = {"f": name, "types": types, "ninjas": ninjas,
               "outcome": outcome, "frames": frames}
        if outcome == "die":
            # WHICH hazard killed the ninja. Without this a losing demo tells you
            # only THAT it died -- and with several hazards on a map that is not
            # enough to attribute a regression to an implementation.
            ninja = getattr(getattr(h, "sim", None), "ninja", None)
            row["cause"] = getattr(ninja, "death_cause", None) or "physics"
        return row
    except Exception as e:  # noqa: BLE001
        return {"f": name, "types": [], "outcome": f"err:{type(e).__name__}", "frames": 0}


# Types with a working implementation in nclone-native. Sourced from the shared
# profile rather than re-listed here: a local copy is exactly how the training
# manifest's allowed-set went stale for an entire entity-porting effort, and this
# script's whole job is to report honestly on what is implemented.
from nclone.constants.entity_types import NATIVE_IMPLEMENTED_TYPES

IMPLEMENTED = set(NATIVE_IMPLEMENTED_TYPES)


def summarize(rows: list[dict], implemented: set[int]) -> None:
    n = len(rows)
    outcomes = Counter(r["outcome"] for r in rows)
    causes = Counter(r.get("cause", "?") for r in rows if r["outcome"] == "die")
    wins = outcomes.get("win", 0)
    print(f"\n{'=' * 66}\n{n} demos: {wins} win ({100 * wins / max(n, 1):.2f}%)")
    for k, v in outcomes.most_common():
        if k != "win":
            print(f"    {k:12s} {v:6d}  ({100 * v / max(n, 1):.2f}%)")

    # THE headline number. Co-op recordings and demos with no tape can never be
    # replayed by a single-ninja sim, so leaving them in the denominator measures
    # the corpus rather than the simulator.
    replayable = [r for r in rows if r["outcome"] in ("win", "die", "timeout")]
    if replayable:
        rw = sum(1 for r in replayable if r["outcome"] == "win")
        skipped = n - len(replayable)
        print(f"\n  REPLAYABLE (excl. {skipped} co-op / no-demo / error): "
              f"{rw}/{len(replayable)} = {100 * rw / len(replayable):.2f}% win"
              f"  <- the number to drive to 100%")

    if causes:
        print("\n  deaths by cause (an expert demo that the real game let through "
              "should not be dying to a hazard at all):")
        for k, v in causes.most_common():
            print(f"    {k:16s} {v:6d}")

    # A demo is "clean" if every entity on its map is implemented. Clean losses are
    # real bugs; unclean losses are expected until the missing type lands.
    clean = [r for r in rows if r["outcome"] in ("win", "die", "timeout")
             and not (set(r["types"]) - implemented)]
    clean_win = sum(1 for r in clean if r["outcome"] == "win")
    print(f"\n  clean maps (all types implemented): {len(clean)}, "
          f"{clean_win} win ({100 * clean_win / max(len(clean), 1):.2f}%)"
          f"  <- residual failures here are sim bugs, not missing entities")

    # Per-missing-type gate size: how many non-winning demos would be unblocked if
    # this type (and nothing else) were the only thing missing.
    missing_counter: Counter[int] = Counter()
    sole_blocker: Counter[int] = Counter()
    for r in rows:
        if r["outcome"] == "win":
            continue
        miss = set(r["types"]) - implemented
        for t in miss:
            missing_counter[t] += 1
        if len(miss) == 1:
            sole_blocker[next(iter(miss))] += 1
    if missing_counter:
        print("\n  losing demos by missing entity type "
              "(present = appears on the map, sole = the ONLY missing type):")
        for t, c in missing_counter.most_common():
            print(f"    type {t:2d}: present {c:5d} ({100 * c / max(n, 1):5.2f}%)"
                  f"   sole blocker {sole_blocker.get(t, 0):5d}")

    # The regression guard for hazard entities. None of the types still being
    # ported has a physical collision (spec §6.6), so they cannot bend the ninja's
    # trajectory -- a CORRECT implementation leaves these rates flat, and a
    # too-lethal one drops them. This is the only signal the corpus gives, since
    # an under-lethal hazard is indistinguishable from an absent one.
    print("\n  win rate on maps CONTAINING each type "
          "(hazard regression guard -- must not drop):")
    per_type: Counter[int] = Counter()
    per_type_win: Counter[int] = Counter()
    for r in rows:
        for t in r["types"]:
            per_type[t] += 1
            if r["outcome"] == "win":
                per_type_win[t] += 1
    for t in sorted(per_type):
        c = per_type[t]
        w = per_type_win.get(t, 0)
        flag = "" if t in implemented else "   (NOT IMPLEMENTED)"
        print(f"    type {t:2d}: {w:5d}/{c:5d} = {100 * w / max(c, 1):6.2f}%{flag}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in-dir")
    ap.add_argument("--files-from",
                    help="Read the demo paths from this file, one per line, instead "
                         "of globbing --in-dir. Globbing a 250k-file corpus on a "
                         "mounted drive costs minutes per run and dominates an A/B; "
                         "this makes repeated measurements of the same subset cheap.")
    ap.add_argument("--out", help="JSONL output path")
    ap.add_argument("--summarize", help="Summarize an existing JSONL and exit")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 6))
    ap.add_argument("--max-frames", type=int, default=100000)
    ap.add_argument("--require-type", type=int, default=None,
                    help="Only simulate demos whose map contains this entity type. "
                         "Decoding still touches every file, but simulation -- the "
                         "expensive part -- is skipped, so a single hazard can be "
                         "A/B'd over a large corpus cheaply.")
    ap.add_argument("--engine", choices=("native", "python"), default="native",
                    help="Which simulator to re-simulate on. 'python' is ~10x slower "
                         "but is the only way to see Python-side entity work, and "
                         "diffing the two localises engine-specific divergence.")
    args = ap.parse_args()

    if args.summarize:
        rows = [json.loads(ln) for ln in Path(args.summarize).read_text().splitlines() if ln.strip()]
        summarize(rows, IMPLEMENTED)
        return 0

    if not (args.in_dir or args.files_from) or not args.out:
        ap.error("--in-dir (or --files-from) and --out are required unless "
                 "--summarize is given")

    if args.files_from:
        files = [ln.strip() for ln in Path(args.files_from).read_text().splitlines()
                 if ln.strip()]
    else:
        files = sorted(glob.glob(os.path.join(args.in_dir, "*")))
        files = [f for f in files if os.path.isfile(f) or os.path.islink(f)]
    files = files[args.offset:]
    if args.limit:
        files = files[: args.limit]
    print(f"Simulating {len(files)} attract demos with {args.workers} workers "
          f"on the {args.engine} engine")

    tasks = [(f, args.max_frames, args.engine, args.require_type) for f in files]
    rows: list[dict] = []
    with mp.Pool(args.workers) as pool, open(args.out, "w") as fh:
        for i, row in enumerate(pool.imap_unordered(_run_one, tasks, chunksize=8), 1):
            if row is None:  # filtered out by --require-type
                continue
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            if i % 1000 == 0:
                w = sum(1 for r in rows if r["outcome"] == "win")
                print(f"  {i}/{len(files)}  win={w}")
                fh.flush()

    summarize(rows, IMPLEMENTED)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
