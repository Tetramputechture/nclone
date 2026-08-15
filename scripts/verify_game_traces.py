#!/usr/bin/env python3
"""Regression guard: replay every captured game trace in BOTH engines and diff.

`docs/game_traces/` holds per-frame ninja state captured out of the running N++
(round 8 §1). Those are real inputs on real trajectories, which makes them a
strictly stronger guard than `diffcheck_entity_parity.py` -- that one samples
official maps with *synthetic* input, and it reported `15/15 maps exact` while
the two engines disagreed on frame 1 of `10132650`
(`docs/ENTITY_RE_ROUND8_FINDINGS.md` §6.1).

Three questions per demo, all of which can regress independently:

1. is the **python** replay bit-exact against the game?
2. is the **native** replay bit-exact against the game?
3. do the two engines agree with **each other**?

(3) is the one that has to hold for every demo: the engines are supposed to be
interchangeable, and nothing else in the repo checks that on real input. (1) and
(2) are still short of 22/22, so they are guarded against a baseline in
`docs/game_traces/expected.json` rather than against perfection -- a demo that is
exact today must stay exact.

    python scripts/verify_game_traces.py              # check against the baseline
    python scripts/verify_game_traces.py --update     # rewrite the baseline
    python scripts/verify_game_traces.py --demo 10132650 -v

Exit status is non-zero if anything regressed, so it can gate a commit.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "docs" / "game_traces"
EXPECTED = TRACE_DIR / "expected.json"
ATTRACTS = Path(os.environ.get("NCLONE_ATTRACTS", "/home/tetra/projects/attracts"))

#: The fields the capture carries. Diffed on their IEEE-754 bit patterns, not on
#: their decimal forms -- these divergences start below 1e-12.
FIELDS = ("xpos", "ypos", "xspeed", "yspeed", "floor_x", "floor_y")


def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def load_jsonl(path: Path) -> dict[int, dict]:
    with open(path) as fh:
        return {r["frame"]: r for r in (json.loads(ln) for ln in fh if ln.strip())}


def replay(demo: str, engine: str) -> dict[int, dict]:
    """Replay one attract demo and return its per-frame ninja state.

    Same schema as the captures, so the two are directly comparable.
    """
    from nclone.replay.npp_attract_decoder import NppAttractDecoder
    from nclone.replay.replay_executor import decode_input_to_controls

    path = ATTRACTS / demo
    dec = NppAttractDecoder()
    tape = dec.decode_npp_attract_file(str(path)).get("demo_data") or []
    mapb = list(dec.create_nclone_map(str(path)))

    if engine == "python":
        from nclone.nplay_headless import NPlayHeadless

        h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    else:
        from nclone.native_headless import NativePlayHeadless

        h = NativePlayHeadless()
    h.load_map_from_map_data(mapb)

    out: dict[int, dict] = {}
    for i, b in enumerate(tape, 1):
        hor, jmp = decode_input_to_controls(int(b))
        h.tick(hor, jmp)
        n = h.sim.ninja
        out[i] = {
            "xpos": float(n.xpos), "ypos": float(n.ypos),
            "xspeed": float(n.xspeed), "yspeed": float(n.yspeed),
            "state": int(n.state),
            "floor_x": float(n.floor_normalized_x),
            "floor_y": float(n.floor_normalized_y),
        }
        if h.ninja_has_won() or h.ninja_has_died():
            break
    return out


def first_divergence(a: dict[int, dict], b: dict[int, dict]) -> int | None:
    """Lowest frame where any field's bit pattern differs, or None if identical
    across every overlapping frame."""
    for f in sorted(set(a) & set(b)):
        if any(bits(a[f][k]) != bits(b[f][k]) for k in FIELDS):
            return f
        if a[f]["state"] != b[f]["state"]:
            return f
    return None


def check(demo: str, verbose: bool = False) -> dict:
    game = load_jsonl(TRACE_DIR / f"{demo}.jsonl")
    py = replay(demo, "python")
    nat = replay(demo, "native")
    res = {
        "python_vs_game": first_divergence(game, py),
        "native_vs_game": first_divergence(game, nat),
        "native_vs_python": first_divergence(py, nat),
        "python_frames": max(py, default=0),
        "native_frames": max(nat, default=0),
    }
    if verbose:
        print(f"  {demo}: {json.dumps(res)}")
    return res


def regressed(demo: str, now: dict, was: dict) -> list[str]:
    """What got worse. A `None` divergence frame means bit-exact, so `None` is the
    best possible value and losing it is always a regression; otherwise diverging
    EARLIER than the baseline is the regression."""
    out = []
    for key in ("python_vs_game", "native_vs_game", "native_vs_python"):
        if key not in was:
            continue  # no baseline for this demo yet -- nothing to regress from
        new, old = now.get(key), was[key]
        if old is None and new is not None:
            out.append(f"{key}: was bit-exact, now diverges at frame {new}")
        elif old is not None and new is not None and new < old:
            out.append(f"{key}: diverges at {new}, was {old}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--demo", action="append",
                    help="Only this demo (repeatable). Default: every trace.")
    ap.add_argument("--update", action="store_true",
                    help="Rewrite the baseline from the current results. Only do "
                         "this once you have read the diff and believe it.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    demos = args.demo or sorted(p.stem for p in TRACE_DIR.glob("*.jsonl"))
    baseline = json.loads(EXPECTED.read_text()) if EXPECTED.exists() else {}

    results, failures = {}, []
    for demo in demos:
        res = check(demo, args.verbose)
        results[demo] = res
        # Cross-engine agreement is not baselined -- it is required. The two
        # simulators are meant to be interchangeable, and every trace passes it
        # as of round 9.
        if res["native_vs_python"] is not None:
            failures.append(f"{demo}: engines disagree at frame {res['native_vs_python']}")
        failures += [f"{demo}: {m}" for m in regressed(demo, res, baseline.get(demo, {}))]

    def tally(key):
        return sum(1 for r in results.values() if r[key] is None)

    print(f"\n{len(results)} traces")
    print(f"  bit-exact vs game, python: {tally('python_vs_game')}/{len(results)}")
    print(f"  bit-exact vs game, native: {tally('native_vs_game')}/{len(results)}")
    print(f"  engines agree:             {tally('native_vs_python')}/{len(results)}")

    if args.update:
        EXPECTED.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(f"\nbaseline written to {EXPECTED.relative_to(ROOT)}")
        return 0

    if failures:
        print("\nREGRESSIONS:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nno regressions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
