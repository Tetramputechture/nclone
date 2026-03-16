#!/usr/bin/env python3
"""
Diagnose whether horizontal-mirror failures are Rust-only divergences (Hypothesis B)
or genuine data bugs visible in the Python sim too (Hypothesis A).

Algorithm
---------
For each attract file (up to --limit):
  1. Decode original → run in Rust.  If Rust says FAIL → skip (not "orig only wins").
  2. Generate mirror in-memory → run in Rust.  If Rust says WIN → skip (not failing mirror).
  3. [Found an "orig wins, mirror fails in Rust" case] → run mirror in Python sim.
     Record whether Python also fails or passes.

Summary table:
  "orig wins / mirror fails in Rust, mirror PASSES in Python" → Hypothesis B (Rust divergence)
  "orig wins / mirror fails in Rust, mirror FAILS in Python"  → Hypothesis A (data bug)

Usage:
    python scripts/diagnose_mirror_failures.py [--limit N] [--verbose] [--debug N]
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nclone.replay.npp_attract_decoder import NppAttractDecoder
from nclone.nsim import Simulator as NcloneSimulator
from nclone.sim_config import SimConfig
from nclone.offline_rl.dataset.augmentation.horizontal_mirror import HorizontalMirror
from nclone.offline_rl.dataset.augmentation.attract_augmentor import _build_map_data

try:
    import nclone_native
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    print("ERROR: nclone_native not available. Build with: cd nclone-native && maturin develop --release")
    sys.exit(1)

ATTRACTS_DIR = Path(__file__).resolve().parent.parent.parent / "attracts_filtered"
MAX_FRAMES = 20_000
_mirror = HorizontalMirror()
_decoder = NppAttractDecoder()


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _decode_input(inp: int) -> Tuple[int, int]:
    jump = 1 if (inp & 0x01) else 0
    right = 1 if (inp & 0x02) else 0
    left = 1 if (inp & 0x04) else 0
    hor = 0 if (left and right) else (-1 if left else (1 if right else 0))
    return hor, jump


def run_rust(map_data: bytes, inputs: List[int]) -> str:
    """Return 'win', 'dead', 'playing', or 'error'."""
    try:
        sim = nclone_native.Simulator.load(list(map_data))
    except Exception as e:
        return "error"
    try:
        for inp in inputs[:MAX_FRAMES]:
            hor, jump = _decode_input(inp)
            sim.tick(hor, jump)
            if sim.has_won():
                return "win"
            if sim.has_died():
                return "dead"
        state = sim.get_state().ninja.state
        if sim.has_won() or state == 8:
            return "win"
        if sim.has_died():
            return "dead"
        return "playing"
    except Exception:
        return "error"


def run_python(map_data: bytes, inputs: List[int]) -> str:
    """Return 'win', 'dead', 'playing', or 'error'."""
    try:
        sc = SimConfig(basic_sim=True, enable_anim=False)
        sim = NcloneSimulator(sc)
        sim.load(list(map_data))
    except Exception:
        return "error"
    try:
        for inp in inputs[:MAX_FRAMES]:
            hor, jump = _decode_input(inp)
            sim.tick(hor, jump)
            s = sim.ninja.state
            if s == 8:
                return "win"
            if s in (6, 7):
                return "dead"
        final = sim.ninja.state
        if final == 8:
            return "win"
        if final in (6, 7):
            return "dead"
        return "playing"
    except Exception:
        return "error"


# ---------------------------------------------------------------------------
# Per-file divergence detail
# ---------------------------------------------------------------------------

@dataclass
class FrameSnap:
    x: float
    y: float
    vx: float
    vy: float
    state: int


def run_rust_detailed(map_data: bytes, inputs: List[int]) -> Tuple[str, List[FrameSnap]]:
    snaps: List[FrameSnap] = []
    try:
        sim = nclone_native.Simulator.load(list(map_data))
    except Exception:
        return "error", snaps
    try:
        for inp in inputs[:MAX_FRAMES]:
            hor, jump = _decode_input(inp)
            sim.tick(hor, jump)
            ns = sim.get_state().ninja
            snaps.append(FrameSnap(ns.xpos, ns.ypos, ns.xspeed, ns.yspeed, ns.state))
            if sim.has_won():
                return "win", snaps
            if sim.has_died():
                return "dead", snaps
        ns = sim.get_state().ninja
        if sim.has_won() or ns.state == 8:
            return "win", snaps
        if sim.has_died():
            return "dead", snaps
        return "playing", snaps
    except Exception:
        return "error", snaps


def run_python_detailed(map_data: bytes, inputs: List[int]) -> Tuple[str, List[FrameSnap]]:
    snaps: List[FrameSnap] = []
    try:
        sc = SimConfig(basic_sim=True, enable_anim=False)
        sim = NcloneSimulator(sc)
        sim.load(list(map_data))
    except Exception:
        return "error", snaps
    try:
        for inp in inputs[:MAX_FRAMES]:
            hor, jump = _decode_input(inp)
            sim.tick(hor, jump)
            n = sim.ninja
            snaps.append(FrameSnap(n.xpos, n.ypos, n.xspeed, n.yspeed, n.state))
            if n.state == 8:
                return "win", snaps
            if n.state in (6, 7):
                return "dead", snaps
        final = sim.ninja.state
        if final == 8:
            return "win", snaps
        if final in (6, 7):
            return "dead", snaps
        return "playing", snaps
    except Exception:
        return "error", snaps


def print_divergence(fname: str, rust_snaps: List[FrameSnap], py_snaps: List[FrameSnap]) -> None:
    """Print the first frame where Rust and Python diverge on the mirror."""
    n = min(len(rust_snaps), len(py_snaps))
    for i in range(n):
        r, p = rust_snaps[i], py_snaps[i]
        dx = abs(r.x - p.x)
        dy = abs(r.y - p.y)
        if max(dx, dy) > 1e-6 or r.state != p.state:
            print(f"    first divergence @ frame {i+1}:")
            print(f"      rust  : pos=({r.x:.8f},{r.y:.8f}) vel=({r.vx:.8f},{r.vy:.8f}) state={r.state}")
            print(f"      python: pos=({p.x:.8f},{p.y:.8f}) vel=({p.vx:.8f},{p.vy:.8f}) state={p.state}")
            # show a few more frames
            for k in range(1, 6):
                if i + k >= n:
                    break
                rk, pk = rust_snaps[i + k], py_snaps[i + k]
                print(f"      f{i+k+1:5d}: rust=({rk.x:.4f},{rk.y:.4f},s={rk.state}) "
                      f"py=({pk.x:.4f},{pk.y:.4f},s={pk.state})")
            return
    print(f"    no positional divergence in first {n} common frames")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose horizontal-mirror Rust vs Python failures")
    parser.add_argument("--limit", type=int, default=None, help="Max files to process")
    parser.add_argument("--verbose", action="store_true", help="Print per-file outcomes")
    parser.add_argument("--debug", type=int, default=3,
                        help="Number of 'orig-only-win' cases to show frame-by-frame debug (0=none)")
    parser.add_argument("--attracts-dir", type=str, default=None)
    args = parser.parse_args()

    attracts_dir = Path(args.attracts_dir) if args.attracts_dir else ATTRACTS_DIR
    files = sorted(attracts_dir.iterdir())
    if args.limit:
        files = files[:args.limit]

    print(f"Attracts dir : {attracts_dir}")
    print(f"Total files  : {len(files)}")
    print()

    # Counters
    n_processed = 0
    n_decode_err = 0
    n_orig_fails_rust = 0      # orig fails in Rust → skip
    n_both_win_rust = 0        # both orig and mirror win in Rust
    n_both_fail_rust = 0       # both fail in Rust

    # "orig wins, mirror fails" in Rust → the interesting set
    mirror_py_win = 0
    mirror_py_dead = 0
    mirror_py_playing = 0
    mirror_py_error = 0

    debug_cases: list = []    # (fname, mirror_map, mirror_inputs, rust_snaps)

    start = time.time()

    for i, fpath in enumerate(files):
        if i % 200 == 0 and i > 0:
            elapsed = time.time() - start
            rate = i / elapsed
            print(f"  [{i}/{len(files)}] {rate:.1f} f/s | "
                  f"orig_fail={n_orig_fails_rust} both_win={n_both_win_rust} "
                  f"both_fail={n_both_fail_rust} "
                  f"interesting: py_win={mirror_py_win} py_fail={mirror_py_dead+mirror_py_playing} py_err={mirror_py_error}")

        # --- Decode ---
        try:
            decoded = _decoder.decode_npp_attract_file(str(fpath))
            if not decoded.get("demo_data"):
                n_decode_err += 1
                continue
            orig_map = _decoder.create_nclone_map(str(fpath))
            orig_inputs = decoded["demo_data"]
        except Exception:
            n_decode_err += 1
            continue

        # --- Generate mirror ---
        try:
            aug_data = _mirror.augment(decoded)
            mirror_map = _build_map_data(aug_data)
            if mirror_map is None:
                n_decode_err += 1
                continue
            mirror_inputs = aug_data["demo_data"]
        except Exception:
            n_decode_err += 1
            continue

        n_processed += 1

        # --- Rust: original ---
        orig_rust = run_rust(orig_map, orig_inputs)
        if orig_rust != "win":
            n_orig_fails_rust += 1
            if args.verbose:
                print(f"  {fpath.name}: orig_rust={orig_rust} [skip]")
            continue

        # --- Rust: mirror ---
        mirror_rust = run_rust(mirror_map, mirror_inputs)

        if mirror_rust == "win":
            n_both_win_rust += 1
            if args.verbose:
                print(f"  {fpath.name}: orig_rust=win mirror_rust=win")
            continue

        if mirror_rust in ("dead", "playing", "error"):
            if mirror_rust in ("dead", "playing"):
                n_both_fail_rust += 1  # counted as "orig only wins" in the problem statement
                # Wait – "both fail" means BOTH orig and mirror fail. Here orig wins.
                # This is the "orig only wins" bucket.
                pass

            # --- Python: mirror ---
            py_out = run_python(mirror_map, mirror_inputs)

            if args.verbose:
                print(f"  {fpath.name}: orig_rust=win mirror_rust={mirror_rust} mirror_py={py_out}")

            if py_out == "win":
                mirror_py_win += 1
            elif py_out == "dead":
                mirror_py_dead += 1
            elif py_out == "playing":
                mirror_py_playing += 1
            else:
                mirror_py_error += 1

            # Collect for detailed debug
            if len(debug_cases) < args.debug:
                debug_cases.append((fpath.name, mirror_map, mirror_inputs))

    elapsed = time.time() - start

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"MIRROR FAILURE DIAGNOSIS  ({n_processed} files, {elapsed:.1f}s)")
    print("=" * 70)
    print(f"  Decode/generate errors  : {n_decode_err}")
    print(f"  Original fails in Rust  : {n_orig_fails_rust}  [excluded from analysis]")
    print()
    print(f"  Rust outcomes (orig=win only):")
    interesting = mirror_py_win + mirror_py_dead + mirror_py_playing + mirror_py_error
    print(f"    Both win  (mirror also wins)         : {n_both_win_rust}")
    print(f"    Orig only wins (mirror fails in Rust): {interesting}")
    print()

    if interesting > 0:
        print(f"  Python sim on the {interesting} mirrored files where Rust says mirror FAILS:")
        print(f"    Python → WIN     : {mirror_py_win:5d}  ({mirror_py_win/interesting*100:.1f}%)")
        print(f"    Python → DEAD    : {mirror_py_dead:5d}  ({mirror_py_dead/interesting*100:.1f}%)")
        print(f"    Python → PLAYING : {mirror_py_playing:5d}  ({mirror_py_playing/interesting*100:.1f}%)")
        print(f"    Python → ERROR   : {mirror_py_error:5d}  ({mirror_py_error/interesting*100:.1f}%)")
        print()
        py_pass = mirror_py_win
        py_fail = mirror_py_dead + mirror_py_playing + mirror_py_error
        if py_pass > 0 and py_fail == 0:
            print("  VERDICT: Hypothesis B confirmed — ALL mirror failures are Rust-only.")
            print("           The mirror data is correct; divergence is in nclone_native physics.")
        elif py_fail > 0 and py_pass == 0:
            print("  VERDICT: Hypothesis A confirmed — ALL mirror failures are Python-visible.")
            print("           There is a data-level bug in the mirroring code.")
        else:
            pct_b = py_pass / interesting * 100
            pct_a = py_fail / interesting * 100
            print(f"  VERDICT: MIXED — {pct_b:.1f}% Rust-only (Hyp B), {pct_a:.1f}% Python-visible (Hyp A).")
            print("           Both sources of failure are present.")

    # -----------------------------------------------------------------------
    # Detailed frame-by-frame debug for a handful of cases
    # -----------------------------------------------------------------------
    if debug_cases:
        print()
        print(f"  --- FRAME-BY-FRAME DEBUG for first {len(debug_cases)} 'orig-only-win' case(s) ---")
        for fname, mmap, minputs in debug_cases:
            print(f"\n  File: {fname}")
            rust_out, rust_snaps = run_rust_detailed(mmap, minputs)
            py_out, py_snaps = run_python_detailed(mmap, minputs)
            print(f"    Rust  outcome: {rust_out} ({len(rust_snaps)} frames)")
            print(f"    Python outcome: {py_out} ({len(py_snaps)} frames)")
            print_divergence(fname, rust_snaps, py_snaps)


if __name__ == "__main__":
    main()
