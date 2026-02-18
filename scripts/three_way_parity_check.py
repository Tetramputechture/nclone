#!/usr/bin/env python3
"""
Three-way simulator parity check: Rust (nclone_native) vs Python (nclone) vs Original Python.

For each attract file in ../attracts_filtered:
1. Decode the attract file to get map data + demo inputs
2. Run all three simulators with identical map + inputs
3. Compare ninja positions and outcomes frame-by-frame
4. Report divergences between all pairs

Usage:
    python scripts/three_way_parity_check.py [--limit N] [--verbose] [--position-only]
"""

import argparse
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- Import nclone-original (needs argv patching for its module-level argparse) ---
_original_argv = sys.argv
sys.argv = [sys.argv[0], "--basic-sim"]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "nclone-original"))
import nsim as nsim_original

sys.argv = _original_argv

# --- Import nclone Python simulator ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nclone.replay.npp_attract_decoder import NppAttractDecoder
from nclone.nsim import Simulator as NcloneSimulator
from nclone.sim_config import SimConfig

# --- Import Rust simulator ---
try:
    import nclone_native
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    print("WARNING: nclone_native not available. Build with: cd nclone-native && maturin develop --release")

ATTRACTS_DIR = Path(__file__).resolve().parent.parent.parent / "attracts_filtered"
MAX_FRAMES = 20000
POS_EPSILON = 1e-6

STATE_NAMES = {
    0: "idle", 1: "running", 2: "ground_sliding", 3: "jumping",
    4: "falling", 5: "wall_sliding", 6: "dead", 7: "awaiting_death",
    8: "celebrating", 9: "disabled",
}


def state_name(s):
    return STATE_NAMES.get(s, f"unknown({s})")


def decode_input(input_byte: int) -> Tuple[int, int]:
    """Decode input byte into (hor_input, jump_input)."""
    jump = 1 if (input_byte & 0x01) else 0
    right = 1 if (input_byte & 0x02) else 0
    left = 1 if (input_byte & 0x04) else 0
    if left and right:
        hor = 0
    elif left:
        hor = -1
    elif right:
        hor = 1
    else:
        hor = 0
    return hor, jump


@dataclass
class FrameSnapshot:
    """Ninja state at a single frame."""
    x: float
    y: float
    vx: float
    vy: float
    state: int


@dataclass
class SimResult:
    """Result from running one simulator on a replay."""
    outcome: str  # "win", "dead", "playing", "error"
    end_frame: int
    final_state: int
    error: Optional[str] = None
    snapshots: List[FrameSnapshot] = field(default_factory=list)


def classify_outcome(state: int) -> Optional[str]:
    """Return outcome string if terminal, else None."""
    if state in (8, 9):
        return "win"
    if state in (6, 7):
        return "dead"
    return None


def run_original(map_data: bytes, demo_inputs: List[int], capture_snapshots: bool = False) -> SimResult:
    """Run nclone-original simulator."""
    try:
        sim = nsim_original.Simulator()
        sim.load(list(map_data))
    except Exception as e:
        return SimResult("error", 0, -1, error=f"load: {e}")

    snapshots = []
    try:
        for frame_idx, inp in enumerate(demo_inputs):
            if frame_idx >= MAX_FRAMES:
                break
            hor, jump = decode_input(inp)
            sim.tick(hor, jump)
            s = sim.ninja.state
            if capture_snapshots:
                snapshots.append(FrameSnapshot(
                    sim.ninja.xpos, sim.ninja.ypos,
                    sim.ninja.xspeed, sim.ninja.yspeed, s,
                ))
            outcome = classify_outcome(s)
            if outcome:
                return SimResult(outcome, frame_idx + 1, s, snapshots=snapshots)

        final = sim.ninja.state
        return SimResult(classify_outcome(final) or "playing", len(demo_inputs), final, snapshots=snapshots)
    except Exception as e:
        return SimResult("error", frame_idx + 1 if 'frame_idx' in dir() else 0, -1, error=f"tick: {e}")


def run_nclone(map_data: bytes, demo_inputs: List[int], capture_snapshots: bool = False) -> SimResult:
    """Run nclone Python simulator."""
    try:
        sc = SimConfig(basic_sim=True, enable_anim=False)
        sim = NcloneSimulator(sc)
        sim.load(list(map_data))
    except Exception as e:
        return SimResult("error", 0, -1, error=f"load: {e}")

    snapshots = []
    try:
        for frame_idx, inp in enumerate(demo_inputs):
            if frame_idx >= MAX_FRAMES:
                break
            hor, jump = decode_input(inp)
            sim.tick(hor, jump)
            s = sim.ninja.state
            if capture_snapshots:
                snapshots.append(FrameSnapshot(
                    sim.ninja.xpos, sim.ninja.ypos,
                    sim.ninja.xspeed, sim.ninja.yspeed, s,
                ))
            outcome = classify_outcome(s)
            if outcome:
                return SimResult(outcome, frame_idx + 1, s, snapshots=snapshots)

        final = sim.ninja.state
        return SimResult(classify_outcome(final) or "playing", len(demo_inputs), final, snapshots=snapshots)
    except Exception as e:
        return SimResult("error", frame_idx + 1 if 'frame_idx' in dir() else 0, -1, error=f"tick: {e}")


def run_rust(map_data: bytes, demo_inputs: List[int], capture_snapshots: bool = False) -> SimResult:
    """Run nclone_native Rust simulator."""
    if not HAS_RUST:
        return SimResult("error", 0, -1, error="nclone_native not available")

    try:
        sim = nclone_native.Simulator.load(list(map_data))
    except Exception as e:
        return SimResult("error", 0, -1, error=f"load: {e}")

    snapshots = []
    try:
        for frame_idx, inp in enumerate(demo_inputs):
            if frame_idx >= MAX_FRAMES:
                break
            hor, jump = decode_input(inp)
            sim.tick(hor, jump)
            sim_state = sim.get_state()
            ninja = sim_state.ninja
            s = ninja.state
            if capture_snapshots:
                snapshots.append(FrameSnapshot(
                    ninja.xpos, ninja.ypos,
                    ninja.xspeed, ninja.yspeed, s,
                ))
            won = sim.has_won() or s == 8
            died = sim.has_died()
            if won:
                return SimResult("win", frame_idx + 1, s, snapshots=snapshots)
            if died:
                return SimResult("dead", frame_idx + 1, s, snapshots=snapshots)

        sim_state = sim.get_state()
        final = sim_state.ninja.state
        won = sim.has_won() or final == 8
        died = sim.has_died()
        if won:
            return SimResult("win", len(demo_inputs), final, snapshots=snapshots)
        if died:
            return SimResult("dead", len(demo_inputs), final, snapshots=snapshots)
        return SimResult("playing", len(demo_inputs), final, snapshots=snapshots)
    except Exception as e:
        return SimResult("error", frame_idx + 1 if 'frame_idx' in dir() else 0, -1, error=f"tick: {e}")


@dataclass
class PairDivergence:
    """Divergence between two simulators."""
    first_diff_frame: int
    pos_diff: float
    sim_a_snap: FrameSnapshot
    sim_b_snap: FrameSnapshot


def find_first_divergence(
    snaps_a: List[FrameSnapshot],
    snaps_b: List[FrameSnapshot],
    epsilon: float = POS_EPSILON,
) -> Optional[PairDivergence]:
    """Find the first frame where two snapshot sequences diverge."""
    n = min(len(snaps_a), len(snaps_b))
    for i in range(n):
        a, b = snaps_a[i], snaps_b[i]
        dx = abs(a.x - b.x)
        dy = abs(a.y - b.y)
        pos_diff = max(dx, dy)
        if pos_diff > epsilon or a.state != b.state:
            return PairDivergence(i + 1, pos_diff, a, b)
    return None


def frame_by_frame_debug(
    map_data: bytes,
    demo_inputs: List[int],
    filename: str,
    limit_frames: int = 10,
):
    """Run all three sims frame-by-frame and print first divergence with context."""
    print(f"\n  --- Frame-by-frame debug for {filename} ---")

    orig_res = run_original(map_data, demo_inputs, capture_snapshots=True)
    nclone_res = run_nclone(map_data, demo_inputs, capture_snapshots=True)
    rust_res = run_rust(map_data, demo_inputs, capture_snapshots=True)

    pairs = [
        ("original", "nclone", orig_res, nclone_res),
        ("original", "rust", orig_res, rust_res),
        ("nclone", "rust", nclone_res, rust_res),
    ]

    for name_a, name_b, res_a, res_b in pairs:
        if res_a.error or res_b.error:
            print(f"  {name_a} vs {name_b}: skipped (error)")
            continue

        div = find_first_divergence(res_a.snapshots, res_b.snapshots)
        if div is None:
            print(f"  {name_a} vs {name_b}: IDENTICAL positions (outcome: {res_a.outcome} vs {res_b.outcome})")
        else:
            a, b = div.sim_a_snap, div.sim_b_snap
            print(f"  {name_a} vs {name_b}: first divergence at frame {div.first_diff_frame} (pos_diff={div.pos_diff:.10f})")
            print(f"    {name_a}: pos=({a.x:.8f}, {a.y:.8f}) vel=({a.vx:.8f}, {a.vy:.8f}) state={state_name(a.state)}")
            print(f"    {name_b}: pos=({b.x:.8f}, {b.y:.8f}) vel=({b.vx:.8f}, {b.vy:.8f}) state={state_name(b.state)}")

            # Show a few frames of context
            frame_start = div.first_diff_frame  # 1-indexed, so index = frame - 1
            idx_start = frame_start - 1
            for k in range(1, min(limit_frames, len(res_a.snapshots) - idx_start, len(res_b.snapshots) - idx_start)):
                idx = idx_start + k
                sa = res_a.snapshots[idx]
                sb = res_b.snapshots[idx]
                pdiff = max(abs(sa.x - sb.x), abs(sa.y - sb.y))
                print(f"    frame {idx + 1}: {name_a}=({sa.x:.4f},{sa.y:.4f},s={state_name(sa.state)}) "
                      f"{name_b}=({sb.x:.4f},{sb.y:.4f},s={state_name(sb.state)}) pdiff={pdiff:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Three-way simulator parity check")
    parser.add_argument("--limit", type=int, default=None, help="Max files to process")
    parser.add_argument("--verbose", action="store_true", help="Print per-file results")
    parser.add_argument("--debug-divergences", type=int, default=3,
                        help="Number of divergent files to debug frame-by-frame (0=none)")
    parser.add_argument("--attracts-dir", type=str, default=None,
                        help="Override attracts directory")
    args = parser.parse_args()

    attracts_dir = Path(args.attracts_dir) if args.attracts_dir else ATTRACTS_DIR
    attract_files = sorted(attracts_dir.iterdir())
    if args.limit:
        attract_files = attract_files[:args.limit]
    total = len(attract_files)

    sim_names = ["original", "nclone", "rust"]
    if not HAS_RUST:
        sim_names = ["original", "nclone"]
        print("Running two-way comparison only (Rust simulator unavailable)")

    print(f"Found {total} attract files in {attracts_dir}")
    print(f"Comparing simulators: {', '.join(sim_names)}")
    print()

    decoder = NppAttractDecoder()

    # Outcome tracking: for each pair, count agreement/disagreement
    pair_keys = []
    for i, a in enumerate(sim_names):
        for b in sim_names[i + 1:]:
            pair_keys.append((a, b))

    # Per-pair outcome matrices
    pair_stats: Dict[Tuple[str, str], Dict[str, int]] = {}
    for pk in pair_keys:
        pair_stats[pk] = {
            "agree_win": 0, "agree_dead": 0, "agree_playing": 0,
            "disagree": 0, "one_error": 0,
        }

    # Per-pair frame divergences (where outcome agrees but frame differs)
    pair_frame_diffs: Dict[Tuple[str, str], List[dict]] = {pk: [] for pk in pair_keys}

    # Overall per-sim stats
    sim_outcomes: Dict[str, Dict[str, int]] = {s: {"win": 0, "dead": 0, "playing": 0, "error": 0} for s in sim_names}

    # Files with disagreements (for debug)
    disagreement_files: List[dict] = []

    decode_errors = 0
    start_time = time.time()

    for i, fpath in enumerate(attract_files):
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            eta = (total - i) / rate if rate > 0 else 0
            n_disagree = sum(s["disagree"] for s in pair_stats.values())
            print(f"  [{i}/{total}] {rate:.1f} files/s, ETA {eta:.0f}s | disagree={n_disagree}")

        # Decode attract file
        try:
            decoded = decoder.decode_npp_attract_file(str(fpath))
            demo_inputs = decoded["demo_data"]
            map_data = decoder.create_nclone_map(str(fpath))
            if not demo_inputs:
                decode_errors += 1
                continue
        except Exception:
            decode_errors += 1
            continue

        # Run all simulators
        results: Dict[str, SimResult] = {}
        results["original"] = run_original(map_data, demo_inputs)
        results["nclone"] = run_nclone(map_data, demo_inputs)
        if HAS_RUST:
            results["rust"] = run_rust(map_data, demo_inputs)

        # Track per-sim outcomes
        for sn in sim_names:
            r = results[sn]
            sim_outcomes[sn][r.outcome] += 1

        if args.verbose:
            parts = [f"{sn}={results[sn].outcome}@{results[sn].end_frame}" for sn in sim_names]
            print(f"  {fpath.name}: {' | '.join(parts)}")

        # Compare pairs
        any_disagree = False
        for a, b in pair_keys:
            ra, rb = results[a], results[b]

            if ra.error or rb.error:
                pair_stats[(a, b)]["one_error"] += 1
                continue

            if ra.outcome == rb.outcome:
                pair_stats[(a, b)][f"agree_{ra.outcome}"] += 1
                if ra.end_frame != rb.end_frame:
                    pair_frame_diffs[(a, b)].append({
                        "file": fpath.name,
                        "outcome": ra.outcome,
                        f"{a}_frame": ra.end_frame,
                        f"{b}_frame": rb.end_frame,
                        "diff": rb.end_frame - ra.end_frame,
                    })
            else:
                pair_stats[(a, b)]["disagree"] += 1
                any_disagree = True

        if any_disagree:
            entry = {"file": fpath.name, "path": str(fpath), "inputs": len(demo_inputs)}
            for sn in sim_names:
                r = results[sn]
                entry[sn] = f"{r.outcome}@{r.end_frame}(s={state_name(r.final_state)})"
            disagreement_files.append(entry)

    elapsed = time.time() - start_time
    processed = total - decode_errors

    # === Print report ===
    print("\n" + "=" * 90)
    print(f"THREE-WAY SIMULATOR PARITY CHECK ({total} files, {elapsed:.1f}s)")
    print("=" * 90)

    print(f"\nFiles processed: {processed}/{total} (decode errors: {decode_errors})")

    print(f"\n--- PER-SIMULATOR OUTCOMES ---")
    print(f"  {'Sim':<12} {'Win':>6} {'Dead':>6} {'Playing':>8} {'Error':>6}")
    for sn in sim_names:
        o = sim_outcomes[sn]
        print(f"  {sn:<12} {o['win']:>6} {o['dead']:>6} {o['playing']:>8} {o['error']:>6}")

    print(f"\n--- PAIR-WISE OUTCOME COMPARISON ---")
    for a, b in pair_keys:
        ps = pair_stats[(a, b)]
        total_compared = ps["agree_win"] + ps["agree_dead"] + ps["agree_playing"] + ps["disagree"]
        agree = ps["agree_win"] + ps["agree_dead"] + ps["agree_playing"]
        pct = agree / total_compared * 100 if total_compared > 0 else 0
        print(f"\n  {a} vs {b}:")
        print(f"    Agree WIN:     {ps['agree_win']}")
        print(f"    Agree DEAD:    {ps['agree_dead']}")
        print(f"    Agree PLAYING: {ps['agree_playing']}")
        print(f"    DISAGREE:      {ps['disagree']}")
        print(f"    Errors:        {ps['one_error']}")
        print(f"    Agreement:     {agree}/{total_compared} ({pct:.1f}%)")

        # Frame-level diffs for agreeing outcomes
        fdiffs = pair_frame_diffs[(a, b)]
        if fdiffs:
            print(f"    Frame-level diffs (same outcome): {len(fdiffs)}")
            for fd in fdiffs[:5]:
                print(f"      {fd['file']}: {fd['outcome']} {a}@{fd[f'{a}_frame']} "
                      f"{b}@{fd[f'{b}_frame']} (diff={fd['diff']:+d})")
            if len(fdiffs) > 5:
                print(f"      ... and {len(fdiffs) - 5} more")

    # Disagreement details
    if disagreement_files:
        print(f"\n--- OUTCOME DISAGREEMENTS ({len(disagreement_files)} files) ---")
        for d in disagreement_files[:30]:
            parts = [f"{sn}={d[sn]}" for sn in sim_names if sn in d]
            print(f"  {d['file']}: {' | '.join(parts)}")
        if len(disagreement_files) > 30:
            print(f"  ... and {len(disagreement_files) - 30} more")

    # Frame-by-frame debug for first N divergent files
    if args.debug_divergences > 0 and disagreement_files:
        n_debug = min(args.debug_divergences, len(disagreement_files))
        print(f"\n--- FRAME-BY-FRAME DEBUG (first {n_debug} disagreements) ---")
        for d in disagreement_files[:n_debug]:
            fpath = Path(d["path"])
            decoded = decoder.decode_npp_attract_file(str(fpath))
            demo_inputs = decoded["demo_data"]
            map_data = decoder.create_nclone_map(str(fpath))
            frame_by_frame_debug(map_data, demo_inputs, d["file"])


if __name__ == "__main__":
    main()
