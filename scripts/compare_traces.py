#!/usr/bin/env python3
"""
Compare the nclone simulation trace against the N++ game trace to find the
root cause of divergence.

Usage:
    python scripts/compare_traces.py <sim_trace.json> <game_trace.csv> [options]

Options:
    --threshold FLOAT   Position/velocity tolerance for "identical" check
                        (default: 1e-6, meaning sub-pixel)
    --verbose-frames N  Print detailed info for the first N diverging frames
                        (default: 30)
    --output CSV        Write full per-frame diff to this CSV file
    --no-header         game_trace.csv has no header row

The sim_trace.json is produced by export_attract_for_game.py.
The game_trace.csv is produced by the CE Lua script ce_record_and_playback.lua.

Output
------
* First diverging frame with before/after values
* Classification of divergence type:
    - POSITION_DRIFT    : gradual floating-point accumulation
    - POSITION_JUMP     : sudden large position error (e.g. tile/wall miss)
    - STATE_MISMATCH    : ninja state (dead/jumping/etc.) differs
    - VELOCITY_ERROR    : velocity diverges but position not yet
* Summary table of divergence magnitude per frame range
* Final outcome comparison (win / dead / playing)
"""

import sys
import json
import csv
import math
from pathlib import Path
from typing import Optional

NINJA_STATE_NAMES = {
    0: "idle", 1: "running", 2: "ground_sliding", 3: "jumping",
    4: "falling", 5: "wall_sliding", 6: "dead", 7: "awaiting_death",
    8: "celebrating", 9: "disabled",
}

TERMINAL_STATES = {6, 7, 8, 9}


# ------------------------------------------------------------------ #
#  Data loading                                                       #
# ------------------------------------------------------------------ #

def load_sim_trace(path: str) -> list[dict]:
    """Load sim_trace.json produced by export_attract_for_game.py."""
    with open(path) as f:
        frames = json.load(f)
    return frames


def load_game_trace(path: str, has_header: bool = True) -> list[dict]:
    """
    Load game_trace.csv produced by ce_record_and_playback.lua.

    Expected columns (with header):
        frame, xpos, ypos, xspeed, yspeed, state,
        airborn, walled, jump_duration
    """
    frames = []
    with open(path, newline="") as f:
        if has_header:
            reader = csv.DictReader(f)
            for row in reader:
                frames.append({
                    "frame":        int(row["frame"]),
                    "xpos":         float(row["xpos"]),
                    "ypos":         float(row["ypos"]),
                    "xspeed":       float(row.get("xspeed", 0)),
                    "yspeed":       float(row.get("yspeed", 0)),
                    "state":        int(row.get("state", -1)),
                    "airborn":      row.get("airborn", "false").lower() == "true",
                    "walled":       row.get("walled", "false").lower() == "true",
                    "jump_duration": int(row.get("jump_duration", 0)),
                })
        else:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if len(row) < 5:
                    continue
                frames.append({
                    "frame":        i + 1,
                    "xpos":         float(row[0]),
                    "ypos":         float(row[1]),
                    "xspeed":       float(row[2]) if len(row) > 2 else 0.0,
                    "yspeed":       float(row[3]) if len(row) > 3 else 0.0,
                    "state":        int(row[4])   if len(row) > 4 else -1,
                    "airborn":      False,
                    "walled":       False,
                    "jump_duration": 0,
                })
    return frames


# ------------------------------------------------------------------ #
#  Comparison logic                                                   #
# ------------------------------------------------------------------ #

def state_name(s: int) -> str:
    return NINJA_STATE_NAMES.get(s, f"?({s})")


def classify_divergence(prev_diff: Optional[float], curr_diff: float,
                        state_mismatch: bool) -> str:
    """Classify the type of divergence at a given frame."""
    if state_mismatch:
        return "STATE_MISMATCH"
    if prev_diff is None:
        return "FIRST_FRAME"
    if curr_diff > 50.0:
        return "POSITION_JUMP"
    if curr_diff > prev_diff * 2.0 and curr_diff > 1.0:
        return "POSITION_JUMP"
    return "POSITION_DRIFT"


def compare_traces(
    sim_frames: list[dict],
    game_frames: list[dict],
    threshold: float = 1e-6,
    verbose_frames: int = 30,
    output_csv: Optional[str] = None,
):
    """
    Do a frame-by-frame comparison and print a structured report.
    Returns a dict of summary statistics.
    """

    n_frames = min(len(sim_frames), len(game_frames))
    if n_frames == 0:
        print("ERROR: One or both trace files are empty.")
        return {}

    if len(sim_frames) != len(game_frames):
        print(f"WARNING: Frame count mismatch: sim={len(sim_frames)} game={len(game_frames)}")
        print(f"         Comparing first {n_frames} frames.\n")

    # ---- Per-frame diff ----
    diffs = []
    first_diff_frame = None
    prev_pos_diff = None
    verbose_printed = 0

    for i in range(n_frames):
        sf = sim_frames[i]
        gf = game_frames[i]

        # Frame numbers should match; warn if not
        if sf["frame"] != gf["frame"]:
            print(f"WARNING: frame numbering mismatch at index {i}: "
                  f"sim_frame={sf['frame']} game_frame={gf['frame']}")

        frame = sf["frame"]
        dx = sf["xpos"] - gf["xpos"]
        dy = sf["ypos"] - gf["ypos"]
        pos_diff = math.hypot(dx, dy)
        dvx = sf["xspeed"] - gf.get("xspeed", sf["xspeed"])
        dvy = sf["yspeed"] - gf.get("yspeed", sf["yspeed"])
        vel_diff = math.hypot(dvx, dvy)
        state_mismatch = sf["state"] != gf["state"]

        has_pos_diff  = pos_diff > threshold
        has_vel_diff  = vel_diff > threshold
        has_any_diff  = has_pos_diff or has_vel_diff or state_mismatch

        div_class = None
        if has_any_diff:
            if first_diff_frame is None:
                first_diff_frame = frame
            div_class = classify_divergence(prev_pos_diff, pos_diff, state_mismatch)

        diffs.append({
            "frame":        frame,
            "dx":           dx,
            "dy":           dy,
            "pos_diff":     pos_diff,
            "dvx":          dvx,
            "dvy":          dvy,
            "vel_diff":     vel_diff,
            "state_mismatch": state_mismatch,
            "sim_state":    sf["state"],
            "game_state":   gf["state"],
            "divergence":   div_class,
        })

        if has_any_diff and verbose_printed < verbose_frames:
            verbose_printed += 1
            print(f"  Frame {frame:5d}  [{div_class}]")
            print(f"    sim : pos=({sf['xpos']:.6f},{sf['ypos']:.6f}) "
                  f"vel=({sf['xspeed']:.6f},{sf['yspeed']:.6f}) "
                  f"state={state_name(sf['state'])}")
            print(f"    game: pos=({gf['xpos']:.6f},{gf['ypos']:.6f}) "
                  f"vel=({gf.get('xspeed',0):.6f},{gf.get('yspeed',0):.6f}) "
                  f"state={state_name(gf['state'])}")
            print(f"    diff: Δpos={pos_diff:.6e}  (Δx={dx:.4e}, Δy={dy:.4e})  "
                  f"Δvel={vel_diff:.4e}")
            if state_mismatch:
                print(f"    *** STATE MISMATCH: sim={state_name(sf['state'])} "
                      f"game={state_name(gf['state'])} ***")
            print()

        if has_pos_diff:
            prev_pos_diff = pos_diff

    # ---- Outcome comparison ----
    sim_last  = sim_frames[-1]
    game_last = game_frames[-1]

    def outcome(frame_list):
        last = frame_list[-1]
        s = last["state"]
        if s == 8:      return "win"
        if s in (6, 7): return "dead"
        return "playing"

    sim_outcome  = outcome(sim_frames)
    game_outcome = outcome(game_frames)

    # ---- Divergence characterisation ----
    total_diverged = sum(1 for d in diffs if d["divergence"] is not None)
    max_pos_diff   = max((d["pos_diff"] for d in diffs), default=0.0)
    divergence_counts = {}
    for d in diffs:
        if d["divergence"]:
            divergence_counts[d["divergence"]] = divergence_counts.get(d["divergence"], 0) + 1

    # Detect if divergence is gradual (drift) or sudden (jump)
    first_jump_frame = None
    for d in diffs:
        if d["divergence"] in ("POSITION_JUMP", "STATE_MISMATCH"):
            first_jump_frame = d["frame"]
            break

    # ---- Summary report ----
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Frames compared    : {n_frames}")
    print(f"Frames diverged    : {total_diverged} ({total_diverged/n_frames*100:.1f}%)")
    print(f"First diff frame   : {first_diff_frame or 'none'}")
    print(f"First jump/state   : {first_jump_frame or 'none (drift only)'}")
    print(f"Max position error : {max_pos_diff:.6e} pixels")
    print()

    print("Divergence types:")
    for dtype, count in sorted(divergence_counts.items()):
        print(f"  {dtype:<25}: {count} frames")
    print()

    print("Outcomes:")
    print(f"  Simulation : {sim_outcome}"
          f" (state={state_name(sim_last['state'])} at frame {sim_last['frame']})")
    print(f"  Game       : {game_outcome}"
          f" (state={state_name(game_last['state'])} at frame {game_last['frame']})")
    outcome_match = (sim_outcome == game_outcome)
    print(f"  Match      : {'YES' if outcome_match else 'NO -- OUTCOMES DIFFER'}")
    print()

    # ---- Drift rate analysis ----
    pos_diff_by_frame = [(d["frame"], d["pos_diff"]) for d in diffs if d["pos_diff"] > 0]
    if len(pos_diff_by_frame) >= 2:
        # Find where drift starts meaningfully (>1e-9) and how fast it grows
        meaningful = [(f, p) for f, p in pos_diff_by_frame if p > 1e-9]
        if meaningful:
            start_f, start_p = meaningful[0]
            end_f, end_p     = meaningful[-1]
            if end_f > start_f:
                frames_span = end_f - start_f
                print(f"Drift analysis (error > 1e-9):")
                print(f"  Start: frame {start_f}  error={start_p:.3e}")
                print(f"  End  : frame {end_f}  error={end_p:.3e}")
                print(f"  Over {frames_span} frames, error grew {end_p/start_p:.1f}x")
                if end_p / start_p < 10:
                    print("  Character: SLOW DRIFT (likely floating-point rounding)")
                elif end_p > 50:
                    print("  Character: SUDDEN JUMP then drift (logic/collision bug)")
                else:
                    print("  Character: MODERATE GROWTH (possibly accumulated physics error)")
                print()

    # ---- Root cause hints ----
    print("Root cause hints:")
    if not first_diff_frame:
        print("  Traces are IDENTICAL -- no physics divergence found.")
        if sim_outcome != game_outcome:
            print("  Outcomes still differ; likely an ENTITY COLLISION difference")
            print("  (mine timing, exit switch range) not visible in position data alone.")
    elif first_jump_frame and first_jump_frame == first_diff_frame:
        print(f"  Divergence starts SUDDENLY at frame {first_diff_frame}.")
        print("  Likely causes: tile collision miss, entity interaction, state transition.")
        # Look at what was happening in the sim at that frame
        sf = sim_frames[first_diff_frame - 1]
        gf = game_frames[first_diff_frame - 1]
        print(f"  Sim  at f{first_diff_frame}: state={state_name(sf['state'])} "
              f"pos=({sf['xpos']:.3f},{sf['ypos']:.3f})")
        print(f"  Game at f{first_diff_frame}: state={state_name(gf['state'])} "
              f"pos=({gf['xpos']:.3f},{gf['ypos']:.3f})")
    else:
        print(f"  Divergence DRIFTS from frame {first_diff_frame} onward.")
        print("  Likely causes: small floating-point constant difference (gravity,")
        print("  drag, friction), or slightly different physics integration order.")

    print()

    # ---- Magnitude histogram ----
    buckets = [
        (1e-15, 1e-9,  "sub-epsilon (< 1e-9)"),
        (1e-9,  1e-4,  "tiny        (1e-9 .. 1e-4)"),
        (1e-4,  0.1,   "small       (1e-4 .. 0.1)"),
        (0.1,   1.0,   "moderate    (0.1  ..  1.0)"),
        (1.0,   10.0,  "large       (1.0  .. 10.0)"),
        (10.0,  1e9,   "huge        (> 10)"),
    ]
    print("Position error magnitude histogram:")
    for lo, hi, label in buckets:
        count = sum(1 for d in diffs if lo <= d["pos_diff"] < hi)
        bar = "#" * min(count, 50)
        print(f"  {label}: {count:5d}  {bar}")
    print()

    # ---- Write output CSV if requested ----
    if output_csv:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "frame", "dx", "dy", "pos_diff", "dvx", "dvy", "vel_diff",
                "state_mismatch", "sim_state", "game_state", "divergence"
            ])
            writer.writeheader()
            writer.writerows(diffs)
        print(f"Full diff written to: {output_csv}")

    return {
        "n_frames": n_frames,
        "total_diverged": total_diverged,
        "first_diff_frame": first_diff_frame,
        "first_jump_frame": first_jump_frame,
        "max_pos_diff": max_pos_diff,
        "sim_outcome": sim_outcome,
        "game_outcome": game_outcome,
        "outcome_match": outcome_match,
        "divergence_counts": divergence_counts,
    }


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compare nclone sim trace vs N++ game trace."
    )
    parser.add_argument("sim_trace",  help="Path to sim_trace.json")
    parser.add_argument("game_trace", help="Path to game_trace.csv")
    parser.add_argument("--threshold",     type=float, default=1e-6,
                        help="Tolerance for position/velocity 'identical' check (default 1e-6)")
    parser.add_argument("--verbose-frames", type=int, default=30,
                        help="Print details for the first N diverging frames (default 30)")
    parser.add_argument("--output",    default=None,
                        help="Write full frame-by-frame diff to this CSV file")
    parser.add_argument("--no-header", action="store_true",
                        help="game_trace.csv has no header row")
    args = parser.parse_args()

    print(f"Sim trace  : {args.sim_trace}")
    print(f"Game trace : {args.game_trace}")
    print(f"Threshold  : {args.threshold}\n")

    sim_frames  = load_sim_trace(args.sim_trace)
    game_frames = load_game_trace(args.game_trace, has_header=not args.no_header)

    print(f"Sim  frames loaded: {len(sim_frames)}")
    print(f"Game frames loaded: {len(game_frames)}")
    print()

    compare_traces(
        sim_frames,
        game_frames,
        threshold=args.threshold,
        verbose_frames=args.verbose_frames,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
