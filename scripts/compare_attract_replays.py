#!/usr/bin/env python3
"""
Compare attract replay outcomes between nclone-original and nclone simulators.

For each attract file in ../attracts_filtered:
1. Decode the attract file to get map data + demo inputs
2. Run both simulators with identical map + inputs
3. Compare outcomes (win/death/playing/error)
4. Report divergences
"""

import sys
import os
import time
import traceback
from pathlib import Path

# Patch sys.argv before importing nclone-original (it uses argparse at module level)
original_argv = sys.argv
sys.argv = [sys.argv[0], "--basic-sim"]

# Add nclone-original to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "nclone-original"))
import nsim as nsim_original

# Restore argv
sys.argv = original_argv

# Now import nclone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nclone.replay.npp_attract_decoder import NppAttractDecoder
from nclone.replay.replay_executor import decode_input_to_controls
from nclone.nsim import Simulator as NcloneSimulator
from nclone.sim_config import SimConfig

ATTRACTS_DIR = Path(__file__).resolve().parent.parent.parent / "attracts_filtered"
MAX_FRAMES = 20000


def decode_input(input_byte):
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


def get_ninja_state_name(state):
    """Map ninja state int to human-readable name."""
    names = {
        0: "idle",
        1: "running",
        2: "ground_sliding",
        3: "jumping",
        4: "falling",
        5: "wall_sliding",
        6: "dead",
        7: "awaiting_death",
        8: "celebrating",
        9: "win",
    }
    return names.get(state, f"unknown({state})")


def run_original(map_data, demo_inputs):
    """Run nclone-original simulator. Returns (outcome, frame, ninja_state, error)."""
    try:
        sim = nsim_original.Simulator()
        sim.load(map_data)
    except Exception as e:
        return "load_error", 0, None, str(e)

    try:
        for frame_idx, inp in enumerate(demo_inputs):
            if frame_idx >= MAX_FRAMES:
                break
            hor, jump = decode_input(inp)
            sim.tick(hor, jump)
            state = sim.ninja.state
            if state in (8, 9):
                return "win", frame_idx + 1, state, None
            if state in (6, 7):
                return "dead", frame_idx + 1, state, None
        # Check final state after all inputs
        final_state = sim.ninja.state
        if final_state in (8, 9):
            return "win", len(demo_inputs), final_state, None
        if final_state in (6, 7):
            return "dead", len(demo_inputs), final_state, None
        return "playing", len(demo_inputs), final_state, None
    except Exception as e:
        return "sim_error", frame_idx + 1, None, str(e)


def run_nclone(map_data, demo_inputs):
    """Run nclone simulator. Returns (outcome, frame, ninja_state, error)."""
    try:
        sc = SimConfig(basic_sim=True, enable_anim=False)
        sim = NcloneSimulator(sc)
        sim.load(list(map_data))
    except Exception as e:
        return "load_error", 0, None, str(e)

    try:
        for frame_idx, inp in enumerate(demo_inputs):
            if frame_idx >= MAX_FRAMES:
                break
            hor, jump = decode_input(inp)
            sim.tick(hor, jump)
            state = sim.ninja.state
            if state in (8, 9):
                return "win", frame_idx + 1, state, None
            if state in (6, 7):
                return "dead", frame_idx + 1, state, None
        final_state = sim.ninja.state
        if final_state in (8, 9):
            return "win", len(demo_inputs), final_state, None
        if final_state in (6, 7):
            return "dead", len(demo_inputs), final_state, None
        return "playing", len(demo_inputs), final_state, None
    except Exception as e:
        return "sim_error", frame_idx + 1, None, str(e)


def main():
    attract_files = sorted(ATTRACTS_DIR.iterdir())
    total = len(attract_files)
    print(f"Found {total} attract files in {ATTRACTS_DIR}")

    decoder = NppAttractDecoder()

    # Counters
    both_win = 0
    both_dead = 0
    both_playing = 0
    orig_win_nclone_dead = 0
    orig_win_nclone_playing = 0
    orig_dead_nclone_win = 0
    orig_dead_nclone_playing = 0
    orig_playing_nclone_win = 0
    orig_playing_nclone_dead = 0
    decode_errors = 0
    orig_errors = 0
    nclone_errors = 0
    both_errors = 0

    # Track divergence details
    divergences = []

    # Track first-divergence frames for matching outcomes
    frame_divergences = []

    start_time = time.time()

    for i, fpath in enumerate(attract_files):
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            eta = (total - i) / rate if rate > 0 else 0
            print(
                f"  [{i}/{total}] {rate:.1f} files/s, ETA {eta:.0f}s | "
                f"both_win={both_win} both_dead={both_dead} diverged={len(divergences)} "
                f"errors={decode_errors + orig_errors + nclone_errors}"
            )

        # Decode
        try:
            decoded = decoder.decode_npp_attract_file(str(fpath))
            demo_inputs = decoded["demo_data"]
            map_data = decoder.create_nclone_map(str(fpath))
            if not demo_inputs:
                decode_errors += 1
                continue
        except Exception as e:
            decode_errors += 1
            continue

        # Run both
        orig_outcome, orig_frame, orig_state, orig_err = run_original(
            map_data, demo_inputs
        )
        nclone_outcome, nclone_frame, nclone_state, nclone_err = run_nclone(
            map_data, demo_inputs
        )

        # Classify
        if orig_err and nclone_err:
            both_errors += 1
            continue
        if orig_err:
            orig_errors += 1
            continue
        if nclone_err:
            nclone_errors += 1
            divergences.append(
                {
                    "file": fpath.name,
                    "orig": f"{orig_outcome}@{orig_frame}",
                    "nclone": f"error: {nclone_err}",
                }
            )
            continue

        if orig_outcome == nclone_outcome:
            if orig_outcome == "win":
                both_win += 1
                if orig_frame != nclone_frame:
                    frame_divergences.append(
                        {
                            "file": fpath.name,
                            "outcome": "win",
                            "orig_frame": orig_frame,
                            "nclone_frame": nclone_frame,
                            "diff": nclone_frame - orig_frame,
                        }
                    )
            elif orig_outcome == "dead":
                both_dead += 1
                if orig_frame != nclone_frame:
                    frame_divergences.append(
                        {
                            "file": fpath.name,
                            "outcome": "dead",
                            "orig_frame": orig_frame,
                            "nclone_frame": nclone_frame,
                            "diff": nclone_frame - orig_frame,
                        }
                    )
            else:
                both_playing += 1
        else:
            key = f"{orig_outcome}_vs_{nclone_outcome}"
            if orig_outcome == "win" and nclone_outcome == "dead":
                orig_win_nclone_dead += 1
            elif orig_outcome == "win" and nclone_outcome == "playing":
                orig_win_nclone_playing += 1
            elif orig_outcome == "dead" and nclone_outcome == "win":
                orig_dead_nclone_win += 1
            elif orig_outcome == "dead" and nclone_outcome == "playing":
                orig_dead_nclone_playing += 1
            elif orig_outcome == "playing" and nclone_outcome == "win":
                orig_playing_nclone_win += 1
            elif orig_outcome == "playing" and nclone_outcome == "dead":
                orig_playing_nclone_dead += 1

            divergences.append(
                {
                    "file": fpath.name,
                    "orig": f"{orig_outcome}@{orig_frame}(state={get_ninja_state_name(orig_state) if orig_state is not None else '?'})",
                    "nclone": f"{nclone_outcome}@{nclone_frame}(state={get_ninja_state_name(nclone_state) if nclone_state is not None else '?'})",
                    "inputs": len(demo_inputs),
                }
            )

    elapsed = time.time() - start_time

    # Print results
    print("\n" + "=" * 80)
    print(f"ATTRACT REPLAY COMPARISON RESULTS ({total} files, {elapsed:.1f}s)")
    print("=" * 80)

    processed = total - decode_errors
    print(f"\nFiles processed: {processed}/{total} (decode errors: {decode_errors})")
    print(f"Original sim errors: {orig_errors}")
    print(f"nclone sim errors: {nclone_errors}")
    print(f"Both errored: {both_errors}")

    print(f"\n--- OUTCOME AGREEMENT ---")
    print(f"Both WIN:     {both_win}")
    print(f"Both DEAD:    {both_dead}")
    print(f"Both PLAYING: {both_playing}")

    print(f"\n--- OUTCOME DIVERGENCES ---")
    print(f"orig=WIN,  nclone=DEAD:    {orig_win_nclone_dead}")
    print(f"orig=WIN,  nclone=PLAYING: {orig_win_nclone_playing}")
    print(f"orig=DEAD, nclone=WIN:     {orig_dead_nclone_win}")
    print(f"orig=DEAD, nclone=PLAYING: {orig_dead_nclone_playing}")
    print(f"orig=PLAY, nclone=WIN:     {orig_playing_nclone_win}")
    print(f"orig=PLAY, nclone=DEAD:    {orig_playing_nclone_dead}")

    total_diverged = len(divergences)
    total_compared = processed - orig_errors - nclone_errors - both_errors
    agreement_pct = (
        (total_compared - total_diverged) / total_compared * 100
        if total_compared > 0
        else 0
    )
    print(f"\nTotal outcome agreement: {total_compared - total_diverged}/{total_compared} ({agreement_pct:.1f}%)")

    if frame_divergences:
        print(f"\n--- FRAME-LEVEL DIVERGENCES (same outcome, different frame) ---")
        print(f"Total: {len(frame_divergences)} files with frame differences")
        for fd in frame_divergences[:30]:
            print(
                f"  {fd['file']}: {fd['outcome']} orig@{fd['orig_frame']} "
                f"nclone@{fd['nclone_frame']} (diff={fd['diff']:+d})"
            )
        if len(frame_divergences) > 30:
            print(f"  ... and {len(frame_divergences) - 30} more")

    if divergences:
        print(f"\n--- FIRST 50 DIVERGENCES (different outcome) ---")
        for d in divergences[:50]:
            print(f"  {d['file']}: orig={d['orig']}  nclone={d['nclone']}")
        if len(divergences) > 50:
            print(f"  ... and {len(divergences) - 50} more")

    # For detailed debugging: find the first divergence where original wins but nclone doesn't
    # and do a frame-by-frame comparison
    if orig_win_nclone_dead > 0 or orig_win_nclone_playing > 0:
        print(f"\n--- FRAME-BY-FRAME DEBUG (first orig-win/nclone-fail) ---")
        for d in divergences:
            if "win" in d["orig"] and ("dead" in d["nclone"] or "playing" in d["nclone"]):
                fname = d["file"]
                fpath = ATTRACTS_DIR / fname
                decoded = decoder.decode_npp_attract_file(str(fpath))
                demo_inputs = decoded["demo_data"]
                map_data = decoder.create_nclone_map(str(fpath))

                # Run both side by side
                orig_sim = nsim_original.Simulator()
                orig_sim.load(map_data)

                sc = SimConfig(basic_sim=True, enable_anim=False)
                nclone_sim = NcloneSimulator(sc)
                nclone_sim.load(list(map_data))

                first_diff_frame = None
                for fi, inp in enumerate(demo_inputs[:MAX_FRAMES]):
                    hor, jump = decode_input(inp)
                    orig_sim.tick(hor, jump)
                    nclone_sim.tick(hor, jump)

                    ox, oy = orig_sim.ninja.xpos, orig_sim.ninja.ypos
                    nx, ny = nclone_sim.ninja.xpos, nclone_sim.ninja.ypos
                    os_ = orig_sim.ninja.state
                    ns = nclone_sim.ninja.state

                    if first_diff_frame is None and (
                        abs(ox - nx) > 0.001 or abs(oy - ny) > 0.001 or os_ != ns
                    ):
                        first_diff_frame = fi + 1
                        print(f"\n  File: {fname}")
                        print(f"  First divergence at frame {first_diff_frame}:")
                        print(f"    orig:   pos=({ox:.6f}, {oy:.6f}) state={get_ninja_state_name(os_)} vel=({orig_sim.ninja.xspeed:.6f}, {orig_sim.ninja.yspeed:.6f})")
                        print(f"    nclone: pos=({nx:.6f}, {ny:.6f}) state={get_ninja_state_name(ns)} vel=({nclone_sim.ninja.xspeed:.6f}, {nclone_sim.ninja.yspeed:.6f})")
                        print(f"    input: byte={inp} hor={hor} jump={jump}")

                        # Print a few more frames for context
                        for fi2 in range(fi + 1, min(fi + 6, len(demo_inputs))):
                            inp2 = demo_inputs[fi2]
                            hor2, jump2 = decode_input(inp2)
                            orig_sim.tick(hor2, jump2)
                            nclone_sim.tick(hor2, jump2)
                            ox2, oy2 = orig_sim.ninja.xpos, orig_sim.ninja.ypos
                            nx2, ny2 = nclone_sim.ninja.xpos, nclone_sim.ninja.ypos
                            print(f"    frame {fi2 + 1}: orig=({ox2:.6f},{oy2:.6f},s={get_ninja_state_name(orig_sim.ninja.state)}) nclone=({nx2:.6f},{ny2:.6f},s={get_ninja_state_name(nclone_sim.ninja.state)})")
                        break

                if first_diff_frame is None:
                    print(f"  File: {fname} - no position divergence found (entity-only?)")
                break  # Only debug the first one


if __name__ == "__main__":
    main()
