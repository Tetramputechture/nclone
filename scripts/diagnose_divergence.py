#!/usr/bin/env python3
"""
Detailed frame-by-frame diagnostic for a single attract file across all three sims.
Prints full ninja state at each frame to pinpoint where and why divergence occurs.
"""

import sys
from pathlib import Path

_original_argv = sys.argv
sys.argv = [sys.argv[0], "--basic-sim"]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "nclone-original"))
import nsim as nsim_original
sys.argv = _original_argv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nclone.replay.npp_attract_decoder import NppAttractDecoder
from nclone.nsim import Simulator as NcloneSimulator
from nclone.sim_config import SimConfig

import nclone_native

ATTRACTS_DIR = Path(__file__).resolve().parent.parent.parent / "attracts_filtered"


def decode_input(input_byte):
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


STATE_NAMES = {
    0: "idle", 1: "running", 2: "ground_slide", 3: "jumping",
    4: "falling", 5: "wall_slide", 6: "dead", 7: "await_death",
    8: "celebrate", 9: "disabled",
}


def diagnose_file(filename, start_frame=0, num_frames=20):
    fpath = ATTRACTS_DIR / filename
    if not fpath.exists():
        print(f"File not found: {fpath}")
        return

    decoder = NppAttractDecoder()
    decoded = decoder.decode_npp_attract_file(str(fpath))
    demo_inputs = decoded["demo_data"]
    map_data = decoder.create_nclone_map(str(fpath))
    map_list = list(map_data)

    # Initialize all three
    py_sim = NcloneSimulator(SimConfig(basic_sim=True, enable_anim=False))
    py_sim.load(list(map_list))

    orig_sim = nsim_original.Simulator()
    orig_sim.load(list(map_list))

    rust_sim = nclone_native.Simulator.load(map_list)

    end_frame = min(start_frame + num_frames, len(demo_inputs))

    # Run up to start_frame without printing
    for i in range(start_frame):
        inp = demo_inputs[i]
        hor, jump = decode_input(inp)
        py_sim.tick(hor, jump)
        orig_sim.tick(hor, jump)
        rust_sim.tick(hor, jump)

    print(f"\n{'='*120}")
    print(f"Diagnosing {filename} | frames {start_frame} to {end_frame} | total inputs: {len(demo_inputs)}")
    print(f"{'='*120}")

    # Print initial state (before any ticks in our range)
    py_n = py_sim.ninja
    orig_n = orig_sim.ninja
    rust_state = rust_sim.get_state()
    rust_n = rust_state.ninja

    print(f"\nState at frame {start_frame} (before tick):")
    print(f"  {'Sim':<8} {'x':>14} {'y':>14} {'vx':>14} {'vy':>14} {'state':>12} {'airborn':>8} {'walled':>8}")
    print(f"  {'py':<8} {py_n.xpos:>14.8f} {py_n.ypos:>14.8f} {py_n.xspeed:>14.8f} {py_n.yspeed:>14.8f} {STATE_NAMES.get(py_n.state,'?'):>12} {str(py_n.airborn):>8} {str(py_n.walled):>8}")
    print(f"  {'orig':<8} {orig_n.xpos:>14.8f} {orig_n.ypos:>14.8f} {orig_n.xspeed:>14.8f} {orig_n.yspeed:>14.8f} {STATE_NAMES.get(orig_n.state,'?'):>12} {str(orig_n.airborn):>8} {str(orig_n.walled):>8}")
    print(f"  {'rust':<8} {rust_n.xpos:>14.8f} {rust_n.ypos:>14.8f} {rust_n.xspeed:>14.8f} {rust_n.yspeed:>14.8f} {STATE_NAMES.get(rust_n.state,'?'):>12} {str(rust_n.airborn):>8} {str(rust_n.walled):>8}")

    for frame_idx in range(start_frame, end_frame):
        inp = demo_inputs[frame_idx]
        hor, jump = decode_input(inp)

        py_sim.tick(hor, jump)
        orig_sim.tick(hor, jump)
        rust_sim.tick(hor, jump)

        py_n = py_sim.ninja
        orig_n = orig_sim.ninja
        rust_state = rust_sim.get_state()
        rust_n = rust_state.ninja

        # Check for divergence
        dx_pr = abs(py_n.xpos - rust_n.xpos)
        dy_pr = abs(py_n.ypos - rust_n.ypos)
        dx_po = abs(py_n.xpos - orig_n.xpos)
        dy_po = abs(py_n.ypos - orig_n.ypos)

        marker = ""
        if dx_pr > 1e-6 or dy_pr > 1e-6:
            marker = " *** RUST DIVERGED"
        if dx_po > 1e-6 or dy_po > 1e-6:
            marker += " *** ORIG DIVERGED"

        print(f"\nFrame {frame_idx + 1} | input: hor={hor:+d} jump={jump}{marker}")
        print(f"  {'Sim':<8} {'x':>14} {'y':>14} {'vx':>14} {'vy':>14} {'state':>12} {'airborn':>8} {'walled':>8} {'floor_n':>14}")

        # Python floor normal
        py_fn = f"({py_n.floor_normalized_x:.4f},{py_n.floor_normalized_y:.4f})" if not py_n.airborn else "n/a"
        print(f"  {'py':<8} {py_n.xpos:>14.8f} {py_n.ypos:>14.8f} {py_n.xspeed:>14.8f} {py_n.yspeed:>14.8f} {STATE_NAMES.get(py_n.state,'?'):>12} {str(py_n.airborn):>8} {str(py_n.walled):>8} {py_fn:>14}")

        orig_fn = f"({orig_n.floor_normalized_x:.4f},{orig_n.floor_normalized_y:.4f})" if not orig_n.airborn else "n/a"
        print(f"  {'orig':<8} {orig_n.xpos:>14.8f} {orig_n.ypos:>14.8f} {orig_n.xspeed:>14.8f} {orig_n.yspeed:>14.8f} {STATE_NAMES.get(orig_n.state,'?'):>12} {str(orig_n.airborn):>8} {str(orig_n.walled):>8} {orig_fn:>14}")

        rust_fn = f"({rust_n.floor_normalized_x:.4f},{rust_n.floor_normalized_y:.4f})" if not rust_n.airborn else "n/a"
        print(f"  {'rust':<8} {rust_n.xpos:>14.8f} {rust_n.ypos:>14.8f} {rust_n.xspeed:>14.8f} {rust_n.yspeed:>14.8f} {STATE_NAMES.get(rust_n.state,'?'):>12} {str(rust_n.airborn):>8} {str(rust_n.walled):>8} {rust_fn:>14}")

        if dx_pr > 1e-6 or dy_pr > 1e-6:
            print(f"  >> Rust drift: dx={dx_pr:.10f} dy={dy_pr:.10f}")

        # Stop early if someone died or won
        any_terminal = (py_n.state in (6, 7, 8) or orig_n.state in (6, 7, 8) or rust_n.state in (6, 7, 8))
        if any_terminal and frame_idx > start_frame + 5:
            print(f"\n  (Terminal state reached, stopping)")
            break


def diagnose_segments(filename):
    """Compare tile segments between Python and Rust sims for a given attract file."""
    fpath = ATTRACTS_DIR / filename
    decoder = NppAttractDecoder()
    decoded = decoder.decode_npp_attract_file(str(fpath))
    map_data = decoder.create_nclone_map(str(fpath))
    map_list = list(map_data)

    py_sim = NcloneSimulator(SimConfig(basic_sim=True, enable_anim=False))
    py_sim.load(list(map_list))

    # Count Python segments
    py_seg_count = 0
    for cell, segs in py_sim.segment_dic.items():
        py_seg_count += len(segs)

    print(f"\nPython sim segment count: {py_seg_count}")

    # Check ninja spawn
    print(f"Python ninja spawn: ({py_sim.ninja.xpos}, {py_sim.ninja.ypos})")

    rust_sim = nclone_native.Simulator.load(map_list)
    rust_state = rust_sim.get_state()
    print(f"Rust ninja spawn: ({rust_state.ninja.xpos}, {rust_state.ninja.ypos})")

    # Print Python segments near the ninja spawn
    nx = int(py_sim.ninja.xpos // 24)
    ny = int(py_sim.ninja.ypos // 24)
    print(f"\nPython segments near spawn cell ({nx},{ny}):")
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            cell = (nx + dx, ny + dy)
            if cell in py_sim.segment_dic:
                segs = py_sim.segment_dic[cell]
                if segs:
                    for s in segs:
                        stype = type(s).__name__
                        if hasattr(s, 'x1'):
                            print(f"  cell {cell}: {stype} ({s.x1:.1f},{s.y1:.1f})->({s.x2:.1f},{s.y2:.1f}) active={s.active}")
                        elif hasattr(s, 'xpos'):
                            print(f"  cell {cell}: {stype} center=({s.xpos:.1f},{s.ypos:.1f}) active={s.active}")


if __name__ == "__main__":
    # Most common divergence: frame 8 floor collision failure
    # Files 10300645 and 10300651 both diverge at frame 8
    target = sys.argv[1] if len(sys.argv) > 1 else "10300645"
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nframes = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    diagnose_segments(target)
    diagnose_file(target, start_frame=start, num_frames=nframes)
