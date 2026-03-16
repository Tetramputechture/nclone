#!/usr/bin/env python3
"""
Deep per-frame comparison of a single attract file across three simulators:
  - nclone-original (reference Python)
  - nclone (current Python)
  - nclone_native (Rust)

Usage: python scripts/compare_single_attract.py [attract_file_path]
"""

import sys
from pathlib import Path

ATTRACT_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/tetra/projects/attracts_filtered/1002581")

# Patch sys.argv before importing nclone-original (uses argparse at module level)
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0], "--basic-sim"]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "nclone-original"))
import nsim as nsim_original
sys.argv = _original_argv

# Import nclone Python simulator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nclone.replay.npp_attract_decoder import NppAttractDecoder
from nclone.nsim import Simulator as NcloneSimulator
from nclone.sim_config import SimConfig

# Import Rust simulator
try:
    import nclone_native
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    print("WARNING: nclone_native not available. Build with: cd nclone-native && maturin develop --release")


NINJA_STATE_NAMES = {
    0: "idle", 1: "running", 2: "ground_sliding", 3: "jumping",
    4: "falling", 5: "wall_sliding", 6: "dead", 7: "awaiting_death",
    8: "celebrating", 9: "disabled",
}

THRESH = 1e-9


def decode_input(inp):
    jump = 1 if (inp & 0x01) else 0
    right = 1 if (inp & 0x02) else 0
    left = 1 if (inp & 0x04) else 0
    hor = 0 if (left and right) else (-1 if left else (1 if right else 0))
    return hor, jump


def state_name(s):
    return NINJA_STATE_NAMES.get(s, f"unknown({s})")


def ninja_snap_orig(sim):
    n = sim.ninja
    return dict(xpos=n.xpos, ypos=n.ypos, xspeed=n.xspeed, yspeed=n.yspeed,
                state=n.state, airborn=n.airborn, walled=n.walled, jump_duration=n.jump_duration)


def ninja_snap_nclone(sim):
    n = sim.ninja
    return dict(xpos=n.xpos, ypos=n.ypos, xspeed=n.xspeed, yspeed=n.yspeed,
                state=n.state, airborn=n.airborn, walled=n.walled, jump_duration=n.jump_duration)


def ninja_snap_rust(sim):
    s = sim.get_state().ninja
    return dict(xpos=s.xpos, ypos=s.ypos, xspeed=s.xspeed, yspeed=s.yspeed,
                state=s.state, airborn=s.airborn, walled=s.walled, jump_duration=s.jump_duration)


def snap_diff_flags(a, b):
    flags = []
    if abs(a["xpos"] - b["xpos"]) > THRESH or abs(a["ypos"] - b["ypos"]) > THRESH:
        flags.append("pos")
    if abs(a["xspeed"] - b["xspeed"]) > THRESH or abs(a["yspeed"] - b["yspeed"]) > THRESH:
        flags.append("vel")
    if a["state"] != b["state"]:
        flags.append("state")
    if a["airborn"] != b["airborn"]:
        flags.append("airborn")
    if a["walled"] != b["walled"]:
        flags.append("walled")
    if a["jump_duration"] != b["jump_duration"]:
        flags.append("jump_dur")
    return flags


def print_snap(label, snap):
    print(f"  {label:<10}: pos=({snap['xpos']:.10f}, {snap['ypos']:.10f}) "
          f"vel=({snap['xspeed']:.10f}, {snap['yspeed']:.10f}) "
          f"state={state_name(snap['state'])} airborn={snap['airborn']} "
          f"walled={snap['walled']} jdur={snap['jump_duration']}")


def compare_pair(name_a, name_b, snaps_a, snaps_b, max_print=20):
    """Compare two lists of per-frame snapshots. Returns (first_diff_frame, total_diffs)."""
    diffs_found = 0
    first_diff_frame = None

    for fi, (a, b) in enumerate(zip(snaps_a, snaps_b)):
        frame = fi + 1
        flags = snap_diff_flags(a, b)
        if flags:
            diffs_found += 1
            if first_diff_frame is None:
                first_diff_frame = frame
            if diffs_found <= max_print:
                print(f"  Frame {frame:5d}  DIFF {name_a} vs {name_b}: {', '.join(flags)}")
                print_snap(name_a, a)
                print_snap(name_b, b)
                if "pos" in flags:
                    dx = b["xpos"] - a["xpos"]
                    dy = b["ypos"] - a["ypos"]
                    print(f"            delta: dx={dx:.2e}  dy={dy:.2e}")
                print()
            elif diffs_found == max_print + 1:
                print(f"  (further {name_a} vs {name_b} diffs suppressed)\n")

    return first_diff_frame, diffs_found


def main():
    fpath = ATTRACT_FILE.resolve()
    print(f"Attract file: {fpath}")

    decoder = NppAttractDecoder()
    decoded = decoder.decode_npp_attract_file(str(fpath))
    demo_inputs = decoded["demo_data"]
    map_data = decoder.create_nclone_map(str(fpath))

    print(f"Decode successful: {len(demo_inputs)} input frames, {len(map_data)} map bytes\n")

    # Initialise simulators
    orig_sim = nsim_original.Simulator()
    orig_sim.load(map_data)

    sc = SimConfig(basic_sim=True, enable_anim=False)
    nclone_sim = NcloneSimulator(sc)
    nclone_sim.load(list(map_data))

    rust_sim = None
    if HAS_RUST:
        rust_sim = nclone_native.Simulator.load(list(map_data))

    sim_names = ["original", "nclone"] + (["rust"] if HAS_RUST else [])
    print(f"Simulators: {', '.join(sim_names)}\n")
    print("Running frame-by-frame simulation...\n")

    orig_snaps, nclone_snaps, rust_snaps = [], [], []
    orig_terminal = nclone_terminal = rust_terminal = None

    for fi, inp in enumerate(demo_inputs):
        hor, jump = decode_input(inp)
        frame = fi + 1

        orig_sim.tick(hor, jump)
        nclone_sim.tick(hor, jump)
        if rust_sim:
            rust_sim.tick(hor, jump)

        os_ = ninja_snap_orig(orig_sim)
        ns_ = ninja_snap_nclone(nclone_sim)
        orig_snaps.append(os_)
        nclone_snaps.append(ns_)

        if rust_sim:
            rs_ = ninja_snap_rust(rust_sim)
            rust_snaps.append(rs_)

        if orig_terminal is None and os_["state"] in (6, 7, 8, 9):
            orig_terminal = (frame, os_["state"])
        if nclone_terminal is None and ns_["state"] in (6, 7, 8, 9):
            nclone_terminal = (frame, ns_["state"])
        if rust_sim and rust_terminal is None and rs_["state"] in (6, 7, 8, 9):
            rust_terminal = (frame, rs_["state"])

    # Fall back to final state
    if orig_terminal is None:
        orig_terminal = (len(demo_inputs), orig_snaps[-1]["state"])
    if nclone_terminal is None:
        nclone_terminal = (len(demo_inputs), nclone_snaps[-1]["state"])
    if HAS_RUST and rust_terminal is None:
        rust_terminal = (len(demo_inputs), rust_snaps[-1]["state"])

    # --- Pairwise comparisons ---
    print("=" * 70)
    print("PAIRWISE FRAME COMPARISONS")
    print("=" * 70 + "\n")

    pairs = [("original", "nclone", orig_snaps, nclone_snaps)]
    if HAS_RUST:
        pairs.append(("original", "rust", orig_snaps, rust_snaps))
        pairs.append(("nclone",   "rust", nclone_snaps, rust_snaps))

    pair_results = {}
    for name_a, name_b, snaps_a, snaps_b in pairs:
        print(f"--- {name_a} vs {name_b} ---")
        first_diff, total_diffs = compare_pair(name_a, name_b, snaps_a, snaps_b)
        pair_results[(name_a, name_b)] = (first_diff, total_diffs)
        if total_diffs == 0:
            print(f"  IDENTICAL across all {len(snaps_a)} frames\n")

    # --- Summary ---
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total input frames: {len(demo_inputs)}\n")

    terminals = {"original": orig_terminal, "nclone": nclone_terminal}
    if HAS_RUST:
        terminals["rust"] = rust_terminal

    for sname, (frame, sstate) in terminals.items():
        print(f"  {sname:<12}: {state_name(sstate)} at frame {frame}")

    print()
    for (na, nb), (first_diff, total_diffs) in pair_results.items():
        if total_diffs == 0:
            print(f"  {na} vs {nb}: IDENTICAL")
        else:
            ta, tb = terminals[na], terminals[nb]
            outcome_match = ta[1] == tb[1]
            print(f"  {na} vs {nb}: first diff at frame {first_diff}, "
                  f"{total_diffs} differing frames, "
                  f"outcomes {'MATCH' if outcome_match else 'DIVERGE'} "
                  f"({state_name(ta[1])}@{ta[0]} vs {state_name(tb[1])}@{tb[0]})")


if __name__ == "__main__":
    main()
