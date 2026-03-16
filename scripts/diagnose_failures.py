#!/usr/bin/env python3
"""
Sample 500 attract files, run only the Python (nclone) simulation,
find failures (runs that don't end with state=8 celebrating/win),
then do verbose frame-by-frame diagnostics on the first failure.
"""

import sys
import random
import struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nclone.replay.npp_attract_decoder import NppAttractDecoder
from nclone.nsim import Simulator as NcloneSimulator
from nclone.sim_config import SimConfig

ATTRACTS_DIR = Path(__file__).resolve().parent.parent.parent / "attracts_filtered"
SAMPLE_SIZE = 500
MAX_FRAMES = 20000
RANDOM_SEED = 42

NINJA_STATE_NAMES = {
    0: "idle", 1: "running", 2: "ground_sliding", 3: "jumping",
    4: "falling", 5: "wall_sliding", 6: "dead", 7: "awaiting_death",
    8: "celebrating", 9: "disabled",
}


def state_name(s):
    return NINJA_STATE_NAMES.get(s, f"unknown({s})")


def decode_input(inp):
    jump = 1 if (inp & 0x01) else 0
    right = 1 if (inp & 0x02) else 0
    left = 1 if (inp & 0x04) else 0
    hor = 0 if (left and right) else (-1 if left else (1 if right else 0))
    return hor, jump


def run_sim(map_data, demo_inputs):
    """Run nclone sim. Returns (outcome, terminal_frame, final_state)."""
    sc = SimConfig(basic_sim=True, enable_anim=False)
    sim = NcloneSimulator(sc)
    sim.load(list(map_data))
    for frame_idx, inp in enumerate(demo_inputs[:MAX_FRAMES]):
        hor, jump = decode_input(inp)
        sim.tick(hor, jump)
        state = sim.ninja.state
        if state == 8:
            return "win", frame_idx + 1, state, sim
        if state in (6, 7):
            return "dead", frame_idx + 1, state, sim
    final_state = sim.ninja.state
    if final_state == 8:
        return "win", len(demo_inputs), final_state, sim
    return "playing", len(demo_inputs), final_state, sim


def verbose_diagnose(fpath, decoder):
    """Run the sim frame-by-frame and print detailed diagnostics around the failure."""
    print(f"\n{'='*70}")
    print(f"VERBOSE DIAGNOSIS: {fpath.name}")
    print(f"{'='*70}")

    # Decode raw file for inspection
    with open(fpath, "rb") as f:
        raw = f.read()
    map_data_len = struct.unpack("<I", raw[0:4])[0]
    demo_data_len = struct.unpack("<I", raw[4:8])[0]
    print(f"Raw file: {len(raw)} bytes | map_data_length={map_data_len} demo_data_length={demo_data_len}")

    decoded = decoder.decode_npp_attract_file(str(fpath))
    demo_inputs = decoded["demo_data"]
    map_data = decoder.create_nclone_map(str(fpath))

    print(f"Decoded: {len(demo_inputs)} input frames, {len(map_data)} map bytes")
    print(f"Ninja spawn: {decoded['ninja_spawn']}")
    print(f"Entities ({len(decoded['entities'])}):")
    for e in decoded["entities"]:
        print(f"  type={e['type']} x={e['x']} y={e['y']} orient={e['orientation']} mode={e['mode']}")

    # Show raw map bytes in the entity section
    print(f"\nMap bytes [1150..1160] (entity section header):")
    print(f"  {list(map_data[1150:1170])}")
    print(f"Map bytes [1228..1250] (around ninja spawn / first entities):")
    print(f"  {list(map_data[1228:1260])}")

    # Run sim, recording per-frame state
    sc = SimConfig(basic_sim=True, enable_anim=False)
    sim = NcloneSimulator(sc)
    sim.load(list(map_data))

    print(f"\nEntities loaded into sim:")
    for etype, elist in sim.entity_dic.items():
        if elist:
            for e in elist:
                pos = getattr(e, 'xpos', '?'), getattr(e, 'ypos', '?')
                active = getattr(e, 'active', '?')
                print(f"  type={etype} class={type(e).__name__} pos={pos} active={active}")

    print(f"\nNinja spawn pos: ({sim.ninja.xpos:.4f}, {sim.ninja.ypos:.4f})")
    print(f"\nRunning {len(demo_inputs)} frames...")

    terminal_frame = None
    terminal_state = None
    prev_state = sim.ninja.state

    # Track entity state changes
    mine_entities = sim.entity_dic.get(1, [])  # Toggle mines
    exit_switch_entities = sim.entity_dic.get(4, [])
    exit_entities = sim.entity_dic.get(3, [])
    gold_entities = sim.entity_dic.get(2, [])

    print(f"  Mines: {len(mine_entities)}, Exit switches: {len(exit_switch_entities)}, "
          f"Exits: {len(exit_entities)}, Gold: {len(gold_entities)}")

    # Track significant events
    events = []

    for frame_idx, inp in enumerate(demo_inputs[:MAX_FRAMES]):
        frame = frame_idx + 1
        hor, jump = decode_input(inp)
        sim.tick(hor, jump)
        state = sim.ninja.state

        # Track state changes
        if state != prev_state:
            events.append((frame, "state_change", prev_state, state,
                          sim.ninja.xpos, sim.ninja.ypos))
            prev_state = state

        # Track entity activations
        for i, e in enumerate(exit_switch_entities):
            if not getattr(e, 'active', True):
                # switch collected
                pass

        if state == 8:
            terminal_frame = frame
            terminal_state = state
            break
        if state in (6, 7):
            terminal_frame = frame
            terminal_state = state
            break

    if terminal_frame is None:
        terminal_frame = len(demo_inputs)
        terminal_state = sim.ninja.state

    print(f"\nTerminal: {state_name(terminal_state)} at frame {terminal_frame}")

    print(f"\nState timeline:")
    for ev in events:
        frame, kind, s0, s1, x, y = ev
        print(f"  frame {frame:5d}: {state_name(s0)} -> {state_name(s1)}  pos=({x:.3f},{y:.3f})")

    # If it failed (not win), re-run and do frame-by-frame around the death
    if terminal_state != 8:
        print(f"\n--- FAILURE DETAILS: replaying around death frame ---")
        sc2 = SimConfig(basic_sim=True, enable_anim=False)
        sim2 = NcloneSimulator(sc2)
        sim2.load(list(map_data))

        death_frame = terminal_frame
        print_start = max(0, death_frame - 15)
        print_end = death_frame + 3

        for frame_idx, inp in enumerate(demo_inputs[:print_end]):
            frame = frame_idx + 1
            hor, jump = decode_input(inp)
            sim2.tick(hor, jump)
            if frame >= print_start:
                n = sim2.ninja
                mine_states = [getattr(e, 'state', '?') for e in mine_entities]
                switch_active = [getattr(e, 'active', '?') for e in exit_switch_entities]
                exit_active = [getattr(e, 'active', '?') for e in exit_entities]
                print(f"  f{frame:5d} inp={inp}(h={hor:+d},j={jump}) "
                      f"pos=({n.xpos:.4f},{n.ypos:.4f}) "
                      f"vel=({n.xspeed:.6f},{n.yspeed:.6f}) "
                      f"state={state_name(n.state)} "
                      f"air={n.airborn} wall={n.walled} "
                      f"switch_active={switch_active} "
                      f"exit_active={exit_active} "
                      f"mines={mine_states}")
            if sim2.ninja.state in (6, 7, 8):
                break

        # Check if the switch was ever collected
        print(f"\nPost-run entity states:")
        for etype, elist in sim2.entity_dic.items():
            for e in elist:
                print(f"  type={etype} class={type(e).__name__} active={getattr(e,'active','?')} "
                      f"pos=({getattr(e,'xpos','?')}, {getattr(e,'ypos','?')})")


def main():
    all_files = sorted(ATTRACTS_DIR.iterdir())
    print(f"Total files: {len(all_files)}")

    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(all_files, min(SAMPLE_SIZE, len(all_files)))
    sample.sort()
    print(f"Sampling {len(sample)} files (seed={RANDOM_SEED})\n")

    decoder = NppAttractDecoder()

    results = {"win": [], "dead": [], "playing": [], "error": []}
    first_failure = None

    for i, fpath in enumerate(sample):
        if i % 50 == 0:
            print(f"  [{i}/{len(sample)}] win={len(results['win'])} "
                  f"dead={len(results['dead'])} playing={len(results['playing'])} "
                  f"error={len(results['error'])}")
        try:
            decoded = decoder.decode_npp_attract_file(str(fpath))
            demo_inputs = decoded["demo_data"]
            if not demo_inputs:
                results["error"].append((fpath.name, "no demo inputs"))
                continue
            map_data = decoder.create_nclone_map(str(fpath))
        except Exception as e:
            results["error"].append((fpath.name, f"decode: {e}"))
            continue

        try:
            outcome, frame, state, sim = run_sim(map_data, demo_inputs)
            results[outcome].append((fpath.name, frame, state))
            if outcome != "win" and first_failure is None:
                first_failure = fpath
        except Exception as e:
            results["error"].append((fpath.name, f"sim: {e}"))
            if first_failure is None:
                first_failure = fpath

    print(f"\n{'='*60}")
    print(f"RESULTS across {len(sample)} sampled files:")
    print(f"{'='*60}")
    print(f"  WIN:     {len(results['win'])}")
    print(f"  DEAD:    {len(results['dead'])}")
    print(f"  PLAYING: {len(results['playing'])}")
    print(f"  ERROR:   {len(results['error'])}")

    win_rate = len(results["win"]) / len(sample) * 100
    print(f"  Success rate: {win_rate:.1f}%")

    if results["dead"]:
        print(f"\nFirst 10 dead cases:")
        for name, frame, state in results["dead"][:10]:
            print(f"  {name}: {state_name(state)} at frame {frame}")

    if results["playing"]:
        print(f"\nFirst 10 playing (no terminal) cases:")
        for name, frame, state in results["playing"][:10]:
            print(f"  {name}: {state_name(state)} at frame {frame}")

    if results["error"]:
        print(f"\nFirst 10 errors:")
        for name, err in results["error"][:10]:
            print(f"  {name}: {err}")

    if first_failure:
        verbose_diagnose(first_failure, decoder)
    else:
        print("\nAll sampled files succeeded!")


if __name__ == "__main__":
    main()
