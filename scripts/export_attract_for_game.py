#!/usr/bin/env python3
"""
Export an attract file into three artifacts for game ground-truth comparison:

1. Level file  -- raw game-format binary that can be placed in the N++ user
                  levels directory so the human can open it in-game.
2. inputs.json -- the per-frame input sequence decoded from the demo section.
3. sim_trace.json -- per-frame ninja state from the nclone Python simulation,
                     used as the reference to compare against the game.

Usage:
    python scripts/export_attract_for_game.py <attract_file> [output_dir]

    attract_file : path to an attract file (e.g. ../attracts_filtered/10131738)
    output_dir   : where to write the three output files (default: ./attract_export/)

Outputs (all written to output_dir):
    <attract_id>.lvl        -- game-native level file (raw map bytes at file[8:])
    <attract_id>_nclone.lvl -- nclone-reconstructed level (from create_nclone_map)
    inputs.json             -- [{frame, input_byte, hor, jump}, ...]
    sim_trace.json          -- [{frame, xpos, ypos, xspeed, yspeed, state,
                                 airborn, walled, jump_duration,
                                 switch_collected, mine_deaths_possible}, ...]
    attract_info.json       -- metadata (spawn, entity counts, total frames, ...)

The level file to try first in the game is <attract_id>.lvl (raw game-native format).
If the game doesn't recognise it, try <attract_id>_nclone.lvl.
"""

import sys
import json
import struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nclone.replay.npp_attract_decoder import NppAttractDecoder
from nclone.nsim import Simulator as NcloneSimulator
from nclone.sim_config import SimConfig

NINJA_STATE_NAMES = {
    0: "idle", 1: "running", 2: "ground_sliding", 3: "jumping",
    4: "falling", 5: "wall_sliding", 6: "dead", 7: "awaiting_death",
    8: "celebrating", 9: "disabled",
}


def decode_input(inp: int):
    jump = 1 if (inp & 0x01) else 0
    right = 1 if (inp & 0x02) else 0
    left = 1 if (inp & 0x04) else 0
    hor = 0 if (left and right) else (-1 if left else (1 if right else 0))
    return hor, jump


def extract_raw_game_map(attract_path: str) -> bytes:
    """
    Build a game-native level file from an attract file.

    Attract file layout:
        [0:4]   uint32 LE  map_data_length
        [4:8]   uint32 LE  demo_data_length
        [8:]    raw_map section (map_data_length bytes)

    Game level file layout (matches test-single-level format):
        [0:4]   uint32 LE  mode  (0 = standard)
        [4:8]   uint32 LE  total_file_size  (= 8 + map_data_length)
        [8:]    raw_map section  (identical to attract_file[8:8+map_data_length])

    So tiles are at game_level[184:1150] = raw_map[176:1142]
                                         = attract_file[184:1150]  ✓
    Ninja spawn at game_level[1231:1233] = attract_file[1231:1233]  ✓
    """
    with open(attract_path, "rb") as f:
        data = f.read()
    map_len = struct.unpack("<I", data[0:4])[0]
    raw_map = data[8:8 + map_len]
    total_size = 8 + map_len
    header = struct.pack("<II", 0, total_size)   # mode=0, file_size
    return header + raw_map


def run_nclone_simulation(map_data: bytes, demo_inputs: list) -> list:
    """
    Run the nclone Python simulation with the given map and inputs.
    Returns a list of per-frame state dicts.
    """
    sc = SimConfig(basic_sim=True, enable_anim=False)
    sim = NcloneSimulator(sc)
    sim.load(list(map_data))

    # Collect initial entity references for tracking
    mines = sim.entity_dic.get(1, [])
    # Type 3 holds both EntityExit and EntityExitSwitch in this codebase
    type3 = sim.entity_dic.get(3, [])
    exit_switch_entities = [e for e in type3 if "Switch" in type(e).__name__]
    type4 = sim.entity_dic.get(4, [])
    exit_switch_entities += type4  # include type-4 switches too

    frames = []
    for frame_idx, inp in enumerate(demo_inputs):
        frame = frame_idx + 1
        hor, jump = decode_input(inp)
        sim.tick(hor, jump)

        n = sim.ninja
        switch_collected = any(
            not getattr(e, "active", True) for e in exit_switch_entities
        )
        active_mines = sum(
            1 for m in mines if getattr(m, "state", 0) == 1
        )
        frames.append({
            "frame": frame,
            "xpos": n.xpos,
            "ypos": n.ypos,
            "xspeed": n.xspeed,
            "yspeed": n.yspeed,
            "state": n.state,
            "state_name": NINJA_STATE_NAMES.get(n.state, str(n.state)),
            "airborn": n.airborn,
            "walled": n.walled,
            "jump_duration": n.jump_duration,
            "switch_collected": switch_collected,
            "active_mines": active_mines,
            "hor_input": hor,
            "jump_input": jump,
            "input_byte": inp,
        })

        if n.state in (6, 7, 8, 9):
            break  # terminal state reached

    return frames


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    attract_path = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path("attract_export")
    output_dir.mkdir(parents=True, exist_ok=True)

    attract_id = attract_path.name
    print(f"Attract file : {attract_path}")
    print(f"Output dir   : {output_dir}")

    decoder = NppAttractDecoder()

    # ------------------------------------------------------------------ #
    # 1.  Decode attract file
    # ------------------------------------------------------------------ #
    decoded = decoder.decode_npp_attract_file(str(attract_path))
    demo_inputs = decoded["demo_data"]
    if not demo_inputs:
        print("ERROR: No demo inputs decoded -- cannot proceed.")
        sys.exit(1)

    print(f"\nDecoded: {len(demo_inputs)} input frames")
    print(f"Spawn   : {decoded['ninja_spawn']} "
          f"(pixel {decoded['ninja_spawn'][0]*6}, {decoded['ninja_spawn'][1]*6})")
    print(f"Entities: {len(decoded['entities'])} total")

    entity_counts = {}
    for e in decoded["entities"]:
        entity_counts[e["type"]] = entity_counts.get(e["type"], 0) + 1
    for t, c in sorted(entity_counts.items()):
        type_names = {1: "toggle_mine", 2: "gold", 3: "exit_door", 4: "exit_switch"}
        print(f"  type {t} ({type_names.get(t, '?')}): {c}")

    # ------------------------------------------------------------------ #
    # 2.  Write game-native level file  (data[8 : 8+map_len])
    # ------------------------------------------------------------------ #
    raw_map = extract_raw_game_map(str(attract_path))
    raw_lvl_path = output_dir / f"{attract_id}.lvl"
    raw_lvl_path.write_bytes(raw_map)
    print(f"\nWrote game-native level  : {raw_lvl_path}  ({len(raw_map)} bytes)")

    # ------------------------------------------------------------------ #
    # 3.  Write nclone-reconstructed level file
    # ------------------------------------------------------------------ #
    nclone_map = decoder.create_nclone_map(str(attract_path))
    nclone_lvl_path = output_dir / f"{attract_id}_nclone.lvl"
    nclone_lvl_path.write_bytes(nclone_map)
    print(f"Wrote nclone-format level: {nclone_lvl_path}  ({len(nclone_map)} bytes)")

    # ------------------------------------------------------------------ #
    # 4.  Write inputs.json
    # ------------------------------------------------------------------ #
    inputs_data = []
    for i, inp in enumerate(demo_inputs):
        hor, jump = decode_input(inp)
        inputs_data.append({
            "frame": i + 1,
            "input_byte": inp,
            "hor": hor,
            "jump": jump,
        })
    inputs_path = output_dir / "inputs.json"
    inputs_path.write_text(json.dumps(inputs_data, indent=2))
    print(f"Wrote inputs             : {inputs_path}  ({len(demo_inputs)} frames)")

    # ------------------------------------------------------------------ #
    # 5.  Run nclone sim and write sim_trace.json
    # ------------------------------------------------------------------ #
    print("\nRunning nclone simulation...")
    frames = run_nclone_simulation(nclone_map, demo_inputs)
    terminal = frames[-1]
    print(f"Terminal state: {terminal['state_name']} at frame {terminal['frame']}")

    sim_trace_path = output_dir / "sim_trace.json"
    sim_trace_path.write_text(json.dumps(frames, indent=2))
    print(f"Wrote sim_trace          : {sim_trace_path}  ({len(frames)} frames)")

    # ------------------------------------------------------------------ #
    # 6.  Write attract_info.json
    # ------------------------------------------------------------------ #
    spawn_x_raw, spawn_y_raw = decoded["ninja_spawn"]
    info = {
        "attract_id": attract_id,
        "attract_path": str(attract_path),
        "total_input_frames": len(demo_inputs),
        "sim_terminal_frame": terminal["frame"],
        "sim_terminal_state": terminal["state"],
        "sim_terminal_state_name": terminal["state_name"],
        "sim_outcome": "win" if terminal["state"] == 8 else (
            "dead" if terminal["state"] in (6, 7) else "playing"
        ),
        "ninja_spawn_raw": [spawn_x_raw, spawn_y_raw],
        "ninja_spawn_pixels": [spawn_x_raw * 6, spawn_y_raw * 6],
        "entity_counts": entity_counts,
        "raw_map_bytes": len(raw_map),
        "nclone_map_bytes": len(nclone_map),
        # CE helper: expected initial xpos/ypos as doubles for memory scanning
        "initial_xpos": float(spawn_x_raw * 6),
        "initial_ypos": float(spawn_y_raw * 6),
    }
    info_path = output_dir / "attract_info.json"
    info_path.write_text(json.dumps(info, indent=2))
    print(f"Wrote attract_info       : {info_path}")

    # ------------------------------------------------------------------ #
    # 7.  Print usage instructions
    # ------------------------------------------------------------------ #
    print(f"""
==========================================================================
NEXT STEPS
==========================================================================

1. Copy the level file to the N++ user levels directory:
       {raw_lvl_path}
   → C:\\Users\\Nick\\Documents\\Metanet\\N++\\levels\\{attract_id}.lvl

   If the game doesn't load it, try the nclone-format version instead:
       {nclone_lvl_path}
   → C:\\Users\\Nick\\Documents\\Metanet\\N++\\levels\\{attract_id}_nclone.lvl

2. Open the level in N++. Check that the spawn position matches:
       pixel ({spawn_x_raw * 6}, {spawn_y_raw * 6})

3. Open Cheat Engine, attach to N++.exe, load and run:
       scripts/ce_record_and_playback.lua

   The script will inject the inputs from inputs.json and record
   the ninja's position each frame to game_trace.csv.

4. After the run completes, compare the traces:
       python scripts/compare_traces.py \\
           {sim_trace_path} \\
           <game_trace.csv>

Simulation outcome: {info['sim_outcome']} at frame {info['sim_terminal_frame']}
==========================================================================
""")


if __name__ == "__main__":
    main()
