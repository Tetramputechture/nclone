#!/usr/bin/env python3
"""
Pinpoint the exact method call that causes the first divergence
between nclone-original and nclone on a single attract file.

Instruments every sub-step of the physics tick and compares state after each.
"""

import sys
import math
from pathlib import Path

sys.argv = [sys.argv[0], "--basic-sim"]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "nclone-original"))
import nsim as nsim_original

sys.argv = [sys.argv[0]]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nclone.replay.npp_attract_decoder import NppAttractDecoder
from nclone.nsim import Simulator as NcloneSim
from nclone.sim_config import SimConfig

ATTRACTS_DIR = Path(__file__).resolve().parent.parent.parent / "attracts_filtered"
FILE_NAME = sys.argv[1] if len(sys.argv) > 1 else "10285366"


def decode_input(inp):
    jump = 1 if (inp & 0x01) else 0
    right = 1 if (inp & 0x02) else 0
    left = 1 if (inp & 0x04) else 0
    if left and right:
        hor = 0
    elif left:
        hor = -1
    elif right:
        hor = 1
    else:
        hor = 0
    return hor, jump


def ninja_state_vec(ninja):
    """Extract full ninja state as a dict for comparison."""
    d = {
        "xpos": ninja.xpos,
        "ypos": ninja.ypos,
        "xspeed": ninja.xspeed,
        "yspeed": ninja.yspeed,
        "state": ninja.state,
        "applied_gravity": ninja.applied_gravity,
    }
    for attr in [
        "xpos_old", "ypos_old", "applied_friction",
        "floor_count", "floor_normal_x", "floor_normal_y",
        "ceiling_count", "ceiling_normal_x", "ceiling_normal_y",
        "x_crush", "y_crush", "crush_len",
        "airborn", "walled", "wall_normal", "jump_input_old",
    ]:
        if hasattr(ninja, attr):
            d[attr] = getattr(ninja, attr)
    return d


def compare_states(orig_s, nclone_s, label=""):
    """Compare two state dicts, return list of differences."""
    diffs = []
    for key in orig_s:
        if key not in nclone_s:
            continue
        ov = orig_s[key]
        nv = nclone_s[key]
        if isinstance(ov, float):
            if ov != nv:
                diffs.append((key, ov, nv, nv - ov))
        elif ov != nv:
            diffs.append((key, ov, nv, None))
    return diffs


def main():
    fpath = ATTRACTS_DIR / FILE_NAME
    if not fpath.exists():
        print(f"File not found: {fpath}")
        return

    decoder = NppAttractDecoder()
    decoded = decoder.decode_npp_attract_file(str(fpath))
    demo = decoded["demo_data"]
    map_data = decoder.create_nclone_map(str(fpath))

    print(f"File: {FILE_NAME}, {len(demo)} frames")

    # Create both simulators
    orig = nsim_original.Simulator()
    orig.load(map_data)

    sc = SimConfig(basic_sim=True, enable_anim=False)
    nclone = NcloneSim(sc)
    nclone.load(list(map_data))

    # Compare initial state
    os0 = ninja_state_vec(orig.ninja)
    ns0 = ninja_state_vec(nclone.ninja)
    diffs = compare_states(os0, ns0, "initial")
    if diffs:
        print(f"INITIAL STATE DIFFERS:")
        for d in diffs:
            print(f"  {d[0]}: orig={d[1]} nclone={d[2]} delta={d[3]}")
    else:
        print("Initial state: IDENTICAL")

    # For each frame, run step-by-step and compare
    for frame_idx, inp in enumerate(demo):
        hor, jump = decode_input(inp)
        frame_num = frame_idx + 1

        # Store inputs
        orig.ninja.hor_input = hor
        orig.ninja.jump_input = jump
        nclone.ninja.hor_input = hor
        nclone.ninja.jump_input = jump

        # --- Step 1: Entity move ---
        for elist in orig.entity_dic.values():
            for entity in elist:
                if entity.is_movable and entity.active:
                    entity.move()
        for elist in nclone.entity_dic.values():
            for entity in elist:
                if entity.is_movable and entity.active:
                    entity.move()

        # --- Step 2: Entity think ---
        for elist in orig.entity_dic.values():
            for entity in elist:
                if entity.is_thinkable and entity.active:
                    entity.think()
        for elist in nclone.entity_dic.values():
            for entity in elist:
                if entity.is_thinkable and entity.active:
                    entity.think()

        if orig.ninja.state == 9 or nclone.ninja.state == 9:
            break

        # Get the actual ninja objects to use
        orig_ninja = orig.ninja if orig.ninja.state != 6 else orig.ninja.ragdoll
        nclone_ninja = nclone.ninja if nclone.ninja.state != 6 else nclone.ninja.ragdoll

        # --- Step 3: integrate ---
        orig_ninja.integrate()
        nclone_ninja.integrate()

        os1 = ninja_state_vec(orig_ninja)
        ns1 = ninja_state_vec(nclone_ninja)
        diffs = compare_states(os1, ns1)
        if diffs:
            print(f"\nFrame {frame_num} DIVERGENCE after integrate():")
            print(f"  input: hor={hor} jump={jump}")
            for d in diffs:
                print(f"  {d[0]}: orig={repr(d[1])} nclone={repr(d[2])} delta={d[3]}")
            # Print pre-integrate state too
            print(f"  (pre-integrate state was identical up to this point)")
            return

        # --- Step 4: pre_collision ---
        orig_ninja.pre_collision()
        nclone_ninja.pre_collision()

        os2 = ninja_state_vec(orig_ninja)
        ns2 = ninja_state_vec(nclone_ninja)
        diffs = compare_states(os2, ns2)
        if diffs:
            print(f"\nFrame {frame_num} DIVERGENCE after pre_collision():")
            print(f"  input: hor={hor} jump={jump}")
            for d in diffs:
                print(f"  {d[0]}: orig={repr(d[1])} nclone={repr(d[2])} delta={d[3]}")
            return

        # --- Step 5: 4x collide_vs_objects + collide_vs_tiles ---
        for collision_iter in range(4):
            orig_ninja.collide_vs_objects()
            nclone_ninja.collide_vs_objects()

            os3 = ninja_state_vec(orig_ninja)
            ns3 = ninja_state_vec(nclone_ninja)
            diffs = compare_states(os3, ns3)
            if diffs:
                print(f"\nFrame {frame_num} DIVERGENCE after collide_vs_objects() iter {collision_iter}:")
                print(f"  input: hor={hor} jump={jump}")
                for d in diffs:
                    print(f"  {d[0]}: orig={repr(d[1])} nclone={repr(d[2])} delta={d[3]}")
                return

            orig_ninja.collide_vs_tiles()
            nclone_ninja.collide_vs_tiles()

            os4 = ninja_state_vec(orig_ninja)
            ns4 = ninja_state_vec(nclone_ninja)
            diffs = compare_states(os4, ns4)
            if diffs:
                print(f"\nFrame {frame_num} DIVERGENCE after collide_vs_tiles() iter {collision_iter}:")
                print(f"  input: hor={hor} jump={jump}")
                for d in diffs:
                    print(f"  {d[0]}: orig={repr(d[1])} nclone={repr(d[2])} delta={d[3]}")
                return

        # --- Step 6: post_collision ---
        orig_ninja.post_collision()
        nclone_ninja.post_collision()

        os5 = ninja_state_vec(orig_ninja)
        ns5 = ninja_state_vec(nclone_ninja)
        diffs = compare_states(os5, ns5)
        if diffs:
            print(f"\nFrame {frame_num} DIVERGENCE after post_collision():")
            print(f"  input: hor={hor} jump={jump}")
            for d in diffs:
                print(f"  {d[0]}: orig={repr(d[1])} nclone={repr(d[2])} delta={d[3]}")
            return

        # --- Step 7: think ---
        # Capture pre-think state to debug conditional path differences
        os_pre_think = ninja_state_vec(orig.ninja)
        ns_pre_think = ninja_state_vec(nclone.ninja)

        orig.ninja.think()
        nclone.ninja.think()

        os6 = ninja_state_vec(orig.ninja)
        ns6 = ninja_state_vec(nclone.ninja)
        diffs = compare_states(os6, ns6)
        if diffs:
            print(f"\nFrame {frame_num} DIVERGENCE after think():")
            print(f"  input: hor={hor} jump={jump}")
            for d in diffs:
                print(f"  {d[0]}: orig={repr(d[1])} nclone={repr(d[2])} delta={d[3]}")
            print(f"\n  PRE-THINK state comparison:")
            pre_diffs = compare_states(os_pre_think, ns_pre_think)
            if pre_diffs:
                for d in pre_diffs:
                    print(f"    {d[0]}: orig={repr(d[1])} nclone={repr(d[2])} delta={d[3]}")
            else:
                print(f"    IDENTICAL (same state going into think)")
            print(f"\n  Key pre-think values (orig):")
            for k in ["state", "airborn", "walled", "wall_normal", "xspeed", "yspeed",
                       "xpos", "ypos", "floor_normalized_x", "floor_normalized_y",
                       "applied_gravity", "applied_friction", "applied_drag",
                       "floor_count", "ceiling_count"]:
                ov = getattr(orig.ninja, k, "N/A")
                nv = getattr(nclone.ninja, k, "N/A") if hasattr(nclone.ninja, k) else "N/A"
                marker = " <-- DIFF" if ov != nv else ""
                print(f"    {k}: orig={repr(ov)} nclone={repr(nv)}{marker}")
            return

        # --- Step 8: update_graphics (original only has this) ---
        orig.ninja.update_graphics()
        # nclone skips this in basic_sim/no-anim mode

        # --- Step 9: Increment frame ---
        orig.frame += 1
        nclone.frame += 1

        # Log
        orig.ninja.log()
        for elist in orig.entity_dic.values():
            for entity in elist:
                entity.log_position()

        if orig.ninja.state in (6, 7, 8, 9) or nclone.ninja.state in (6, 7, 8, 9):
            print(f"Frame {frame_num}: orig_state={orig.ninja.state} nclone_state={nclone.ninja.state}")
            break

    print(f"\nCompleted {frame_num} frames with NO divergence detected")
    print(f"  orig: state={orig.ninja.state} pos=({orig.ninja.xpos}, {orig.ninja.ypos})")
    print(f"  nclone: state={nclone.ninja.state} pos=({nclone.ninja.xpos}, {nclone.ninja.ypos})")


if __name__ == "__main__":
    main()
