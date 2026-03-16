#!/usr/bin/env python3
"""Diagnose why d_max_exit = 0.0 on the basics level surface graph.

Runs 6 checks to pinpoint whether the failure is:
  H1: No surface points near the exit door (goal node identification failure)
  H2: Goal nodes exist but have zero incoming edges (edge enumeration failure)
  H3: Exit only reachable while airborne (structural limitation)

Usage:
    python scripts/diagnose_surface_graph.py
    python scripts/diagnose_surface_graph.py --map test-single-level/basics \
        --npz value_functions/basics_surface_pbrs.npz \
        --replay datasets/000basics/20260223_164858_basics.replay
"""

import argparse
import math
import os
import sys
from collections import defaultdict, deque
from typing import List, Tuple, Optional

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _banner(text: str) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}")


def _section(text: str) -> None:
    print(f"\n--- {text} ---")


def load_surface_pbrs(npz_path: str):
    """Load surface graph and d* arrays from .npz file."""
    from nclone.value_function.surface_extraction import SurfacePoint
    from nclone.value_function.surface_graph import build_graph

    data = np.load(npz_path, allow_pickle=True)
    n = len(data["surface_points_x"])
    surface_points = [
        SurfacePoint(
            x=float(data["surface_points_x"][i]),
            y=float(data["surface_points_y"][i]),
            normal_x=float(data["surface_normals_x"][i]),
            normal_y=float(data["surface_normals_y"][i]),
            surface_type=int(data["surface_types"][i]),
            segment_id=int(data["surface_segment_ids"][i]),
            is_edge=bool(data["surface_is_edge"][i]),
        )
        for i in range(n)
    ]
    graph = build_graph(surface_points)
    if "edges_src" in data:
        graph.edges_src = data["edges_src"]
        graph.edges_dst = data["edges_dst"]
        graph.edges_weight = data["edges_weight"]
    d_star_switch = data["d_star_switch"].astype(np.float32)
    d_star_exit = data["d_star_exit"].astype(np.float32)
    meta = data["metadata"].item() if "metadata" in data else {}
    print(f"Loaded surface graph from: {npz_path}")
    print(f"  Surface points: {n:,}  |  Graph nodes: {graph.n_nodes:,}  "
          f"|  Edges: {len(graph.edges_src):,}")
    return graph, d_star_switch, d_star_exit, meta


def check1_goal_node_identification(graph, exit_pos, switch_pos):
    """Check 1: Verify exit goal nodes exist and are well-placed."""
    from nclone.value_function.surface_graph import (
        find_goal_nodes, find_nearest_point,
    )
    from nclone.constants.physics_constants import (
        NINJA_RADIUS, EXIT_SWITCH_RADIUS, EXIT_DOOR_RADIUS,
    )

    _section("Check 1: Goal Node Identification")

    goal_nodes_switch = find_goal_nodes(graph, switch_pos[0], switch_pos[1], "switch")
    goal_nodes_exit = find_goal_nodes(graph, exit_pos[0], exit_pos[1], "exit")

    print(f"Switch position:  ({switch_pos[0]:.1f}, {switch_pos[1]:.1f})")
    print(f"Exit position:    ({exit_pos[0]:.1f}, {exit_pos[1]:.1f})")
    print(f"Switch contact radius: {NINJA_RADIUS + EXIT_SWITCH_RADIUS:.1f} px")
    print(f"Exit contact radius:   {NINJA_RADIUS + EXIT_DOOR_RADIUS:.1f} px")
    print(f"Goal nodes (switch): {len(goal_nodes_switch)}")
    print(f"Goal nodes (exit):   {len(goal_nodes_exit)}")

    if goal_nodes_exit:
        print(f"\nExit goal surface points (unique):")
        max_bins = graph.max_vel_bins
        seen_pts = set()
        for nid in goal_nodes_exit:
            pt_idx = nid // max_bins
            if pt_idx not in seen_pts:
                seen_pts.add(pt_idx)
                sp = graph.surface_points[pt_idx]
                stype = "floor" if sp.surface_type == 0 else "wall"
                d = math.sqrt((sp.x - exit_pos[0])**2 + (sp.y - exit_pos[1])**2)
                print(f"  pt[{pt_idx}] ({sp.x:.1f}, {sp.y:.1f}) type={stype} "
                      f"dist_to_exit={d:.1f}px is_edge={sp.is_edge}")
    else:
        print("\n  [!] NO exit goal nodes found -- H1 confirmed or contact range too small")

    # Also find the nearest surface point to exit regardless of contact range
    nearest_idx = find_nearest_point(graph, exit_pos[0], exit_pos[1])
    if nearest_idx is not None:
        sp = graph.surface_points[nearest_idx]
        d = math.sqrt((sp.x - exit_pos[0])**2 + (sp.y - exit_pos[1])**2)
        stype = "floor" if sp.surface_type == 0 else "wall"
        print(f"\nNearest surface point to exit (ignoring range):")
        print(f"  pt[{nearest_idx}] ({sp.x:.1f}, {sp.y:.1f}) type={stype} "
              f"dist={d:.1f}px is_edge={sp.is_edge}")
    else:
        print("\n  [!] No surface point found anywhere near exit!")

    return goal_nodes_switch, goal_nodes_exit


def check2_edge_connectivity(graph, goal_nodes_exit, goal_nodes_switch):
    """Check 2: How many edges have exit goal nodes as source/destination."""
    _section("Check 2: Incoming/Outgoing Edge Count for Exit Goal Nodes")

    if not goal_nodes_exit:
        print("  [!] No exit goal nodes -- skipping edge check")
        return

    exit_goal_set = set(int(n) for n in goal_nodes_exit)
    switch_goal_set = set(int(n) for n in goal_nodes_switch)

    edges_src = graph.edges_src
    edges_dst = graph.edges_dst
    edges_weight = graph.edges_weight

    incoming = []
    outgoing = []
    for i in range(len(edges_src)):
        s, d, w = int(edges_src[i]), int(edges_dst[i]), float(edges_weight[i])
        if d in exit_goal_set:
            incoming.append((s, d, w))
        if s in exit_goal_set:
            outgoing.append((s, d, w))

    print(f"Total edges in graph: {len(edges_src):,}")
    print(f"Incoming to exit goal nodes: {len(incoming)}")
    print(f"Outgoing from exit goal nodes: {len(outgoing)}")

    if len(incoming) == 0:
        print("\n  [!] ZERO incoming edges to exit goal nodes -- H2 confirmed")
        print("  Edge enumeration never landed at a surface node near the exit.")
    else:
        print(f"\n  Sample incoming edges (first 5):")
        for s, d, w in incoming[:5]:
            max_bins = graph.max_vel_bins
            src_pt = s // max_bins
            sp = graph.surface_points[src_pt]
            print(f"    node {s} ({sp.x:.1f},{sp.y:.1f}) -> exit node {d} [{w:.1f} frames]")

    if outgoing:
        print(f"\n  Sample outgoing edges (first 5):")
        for s, d, w in outgoing[:5]:
            max_bins = graph.max_vel_bins
            dst_pt = d // max_bins
            dp = graph.surface_points[dst_pt]
            print(f"    exit node {s} -> node {d} ({dp.x:.1f},{dp.y:.1f}) [{w:.1f} frames]")

    return incoming, outgoing


def check3_nearest_surface_distance(graph, exit_pos, switch_pos):
    """Check 3: Distance from nearest surface point to exit and switch."""
    from nclone.value_function.surface_graph import find_nearest_point
    from nclone.value_function.surface_extraction import SURFACE_FLOOR, SURFACE_WALL

    _section("Check 3: Nearest Surface Point Distances")

    for label, pos in [("switch", switch_pos), ("exit", exit_pos)]:
        print(f"\n{label.capitalize()} position: ({pos[0]:.1f}, {pos[1]:.1f})")
        for stype, sname in [(None, "any"), (SURFACE_FLOOR, "floor"), (SURFACE_WALL, "wall")]:
            idx = find_nearest_point(graph, pos[0], pos[1], stype)
            if idx is not None:
                sp = graph.surface_points[idx]
                d = math.sqrt((sp.x - pos[0])**2 + (sp.y - pos[1])**2)
                actual_type = "floor" if sp.surface_type == SURFACE_FLOOR else "wall"
                print(f"  Nearest {sname} surface: ({sp.x:.1f},{sp.y:.1f}) "
                      f"type={actual_type} dist={d:.1f}px edge={sp.is_edge}")
            else:
                print(f"  Nearest {sname} surface: NONE FOUND")


def check4_replay_transitions(replay_path: str, graph, exit_pos, switch_pos):
    """Check 4: Replay a demo and find every surface-to-surface landing."""
    from nclone.value_function.surface_graph import find_nearest_point, find_nearest_node
    from nclone.replay.gameplay_recorder import CompactReplay
    from nclone.replay.replay_executor import decode_input_to_controls

    _section("Check 4: Replay Demo -- Surface-to-Surface Transitions")

    if not os.path.exists(replay_path):
        print(f"  [!] Replay not found: {replay_path}")
        return

    try:
        import nclone_native
    except ImportError:
        print("  [!] nclone_native not available -- skipping replay check")
        return

    with open(replay_path, "rb") as f:
        data = f.read()
    replay = CompactReplay.from_binary(data, episode_id="diagnostic")
    print(f"Loaded replay: {replay_path}")
    print(f"  Map data: {len(replay.map_data)} bytes")
    print(f"  Input sequence: {len(replay.input_sequence)} frames")

    sim = nclone_native.Simulator.load(list(replay.map_data))

    # State codes from nclone-native/src/ninja.rs:
    #   0=Immobile (floor), 1=Running (floor), 2=GroundSlide (floor)
    #   3=Jumping (air), 4=Falling (air), 5=WallSliding
    #   6=Dead, 7=AwaitingDeath
    FLOOR_STATES = {0, 1, 2}
    WALL_STATES = {5}
    AIR_STATES = {3, 4}
    SURFACE_STATES = FLOOR_STATES | WALL_STATES
    DEATH_STATES = {6, 7}

    def state_label(code):
        names = {0: "immobile", 1: "running", 2: "groundslide",
                 3: "jumping", 4: "falling", 5: "wallslide",
                 6: "dead", 7: "awaiting_death"}
        return names.get(code, f"state_{code}")

    transitions = []
    exit_area_frames = []
    prev_in_air = False
    prev_contact_type = None
    won = False

    exit_threshold = 50.0
    switch_threshold = 30.0

    for frame_idx, input_byte in enumerate(replay.input_sequence):
        h, j = decode_input_to_controls(input_byte)
        sim.tick(h, j)

        state_code = sim.get_ninja_state_code()
        if state_code in DEATH_STATES:
            print(f"  [!] Ninja died at frame {frame_idx}")
            break

        pos = sim.get_ninja_pos()
        vel = sim.get_ninja_velocity()
        x, y, vx, vy = pos[0], pos[1], vel[0], vel[1]

        if sim.has_won():
            won = True
            print(f"  Ninja WON at frame {frame_idx}, pos=({x:.1f},{y:.1f})")
            break

        in_air = state_code in AIR_STATES
        on_surface = state_code in SURFACE_STATES

        # Detect surface-to-surface landing (was airborne, now on surface)
        if prev_in_air and on_surface:
            contact_type = "floor" if state_code in FLOOR_STATES else "wall"
            nearest_pt = find_nearest_point(graph, x, y,
                                            0 if contact_type == "floor" else 1)
            if nearest_pt is not None:
                sp = graph.surface_points[nearest_pt]
                dist_to_node = math.sqrt((sp.x - x)**2 + (sp.y - y)**2)
            else:
                dist_to_node = float("inf")

            transitions.append({
                "frame": frame_idx,
                "x": x, "y": y, "vx": vx, "vy": vy,
                "contact": contact_type,
                "nearest_pt": nearest_pt,
                "dist_to_node": dist_to_node,
            })

        prev_in_air = in_air

        # Track frames near exit
        dist_exit = math.sqrt((x - exit_pos[0])**2 + (y - exit_pos[1])**2)
        if dist_exit < exit_threshold:
            exit_area_frames.append({
                "frame": frame_idx,
                "x": x, "y": y, "vx": vx, "vy": vy,
                "state": state_label(state_code),
                "dist_to_exit": dist_exit,
            })

    if not won:
        print("  [!] Replay did not result in a win -- replay may be invalid")

    print(f"\nTotal surface-to-surface landings: {len(transitions)}")
    for t in transitions:
        flag = ""
        if t["nearest_pt"] is None:
            flag = " [!] NO SURFACE NODE FOUND"
        elif t["dist_to_node"] > 20:
            flag = f" [!] FAR FROM GRAPH ({t['dist_to_node']:.1f}px)"
        print(f"  frame={t['frame']:4d} pos=({t['x']:6.1f},{t['y']:6.1f}) "
              f"vel=({t['vx']:5.2f},{t['vy']:5.2f}) {t['contact']}{flag}")

    print(f"\nFrames within {exit_threshold}px of exit: {len(exit_area_frames)}")
    for f in exit_area_frames:
        print(f"  frame={f['frame']:4d} pos=({f['x']:6.1f},{f['y']:6.1f}) "
              f"vel=({f['vx']:5.2f},{f['vy']:5.2f}) state={f['state']} "
              f"dist_to_exit={f['dist_to_exit']:.1f}px")

    return transitions, exit_area_frames


def check5_exit_approach_detail(replay_path: str, exit_pos, switch_pos):
    """Check 5: Detailed frame-by-frame trace as ninja approaches the exit."""
    from nclone.replay.gameplay_recorder import CompactReplay
    from nclone.replay.replay_executor import decode_input_to_controls

    _section("Check 5: Detailed Exit Approach Trace")

    if not os.path.exists(replay_path):
        print(f"  [!] Replay not found: {replay_path}")
        return

    try:
        import nclone_native
    except ImportError:
        print("  [!] nclone_native not available -- skipping")
        return

    with open(replay_path, "rb") as f:
        data = f.read()
    replay = CompactReplay.from_binary(data, episode_id="diagnostic")

    sim = nclone_native.Simulator.load(list(replay.map_data))

    state_names = {
        0: "immobile", 1: "running", 2: "groundslide",
        3: "jumping",  4: "falling",  5: "wallslide",
        6: "dead",     7: "awaiting_death"
    }

    CLOSE_THRESHOLD = 120.0
    last_far_frame = None

    all_states = []
    for frame_idx, input_byte in enumerate(replay.input_sequence):
        h, j = decode_input_to_controls(input_byte)
        sim.tick(h, j)

        if sim.get_ninja_state_code() in {6, 7}:
            break

        pos = sim.get_ninja_pos()
        vel = sim.get_ninja_velocity()
        x, y, vx, vy = pos[0], pos[1], vel[0], vel[1]
        sc = sim.get_ninja_state_code()
        dist_exit = math.sqrt((x - exit_pos[0])**2 + (y - exit_pos[1])**2)
        dist_switch = math.sqrt((x - switch_pos[0])**2 + (y - switch_pos[1])**2)
        won = sim.has_won()

        all_states.append((frame_idx, x, y, vx, vy, sc, dist_exit, dist_switch, won))

        if won:
            break

    # Find the first frame within CLOSE_THRESHOLD of exit, then show from 20 frames before
    first_close = None
    for i, (frame, x, y, vx, vy, sc, dist_exit, dist_switch, won) in enumerate(all_states):
        if dist_exit < CLOSE_THRESHOLD:
            first_close = max(0, i - 20)
            break

    if first_close is None:
        print(f"  [!] Ninja never came within {CLOSE_THRESHOLD}px of exit in this replay")
        # Show last 30 frames anyway
        first_close = max(0, len(all_states) - 30)

    print(f"\nDetailed trace (starting {CLOSE_THRESHOLD}px approach to exit):")
    print(f"  {'frame':>5}  {'x':>7}  {'y':>7}  {'vx':>6}  {'vy':>6}  "
          f"{'state':<12}  {'d_exit':>7}  {'won':>4}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}  "
          f"{'-'*12}  {'-'*7}  {'-'*4}")

    for frame, x, y, vx, vy, sc, dist_exit, dist_switch, won in all_states[first_close:]:
        sname = state_names.get(sc, f"state{sc}")
        print(f"  {frame:5d}  {x:7.1f}  {y:7.1f}  {vx:6.3f}  {vy:6.3f}  "
              f"{sname:<12}  {dist_exit:7.1f}  {'YES' if won else '':>4}")
        if won:
            break


def check6_connectivity(graph, goal_nodes_exit, goal_nodes_switch):
    """Check 6: Connected components analysis."""
    _section("Check 6: Graph Connectivity Analysis")

    edges_src = graph.edges_src
    edges_dst = graph.edges_dst
    n_nodes = graph.n_nodes

    # Build undirected adjacency for connectivity analysis
    adj = defaultdict(set)
    for i in range(len(edges_src)):
        s, d = int(edges_src[i]), int(edges_dst[i])
        adj[s].add(d)
        adj[d].add(s)

    # Find connected components via BFS
    visited = {}  # node -> component_id
    comp_id = 0
    components = []
    all_nodes_with_edges = set(adj.keys())

    for start in all_nodes_with_edges:
        if start in visited:
            continue
        # BFS
        component = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited[node] = comp_id
            component.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(component)
        comp_id += 1

    # Sort by size descending
    components.sort(key=len, reverse=True)

    print(f"Nodes with at least one edge: {len(all_nodes_with_edges):,} / {n_nodes:,}")
    print(f"Connected components: {len(components)}")
    print(f"Largest component size: {len(components[0]):,} nodes" if components else "  (none)")

    if len(components) > 1:
        print(f"\nComponent size distribution (top 10):")
        for i, comp in enumerate(components[:10]):
            print(f"  Component {i}: {len(comp):,} nodes")

    # Check if switch and exit goal nodes are in the same component
    if goal_nodes_switch and goal_nodes_exit:
        switch_comp = visited.get(int(goal_nodes_switch[0]))
        exit_comp = visited.get(int(goal_nodes_exit[0]))

        if switch_comp is None:
            print(f"\nSwitch goal node {goal_nodes_switch[0]} has NO edges (isolated)")
        else:
            print(f"\nSwitch goal nodes in component: {switch_comp} "
                  f"(size {len(components[switch_comp]):,})")

        if exit_comp is None:
            print(f"Exit goal node {goal_nodes_exit[0]} has NO edges (isolated)")
            print("  [!] Exit goal nodes are completely disconnected from the edge graph!")
        else:
            print(f"Exit goal nodes in component: {exit_comp} "
                  f"(size {len(components[exit_comp]):,})")

        same_component = (switch_comp is not None and exit_comp is not None
                          and switch_comp == exit_comp)
        print(f"\nSwitch and exit in same connected component: "
              f"{'YES' if same_component else 'NO [!]'}")

        if not same_component:
            print("\n  This confirms the exit is in a disconnected or isolated component.")
            print("  Backward Dijkstra from exit can't reach switch nodes => d_max_exit = 0")

    # Check isolation of exit goal nodes specifically
    if goal_nodes_exit:
        exit_goal_set = set(int(n) for n in goal_nodes_exit)
        isolated_exit_nodes = [n for n in exit_goal_set if n not in adj]
        has_edges_exit_nodes = [n for n in exit_goal_set if n in adj]
        print(f"\nExit goal nodes with zero edges: {len(isolated_exit_nodes)} / {len(exit_goal_set)}")
        print(f"Exit goal nodes with edges: {len(has_edges_exit_nodes)} / {len(exit_goal_set)}")


def print_verdict(goal_nodes_exit):
    """Print final verdict."""
    _section("VERDICT")

    if not goal_nodes_exit:
        print("H1: Surface extraction did not place any nodes within contact range of the exit.")
        print("    The exit position may be floating (airborne) or surface points are too sparse.")
    else:
        print("H2 or H3: Exit goal nodes exist but are unreachable from the rest of the graph.")
        print("  - H2 if exit goal nodes have ZERO incoming edges: edge enumeration never")
        print("       simulated a trajectory that landed at a surface node near the exit.")
        print("  - H3 if exit goal nodes exist and have edges but are in a disconnected")
        print("       component: the graph is fragmented and the exit island can't be reached.")
        print("  See Checks 2 and 6 above for which applies.")


def main():
    parser = argparse.ArgumentParser(description="Diagnose surface graph exit failure")
    parser.add_argument("--map", default="test-single-level/basics",
                        help="Path to level .map file (default: test-single-level/basics)")
    parser.add_argument("--npz", default="value_functions/basics_surface_pbrs.npz",
                        help="Path to surface_pbrs.npz "
                             "(default: value_functions/basics_surface_pbrs.npz)")
    parser.add_argument("--replay", default="datasets/000basics/20260223_164858_basics.replay",
                        help="Path to a known-good replay file")
    args = parser.parse_args()

    _banner("DIAGNOSIS: basics surface graph (d_max_exit = 0.0)")

    # Load graph
    graph, d_star_switch, d_star_exit, meta = load_surface_pbrs(args.npz)
    print(f"  d_max_switch: {meta.get('d_max_switch', '?')}")
    print(f"  d_max_exit:   {meta.get('d_max_exit', '?')}")

    finite_sw = np.sum(np.isfinite(d_star_switch))
    finite_ex = np.sum(np.isfinite(d_star_exit))
    print(f"  Finite switch nodes: {finite_sw:,} / {graph.n_nodes:,}")
    print(f"  Finite exit nodes:   {finite_ex:,} / {graph.n_nodes:,}")

    # Load level to get goal positions
    from nclone.value_function.physics_step_wrapper import PhysicsStepWrapper
    pw = PhysicsStepWrapper(map_path=args.map)
    exit_pos = pw.get_goal_position("exit")
    switch_pos = pw.get_goal_position("switch")

    # Run all checks
    goal_nodes_switch, goal_nodes_exit = check1_goal_node_identification(
        graph, exit_pos, switch_pos)

    check2_edge_connectivity(graph, goal_nodes_exit, goal_nodes_switch)

    check3_nearest_surface_distance(graph, exit_pos, switch_pos)

    check4_replay_transitions(args.replay, graph, exit_pos, switch_pos)

    check5_exit_approach_detail(args.replay, exit_pos, switch_pos)

    check6_connectivity(graph, goal_nodes_exit, goal_nodes_switch)

    print_verdict(goal_nodes_exit)

    _banner("DIAGNOSIS COMPLETE")


if __name__ == "__main__":
    main()
