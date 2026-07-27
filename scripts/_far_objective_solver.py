"""Offline round-trip 'route-and-solve' solver for FAR_OBJECTIVE (loop) levels.

Produces a PHYSICS-VALID winning action sequence (spawn -> switch -> ACTIVATE ->
exit -> WIN), validated by replaying it on the native sim and confirming the
episode terminates with player_won.

Design (two-stage round-trip):
  Stage A: discover a switch-activating prefix. We reuse the switch-goal RRT from
    StartStateCurriculum.generate (free-space-guided over tick_physics_mines_only),
    then replay each prefix on FULL native `tick` and keep the ones that actually
    consume the exit switch (entity-4 active flips True->False). Pick the survivor
    with the smallest free-space cost-to-go to the EXIT.
  Stage B: replay prefix A on full `tick` to reach the POST-SWITCH sim state
    (door now open), snapshot it with save_full_state, and run a second RRT that
    STARTS FROM THAT STATE and expands via full `tick` (so the opened-exit/
    activated-switch entity state persists), guided by free-space cost-to-go to the
    EXIT. Stop as soon as a node WINS.
  Concatenate A+B. VALIDATE by replaying A+B from a fresh fast_reset on full
    physics and confirming has_won(). Discard non-winning candidates.

The whole solver only ever asserts a win via full-physics replay, so a returned
demo is a true win certificate (no silent non-winning demos).
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

from nclone.value_function.free_space_distance import FreeSpaceDistanceSource
from nclone.gym_environment.start_state_curriculum import StartStateCurriculum, ACT

_LEVEL_W_PX = 44 * 24.0
_LEVEL_H_PX = 25 * 24.0
_CELLG = 12.0


def _full_replay_win(sim, actions: List[int]) -> Tuple[bool, bool, int]:
    """Replay `actions` from a fresh fast_reset on full physics.

    Returns (won, died, frames_used). Stops early on win/death."""
    sim.fast_reset()
    for i, a in enumerate(actions):
        hin, jin = ACT.get(int(a), (0, 0))
        sim.tick(hin, jin)
        if sim.has_won():
            return True, False, i + 1
        if sim.has_died():
            return False, True, i + 1
    return sim.has_won(), sim.has_died(), len(actions)


def _switch_consumed(sim) -> bool:
    return any(e.entity_type == 4 and not e.active for e in sim.get_state().entities)


def _best_switch_prefix(
    map_data: List[int],
    fs: FreeSpaceDistanceSource,
    *,
    n_states: int,
    max_iter: int,
    kext: int,
    seed: int,
    log,
) -> Optional[Tuple[List[int], Tuple[float, float]]]:
    """Stage A. Return (prefix, post_switch_xy) for the switch-consuming prefix
    with the smallest free-space exit cost-to-go, or None."""
    import nclone_native

    switch_cur = StartStateCurriculum.generate(
        map_data, goal_kind="switch", n_states=n_states,
        max_iter=max_iter, kext=kext, seed=seed, log=log,
    )
    if not switch_cur.states:
        log("  [A] switch RRT found no states")
        return None

    sim = nclone_native.Simulator.load(list(map_data))
    best = None
    best_cost = 1e18
    n_consuming = 0
    for prefix, _ in switch_cur.states:
        sim.fast_reset()
        ok = True
        for a in prefix:
            hin, jin = ACT.get(int(a), (0, 0))
            sim.tick(hin, jin)
            if sim.has_died():
                ok = False
                break
            if sim.has_won():
                break
        if not ok or not _switch_consumed(sim):
            continue
        n_consuming += 1
        px, py = sim.get_ninja_pos()
        c = fs.cost_to_go(px, py, True)
        if not math.isfinite(c):
            c = fs._d_max_exit
        if c < best_cost:
            best_cost = c
            best = (list(prefix), (px, py))
    log(f"  [A] {n_consuming}/{len(switch_cur.states)} switch-consuming prefixes; "
        f"best exit-cost {best_cost:.0f}")
    return best


def _stage_b_to_exit(
    map_data: List[int],
    fs: FreeSpaceDistanceSource,
    prefix_a: List[int],
    *,
    max_iter: int,
    kext: int,
    seed: int,
    log,
) -> Optional[List[int]]:
    """Stage B. RRT from the POST-SWITCH state to the exit via full `tick`.

    Returns the Stage-B action suffix (NOT including prefix_a) that wins, or None.
    """
    import nclone_native

    sim = nclone_native.Simulator.load(list(map_data))

    # Replay prefix A on full physics to reach the post-switch state.
    sim.fast_reset()
    for a in prefix_a:
        hin, jin = ACT.get(int(a), (0, 0))
        sim.tick(hin, jin)
        if sim.has_died():
            log("  [B] prefix A died on replay (unexpected)")
            return None
        if sim.has_won():
            # Already won during stage A (switch and exit coincident). Empty suffix.
            return []
    if not _switch_consumed(sim):
        log("  [B] switch not consumed after prefix A replay")
        return None

    start_state = sim.save_full_state()
    sx, sy = sim.get_ninja_pos()

    exit_xy = next(((float(e.x), float(e.y))
                    for e in sim.get_state().entities if e.entity_type == 3), None)
    if exit_xy is None:
        log("  [B] no exit entity")
        return None

    rng = random.Random(seed + 7)

    def progress(x, y):
        c = fs.cost_to_go(x, y, True)
        return c if math.isfinite(c) else fs._d_max_exit

    # node: [full_state_bytes, x, y, parent, action]
    nodes = [[start_state, sx, sy, -1, None]]
    buckets = {(int(sx / _CELLG), int(sy / _CELLG)): [0]}
    closest_ni, closest_d = 0, 1e18
    win_ni = None

    for it in range(max_iter):
        r = rng.random()
        if r < 0.3:
            fx, fy = nodes[closest_ni][1], nodes[closest_ni][2]
            tx = fx + rng.gauss(0, 45) + (exit_xy[0] - fx) * 0.4
            ty = fy + rng.gauss(0, 45) + (exit_xy[1] - fy) * 0.4
        elif r < 0.55:
            tx, ty = exit_xy
        else:
            tx, ty = rng.uniform(0, _LEVEL_W_PX), rng.uniform(0, _LEVEL_H_PX)
        bx, by = int(tx / _CELLG), int(ty / _CELLG)
        best, bd = None, 1e18
        for rr in range(0, 7):
            for dx in range(-rr, rr + 1):
                for dy in (range(-rr, rr + 1) if abs(dx) == rr else (-rr, rr)):
                    for ni in buckets.get((bx + dx, by + dy), ()):
                        d = (nodes[ni][1] - tx) ** 2 + (nodes[ni][2] - ty) ** 2
                        if d < bd:
                            bd, best = d, ni
            if best is not None and rr >= 2:
                break
        if best is None:
            continue
        base = nodes[best][0]
        ba, bdd, bstate, bxy, bwon = None, 1e18, None, None, False
        for a, (hin, jin) in ACT.items():
            sim.restore_full_state(base)
            dead = won = False
            for _ in range(kext):
                sim.tick(hin, jin)
                if sim.has_won():
                    won = True
                    break
                if sim.has_died():
                    dead = True
                    break
            if dead:
                continue
            px, py = sim.get_ninja_pos()
            d = (px - tx) ** 2 + (py - ty) ** 2
            if won:
                # Prefer a winning expansion immediately.
                ba, bstate, bxy, bwon = a, sim.save_full_state(), (px, py), True
                break
            if d < bdd:
                bdd, ba, bstate, bxy = d, a, sim.save_full_state(), (px, py)
        if ba is None:
            continue
        nid = len(nodes)
        nodes.append([bstate, bxy[0], bxy[1], best, ba])
        buckets.setdefault((int(bxy[0] / _CELLG), int(bxy[1] / _CELLG)), []).append(nid)
        gd = (bxy[0] - exit_xy[0]) ** 2 + (bxy[1] - exit_xy[1]) ** 2
        if gd < closest_d:
            closest_d, closest_ni = gd, nid
        if bwon:
            win_ni = nid
            log(f"  [B] WON at it={it}, {len(nodes)} nodes")
            break
        if it % 50000 == 0 and it:
            log(f"  [B] explore {it} it, {len(nodes)} nodes, "
                f"closest {math.sqrt(closest_d):.0f}px (exit prog {progress(*exit_xy):.0f})")

    if win_ni is None:
        log(f"  [B] no win after {max_iter} it; closest {math.sqrt(closest_d):.0f}px")
        return None

    # Reconstruct the stage-B suffix.
    seq = []
    ni = win_ni
    while nodes[ni][3] != -1:
        seq.append(nodes[ni][4])
        ni = nodes[ni][3]
    seq.reverse()
    suffix = []
    for a in seq:
        suffix.extend([a] * kext)
    return suffix


def solve_far_objective(
    map_data: List[int],
    *,
    n_states: int = 300,
    a_max_iter: int = 120000,
    b_max_iter: int = 200000,
    kext: int = 4,
    seed: int = 0,
    n_seeds: int = 3,
    log=lambda m: None,
) -> Optional[dict]:
    """Full round-trip solve. Returns a dict with the validated winning demo, or
    None if no validated win was found.

    Tries up to ``n_seeds`` RRT seeds (the switch-touch and exit contacts are
    precise, so a single seed can be unlucky on an otherwise solvable level);
    returns the first seed that yields a full-physics-validated win.

    Returned dict:
      {actions: List[int], prefix_a_len, suffix_b_len, frames, seed, won: True}
    """
    import nclone_native

    fs = FreeSpaceDistanceSource(list(map_data))
    if not fs.available:
        log("  free-space source unavailable")
        return None

    sim = nclone_native.Simulator.load(list(map_data))
    for s in range(seed, seed + max(1, n_seeds)):
        a = _best_switch_prefix(map_data, fs, n_states=n_states, max_iter=a_max_iter,
                                kext=kext, seed=s, log=log)
        if a is None:
            continue
        prefix_a, _ = a

        suffix_b = _stage_b_to_exit(map_data, fs, prefix_a, max_iter=b_max_iter,
                                    kext=kext, seed=s, log=log)
        if suffix_b is None:
            continue

        actions = list(prefix_a) + list(suffix_b)

        # Mandatory: validate the concatenated sequence wins from spawn on full
        # physics. A returned demo is always a true win certificate.
        won, died, frames = _full_replay_win(sim, actions)
        if not won:
            log(f"  [V] seed {s} concatenated replay did NOT win "
                f"(died={died}, frames={frames})")
            continue
        actions = actions[:frames]  # truncate to the exact winning frame
        log(f"  [V] VALIDATED win (seed {s}) in {frames} frames "
            f"(A={len(prefix_a)} + B={len(suffix_b)})")
        return {
            "actions": [int(x) for x in actions],
            "prefix_a_len": len(prefix_a),
            "suffix_b_len": len(suffix_b),
            "frames": frames,
            "seed": s,
            "won": True,
        }
    return None
