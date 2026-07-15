"""Start-state curriculum: faithful reachable mid-route start states for any level.

A general capability for the RL pipeline:

* **Generalized training** — sampling episodes from a diversity of reachable
  states (not just spawn) improves exploration and robustness across procgen.
* **Per-level fine-tuning** — a *reverse* curriculum (start near the goal, schedule
  the start back toward spawn) cracks hard levels whose win is unreachable by
  random exploration, by letting the agent learn the final maneuvers first.

Start states are stored as **frame-action prefixes** (lists of action ids 0-5),
NOT engine state snapshots — so they replay faithfully on whatever sim the env
uses (`env.reset(options={'start_state_curriculum': cur, 'start_state_difficulty': d})`
replays the prefix on the env's own physics). This sidesteps both the native↔Python
parity gap and the entity-mutation gap in `save/restore_ninja_state`.

Prefixes are discovered by a fast kinodynamic RRT over the native
``tick_physics_mines_only`` (tile + mine-death, NO entity mutation → ninja-only
save/restore round-trips, so the exploration is faithful and O(1) per node). Each
prefix is tagged with the free-space cost-to-go to the goal ("progress"); the
curriculum samples by a ``difficulty`` knob (0 = nearest the goal, 1 = the full
range). On replay the env truncates at the last surviving frame, so a prefix is
always a valid start state even if it drifts on a different engine.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..value_function.free_space_distance import FreeSpaceDistanceSource

ACT = {0: (0, 0), 1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (1, 1), 5: (-1, 1)}
_LEVEL_W_PX = 44 * 24.0
_LEVEL_H_PX = 25 * 24.0
_CELLG = 12.0


def _exit_guided_tree(map_data, fs, prefix_a, *, max_iter: int, kext: int,
                      seed: int, log=lambda m: None):
    """Exit-guided RRT from the POST-SWITCH state, returning the WHOLE tree.

    Replays ``prefix_a`` on full ``tick`` to reach the post-switch state (switch
    consumed, door open), snapshots it, then expands an RRT toward the exit via
    full ``tick`` so the activated-switch / open-door entity state persists. This
    is the same expansion the demo solver's Stage B uses, but instead of keeping
    only the winning leaf it returns EVERY node — each is a graduated return-leg
    position spanning the switch (full leg) down to the exit. The caller turns
    each node into a ``(prefix_a + node_path, exit_cost_to_go)`` start state, which
    gives the post-switch reverse curriculum the near-exit states it was missing.

    Returns ``(nodes, exit_xy)`` where ``nodes`` is a list of
    ``[full_state, x, y, parent, action]``, or ``(None, None)`` if the
    post-switch state can't be reached (prefix died/won, switch not consumed, no
    exit entity). Does NOT require a win — a partial tree still yields useful
    graduated states on levels the planner can't fully solve."""
    import nclone_native

    sim = nclone_native.Simulator.load(list(map_data))
    sim.fast_reset()
    for a in prefix_a:
        hin, jin = ACT.get(int(a), (0, 0))
        sim.tick(hin, jin)
        if sim.has_died() or sim.has_won():
            return None, None
    if not any(e.entity_type == 4 and not e.active for e in sim.get_state().entities):
        return None, None
    exit_xy = next(((float(e.x), float(e.y))
                    for e in sim.get_state().entities if e.entity_type == 3), None)
    if exit_xy is None:
        return None, None

    start_state = sim.save_full_state()
    sx, sy = sim.get_ninja_pos()
    rng = random.Random(seed + 7)

    # node: [full_state_bytes, x, y, parent, action]
    nodes = [[start_state, sx, sy, -1, None]]
    buckets = {(int(sx / _CELLG), int(sy / _CELLG)): [0]}
    closest_ni, closest_d = 0, 1e18

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
        ba, bdd, bstate, bxy = None, 1e18, None, None
        for a, (hin, jin) in ACT.items():
            sim.restore_full_state(base)
            dead = False
            for _ in range(kext):
                sim.tick(hin, jin)
                if sim.has_won():
                    break
                if sim.has_died():
                    dead = True
                    break
            if dead:
                continue
            px, py = sim.get_ninja_pos()
            d = (px - tx) ** 2 + (py - ty) ** 2
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
        if it % 50000 == 0 and it:
            log(f"  [post-switch tree] {it} it, {len(nodes)} nodes, "
                f"closest {math.sqrt(closest_d):.0f}px")

    return nodes, exit_xy


def _node_prefix(nodes, ni: int, kext: int) -> List[int]:
    """Action prefix (root -> node ``ni``), each tree edge repeated ``kext``x."""
    seq = []
    while nodes[ni][3] != -1:
        seq.append(nodes[ni][4])
        ni = nodes[ni][3]
    seq.reverse()
    out: List[int] = []
    for a in seq:
        out.extend([a] * kext)
    return out


@dataclass
class StartStateCurriculum:
    """Reachable start states (frame-action prefixes) tagged by progress-to-goal.

    ``states`` is a list of ``(action_prefix, progress)`` sorted by progress
    ascending (0 == at the goal). ``max_progress`` normalizes difficulty."""

    states: List[Tuple[List[int], float]] = field(default_factory=list)
    max_progress: float = 1.0
    goal_kind: str = "switch"

    def __len__(self):
        return len(self.states)

    def sample(self, difficulty: float = 1.0, rng: Optional[random.Random] = None,
               band: float = 0.18) -> dict:
        """Reverse-curriculum sample. ``difficulty`` in [0,1] maps to a start
        progress-level of ``difficulty * max_progress``: 0 -> at the goal, 1 -> at
        spawn. We draw from states within ``band`` of that level so the agent
        practises starts at a controlled distance from the goal."""
        if not self.states:
            return {}
        rng = rng or random
        target = max(0.0, min(1.0, difficulty)) * self.max_progress
        half = band * self.max_progress
        cands = [i for i, (_, p) in enumerate(self.states)
                 if target - half <= p <= target + half]
        if not cands:
            cands = [min(range(len(self.states)),
                         key=lambda i: abs(self.states[i][1] - target))]
        idx = rng.choice(cands)
        return {"start_actions": list(self.states[idx][0]), "progress": self.states[idx][1]}

    def save(self, path: str) -> None:
        import json
        json.dump({"max_progress": self.max_progress, "goal_kind": self.goal_kind,
                   "states": [[list(a), p] for a, p in self.states]}, open(path, "w"))

    @classmethod
    def load(cls, path: str) -> "StartStateCurriculum":
        import json
        d = json.load(open(path))
        return cls(states=[(a, p) for a, p in d["states"]],
                   max_progress=d["max_progress"], goal_kind=d["goal_kind"])

    @classmethod
    def generate(cls, map_data, *, goal_kind: str = "switch", n_states: int = 2000,
                 max_iter: int = 300000, kext: int = 4, seed: int = 0,
                 log=lambda m: None) -> "StartStateCurriculum":
        """Explore reachable mid-route states via a faithful RRT and tag by
        free-space cost-to-go. Returns action prefixes (each repeated ``kext`` per
        tree edge). ``goal_kind`` in {"switch","exit"}."""
        import nclone_native
        rng = random.Random(seed)
        fs = FreeSpaceDistanceSource(list(map_data))
        sim = nclone_native.Simulator.load(list(map_data))
        st0 = sim.get_state()
        goal = next(((float(e.x), float(e.y)) for e in st0.entities
                     if e.entity_type == (4 if goal_kind == "switch" else 3)), None)
        if goal is None or not fs.available:
            return cls()

        def progress(x, y):
            # Raw free-space cost-to-go; INFINITE means the goal is unreachable
            # from (x, y) — i.e. the position is outside the goal's reachable
            # region. Such states must NEVER become start states (see the filter
            # in the tagging step below), so we return inf rather than clamping.
            return fs.cost_to_go(x, y, goal_kind == "exit")

        sim.fast_reset()
        x0, y0 = sim.get_ninja_pos()
        # node: [state_bytes, x, y, parent, action]
        nodes = [[sim.save_ninja_state(), x0, y0, -1, None]]
        buckets = {(int(x0 / _CELLG), int(y0 / _CELLG)): [0]}
        closest_ni, closest_d = 0, 1e18
        for it in range(max_iter):
            r = rng.random()
            if r < 0.3:
                fx, fy = nodes[closest_ni][1], nodes[closest_ni][2]
                tx = fx + rng.gauss(0, 45) + (goal[0] - fx) * 0.3
                ty = fy + rng.gauss(0, 45) + (goal[1] - fy) * 0.3
            elif r < 0.5:
                tx, ty = goal
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
            ba, bdd, bstate, bxy = None, 1e18, None, None
            for a, (hin, jin) in ACT.items():
                sim.restore_ninja_state(base)
                dead = False
                for _ in range(kext):
                    sim.tick_physics_mines_only(hin, jin)
                    if sim.has_died():
                        dead = True
                        break
                if dead:
                    continue
                px, py = sim.get_ninja_pos()
                d = (px - tx) ** 2 + (py - ty) ** 2
                if d < bdd:
                    bdd, ba, bstate, bxy = d, a, sim.save_ninja_state(), (px, py)
            if ba is None:
                continue
            nid = len(nodes)
            nodes.append([bstate, bxy[0], bxy[1], best, ba])
            buckets.setdefault((int(bxy[0] / _CELLG), int(bxy[1] / _CELLG)), []).append(nid)
            gd = (bxy[0] - goal[0]) ** 2 + (bxy[1] - goal[1]) ** 2
            if gd < closest_d:
                closest_d, closest_ni = gd, nid
            if it % 50000 == 0:
                log(f"  curriculum explore {it} it, {len(nodes)} nodes, "
                    f"closest {math.sqrt(closest_d):.0f}px")
            if len(nodes) >= max(n_states * 8, 20000):
                break

        def prefix(ni):
            seq = []
            while nodes[ni][3] != -1:
                seq.append(nodes[ni][4])
                ni = nodes[ni][3]
            seq.reverse()
            out = []
            for a in seq:
                out.extend([a] * kext)
            return out

        # REACHABILITY GUARD (RFCL invariant): only keep states from which the
        # goal is still reachable (FINITE free-space cost-to-go). The RRT explores
        # over physics and can reach dead-end pits/ledges from which the switch or
        # exit can no longer be reached; those have infinite cost-to-go. Seeding an
        # episode from such a state moves the "spawn" OUTSIDE the reachable area —
        # producing an unsolvable episode, inf PBRS/goal-distance signal, and the
        # "position far outside graph bounds" log flood. Discard them here so the
        # reverse curriculum can never place the ninja outside the reachable region.
        tagged = []
        for i in range(1, len(nodes)):
            c = progress(nodes[i][1], nodes[i][2])
            if not math.isfinite(c):
                continue
            tagged.append((prefix(i), c))
        n_dropped = (len(nodes) - 1) - len(tagged)
        tagged.sort(key=lambda t: t[1])
        if len(tagged) > n_states:
            idx = np.linspace(0, len(tagged) - 1, n_states).astype(int)
            tagged = [tagged[i] for i in idx]
        max_p = max((p for _, p in tagged), default=1.0) or 1.0
        log(f"  curriculum: {len(tagged)} states (dropped {n_dropped} "
            f"goal-unreachable), progress 0..{max_p:.0f}")
        return cls(states=tagged, max_progress=max_p, goal_kind=goal_kind)

    @classmethod
    def generate_post_switch(cls, map_data, *, n_states: int = 400,
                             max_iter: int = 120000, b_max_iter: int = 40000,
                             kext: int = 4, seed: int = 0,
                             log=lambda m: None) -> "StartStateCurriculum":
        """Build a POST-SWITCH return-leg reverse curriculum.

        The motivating bug: on FAR_OBJECTIVE (loop) levels the agent reaches the
        switch but never completes the return trip to the exit (``exit_given_switch
        == 0``) — the post-switch return policy is barely trained because it
        rarely reaches that state. Seeding episodes *with the switch already
        activated* lets it practise the return leg directly.

        The switch-reaching RRT (:meth:`generate`) explores over
        ``tick_physics_mines_only``, which never mutates entities, so its prefixes
        only reach the switch *position* — they don't activate it. We replay each
        switch-goal prefix on the FULL native physics (``tick``, which DOES process
        the switch/door) and keep the ones that actually consume the exit switch
        (entity-4 ``active`` flips ``True -> False``).

        Those switch-consuming prefixes all END essentially AT the switch, so on
        their own they only populate the curriculum at ~max exit-progress (the
        FULL return leg) — leaving the ``difficulty`` knob with no near-exit states
        to anneal *to*, which is exactly why the reverse curriculum was inert on
        loop levels. So we additionally run an EXIT-GUIDED RRT from the post-switch
        state of the best-connected switch prefix (:func:`_exit_guided_tree`) and
        harvest EVERY node of that tree as a graduated start state, tagged by its
        free-space cost-to-go to the exit. This fills the return leg from the
        switch (full leg, difficulty 1) down to one hop from the exit (difficulty
        0 — an on-policy-discoverable win that finally gives a real reward
        gradient). The two sources are merged and sub-sampled.

        Prefixes are replayed faithfully on the env's own physics on reset (see
        ``NppEnvironment._maybe_apply_start_state``), so a survivor always
        re-activates the switch (and the harvested suffix re-traverses the return
        leg) there too."""
        import nclone_native

        # Phase 1: discover prefixes that reach the switch position.
        switch_cur = cls.generate(map_data, goal_kind="switch", n_states=n_states,
                                  max_iter=max_iter, kext=kext, seed=seed, log=log)
        if not switch_cur.states:
            return cls(goal_kind="exit")

        fs = FreeSpaceDistanceSource(list(map_data))
        sim = nclone_native.Simulator.load(list(map_data))

        def switch_consumed() -> bool:
            # The exit switch (entity_type 4) is ``active`` until touched; once
            # the ninja activates it the flag flips to False and the exit opens.
            return any(e.entity_type == 4 and not e.active
                       for e in sim.get_state().entities)

        def exit_progress(x, y) -> float:
            # Raw exit-leg cost-to-go; INFINITE == the exit is unreachable from
            # (x, y). Callers DROP inf states (below) so a post-switch start is
            # never placed outside the exit's reachable region (RFCL invariant).
            return fs.cost_to_go(x, y, True)  # switch_activated=True -> exit leg

        # Phase 2: validate each prefix on FULL physics; keep switch-consuming
        # ones (re-tagged by exit cost-to-go) and track the best-connected one as
        # the anchor for the exit-guided tree.
        coarse: List[Tuple[List[int], float]] = []
        anchor_prefix: Optional[List[int]] = None
        anchor_cost = 1e18
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
            if not ok or not switch_consumed():
                continue
            px, py = sim.get_ninja_pos()
            cost = exit_progress(px, py)
            if not math.isfinite(cost):
                continue  # exit unreachable from here -> never a valid start
            coarse.append((list(prefix), cost))
            if cost < anchor_cost:
                anchor_cost, anchor_prefix = cost, list(prefix)

        if not coarse:
            log("  post-switch curriculum: 0 switch-consuming prefixes found")
            return cls(goal_kind="exit")

        # Phase 3: harvest the exit-guided tree from the best anchor's post-switch
        # state -> graduated near-exit..full-leg states (the missing coverage).
        tree_states: List[Tuple[List[int], float]] = []
        if anchor_prefix is not None:
            try:
                nodes, _exit_xy = _exit_guided_tree(
                    map_data, fs, anchor_prefix, max_iter=b_max_iter,
                    kext=kext, seed=seed, log=log)
            except Exception as exc:  # noqa: BLE001 - never break the build
                nodes = None
                log(f"  post-switch tree failed: {exc}")
            if nodes:
                for ni in range(1, len(nodes)):  # skip root (== anchor, in coarse)
                    nx, ny = nodes[ni][1], nodes[ni][2]
                    ecost = exit_progress(nx, ny)
                    if not math.isfinite(ecost):
                        continue  # exit unreachable from here -> never a valid start
                    suffix = _node_prefix(nodes, ni, kext)
                    tree_states.append((anchor_prefix + suffix, ecost))

        tagged = coarse + tree_states
        tagged.sort(key=lambda t: t[1])
        if len(tagged) > n_states:
            idx = np.linspace(0, len(tagged) - 1, n_states).astype(int)
            tagged = [tagged[i] for i in idx]
        max_p = max((p for _, p in tagged), default=1.0) or 1.0
        log(f"  post-switch curriculum: {len(coarse)} switch states + "
            f"{len(tree_states)} return-leg tree states -> {len(tagged)} total, "
            f"exit-progress 0..{max_p:.0f}")
        return cls(states=tagged, max_progress=max_p, goal_kind="exit")
