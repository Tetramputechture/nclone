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
            c = fs.cost_to_go(x, y, goal_kind == "exit")
            return c if math.isfinite(c) else fs._d_max_switch

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

        tagged = [(prefix(i), progress(nodes[i][1], nodes[i][2]))
                  for i in range(1, len(nodes))]
        tagged.sort(key=lambda t: t[1])
        if len(tagged) > n_states:
            idx = np.linspace(0, len(tagged) - 1, n_states).astype(int)
            tagged = [tagged[i] for i in idx]
        max_p = max((p for _, p in tagged), default=1.0) or 1.0
        log(f"  curriculum: {len(tagged)} states, progress 0..{max_p:.0f}")
        return cls(states=tagged, max_progress=max_p, goal_kind=goal_kind)
