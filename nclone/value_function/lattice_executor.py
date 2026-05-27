"""Production API for the lattice + micro-search route planner/executor.

This is the thin, importable entry point to the planner validated in
``tools/lattice_audit``: backward-Dijkstra surface-graph ``d*`` over the jointly
optimal switch→exit tree → per-arc route → closed-loop + micro-search executor
that emits a **winning per-frame input tape**.  Because native↔Python physics are
bit-identical (validated by ``route_executor_check``), the tape replays
deterministically on the gym ``nsim`` engine.

Consumers:
  - completion videos (``tools/lattice_video``),
  - open-loop tape replay in ``npp-rl`` (the static-level deploy path; the
    per-frame tape is frame-skip 1, so an agent controller running on a
    frame-skip>1 env should re-plan with ``macro=frame_skip`` — TODO).

Example::

    from nclone.value_function.lattice_executor import plan_winning_tape
    res = plan_winning_tape("nclone/maps/official/SL/001 take the easy way out")
    if res["won"]:
        tape = res["tape"]   # list[(h_input, jump_input)] per frame, from spawn
"""

from __future__ import annotations

from typing import List, Optional, Tuple


def plan_winning_tape(map_path: str, workers: int = 4,
                      use_cache: bool = True) -> dict:
    """Plan the lattice route and micro-search it into a winning input tape.

    Returns the planner result dict with keys ``won``, ``died``, ``frames``,
    ``switch_frame``, ``min_clear``, ``n_arcs``, ``n_reached`` and ``tape`` (the
    per-frame ``(h_input, jump_input)`` sequence from spawn — empty when the
    level is unreachable on the lattice).
    """
    # Import lazily: the validated implementation currently lives in the audit
    # tool (which also carries the diagnostic open-loop / closed-loop replays).
    # Keeping this facade lets production code depend on a stable API while the
    # heavier executor extraction happens incrementally.
    from .tools.lattice_audit import LatticeAudit

    audit = LatticeAudit(map_path, workers=workers)
    return audit.plan_tape(use_cache=use_cache)


def tape_to_actions(tape: List[Tuple[int, int]]) -> List[int]:
    """Map a per-frame ``(h_input, jump_input)`` tape to 6-action env indices.

    0=NOOP 1=LEFT 2=RIGHT 3=JUMP 4=JUMP+LEFT 5=JUMP+RIGHT — matching the nclone
    BaseNppEnvironment / npp-rl controllers.
    """
    out: List[int] = []
    for (h, j) in tape:
        if j:
            out.append(4 if h < 0 else (5 if h > 0 else 3))
        else:
            out.append(1 if h < 0 else (2 if h > 0 else 0))
    return out
