"""Sim-replay verification for planner-generated action sequences.

The planner produces an action sequence derived from
:mod:`nclone.planning.optimal_route.physics` (a faithful but simplified
model of :class:`nclone.ninja.Ninja`). To guarantee that the reported
``optimal_frames`` count is achievable in the real simulator, we replay the
actions through a fresh :class:`nclone.nsim.Simulator` and record the
frame count at which the ninja enters the "celebrating" (won) state.

Notes
-----

* ``Simulator.tick(hor_input, jump_input)`` increments ``sim.frame`` once per
  call. After a successful plan, ``sim.frame`` equals the number of frames
  consumed, and ``sim.ninja.has_won()`` becomes True on the tick that
  triggered the win condition.

* If the action sequence completes before the ninja wins, we continue
  ticking with ``(0, 0)`` inputs for a small tail grace period. This
  absorbs sub-pixel timing differences between the planner sim and the
  real sim (the planner uses a near-identical integrator, but collision
  clamping is simplified).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ...nsim import Simulator
from ...sim_config import SimConfig

from .types import ActionFrame


REPLAY_TAIL_FRAMES = 30


def replay_and_verify(
    map_data: List[int],
    action_sequence: List[ActionFrame],
    max_tail_frames: int = REPLAY_TAIL_FRAMES,
) -> Tuple[bool, Optional[int]]:
    """Replay ``action_sequence`` in a fresh simulator and return verification.

    Returns a tuple ``(won, frames)`` where:

    * ``won`` is ``True`` if the ninja reached the "celebrating" state.
    * ``frames`` is the number of simulator ticks consumed when the win
      triggered, or ``None`` when the replay did not end in a win.
    """
    sim = Simulator(SimConfig(enable_anim=False, log_data=False, debug=False))
    sim.load(list(map_data))

    for hor, jump in action_sequence:
        sim.tick(int(hor), 1 if jump else 0)
        if sim.ninja.has_won():
            return True, sim.frame
        if sim.ninja.has_died():
            return False, None

    for _ in range(max_tail_frames):
        sim.tick(0, 0)
        if sim.ninja.has_won():
            return True, sim.frame
        if sim.ninja.has_died():
            return False, None

    return False, None
