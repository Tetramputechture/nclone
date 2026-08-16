"""Which locked doors are NECESSARY, and can the necessary set actually be opened?

`solve_door_dependencies` has to answer two different questions about a door and
it used to conflate them:

  SUFFICIENCY  -- with this set of doors open, is the exit reachable?
  FEASIBILITY  -- starting from spawn with everything shut, can this set be
                  opened at all?

The old rule greedily dropped any door whose removal still left the exit
reachable, which tests only sufficiency: `exit_ok(cand)` assumes `cand` is open
without asking how you would have opened it. On a prerequisite chain that drops
the prerequisite, and the order loop then declares a SOLVABLE level unsolvable.

These tests pin the behaviour on the shapes real multi-door levels have:
prerequisite chains, irrelevant doors, several valid orderings, and doors that
gate other doors' switches rather than the exit.
"""

from __future__ import annotations

import numpy as np
import pytest

from nclone.value_function.door_dependency import (
    solve_door_dependencies,
    DoorDescriptor,
)

# Fixed tags so a toy oracle can identify what it is being asked about.
SPAWN = (0.0, 0.0)
EXIT_DOOR = (900.0, 0.0)
EXIT_SWITCH = (910.0, 0.0)


class ReachOracle:
    """Reachability is a pure function of the open-door set.

    `rule(target, open_set) -> bool` decides everything, so a test states its
    level as a truth table rather than as geometry.
    """

    def __init__(self, rule, switches):
        self._rule = rule
        self._switches = {(round(x), round(y)): i for i, (x, y) in enumerate(switches)}
        self.configs = set()

    def config(self, open_set):
        key = frozenset(open_set)
        self.configs.add(key)
        return key

    def reachable_from(self, cfg, x, y):
        return cfg

    def reaches(self, cfg, reach, x, y, kind):
        tag = (round(x), round(y))
        if tag == (round(EXIT_DOOR[0]), round(EXIT_DOOR[1])):
            return self._rule("exit", set(reach))
        if tag == (round(EXIT_SWITCH[0]), round(EXIT_SWITCH[1])):
            return self._rule("switch", set(reach))
        return self._rule(self._switches[tag], set(reach))

    def leg_field(self, cfg, x, y, kind):
        return np.zeros((4, 4), dtype=np.float32)


def _solve(rule, n_doors):
    switches = [(10.0 * (i + 1), 0.0) for i in range(n_doors)]
    doors = [DoorDescriptor(switch_x=x, switch_y=y) for x, y in switches]
    oracle = ReachOracle(rule, switches)
    plan = solve_door_dependencies(
        None, doors,
        spawn_x=SPAWN[0], spawn_y=SPAWN[1],
        switch_x=EXIT_SWITCH[0], switch_y=EXIT_SWITCH[1],
        exit_x=EXIT_DOOR[0], exit_y=EXIT_DOOR[1],
        oracle=oracle,
    )
    return plan, oracle


def test_prerequisite_chain_is_solvable_and_both_doors_are_necessary():
    """A opens the way to B's switch; B opens the exit. Both are necessary.

    The old rule dropped A -- the exit is reachable "with B open" -- and then
    reported the level unsolvable with reason "unexpected dependency cycle".
    """
    def rule(target, open_):
        if target == 0:                      # A's switch: from spawn
            return True
        if target == 1:                      # B's switch: needs A
            return 0 in open_
        return 1 in open_                    # exit + exit switch: need B

    plan, _ = _solve(rule, 2)
    assert plan.solvable, plan.reason
    assert set(plan.necessary_order) == {0, 1}
    assert plan.necessary_order.index(0) < plan.necessary_order.index(1), (
        "A must be opened before B -- B's switch is unreachable until A is open"
    )


def test_an_irrelevant_door_is_not_credited():
    """Door 1 gates nothing. Rewarding it would pay for a detour."""
    def rule(target, open_):
        if target in (0, 1):
            return True                      # both switches reachable from spawn
        return 0 in open_                    # only door 0 gates the exit

    plan, _ = _solve(rule, 2)
    assert plan.solvable, plan.reason
    assert plan.necessary_order == [0]


def test_a_door_that_gates_only_another_switch_is_still_necessary():
    """Door 0 does not touch the exit route -- it only unlocks door 1's switch."""
    def rule(target, open_):
        if target == 0:
            return True
        if target == 1:
            return 0 in open_
        # The exit needs door 1. Door 0 alone does nothing for it.
        return 1 in open_

    plan, _ = _solve(rule, 2)
    assert plan.solvable, plan.reason
    assert set(plan.necessary_order) == {0, 1}


def test_no_doors_needed_when_the_exit_is_open_from_spawn():
    def rule(target, open_):
        return True                          # everything reachable, always

    plan, _ = _solve(rule, 3)
    assert plan.solvable, plan.reason
    assert plan.necessary_order == []


def test_two_independent_gates_are_both_necessary_in_either_order():
    """Two doors each independently gate the exit; either order works."""
    def rule(target, open_):
        if target in (0, 1):
            return True
        return {0, 1} <= open_

    plan, _ = _solve(rule, 2)
    assert plan.solvable, plan.reason
    assert set(plan.necessary_order) == {0, 1}


def test_genuinely_unreachable_exit_is_still_reported_unsolvable():
    """The fix must not turn every level solvable."""
    def rule(target, open_):
        if target == 0:
            return True
        return False                         # exit never reachable

    plan, _ = _solve(rule, 1)
    assert not plan.solvable
    assert "every openable door" in plan.reason


def test_a_switch_behind_its_own_door_is_unopenable_and_not_required():
    """Door 1's switch sits behind door 1. It can never be opened.

    The exit does not need it, so the level is solvable and door 1 is not
    credited -- the closure simply never contains it.
    """
    def rule(target, open_):
        if target == 0:
            return True
        if target == 1:
            return 1 in open_                # circular: unreachable forever
        return 0 in open_                    # exit needs only door 0

    plan, _ = _solve(rule, 2)
    assert plan.solvable, plan.reason
    assert plan.necessary_order == [0]


def test_necessity_costs_a_linear_number_of_configurations():
    """O(n) oracle configs, not O(n^2).

    Official door levels average 7.5 locked doors but the tail reaches 380 on one
    map, where a per-candidate feasibility re-check would be ~144k floods at
    ~0.07 s each. This is the guard on that.
    """
    N = 12

    def rule(target, open_):
        if target == "exit" or target == "switch":
            return 0 in open_
        return True

    plan, oracle = _solve(rule, N)
    assert plan.solvable, plan.reason
    assert plan.necessary_order == [0]
    # Closure pass + one config per door + the ordering/leg passes. Comfortably
    # linear; O(n^2) would be >=144 here.
    assert len(oracle.configs) <= 4 * N, (
        f"{len(oracle.configs)} configurations for {N} doors is not linear"
    )
