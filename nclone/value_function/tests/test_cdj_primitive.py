"""Validation tests for the Corner Double Jump (CDJ) composite primitive.

Grounded in the validated human replay
``datasets/human_replays/20260525_130015_di-required.replay`` (see
``docs/CDJ_REPLAY_HANDOFF.md``). The tests assert:

  1. the parameterized generator reproduces the replay's trick inputs exactly;
  2. the canonical CDJ climbs to the di-required switch height (~200px) and the
     full recorded replay still replays to a WIN (regression on the recorder fix
     + sim determinism);
  3. the buffer-intersection MECHANISM: a wall jump fires first while
     ``floor_buffer`` survives it, then a floor jump stacks (|yspeed| jumps).
"""

import os
import struct
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from nclone.nplay_headless import NPlayHeadless
from nclone.value_function.cdj_primitive import (
    DI_REQUIRED_CDJ,
    CDJParams,
    cdj_core_inputs,
    cdj_input_sequence,
    execute_cdj,
    simulate_cdj,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPLAY = _REPO_ROOT / "datasets/human_replays/20260525_130015_di-required.replay"


def _decode_byte(b: int):
    return (1 if b & 2 else (-1 if b & 4 else 0), b & 1)


def _load_replay():
    """Return (map_data, decoded_inputs, success_flag) from the .replay file."""
    data = _REPLAY.read_bytes()
    _ver, mlen, ilen = struct.unpack("<III", data[0:12])
    success = data[12]
    map_data = list(data[13 : 13 + mlen])
    inputs = [_decode_byte(b) for b in data[13 + mlen : 13 + mlen + ilen]]
    return map_data, inputs, success


def _new_env(map_data):
    nph = NPlayHeadless(render_mode="rgb_array")
    nph.load_map_from_map_data(map_data)
    return nph


@unittest.skipUnless(_REPLAY.exists(), f"replay not found: {_REPLAY}")
class TestCDJPrimitive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map_data, cls.replay_inputs, cls.success = _load_replay()

    # 1. Generator fidelity --------------------------------------------------
    def test_generator_matches_replay_trick_inputs(self):
        """The canonical generator reproduces the replay through the apex."""
        seq = cdj_input_sequence(DI_REQUIRED_CDJ)
        # approach(8) + floor jump(1) + climb(7) + core(press+release+hold 74) = 92
        self.assertEqual(len(seq), 92)
        self.assertEqual(seq, self.replay_inputs[:92])

    def test_core_structure(self):
        """The trick core is press, release, then hold (the user's pattern)."""
        core = cdj_core_inputs(approach_dir=1, hold_frames=5)
        self.assertEqual(core[0], (1, 1))   # press: buffer the jump
        self.assertEqual(core[1], (1, 0))   # release: buffered WALL jump fires
        self.assertEqual(core[2:], [(1, 1)] * 5)  # press+hold: FLOOR jump stacks
        self.assertEqual(len(core), 7)

    # 2. Outcome: climb + win ------------------------------------------------
    def test_canonical_cdj_climbs_to_switch_height(self):
        """A single CDJ from spawn climbs ~200px (reaches the di-required switch).

        This contradicts the earlier brute-force cap of +149.7px: the human's
        floor-jump-up-the-wall approach + 74-frame hold reaches the switch with
        one CDJ (no chaining needed).
        """
        nph = _new_env(self.map_data)
        res = simulate_cdj(nph, DI_REQUIRED_CDJ)
        self.assertFalse(res.died, "CDJ composite should not die mid-trick")
        self.assertGreaterEqual(
            res.climb_px, 200.0, f"CDJ only climbed {res.climb_px:.1f}px"
        )

    def test_full_replay_wins(self):
        """The full recorded replay replays 1-byte-per-tick to a WIN."""
        self.assertEqual(self.success, 1, "replay was not recorded as a live win")
        nph = _new_env(self.map_data)
        won = False
        for h, j in self.replay_inputs:
            nph.tick(h, j)
            if nph.ninja_has_won():
                won = True
                break
            if nph.ninja_has_died():
                break
        self.assertTrue(won, "validated CDJ replay should replay to a win")

    # 3. Buffer-intersection mechanism --------------------------------------
    def test_buffer_intersection_stacks_impulses(self):
        """Wall jump fires first (floor_buffer survives), then floor jump stacks.

        Verifies the WCu mechanism end-to-end on the real sim: at the wall-jump
        frame ``floor_buffer`` is still live (>= 0), and the immediately
        following floor jump makes ``yspeed`` markedly more negative (the stack),
        producing an upward speed faster than any single jump (~-2.75).
        """
        nph = _new_env(self.map_data)
        n = nph.sim.ninja
        log = []  # (xspeed, yspeed, floor_buffer, state) per tick
        for h, j in cdj_input_sequence(DI_REQUIRED_CDJ):
            nph.tick(h, j)
            log.append((n.xspeed, n.yspeed, n.floor_buffer, n.state))

        # The wall jump pushes off the right wall: xspeed flips strongly negative.
        wall_idx = next(
            (i for i, e in enumerate(log) if e[0] < -0.5 and e[3] == 3), None
        )
        self.assertIsNotNone(wall_idx, "no wall jump detected (xspeed never < -0.5)")

        # floor_buffer must SURVIVE the wall jump (wall_jump does not reset it).
        self.assertGreaterEqual(
            log[wall_idx][2], 0, "floor_buffer did not survive the wall jump"
        )

        # The next frame is the stacking floor jump: yspeed gets markedly faster.
        self.assertLess(wall_idx + 1, len(log))
        ys_wall = log[wall_idx][1]
        ys_stack = log[wall_idx + 1][1]
        self.assertLess(
            ys_stack, ys_wall - 0.5,
            f"no stack: yspeed {ys_wall:.2f} -> {ys_stack:.2f}",
        )

        # The stacked upward speed exceeds any single jump impulse (~-2.75).
        peak_up = min(e[1] for e in log)
        self.assertLessEqual(
            peak_up, -3.5, f"peak upward speed {peak_up:.2f} too weak to be a stack"
        )

    # 4. Adaptive closed-loop primitive ------------------------------------
    def test_adaptive_cdj_detects_corner_and_climbs(self):
        """The sim-feedback CDJ auto-detects the corner and climbs to the switch.

        No hard-coded climb length: the climb stops on the first re-grounded
        frame after wall contact, so the trick generalizes across wall heights.
        """
        nph = _new_env(self.map_data)
        ex = execute_cdj(nph, approach_dir=1, hold_frames=74)
        self.assertTrue(ex.corner_found, "adaptive climb did not detect the corner")
        self.assertFalse(ex.result.died)
        self.assertGreaterEqual(ex.result.climb_px, 200.0)

    def test_adaptive_cdj_matches_validated_tape(self):
        """On di-required the adaptive detector reproduces the validated tape."""
        nph = _new_env(self.map_data)
        ex = execute_cdj(nph, approach_dir=1, hold_frames=74)
        self.assertEqual(ex.climb_frames, 7)
        self.assertEqual(ex.tape, cdj_input_sequence(DI_REQUIRED_CDJ))

    def test_adaptive_cdj_with_traverse_wins(self):
        """Adaptive CDJ + the replay's post-apex traverse replays to a WIN."""
        nph = _new_env(self.map_data)
        traverse = self.replay_inputs[92:]
        ex = execute_cdj(nph, approach_dir=1, hold_frames=74, post_inputs=traverse)
        self.assertTrue(ex.result.won, "adaptive CDJ + traverse should win")


_DI_MAP = _REPO_ROOT / "nclone/test_maps/di-required"


@unittest.skipUnless(_DI_MAP.exists(), f"map not found: {_DI_MAP}")
class TestCDJExecutorHook(unittest.TestCase):
    """The lattice executor falls back to the CDJ trick + forward micro-search
    on di-required (air switch the lattice cannot route) and wins autonomously."""

    def test_plan_tape_wins_di_required_via_cdj(self):
        from nclone.value_function.tools.lattice_audit import LatticeAudit

        audit = LatticeAudit(str(_DI_MAP))
        res = audit.plan_tape(use_cache=True)
        self.assertTrue(res.get("won"), "executor did not complete di-required")
        self.assertEqual(res.get("via"), "cdj", "should win via the CDJ fallback")
        # Mine-safe: min clearance must exceed the deadly-mine kill radius (~14px).
        self.assertGreater(res.get("min_clear", 0.0), 14.0)

        # The emitted tape replays standalone to a win on a fresh native sim.
        import nclone_native as N
        from nclone.value_function.tools.physics_route_spike import _load_map_data

        sim = N.Simulator.load(_load_map_data(str(_DI_MAP)))
        won = False
        for h, j in res["tape"]:
            sim.tick(h, j)
            if sim.has_won():
                won = True
                break
            if sim.has_died():
                break
        self.assertTrue(won, "emitted tape did not replay to a win")


class TestCDJChain(unittest.TestCase):
    """The lattice ``d*`` composes TWO Corner Double Jumps in sequence on the
    ``cdj-chain`` staircase: spawn --CDJ#1--> mid platform (switch) --CDJ#2-->
    top shelf (exit).

    Guards the two fixes that unlocked chaining:
      1. reachable-launch enumeration — a mid-route trick edge launches from a
         forward-reachable node (not the wall-edge phantom), so backward
         ``d_exit`` propagates through it to the switch (otherwise the switch
         reads as unreachable and ``compute_two_phase`` raises);
      2. shelf geometry — CDJ#2's near-vertical apex lands ON the shelf (left
         edge at the landing column), not one tile short on its left face.

    Integration-scale: the lattice precompute runs in-process (cached in /tmp),
    so the first run is ~heavy; subsequent runs reuse the cache.
    """

    def test_chain_route_composes_two_cdjs_and_wins(self):
        from nclone.value_function.tools.cdj_chain_map import build_cdj_chain_map
        from nclone.value_function.tools.lattice_audit import LatticeAudit
        import nclone_native as N

        md = build_cdj_chain_map()
        audit = LatticeAudit(map_data=md, name="cdj-chain")

        # (1) The jointly-optimal spawn route must compose TWO trick (CDJ) arcs.
        pc = audit.precompute(use_cache=True)
        phase0, phase1, _info = audit.build_route(pc)
        route = phase0 + phase1
        n_trick = sum(1 for a in route if getattr(a, "trick_tape", None))
        self.assertEqual(n_trick, 2, f"expected 2 composed CDJ arcs, got {n_trick}")

        # (2) End-to-end: the executor wins (switch first) and the emitted tape
        # replays standalone to a win on a fresh native sim.
        res = audit.plan_tape(use_cache=True)
        self.assertTrue(res.get("won"), "chain executor did not win")
        self.assertNotEqual(
            res.get("via"), "cdj",
            "should win via the composed lattice trick edges, not the "
            "spawn-only CDJ fallback")
        self.assertIsNotNone(res.get("switch_frame"), "switch not collected")

        sim = N.Simulator.load(md)
        won = False
        for h, j in res["tape"]:
            sim.tick(h, j)
            if sim.has_won():
                won = True
                break
            if sim.has_died():
                break
        self.assertTrue(won, "emitted chain tape did not replay to a win")


if __name__ == "__main__":
    unittest.main()
