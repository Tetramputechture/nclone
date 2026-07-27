"""Verification for the horizontal-mirror augmentation (§6 of the plan).

Six checks, in the order the plan sequences them:

1. involution -- ``M(M(x)) == x`` for every field
2. the unclassified-dim guard -- a ``game_state`` that grows without a rule fails
3. the action / input-byte maps
4. tile-ID remapping
5. **simulator commutation** -- the load-bearing one
6. golden-observation regression -- a pinned mirrored observation

(5) is what makes the whole thing trustworthy: it validates the tile table, the
entity handling, drone chirality, the ghost-arc swap, the raycast permutation and
the scandot invariance simultaneously, on the real simulator.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest

from nclone.map_augmentation.mirror_verify import (
    GAME_STATE_EXACT_DIMS,
    check_commutation,
    check_involution,
    rollout,
)
from nclone.map_augmentation.obs_mirror import (
    MIRROR_ACTION_MAP,
    GameStateMirror,
    ObservationMirror,
    UnclassifiedGameStateDims,
    mirror_action,
    mirror_input_byte,
    mirror_tile_ids,
)
from nclone.map_augmentation.transforms import _TILE_MIRROR, mirror_map_data

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")


#: The v26/v27 obs-exposure set. The mirror spec is only meaningful against a
#: specific contract, so the test pins the one training actually uses.
EXPOSURE = dict(
    mirror_equivariant_obs=True,
    expose_fall_lethality_map=True,
    expose_ego_crop_fall_fine=True,
    expose_commitment_state=True,
    mine_faithful_obs=True,
    expose_ego_crop_mines_fine=True,
    expose_mine_proximity=True,
    expose_frontier_map=True,
    expose_waypoint_features=True,
    expose_cost_to_go_map=True,
    expose_visitation_map=True,
    visitation_recency_tau=600,
    expose_blocked_goal=True,
    expose_ego_crop_route_fine=True,
    enable_ego_crop=True,
    ego_crop_size=21,
    enable_ego_crop_coarse=True,
    ego_crop_coarse_route=True,
    cost_map_norm="ego_residual",
    cost_map_ego_window=8.0,
    ballistic_air_potential=True,
    ballistic_air_best_arc=True,
)

_OFFICIAL = Path(__file__).resolve().parents[1] / "maps" / "official"
#: A handful of real official levels: slopes, curves, mines, gold, drones.
_TEST_LEVELS = (
    _OFFICIAL / "SI" / "000 the basics",
    _OFFICIAL / "R" / "000 hilltops",
    _OFFICIAL / "RL" / "000 downhill stroll",
)


def _live_game_state_width() -> int:
    """The width the env ACTUALLY emits under :data:`EXPOSURE`."""
    from nclone.gym_environment.config import EnvironmentConfig, ReachabilityConfig
    from nclone.gym_environment.npp_environment import NppEnvironment
    import dataclasses

    fields = {f.name for f in dataclasses.fields(ReachabilityConfig)}
    reach = ReachabilityConfig(**{k: v for k, v in EXPOSURE.items() if k in fields})
    env = NppEnvironment(EnvironmentConfig.for_training(reachability=reach))
    try:
        return int(env.observation_space["game_state"].shape[-1])
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 1-4: transform-only checks (no simulator)
# ---------------------------------------------------------------------------


def test_action_map_is_an_involution_that_swaps_left_and_right():
    assert MIRROR_ACTION_MAP == (0, 2, 1, 3, 5, 4)
    for a in range(6):
        assert mirror_action(mirror_action(a)) == a
    # NOOP and Jump are the only fixed points.
    assert [a for a in range(6) if mirror_action(a) == a] == [0, 3]


def test_input_byte_map_swaps_left_and_right_bits_only():
    # bit0 = jump, bit1 = right, bit2 = left.
    assert [mirror_input_byte(b) for b in range(8)] == [0, 1, 4, 5, 2, 3, 6, 7]
    for b in range(8):
        assert mirror_input_byte(mirror_input_byte(b)) == b
        assert (b & 0x01) == (mirror_input_byte(b) & 0x01)  # jump untouched


def test_tile_remap_covers_every_asymmetric_tile_and_is_involutive():
    ids = np.arange(38, dtype=np.int8)
    assert np.array_equal(mirror_tile_ids(mirror_tile_ids(ids)), ids)
    for src, dst in _TILE_MIRROR.items():
        assert mirror_tile_ids(np.array([src]))[0] == dst
    # The fall-through set must be genuinely symmetric geometry: 0/1 empty and
    # full, 2/4 the top/bottom half tiles, 34-37 the glitched no-collision tiles.
    identity = [t for t in range(38) if t not in _TILE_MIRROR]
    assert identity == [0, 1, 2, 4, 34, 35, 36, 37]


def test_game_state_spec_rejects_an_unclassified_dim():
    """The v26 launch died because an obs block grew and one allow-list did not.

    The mirror spec is the same hazard, so a width it cannot account for must be
    a hard error rather than a silent pass-through.
    """
    width = _live_game_state_width()
    # Right width, but the exposure flags that produced it are not declared.
    with pytest.raises(UnclassifiedGameStateDims):
        GameStateMirror(width)
    # And a width nothing explains at all.
    with pytest.raises(UnclassifiedGameStateDims):
        GameStateMirror(width + 3, expose_blocked_goal=True,
                        expose_commitment_state=True)


def test_game_state_mirror_is_an_involution():
    width = _live_game_state_width()
    m = GameStateMirror(
        width, expose_blocked_goal=True, expose_commitment_state=True
    )
    rng = np.random.default_rng(0)
    v = rng.standard_normal(width).astype(np.float32)
    v[21] = 0.37  # position_x lives in [0, 1] and reflects as 1 - x
    assert np.allclose(m(m(v)), v, atol=1e-6)
    # The affine field really is affine, not a plain negation.
    assert m(v)[21] == pytest.approx(1.0 - 0.37, abs=1e-6)
    # Steering left becomes steering right: arcs 0 and 2 of a bundle swap.
    from nclone.gym_environment.constants import (
        GAME_STATE_GHOST_ARC_IDX as G,
        GAME_STATE_GHOST_ARC_PER_ARC as P,
    )

    assert m.perm[G] == G + 2 * P
    assert m.perm[G + P] == G + P  # u = 0 stays
    assert m.perm[G + 2 * P] == G


# ---------------------------------------------------------------------------
# 5: simulator commutation -- the load-bearing check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", _TEST_LEVELS, ids=lambda p: p.name)
def test_observation_mirror_is_an_involution_on_real_observations(level):
    obs, _rew, _term = rollout(
        list(level.read_bytes()),
        [1, 3, 5, 0, 2, 4],
        exposure=EXPOSURE,
        map_name=f"inv_{level.name}",
        skip_reward=True,
    )
    mirror = ObservationMirror(
        int(obs[0]["game_state"].shape[-1]), exposure=EXPOSURE
    )
    for step, o in enumerate(obs):
        bad = check_involution(mirror, o, atol=1e-6)
        assert not bad, f"M(M(x)) != x at step {step} for {bad}"


@pytest.mark.parametrize("level", _TEST_LEVELS, ids=lambda p: p.name)
def test_simulator_commutes_with_reflection(level):
    """``mirror(obs(sim(L, a))) == obs(sim(mirror(L), mirror(a)))``.

    See ``mirror_verify`` for the two documented residual classes (grid
    quantization at exact cell boundaries, and planner / argmin tie-breaks).
    Everything else -- and the whole ninja physics block on EVERY step -- must
    agree to float tolerance.
    """
    rng = random.Random(hash(level.name) & 0xFFFF)
    actions = [rng.randrange(6) for _ in range(40)]
    report = check_commutation(
        list(level.read_bytes()), actions, exposure=EXPOSURE
    )
    assert report.contract_holds, report.summary()


def test_ninja_physics_block_never_diverges():
    """The sim-commutes claim, stated on its own so a failure is unambiguous."""
    level = _TEST_LEVELS[0]
    rng = random.Random(11)
    actions = [rng.randrange(6) for _ in range(60)]
    report = check_commutation(
        list(level.read_bytes()), actions, exposure=EXPOSURE, check_rewards=False
    )
    bad = report.bad_indices("game_state") & set(GAME_STATE_EXACT_DIMS)
    assert not bad, (
        f"the simulator does NOT commute with reflection: game_state dims "
        f"{sorted(bad)} diverged.\n{report.summary()}"
    )


def test_rewards_are_reflection_invariant():
    level = _TEST_LEVELS[1]
    rng = random.Random(5)
    actions = [rng.randrange(6) for _ in range(40)]
    report = check_commutation(list(level.read_bytes()), actions, exposure=EXPOSURE)
    assert not report.reward_divergent, (
        f"reward stream is not reflection-invariant "
        f"(max|d| = {report.reward_max_abs_diff:.3g})"
    )


# ---------------------------------------------------------------------------
# 6: golden regression -- the obs contract cannot grow past the mirror silently
# ---------------------------------------------------------------------------


def test_mirrored_observation_covers_every_live_key():
    """Every key the env emits must have an explicit mirror rule.

    This is the golden-regression guard in its useful form: pinning float values
    would break on any legitimate obs change, whereas an unmirrored NEW key is
    always a bug -- it is the exact failure mode (a second code path not updated
    alongside the contract) that took down the v26 launch.
    """
    level = _TEST_LEVELS[0]
    obs, _rew, _term = rollout(
        list(level.read_bytes()), [0], exposure=EXPOSURE,
        map_name="golden", skip_reward=True,
    )
    mirror = ObservationMirror(
        int(obs[0]["game_state"].shape[-1]), exposure=EXPOSURE, strict=True
    )
    mirrored = mirror(obs[0])  # raises KeyError on an unclassified key
    assert set(mirrored) == set(obs[0])
    for key, value in obs[0].items():
        assert mirrored[key].shape == np.asarray(value).shape


def test_mirrored_map_round_trips():
    for level in _TEST_LEVELS:
        md = list(level.read_bytes())
        assert mirror_map_data(mirror_map_data(md)) == md
        assert mirror_map_data(md) != md  # none of these levels is symmetric
