"""Every obs-affecting exposure flag must survive the ReplayExecutor allow-list.

``ReplayExecutor._OBS_EXPOSURE_FIELDS`` is an allow-list: a reachability flag not
named in it is silently dropped when demos are re-simulated. When the flag grows
``game_state``, the demo obs is then NARROWER than the live policy space and the
first DAPG minibatch dies deep inside the hazard encoder's FiLM with

    RuntimeError: mat1 and mat2 shapes cannot be multiplied (Nx263 and 269x64)

which names neither the flag nor the file that dropped it.

This has now happened six times — ``expose_blocked_goal``,
``expose_ego_crop_route_fine``, ``expose_wall_contact_tau``,
``expose_precision_state``, ``expose_time_phase``, and (v26)
``expose_commitment_state``, the last of which killed a launched 100M-step run
30k steps in. Each was fixed reactively by appending to the tuple. These tests
make the invariant enforceable instead: adding an obs-growing flag without
wiring it here fails in CI, not on the GPU.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

from nclone.gym_environment.config import EnvironmentConfig, ReachabilityConfig
from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.replay.replay_executor import ReplayExecutor


# Flags that grow game_state. Each entry is (flag, expected added dims).
GAME_STATE_GROWING = [
    ("expose_blocked_goal", 1),
    ("expose_wall_contact_tau", 1),
    ("expose_precision_state", NppEnvironment.PRECISION_STATE_DIM),
    ("expose_time_phase", NppEnvironment.TIME_PHASE_DIM),
    ("expose_commitment_state", NppEnvironment.COMMITMENT_STATE_DIM),
]

# Flags that add whole obs KEYS rather than game_state dims.
KEY_ADDING = [
    ("expose_cost_to_go_map", "cost_to_go_map"),
    ("expose_visitation_map", "visitation_map"),
    ("expose_frontier_map", "frontier_map"),
    ("expose_ego_crop_route_fine", "ego_crop_route_fine"),
    ("expose_fall_lethality_map", "fall_lethality_map"),
    ("expose_ego_crop_fall_fine", "ego_crop_fall_fine"),
]


@pytest.mark.parametrize("flag,_dims", GAME_STATE_GROWING)
def test_game_state_growing_flag_is_in_the_allow_list(flag, _dims):
    assert flag in ReplayExecutor._OBS_EXPOSURE_FIELDS, (
        f"{flag} grows game_state but is not in ReplayExecutor."
        f"_OBS_EXPOSURE_FIELDS, so demos will re-simulate at the OLD width and "
        f"the first DAPG minibatch will die on a mat1/mat2 shape mismatch"
    )


@pytest.mark.parametrize("flag,_key", KEY_ADDING)
def test_key_adding_flag_is_in_the_allow_list(flag, _key):
    assert flag in ReplayExecutor._OBS_EXPOSURE_FIELDS, (
        f"{flag} adds an obs key but is not in ReplayExecutor."
        f"_OBS_EXPOSURE_FIELDS, so every demo would train that stream on a "
        f"constant neutral prior"
    )


def _gs_dim(**reach) -> int:
    cfg = EnvironmentConfig.for_training(
        frame_skip=1, sim_backend="python", reachability=ReachabilityConfig(**reach)
    )
    env = NppEnvironment(config=cfg)
    return int(env.observation_space.spaces["game_state"].shape[-1])


@pytest.mark.parametrize("flag,dims", GAME_STATE_GROWING)
def test_flag_actually_grows_game_state_by_the_documented_amount(flag, dims):
    """Guards the table above from drifting away from the env."""
    assert _gs_dim(**{flag: True}) - _gs_dim() == dims


@pytest.mark.parametrize("flag,key", KEY_ADDING)
def test_flag_actually_adds_its_key(flag, key):
    cfg = EnvironmentConfig.for_training(
        frame_skip=1, sim_backend="python",
        reachability=ReachabilityConfig(**{flag: True}),
    )
    env = NppEnvironment(config=cfg)
    assert key in env.observation_space.spaces


def test_allow_list_covers_every_exposure_flag_the_trainer_sends():
    """The real invariant: whatever npp-rl puts in its obs_exposure dict must be
    understood here. Skips cleanly when npp-rl is not importable (nclone is used
    standalone too)."""
    try:
        from npp_rl.training.trainer import Trainer, TrainerConfig
    except Exception:  # pragma: no cover - npp-rl not on the path
        pytest.skip("npp-rl not importable")

    trainer = Trainer.__new__(Trainer)
    trainer.config = TrainerConfig(
        benchmark_root="/tmp", output_dir="/tmp", experiment_name="contract",
        **{f: True for f in (
            "expose_blocked_goal", "expose_wall_contact_tau",
            "expose_precision_state", "expose_time_phase",
            "expose_commitment_state", "expose_fall_lethality_map",
            "expose_ego_crop_fall_fine", "expose_ego_crop_route_fine",
        )}
    )
    sent = {k for k, v in trainer._obs_exposure().items() if v}
    known = set(ReplayExecutor._OBS_EXPOSURE_FIELDS)
    dropped = sent - known
    assert not dropped, (
        f"npp-rl sends obs_exposure flags nclone's ReplayExecutor silently "
        f"drops: {sorted(dropped)}. Demos will re-simulate under a DIFFERENT "
        f"observation contract than the live policy."
    )
